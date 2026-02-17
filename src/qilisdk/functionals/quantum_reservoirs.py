# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterator

import numpy as np

from qilisdk.analog import Hamiltonian, Schedule
from qilisdk.analog.hamiltonian import PauliOperator
from qilisdk.core import Domain, Parameter, QTensor, tensor_prod
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.types import RealNumber
from qilisdk.digital import Circuit, M
from qilisdk.functionals.functional import PrimitiveFunctional


class ReservoirInput(Parameter):
    """Input-only parameter used to inject layer-wise data into a reservoir program.

    This parameter behaves like a standard :class:`~qilisdk.core.variables.Parameter` but is
    always created as non-trainable, so optimizers do not update it.
    """

    def __init__(
        self,
        label: str,
        value: RealNumber,
        domain: Domain = Domain.REAL,
        bounds: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """Create a non-trainable reservoir input parameter.

        Args:
            label: Parameter label used to reference input values.
            value: Default numerical value.
            domain: Allowed numeric domain.
            bounds: Optional lower/upper parameter bounds.
        """
        super().__init__(
            label=label,
            value=value,
            domain=domain,
            bounds=bounds,
            trainable=False,
        )

    __hash__ = Parameter.__hash__


class ReservoirLayer(Parameterizable):
    """Single reservoir layer template.

    A reservoir layer can contain up to three ordered stages:
    ``pre_processing -> reservoir_dynamics -> post_processing``.
    It also stores the observables measured after each layer and optional qubits to reset.
    """

    def __init__(
        self,
        evolution_dynamics: Schedule,
        observables: list[QTensor | Hamiltonian | PauliOperator],
        input_encoding: Circuit | None = None,
        output_encoding: Circuit | None = None,
        qubits_to_reset: list[int] | None = None,
    ) -> None:
        """Build a reservoir pass description.

        Args:
            evolution_dynamics: Main dynamics block, either analog schedule or digital circuit.
            observables: Observables sampled at the end of the pass.
            input_encoding: Optional single-qubit pre-processing circuit.
            output_encoding: Optional single-qubit post-processing circuit.
            qubits_to_reset: Optional qubits reset between consecutive layers.
        """

        super().__init__()
        self._input_encoding: Circuit | None = input_encoding
        self._evolution_dynamics: Schedule = evolution_dynamics
        self._output_encoding: Circuit | None = output_encoding
        self._qubits_to_reset: list[int] | None = qubits_to_reset
        self._full_param_list: dict[str, Parameter] = {}
        self._nqubits = self._evolution_dynamics.nqubits

        if input_encoding:
            self._validate_input_encoding(input_encoding)

        if output_encoding:
            self._validate_output_encoding(output_encoding)

        self._qtensor_observables: list[QTensor] = []

        for observable in observables:
            if isinstance(observable, PauliOperator):
                self._qtensor_observables.append(observable.to_hamiltonian().to_qtensor(self._nqubits))
            elif isinstance(observable, Hamiltonian):
                self._qtensor_observables.append(observable.to_qtensor(self._nqubits))
            elif isinstance(observable, QTensor):
                self._qtensor_observables.append(self._process_qtensor(observable))

        self._observables = observables

    def _validate_input_encoding(self, pre_processing: Circuit) -> None:
        """Validate and normalize the optional pre-processing circuit.

        Raises:
            ValueError: If measurements are present, multi-qubit gates are used, or the
                circuit acts on more qubits than the reservoir dynamics.
        """
        if any(isinstance(g, M) for g in pre_processing.gates):
            raise ValueError("Pre-Processing Circuit can't contain measurements")
        if pre_processing.nqubits > self._evolution_dynamics.nqubits:
            raise ValueError("Pre-Processing Circuit acts on more qubits than defined by the reservoir dynamics.")
        if any(g.nqubits > 1 for g in pre_processing.gates):
            raise ValueError("Only single qubit gates are allowed in the pre-processing circuit.")
        if pre_processing.nqubits < self.nqubits:
            self._input_encoding = Circuit(self._nqubits)
            self._input_encoding.add(pre_processing.gates)

    def _validate_output_encoding(self, post_processing: Circuit) -> None:
        """Validate and normalize the optional post-processing circuit.

        Raises:
            ValueError: If measurements are present, multi-qubit gates are used, or the
                circuit acts on more qubits than the reservoir dynamics.
        """
        if any(isinstance(g, M) for g in post_processing.gates):
            raise ValueError("Post-Processing Circuit can't contain measurements")
        if post_processing.nqubits > self._evolution_dynamics.nqubits:
            raise ValueError("Post-Processing Circuit acts on more qubits than defined by the reservoir dynamics.")
        if any(g.nqubits > 1 for g in post_processing.gates):
            raise ValueError("Only single qubit gates are allowed in the post-processing circuit.")
        if post_processing.nqubits < self.nqubits:
            self._output_encoding = Circuit(self._nqubits)
            self._output_encoding.add(post_processing.gates)

    def _process_qtensor(self, observable: QTensor) -> QTensor:
        """Pad observable tensors with identities to match the reservoir width.

        Returns:
            Observable expanded (if needed) to match the number of reservoir qubits.

        Raises:
            ValueError: If the observable acts on more qubits than the reservoir.
        """
        if observable.nqubits < self._nqubits:
            padding = [QTensor(np.identity(2)) for _ in range(self._nqubits - observable.nqubits)]
            full_list = [observable]
            full_list.extend(padding)
            _observable = tensor_prod(full_list)
            return _observable
        if observable.nqubits > self._nqubits:
            raise ValueError("Observable acts on more qubits than the system contains")
        return observable

    @property
    def input_parameter_names(self) -> list[str]:
        """Return parameter names that are input-driven (non-trainable)."""
        trainable_params = self.get_trainable_parameter_names()
        return [name for name in self.get_parameter_names() if name not in trainable_params]

    @property
    def nparameters(self) -> int:
        """Total parameter count across pre-processing, dynamics, and post-processing."""
        return (
            (self._input_encoding.nparameters if self._input_encoding else 0)
            + self._evolution_dynamics.nparameters
            + (self._output_encoding.nparameters if self._output_encoding else 0)
        )

    @property
    def nqubits(self) -> int:
        """Number of qubits acted on by this pass."""
        return self._nqubits

    @property
    def input_encoding(self) -> Circuit | None:
        """Optional pre-processing stage."""
        return self._input_encoding

    @property
    def output_encoding(self) -> Circuit | None:
        """Optional post-processing stage."""
        return self._output_encoding

    @property
    def observables_as_qtensor(self) -> list[QTensor]:
        """Measured observables converted to qubit-sized ``QTensor`` operators."""
        return self._qtensor_observables

    @property
    def qubits_to_reset(self) -> list[int] | None:
        """Optional qubits reset between layers."""
        return self._qubits_to_reset

    @property
    def observables(self) -> list[QTensor | Hamiltonian | PauliOperator] | None:
        """Original observable list provided by the user."""
        return self._observables

    @property
    def evolution_dynamics(self) -> Schedule:
        """Main dynamics stage for each pass."""
        return self._evolution_dynamics

    def get_parameter_values(self) -> list[float]:
        """Return current parameter values of the pre-processing and dynamics blocks."""
        return (
            self._input_encoding.get_parameter_values() if self._input_encoding else []
        ) + self._evolution_dynamics.get_parameter_values()

    def get_parameter_names(self) -> list[str]:
        """Return parameter names in pass execution order."""
        return (
            (self._input_encoding.get_parameter_names() if self._input_encoding else [])
            + self._evolution_dynamics.get_parameter_names()
            + (self._output_encoding.get_parameter_names() if self._output_encoding else [])
        )

    def get_trainable_parameter_names(self) -> list[str]:
        """Return trainable parameter names in pass execution order."""
        return (
            (self._input_encoding.get_trainable_parameter_names() if self._input_encoding else [])
            + self._evolution_dynamics.get_trainable_parameter_names()
            + (self._output_encoding.get_trainable_parameter_names() if self._output_encoding else [])
        )

    def get_parameters(self) -> dict[str, RealNumber]:
        """Return current parameter mapping aggregated from all enabled stages."""
        out = self._input_encoding.get_parameters() if self._input_encoding else {}
        out.update(self._evolution_dynamics.get_parameters())
        out.update(self._output_encoding.get_parameters() if self._output_encoding else {})
        return out

    def get_trainable_parameters(self) -> dict[str, RealNumber]:
        """Return trainable parameter mapping aggregated from all enabled stages."""
        out = self._input_encoding.get_trainable_parameters() if self._input_encoding else {}
        out.update(self._evolution_dynamics.get_trainable_parameters())
        out.update(self._output_encoding.get_trainable_parameters() if self._output_encoding else {})
        return out

    def set_parameter_values(self, values: list[float]) -> None:
        """Set all pass parameters from an ordered value list.

        Raises:
            ValueError: If the provided number of values does not match ``nparameters``.
        """
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but this object has {self.nparameters} parameters.")
        param_names = self.get_parameter_names()
        value_dict = {param_names[i]: values[i] for i in range(len(values))}
        self.set_parameters(value_dict)

    def set_parameters(self, parameters: dict[str, float]) -> None:
        """Update stage parameters from a label-to-value mapping."""
        if self._input_encoding:
            self._input_encoding.set_parameters(
                {k: v for k, v in parameters.items() if k in self._input_encoding.get_parameter_names()}
            )
        self._evolution_dynamics.set_parameters(
            {k: v for k, v in parameters.items() if k in self._evolution_dynamics.get_parameter_names()}
        )
        if self._output_encoding:
            self._output_encoding.set_parameters(
                {k: v for k, v in parameters.items() if k in self._output_encoding.get_parameter_names()}
            )

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds aggregated from all enabled stages."""
        out = self._input_encoding.get_parameter_bounds() if self._input_encoding else {}
        out.update(self._evolution_dynamics.get_parameter_bounds())
        out.update(self._output_encoding.get_parameter_bounds() if self._output_encoding else {})
        return out

    def get_trainable_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return trainable parameter bounds aggregated from all enabled stages."""
        out = self._input_encoding.get_trainable_parameter_bounds() if self._input_encoding else {}
        out.update(self._evolution_dynamics.get_trainable_parameter_bounds())
        out.update(self._output_encoding.get_trainable_parameter_bounds() if self._output_encoding else {})
        return out

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Set parameter bounds in each stage for matching parameter names."""
        if self._input_encoding:
            self._input_encoding.set_parameter_bounds(
                {k: v for k, v in ranges.items() if k in self._input_encoding.get_parameter_names()}
            )
        self._evolution_dynamics.set_parameter_bounds(
            {k: v for k, v in ranges.items() if k in self._evolution_dynamics.get_parameter_names()}
        )
        if self._output_encoding:
            self._output_encoding.set_parameter_bounds(
                {k: v for k, v in ranges.items() if k in self._output_encoding.get_parameter_names()}
            )

    def __len__(self) -> int:
        """
        Get the total number of steps in the reservoir pass (maximum of 3)

        Returns:
            int: The number of steps in the reservoir pass.
        """
        return int(self._input_encoding is not None) + int(self._output_encoding is not None) + 1

    def __iter__(self) -> Iterator[Circuit | Schedule]:
        """
        Return an iterator over the steps in the reservoir pass.

        Yields:
            Iterator[Circuit | Schedule]: The steps in the pass
        """
        if self._input_encoding:
            yield self._input_encoding
        yield self._evolution_dynamics
        if self._output_encoding:
            yield self._output_encoding


class QuantumReservoir(PrimitiveFunctional):
    """Reservoir functional executed over a sequence of input layers.

    Each element in ``input_per_layer`` is applied to the underlying
    :class:`ReservoirPass` before executing one pass of dynamics and measurement.
    """

    def __init__(
        self,
        initial_state: QTensor,
        reservoir_layer: ReservoirLayer,
        input_per_layer: list[dict[str, float]],
        store_final_state: bool = False,
        store_intermideate_states: bool = False,
        nshots: int = 0,
    ) -> None:
        """Construct a quantum reservoir functional.

        Args:
            initial_state: Initial state before the first layer.
            reservoir_layer: Reservoir pass definition repeated at each layer.
            input_per_layer: Input parameter assignments for each layer.
            store_final_state: Whether to store the final state after the last layer.
            store_intermideate_states: Whether to store layer-by-layer intermediate states.
            nshots: Number of measurement shots for dynamics executions that use sampling.

        Raises:
            ValueError: If the initial state qubit count does not match the reservoir pass.
        """
        super().__init__()
        self._initial_state = initial_state
        self._reservoir_layer = reservoir_layer
        self._input_per_layer = input_per_layer
        self._store_final_state = store_final_state
        self._store_intermideate_states = store_intermideate_states
        self._nshots = nshots
        if self._reservoir_layer.nqubits != self._initial_state.nqubits:
            raise ValueError(
                f"invalid initial state: the initial state acts on {self._initial_state.nqubits} qubits while the reservoir is defined with {self._reservoir_layer.nqubits} qubits."
            )

    @property
    def nqubits(self) -> int:
        """Number of qubits in the reservoir."""
        return self._reservoir_layer.nqubits

    @property
    def initial_state(self) -> QTensor:
        """Initial quantum state used before the first layer."""
        return self._initial_state

    @property
    def reservoir_layer(self) -> ReservoirLayer:
        """Reservoir pass definition applied at each layer."""
        return self._reservoir_layer

    @property
    def input_per_layer(self) -> list[dict[str, float]]:
        """Layer-ordered input parameter assignments."""
        return self._input_per_layer

    @property
    def store_final_state(self) -> bool:
        """Whether to include the final reservoir state in the result."""
        return self._store_final_state

    @property
    def store_intermideate_states(self) -> bool:
        """Whether to include intermediate per-layer states in the result."""
        return self._store_intermideate_states

    @property
    def nshots(self) -> int:
        """Number of shots used by backend executions."""
        return self._nshots

    @property
    def input_parameter_names(self) -> list[str]:
        """Input-driven (non-trainable) parameter names expected per layer."""
        return self._reservoir_layer.input_parameter_names
