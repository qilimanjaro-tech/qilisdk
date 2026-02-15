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


class ReservoirPass(Parameterizable):
    """Single reservoir layer template.

    A reservoir pass can contain up to three ordered stages:
    ``pre_processing -> reservoir_dynamics -> post_processing``.
    It also stores the observables measured after each pass and optional qubits to reset.
    """

    def __init__(
        self,
        reservoir_dynamics: Schedule,
        measured_observables: list[QTensor | Hamiltonian | PauliOperator],
        post_processing: Circuit | None = None,
        pre_processing: Circuit | None = None,
        qubits_to_reset: list[int] | None = None,
    ) -> None:
        """Build a reservoir pass description.

        Args:
            reservoir_dynamics: Main dynamics block, either analog schedule or digital circuit.
            measured_observables: Observables sampled at the end of the pass.
            post_processing: Optional single-qubit post-processing circuit.
            pre_processing: Optional single-qubit pre-processing circuit.
            qubits_to_reset: Optional qubits reset between consecutive layers.
        """

        self._pre_processing: Circuit | None = pre_processing
        self._reservoir_dynamics: Schedule = reservoir_dynamics
        self._post_processing: Circuit | None = post_processing
        self._qubits_to_reset: list[int] | None = qubits_to_reset
        super().__init__()
        self._full_param_list: dict[str, Parameter] = {}
        self._nqubits = self._reservoir_dynamics.nqubits

        if pre_processing:
            self._validate_pre_processing(pre_processing)

        if post_processing:
            self._validate_post_processing(post_processing)

        self._qtensor_observables: list[QTensor] = []

        for observable in measured_observables:
            if isinstance(observable, PauliOperator):
                self._qtensor_observables.append(observable.to_hamiltonian().to_qtensor(self._nqubits))
            elif isinstance(observable, Hamiltonian):
                self._qtensor_observables.append(observable.to_qtensor(self._nqubits))
            elif isinstance(observable, QTensor):
                self._qtensor_observables.append(self._process_qtensor(observable))

        self._measured_observables = measured_observables

    def _validate_pre_processing(self, pre_processing: Circuit) -> None:
        """Validate and normalize the optional pre-processing circuit.

        Raises:
            ValueError: If measurements are present, multi-qubit gates are used, or the
                circuit acts on more qubits than the reservoir dynamics.
        """
        if any(isinstance(g, M) for g in pre_processing.gates):
            raise ValueError("Pre-Processing Circuit can't contain measurements")
        if pre_processing.nqubits > self._reservoir_dynamics.nqubits:
            raise ValueError("Pre-Processing Circuit acts on more qubits than defined by the reservoir dynamics.")
        if any(g.nqubits > 1 for g in pre_processing.gates):
            raise ValueError("Only single qubit gates are allowed in the pre-processing circuit.")
        if pre_processing.nqubits < self.nqubits:
            self._pre_processing = Circuit(self._nqubits)
            self._pre_processing.add(pre_processing.gates)

    def _validate_post_processing(self, post_processing: Circuit) -> None:
        """Validate and normalize the optional post-processing circuit.

        Raises:
            ValueError: If measurements are present, multi-qubit gates are used, or the
                circuit acts on more qubits than the reservoir dynamics.
        """
        if any(isinstance(g, M) for g in post_processing.gates):
            raise ValueError("Post-Processing Circuit can't contain measurements")
        if post_processing.nqubits > self._reservoir_dynamics.nqubits:
            raise ValueError("Post-Processing Circuit acts on more qubits than defined by the reservoir dynamics.")
        if any(g.nqubits > 1 for g in post_processing.gates):
            raise ValueError("Only single qubit gates are allowed in the post-processing circuit.")
        if post_processing.nqubits < self.nqubits:
            self._post_processing = Circuit(self._nqubits)
            self._post_processing.add(post_processing.gates)

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
            (self._pre_processing.nparameters if self._pre_processing else 0)
            + self._reservoir_dynamics.nparameters
            + (self._post_processing.nparameters if self._post_processing else 0)
        )

    @property
    def nqubits(self) -> int:
        """Number of qubits acted on by this pass."""
        return self._nqubits

    @property
    def pre_processing(self) -> Circuit | None:
        """Optional pre-processing stage."""
        return self._pre_processing

    @property
    def post_processing(self) -> Circuit | None:
        """Optional post-processing stage."""
        return self._post_processing

    @property
    def observables_as_qtensor(self) -> list[QTensor]:
        """Measured observables converted to qubit-sized ``QTensor`` operators."""
        return self._qtensor_observables

    @property
    def qubits_to_reset(self) -> list[int] | None:
        """Optional qubits reset between layers."""
        return self._qubits_to_reset

    @property
    def measured_observables(self) -> list[QTensor | Hamiltonian | PauliOperator] | None:
        """Original observable list provided by the user."""
        return self._measured_observables

    @property
    def reservoir_dynamics(self) -> Schedule:
        """Main dynamics stage for each pass."""
        return self._reservoir_dynamics

    def get_parameter_values(self) -> list[float]:
        """Return current parameter values of the pre-processing and dynamics blocks."""
        return (
            self._pre_processing.get_parameter_values() if self._pre_processing else []
        ) + self._reservoir_dynamics.get_parameter_values()

    def get_parameter_names(self) -> list[str]:
        """Return parameter names in pass execution order."""
        return (
            (self._pre_processing.get_parameter_names() if self._pre_processing else [])
            + self._reservoir_dynamics.get_parameter_names()
            + (self._post_processing.get_parameter_names() if self._post_processing else [])
        )

    def get_trainable_parameter_names(self) -> list[str]:
        """Return trainable parameter names in pass execution order."""
        return (
            (self._pre_processing.get_trainable_parameter_names() if self._pre_processing else [])
            + self._reservoir_dynamics.get_trainable_parameter_names()
            + (self._post_processing.get_trainable_parameter_names() if self._post_processing else [])
        )

    def get_parameters(self) -> dict[str, RealNumber]:
        """Return current parameter mapping aggregated from all enabled stages."""
        out = self._pre_processing.get_parameters() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_parameters())
        out.update(self._post_processing.get_parameters() if self._post_processing else {})
        return out

    def get_trainable_parameters(self) -> dict[str, RealNumber]:
        """Return trainable parameter mapping aggregated from all enabled stages."""
        out = self._pre_processing.get_trainable_parameters() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_trainable_parameters())
        out.update(self._post_processing.get_trainable_parameters() if self._post_processing else {})
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
        if self._pre_processing:
            self._pre_processing.set_parameters(
                {k: v for k, v in parameters.items() if k in self._pre_processing.get_parameter_names()}
            )
        self._reservoir_dynamics.set_parameters(
            {k: v for k, v in parameters.items() if k in self._reservoir_dynamics.get_parameter_names()}
        )
        if self._post_processing:
            self._post_processing.set_parameters(
                {k: v for k, v in parameters.items() if k in self._post_processing.get_parameter_names()}
            )

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds aggregated from all enabled stages."""
        out = self._pre_processing.get_parameter_bounds() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_parameter_bounds())
        out.update(self._post_processing.get_parameter_bounds() if self._post_processing else {})
        return out

    def get_trainable_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return trainable parameter bounds aggregated from all enabled stages."""
        out = self._pre_processing.get_trainable_parameter_bounds() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_trainable_parameter_bounds())
        out.update(self._post_processing.get_trainable_parameter_bounds() if self._post_processing else {})
        return out

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Set parameter bounds in each stage for matching parameter names."""
        if self._pre_processing:
            self._pre_processing.set_parameter_bounds(
                {k: v for k, v in ranges.items() if k in self._pre_processing.get_parameter_names()}
            )
        self._reservoir_dynamics.set_parameter_bounds(
            {k: v for k, v in ranges.items() if k in self._reservoir_dynamics.get_parameter_names()}
        )
        if self._post_processing:
            self._post_processing.set_parameter_bounds(
                {k: v for k, v in ranges.items() if k in self._post_processing.get_parameter_names()}
            )

    def __len__(self) -> int:
        """
        Get the total number of steps in the reservoir pass (maximum of 3)

        Returns:
            int: The number of steps in the reservoir pass.
        """
        return int(self._pre_processing is not None) + int(self._post_processing is not None) + 1

    def __iter__(self) -> Iterator[Circuit | Schedule]:
        """
        Return an iterator over the steps in the reservoir pass.

        Yields:
            Iterator[Circuit | Schedule]: The steps in the pass
        """
        if self._pre_processing:
            yield self._pre_processing
        yield self._reservoir_dynamics
        if self._post_processing:
            yield self._post_processing


class QuantumReservoir(PrimitiveFunctional):
    """Reservoir functional executed over a sequence of input layers.

    Each element in ``input_per_layer`` is applied to the underlying
    :class:`ReservoirPass` before executing one pass of dynamics and measurement.
    """

    def __init__(
        self,
        initial_state: QTensor,
        reservoir_pass: ReservoirPass,
        input_per_pass: list[dict[str, float]],
        store_final_state: bool = False,
        store_intermideate_states: bool = False,
        nshots: int = 0,
    ) -> None:
        """Construct a quantum reservoir functional.

        Args:
            initial_state: Initial state before the first layer.
            reservoir_pass: Reservoir pass definition repeated at each layer.
            input_per_pass: Input parameter assignments for each layer.
            store_final_state: Whether to store the final state after the last layer.
            store_intermideate_states: Whether to store layer-by-layer intermediate states.
            nshots: Number of measurement shots for dynamics executions that use sampling.

        Raises:
            ValueError: If the initial state qubit count does not match the reservoir pass.
        """
        self._initial_state = initial_state
        self._reservoir_pass = reservoir_pass
        self._input_per_pass = input_per_pass
        self._store_final_state = store_final_state
        self._store_intermideate_states = store_intermideate_states
        self._nshots = nshots
        if self._reservoir_pass.nqubits != self._initial_state.nqubits:
            raise ValueError(
                f"invalid initial state: the initial state acts on {self._initial_state.nqubits} qubits while the reservoir is defined with {self._reservoir_pass.nqubits} qubits."
            )

    @property
    def nqubits(self) -> int:
        """Number of qubits in the reservoir."""
        return self._reservoir_pass.nqubits

    @property
    def initial_state(self) -> QTensor:
        """Initial quantum state used before the first layer."""
        return self._initial_state

    @property
    def reservoir_pass(self) -> ReservoirPass:
        """Reservoir pass definition applied at each layer."""
        return self._reservoir_pass

    @property
    def input_per_pass(self) -> list[dict[str, float]]:
        """Layer-ordered input parameter assignments."""
        return self._input_per_pass

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
        return self._reservoir_pass.input_parameter_names
