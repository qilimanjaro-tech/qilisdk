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

from .quantum_reservoirs_result import QuantumReservoirResult


class ReservoirInput(Parameter):
    """Input-only parameter used to inject layer-wise data into a reservoir program.

    This parameter behaves like a standard :class:`~qilisdk.core.variables.Parameter` but is
    always created as non-trainable, so optimizers do not update it.
    """

    def __init__(
        self,
        label: str,
        value: RealNumber,
    ) -> None:
        """Create a non-trainable reservoir input parameter.

        Args:
            label: Parameter label used to reference input values.
            value: Default numerical value.
        """
        super().__init__(
            label=label,
            value=value,
            domain=Domain.REAL,
            bounds=(None, None),
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
            evolution_dynamics: Main analog schedule block.
            observables: Observables sampled at the end of the pass.
            input_encoding: Optional single-qubit pre-processing circuit.
            output_encoding: Optional single-qubit post-processing circuit.
            qubits_to_reset: Optional qubits reset between consecutive layers.

        Raises:
            ValueError: If an observable type is unsupported.
        """

        super().__init__()
        self._input_encoding: Circuit | None = input_encoding
        self._evolution_dynamics: Schedule = evolution_dynamics
        self._output_encoding: Circuit | None = output_encoding
        self._qubits_to_reset: list[int] | None = qubits_to_reset
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
            else:
                raise ValueError(
                    "Unsupported observable type. Expected QTensor, Hamiltonian, or PauliOperator, "
                    f"received {type(observable).__name__}."
                )

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
        self._input_encoding = Circuit(self._nqubits, parameter_prefix="input_encoding_")
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
        self._output_encoding = Circuit(self._nqubits, parameter_prefix="output_encoding_")
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
        return self.get_parameter_names(
            trainable=False, parameter_filter=lambda param: isinstance(param, ReservoirInput)
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
    def pre_processing(self) -> Circuit | None:
        """Backward-compatible alias for ``input_encoding``."""
        return self._input_encoding

    @property
    def output_encoding(self) -> Circuit | None:
        """Optional post-processing stage."""
        return self._output_encoding

    @property
    def post_processing(self) -> Circuit | None:
        """Backward-compatible alias for ``output_encoding``."""
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

    @property
    def reservoir_dynamics(self) -> Schedule:
        """Backward-compatible alias for ``evolution_dynamics``."""
        return self._evolution_dynamics

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        if self._input_encoding:
            yield self._input_encoding
        yield self._evolution_dynamics
        if self._output_encoding:
            yield self._output_encoding

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


class QuantumReservoir(PrimitiveFunctional[QuantumReservoirResult]):
    """Reservoir functional executed over a sequence of input layers.

    Each element in ``input_per_layer`` is applied to the underlying
    :class:`ReservoirLayer` before executing one pass of dynamics and measurement.
    """

    def __init__(
        self,
        initial_state: QTensor,
        reservoir_layer: ReservoirLayer | None = None,
        input_per_layer: list[dict[str, float]] | None = None,
        store_final_state: bool = False,
        store_intermediate_states: bool = False,
        nshots: int = 0,
    ) -> None:
        """Construct a quantum reservoir functional.

        Args:
            initial_state: Initial state before the first layer.
            reservoir_layer: Reservoir pass definition repeated at each layer.
            input_per_layer: Input parameter assignments for each layer.
            store_final_state: Whether to store the final state after the last layer.
            store_intermediate_states: Whether to store layer-by-layer intermediate states.
            nshots: Number of measurement shots for dynamics executions that use sampling.

        Raises:
            ValueError: If the initial state qubit count does not match the reservoir pass.
        """
        super().__init__()
        if reservoir_layer is None:
            raise ValueError("`reservoir_layer` must be provided.")
        if input_per_layer is None:
            raise ValueError("`input_per_layer` must be provided.")
        if len(input_per_layer) == 0:
            raise ValueError("`input_per_layer` must contain at least one layer.")

        self._initial_state = initial_state
        self._reservoir_layer = reservoir_layer
        self._input_per_layer = input_per_layer
        self._store_final_state = store_final_state
        self._store_intermediate_states = store_intermediate_states
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
    def store_intermediate_states(self) -> bool:
        """Whether to include intermediate per-layer states in the result."""
        return self._store_intermediate_states

    @property
    def nshots(self) -> int:
        """Number of shots used by backend executions."""
        return self._nshots

    @property
    def input_parameter_names(self) -> list[str]:
        """Input-driven (non-trainable) parameter names expected per layer."""
        return self._reservoir_layer.input_parameter_names

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        yield self._reservoir_layer
