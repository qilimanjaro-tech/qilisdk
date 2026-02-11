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
from enum import Enum
from typing import Generator, Iterator

import numpy as np

from qilisdk.analog import Hamiltonian, Schedule
from qilisdk.analog.hamiltonian import PauliOperator
from qilisdk.core import Domain, Parameter, QTensor, tensor_prod
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.types import RealNumber
from qilisdk.digital import Circuit, M


class ReservoirInput(Parameter):
    def __init__(
        self,
        label: str,
        value: RealNumber,
        domain: Domain = Domain.REAL,
        bounds: tuple[float | None, float | None] = (None, None),
    ) -> None:
        super().__init__(
            label=label,
            value=value,
            domain=domain,
            bounds=bounds,
            trainable=False,
        )


class ReservoirPass(Parameterizable):
    def __init__(
        self,
        reservoir_dynamics: Schedule | Circuit,
        measured_observables: list[QTensor | Hamiltonian | PauliOperator],
        post_processing: Circuit | None = None,
        pre_processing: Circuit | None = None,
        qubits_to_reset: list[int] | None = None,
    ) -> None:

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
        trainable_params = self.get_trainable_parameter_names()
        return [name for name in self.get_parameter_names() if name not in trainable_params]

    @property
    def nparameters(self) -> int:
        return (
            (self._pre_processing.nparameters if self._pre_processing else 0)
            + self._reservoir_dynamics.nparameters
            + (self._post_processing.nparameters if self._post_processing else 0)
        )

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @property
    def pre_processing(self) -> Circuit | None:
        return self._pre_processing

    @property
    def post_processing(self) -> Circuit | None:
        return self._post_processing

    @property
    def observables_as_qtensor(self) -> list[QTensor]:
        return self._qtensor_observables

    @property
    def qubits_to_reset(self) -> list[int] | None:
        return self._qubits_to_reset

    @property
    def measured_observables(self) -> list[QTensor | Hamiltonian | PauliOperator] | None:
        return self._measured_observables

    @property
    def reservoir_dynamics(self) -> Schedule | Circuit:
        return self._reservoir_dynamics

    def get_parameter_values(self) -> list[float]:
        return (
            self._pre_processing.get_parameter_values() if self._pre_processing else []
        ) + self._reservoir_dynamics.get_parameter_values()

    def get_parameter_names(self) -> list[str]:
        return (
            (self._pre_processing.get_parameter_names() if self._pre_processing else [])
            + self._reservoir_dynamics.get_parameter_names()
            + (self._post_processing.get_parameter_names() if self._post_processing else [])
        )

    def get_trainable_parameter_names(self) -> list[str]:
        return (
            (self._pre_processing.get_trainable_parameter_names() if self._pre_processing else [])
            + self._reservoir_dynamics.get_trainable_parameter_names()
            + (self._post_processing.get_trainable_parameter_names() if self._post_processing else [])
        )

    def get_parameters(self) -> dict[str, RealNumber]:
        out = self._pre_processing.get_parameters() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_parameters())
        out.update(self._post_processing.get_parameters() if self._post_processing else {})
        return out

    def get_trainable_parameters(self) -> dict[str, RealNumber]:
        out = self._pre_processing.get_trainable_parameters() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_trainable_parameters())
        out.update(self._post_processing.get_trainable_parameters() if self._post_processing else {})
        return out

    def set_parameter_values(self, values: list[float]) -> None:
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but this object has {self.nparameters} parameters.")
        param_names = self.get_parameter_names()
        value_dict = {param_names[i]: values[i] for i in range(len(values))}
        self.set_parameters(value_dict)

    def set_parameters(self, parameters: dict[str, float]) -> None:
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
        out = self._pre_processing.get_parameter_bounds() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_parameter_bounds())
        out.update(self._post_processing.get_parameter_bounds() if self._post_processing else {})
        return out

    def get_trainable_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        out = self._pre_processing.get_trainable_parameter_bounds() if self._pre_processing else {}
        out.update(self._reservoir_dynamics.get_trainable_parameter_bounds())
        out.update(self._post_processing.get_trainable_parameter_bounds() if self._post_processing else {})
        return out

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
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


class QuantumReservoir:
    def __init__(
        self,
        initial_state: QTensor,
        reservoir_pass: ReservoirPass,
        input_per_layer: list[dict[str, float]],
        store_final_state: bool = False,
        store_intermideate_states: bool = False,
        nshots: int = 0,
    ) -> None:
        self._initial_state = initial_state
        self._reservoir_pass = reservoir_pass
        self._input_per_layer = input_per_layer
        self._store_final_state = store_final_state
        self._store_intermideate_states = store_intermideate_states
        self._nshots = nshots
        if self._reservoir_pass.nqubits != self._initial_state.nqubits:
            raise ValueError(
                f"invalid initial state: the initial state acts on {self._initial_state.nqubits} qubits while the reservoir is defined with {self._reservoir_pass.nqubits} qubits."
            )

    @property
    def nqubits(self) -> int:
        return self._reservoir_pass.nqubits

    @property
    def initial_state(self) -> QTensor:
        return self._initial_state

    @property
    def reservoir_pass(self) -> ReservoirPass:
        return self._reservoir_pass

    @property
    def input_per_layer(self) -> list[dict[str, float]]:
        return self._input_per_layer

    @property
    def store_final_state(self) -> bool:
        return self._store_final_state

    @property
    def store_intermideate_states(self) -> bool:
        return self._store_intermideate_states

    @property
    def nshots(self) -> int:
        return self._nshots

    @property
    def input_parameter_names(self) -> list[str]:
        return self._reservoir_pass.input_parameter_names
