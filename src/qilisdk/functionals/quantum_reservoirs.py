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

import numpy as np

from qilisdk.analog import Hamiltonian, Schedule
from qilisdk.analog.hamiltonian import PauliOperator
from qilisdk.core import Domain, Parameter, QTensor, tensor_prod
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.types import RealNumber
from qilisdk.digital import Circuit


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
        evolution_schedule: Schedule,
        measured_observables: list[QTensor | Hamiltonian | PauliOperator],
        state_encoding: Circuit | None = None,
        qubits_to_reset: list[int] | None = None,
    ) -> None:
        self.state_encoding: Circuit | None = state_encoding
        self.evlution_schedule: Schedule = evolution_schedule
        self.qubits_to_reset: list[int] | None = qubits_to_reset
        super().__init__()
        self._full_param_list: dict[str, Parameter] = {}
        self._nqubits = self.evlution_schedule.nqubits
        if state_encoding:
            if state_encoding.nqubits < self._nqubits:
                self.state_encoding = Circuit(self._nqubits)
                self.state_encoding.add(state_encoding.gates)
            elif state_encoding.nqubits > self._nqubits:
                raise ValueError(
                    "Encoding Circuit acts on more qubits than the number of qubits specified by the evolution schedule."
                )
        self.measured_observables: list[QTensor | Hamiltonian | PauliOperator] = []
        identity = QTensor(np.identity(2))
        for observable in measured_observables:
            if isinstance(observable, PauliOperator):
                self.measured_observables.append(observable.to_hamiltonian().to_qtensor(self._nqubits))
            elif isinstance(observable, Hamiltonian):
                self.measured_observables.append(observable.to_qtensor(self._nqubits))
            elif isinstance(observable, QTensor):
                if observable.nqubits < self._nqubits:
                    padding = [identity for _ in range(self._nqubits - observable.nqubits)]
                    full_list = [observable]
                    full_list.extend(padding)
                    _observable = tensor_prod(full_list)
                    self.measured_observables.append(_observable)
                elif observable.nqubits > self._nqubits:
                    raise ValueError("Observable acts on more qubits than the system contains")
                else:
                    self.measured_observables.append(observable)

    @property
    def input_parameter_names(self) -> list[str]:
        trainable_params = self.get_trainable_parameter_names()
        return [name for name in self.get_parameter_names() if name not in trainable_params]

    @property
    def nparameters(self) -> int:
        return (self.state_encoding.nparameters if self.state_encoding else 0) + self.evlution_schedule.nparameters

    @property
    def nqubits(self) -> int:
        return self._nqubits

    def get_parameter_values(self) -> list[float]:
        return (
            self.state_encoding.get_parameter_values() if self.state_encoding else []
        ) + self.evlution_schedule.get_parameter_values()

    def get_parameter_names(self) -> list[str]:
        return (
            self.state_encoding.get_parameter_names() if self.state_encoding else []
        ) + self.evlution_schedule.get_parameter_names()

    def get_trainable_parameter_names(self) -> list[str]:
        return (
            self.state_encoding.get_trainable_parameter_names() if self.state_encoding else []
        ) + self.evlution_schedule.get_trainable_parameter_names()

    def get_parameters(self) -> dict[str, RealNumber]:
        out = self.state_encoding.get_parameters() if self.state_encoding else {}
        out.update(self.evlution_schedule.get_parameters())
        return out

    def get_trainable_parameters(self) -> dict[str, RealNumber]:
        out = self.state_encoding.get_trainable_parameters() if self.state_encoding else {}
        out.update(self.evlution_schedule.get_trainable_parameters())
        return out

    def set_parameter_values(self, values: list[float]) -> None:
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but this object has {self.nparameters} parameters.")
        param_names = self.get_parameter_names()
        value_dict = {param_names[i]: values[i] for i in range(len(values))}
        self.set_parameters(value_dict)

    def set_parameters(self, parameters: dict[str, float]) -> None:
        if self.state_encoding:
            self.state_encoding.set_parameters(
                {k: v for k, v in parameters.items() if k in self.state_encoding.get_parameter_names()}
            )
        self.evlution_schedule.set_parameters(
            {k: v for k, v in parameters.items() if k in self.evlution_schedule.get_parameter_names()}
        )

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        out = self.state_encoding.get_parameter_bounds() if self.state_encoding else {}
        out.update(self.evlution_schedule.get_parameter_bounds())
        return out

    def get_trainable_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        out = self.state_encoding.get_trainable_parameter_bounds() if self.state_encoding else {}
        out.update(self.evlution_schedule.get_trainable_parameter_bounds())
        return out

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        if self.state_encoding:
            self.state_encoding.set_parameter_bounds(
                {k: v for k, v in ranges.items() if k in self.state_encoding.get_parameter_names()}
            )
        self.evlution_schedule.set_parameter_bounds(
            {k: v for k, v in ranges.items() if k in self.evlution_schedule.get_parameter_names()}
        )


class QuantumReservoir:
    def __init__(
        self,
        intial_state: QTensor,
        reservoir_pass: ReservoirPass,
        input_per_layer: list[dict[str, float]],
        store_final_state: bool = False,
        store_intermideate_states: bool = False,
        nshots: int = 0,
    ) -> None:
        self._intial_state = intial_state
        self._reservoir_pass = reservoir_pass
        self._input_per_layer = input_per_layer
        self._store_final_state = store_final_state
        self._store_intermideate_states = store_intermideate_states
        self._nshots = nshots
        if self._reservoir_pass.nqubits != self._intial_state.nqubits:
            raise ValueError(
                f"invalid initial state: the initial state acts on {self._intial_state.nqubits} qubits while the reservoir is defined with {self._reservoir_pass.nqubits} qubits."
            )

    @property
    def nqubits(self) -> int:
        return self._reservoir_pass.nqubits

    @property
    def intial_state(self) -> QTensor:
        return self._intial_state

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
