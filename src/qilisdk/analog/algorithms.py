# Copyright 2025 Qilimanjaro Quantum Tech
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
from qilisdk.analog.analog_backend import AnalogBackend
from qilisdk.analog.analog_result import AnalogResult
from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.analog.schedule import Schedule

Complex = int | float | complex


class AnalogAlgorithm:
    def __init__(self, backend: AnalogBackend) -> None:
        self.backend = backend


class TimeEvolution(AnalogAlgorithm):
    def __init__(
        self,
        backend: AnalogBackend,
        schedule: Schedule,
        observables: list[PauliOperator | Hamiltonian],
        initial_state: QuantumObject,
        n_shots: int = 1000,
    ) -> None:
        """
        Args:
            schedule (Schedule): The evolution schedule over time.
            observables (list[PauliOperator  |  Hamiltonian]): a list of observables to measure at the end of the evolution.
            initial_state (QuantumObject): the initial state of the evolution. Defaults to None.
            n_shots (int, optional): the number of shots. Defaults to 1000.
        """
        super().__init__(backend)
        self.initial_state = initial_state
        if not isinstance(self.initial_state, QuantumObject):
            raise NotImplementedError("currently only QuantumObjects are accepted as initial states.")
        self.schedule = schedule
        self.observables = observables
        self.n_shots = n_shots

    def evolve(self, store_intermediate_results: bool = False, **kwargs: dict) -> AnalogResult:
        return self.backend.evolve(
            schedule=self.schedule,
            initial_state=self.initial_state,
            observables=self.observables,
            store_intermediate_results=store_intermediate_results,
            **kwargs,
        )
