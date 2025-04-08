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
from abc import abstractmethod

from qilisdk.analog.analog_result import AnalogResult
from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.analog.schedule import Schedule
from qilisdk.common.backend import Backend


class AnalogBackend(Backend):
    @abstractmethod
    def evolve(
        self,
        schedule: Schedule,
        initial_state: QuantumObject,
        observables: list[PauliOperator | Hamiltonian],
        store_intermediate_results: bool = False,
    ) -> AnalogResult:
        """
        Computes the time evolution under of an initial state under the given schedule.

        Args:
            schedule (Schedule): The evolution schedule of the system.
            initial_state (QuantumObject): the initial state of the evolution.
            observables (list[PauliOperator  |  Hamiltonian]): the list of observables to be measured at the end of the evolution.
            store_intermediate_results (bool): A flag to store the intermediate results along the evolution.

        Returns:
            AnalogResult: The results of the evolution.
        """
