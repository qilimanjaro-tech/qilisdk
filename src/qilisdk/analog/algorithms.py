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


from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.analog.schedule import Schedule
from qilisdk.common.backend import AnalogBackend
from qilisdk.digital.circuit import Circuit

Complex = int | float | complex


class AnalogAlgorithm:
    def __init__(self, backend: AnalogBackend) -> None:
        self.backend = backend


class TimeEvolution(AnalogAlgorithm):

    def __init__(
        self,
        backend: AnalogBackend,
        schedule: Schedule,
        observables: list[QuantumObject | PauliOperator | Hamiltonian],
        initial_state: QuantumObject | Circuit | None = None,
        n_shots: int = 1000,
    ) -> None:
        """
        Args:
            schedule (Schedule): The evolution schedule over time.
            observables (list[QuantumObject  |  PauliOperator  |  Hamiltonian]): a list of observables to measure at the end of the evolution.
            initial_state (QuantumObject | Circuit | None, optional): the initial state of the evolution. Defaults to None.
            n_shots (int, optional): the number of shots. Defaults to 1000.

        Raises:
            ValueError: if one of the observables provided are not valid.
        """
        super().__init__(backend)
        self.initial_state = initial_state
        if not isinstance(self.initial_state, QuantumObject):
            raise NotImplementedError("currently only QuantumObjects are accepted as initial states.")
        self.schedule = schedule
        self.observables = []
        for obs in observables:
            if isinstance(obs, PauliOperator):
                self.observables.append(QuantumObject(obs.matrix))  # append Identities to this.
            elif isinstance(obs, Hamiltonian):
                self.observables.append(QuantumObject(obs.to_matrix()))  # check the number of qubits.
            elif isinstance(obs, QuantumObject):
                self.observables.append(obs)  # check the number of qubits.
            else:
                raise ValueError(
                    "Only PauliOperators, Hamiltonians, and QuantumObjects are considered valid observables."
                )
        self.n_shots = n_shots

    def evolve(self) -> list[Complex]:
        return self.backend.evolve()
