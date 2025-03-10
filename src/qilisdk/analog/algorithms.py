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


import numpy as np
from scipy.sparse import identity, spmatrix

from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.analog.quantum_objects import QuantumObject, tensor
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
                _obs_ham = Hamiltonian({(obs,): 1})
                if _obs_ham.nqubits < self.schedule.nqubits:
                    _obs = _add_padding(_obs_ham.to_matrix(), self.schedule.nqubits - _obs_ham.nqubits)
                else:
                    _obs = QuantumObject(_obs_ham.to_matrix())
                self.observables.append(QuantumObject(obs.matrix))  # append Identities to this.
            elif isinstance(obs, Hamiltonian):
                if obs.nqubits < self.schedule.nqubits:
                    _obs = _add_padding(obs.to_matrix(), self.schedule.nqubits - obs.nqubits)
                else:
                    _obs = QuantumObject(obs.to_matrix())
                self.observables.append(_obs)
            elif isinstance(obs, QuantumObject):
                if obs.nqubits < self.schedule.nqubits:
                    _obs = _add_padding(obs.data, self.schedule.nqubits - obs.nqubits)
                else:
                    _obs = obs
                self.observables.append(_obs)  # check the number of qubits.
            else:
                raise ValueError(
                    "Only PauliOperators, Hamiltonians, and QuantumObjects are considered valid observables."
                )
        self.n_shots = n_shots

    def evolve(self) -> list[Complex]:
        return self.backend.evolve()


def _add_padding(matrix: np.ndarray | spmatrix, num_missing_dim: int) -> QuantumObject:
    if num_missing_dim == 0:
        return QuantumObject(matrix)
    return tensor([QuantumObject(matrix), QuantumObject(identity(2**num_missing_dim))])
