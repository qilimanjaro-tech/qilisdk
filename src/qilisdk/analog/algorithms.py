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

from qilisdk.analog.analog_backend import AnalogBackend
from qilisdk.analog.analog_result import AnalogResult
from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.analog.schedule import Schedule
from qilisdk.common.algorithm import Algorithm
from qilisdk.yaml import yaml

Complex = int | float | complex


@yaml.register_class
class AnalogAlgorithm(Algorithm):
    """
    Abstract base class for analog quantum algorithms.

    This class provides the foundational interface for analog quantum algorithms
    that simulate the dynamics of quantum systems. It holds a reference to an
    AnalogBackend instance responsible for executing quantum operations. Subclasses
    must implement the evolve() method to perform the algorithm-specific simulation
    and return an AnalogResult.

    Attributes:
        backend (AnalogBackend): The backend instance used to execute the quantum
            simulation operations.
    """

    @abstractmethod
    def evolve(self, backend: AnalogBackend, store_intermediate_results: bool = False) -> AnalogResult:
        """
        Execute the analog quantum algorithm's evolution process.

        This abstract method defines the interface for simulating the time evolution
        of a quantum system. Subclasses must implement this method to perform the
        simulation using the provided backend, and to return the simulation results
        encapsulated in an AnalogResult object.

        Args:
            store_intermediate_results (bool, optional): If True, the algorithm should
                store and include intermediate results during the simulation. Defaults to False.

        Returns:
            AnalogResult: The result of the simulation, which includes the final state,
            measured observables, and any intermediate data if requested.
        """


@yaml.register_class
class TimeEvolution(AnalogAlgorithm):
    def __init__(
        self,
        schedule: Schedule,
        observables: list[PauliOperator | Hamiltonian],
        initial_state: QuantumObject,
        n_shots: int = 1000,
    ) -> None:
        """
        Initialize the TimeEvolution simulation.

        Args:
            backend (AnalogBackend): The backend to use for simulating the dynamics.
            schedule (Schedule): The evolution schedule defining the time-dependent Hamiltonian.
            observables (list[PauliOperator | Hamiltonian]): A list of observables to measure at the end of the evolution.
            initial_state (QuantumObject): The initial quantum state from which the simulation starts.
            n_shots (int, optional): The number of simulation repetitions (shots). Defaults to 1000.
        """
        super().__init__()
        self.initial_state = initial_state
        self.schedule = schedule
        self.observables = observables
        self.n_shots = n_shots

    def evolve(self, backend: AnalogBackend, store_intermediate_results: bool = False) -> AnalogResult:
        """
        Execute the time evolution algorithm.

        This method performs the quantum dynamics evolution according to
        the provided schedule, initial state, and observables. The computation
        is delegated to the configured backend, which may optionally store
        intermediate results during the evolution.

        Args:
            store_intermediate_results (bool, optional): If True, the algorithm
                will store intermediate results during the time evolution.
                Defaults to False.

        Returns:
            AnalogResult: The result of the evolution, including the final state
            and measured observables (and intermediate results if requested).
        """
        return backend.evolve(
            schedule=self.schedule,
            initial_state=self.initial_state,
            observables=self.observables,
            store_intermediate_results=store_intermediate_results,
        )
