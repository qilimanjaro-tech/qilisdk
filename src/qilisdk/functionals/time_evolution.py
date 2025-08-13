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
from qilisdk.analog.schedule import Schedule
from qilisdk.common.model import Model
from qilisdk.common.quantum_objects import QuantumObject
from qilisdk.common.variables import Number
from qilisdk.functionals.functional import Functional
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.yaml import yaml

Complex = int | float | complex


@yaml.register_class
class TimeEvolution(Functional[TimeEvolutionResult]):
    result_type = TimeEvolutionResult

    def __init__(
        self,
        schedule: Schedule,
        observables: list[PauliOperator | Hamiltonian],
        initial_state: QuantumObject,
        nshots: int = 1000,
        store_intermediate_results: bool = False,
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
        self.nshots = nshots
        self.store_intermediate_results = store_intermediate_results

    def get_parameters(self) -> dict[str, Number]:  # noqa: PLR6301
        return {}

    def set_parameters(self, parameters: dict[str, Number]) -> None: ...

    def get_parameter_names(self) -> list[str]:  # noqa: PLR6301
        return []

    def get_parameter_values(self) -> list[Number]:  # noqa: PLR6301
        return []

    def compute_cost(self, results: TimeEvolutionResult, cost_model: Model) -> float:
        raise NotImplementedError
