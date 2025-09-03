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
from typing import ClassVar

from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.analog.schedule import Schedule
from qilisdk.common.qtensor import QTensor
from qilisdk.common.variables import RealNumber
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.yaml import yaml


@yaml.register_class
class TimeEvolution(PrimitiveFunctional[TimeEvolutionResult]):
    result_type: ClassVar[type[TimeEvolutionResult]] = TimeEvolutionResult

    def __init__(
        self,
        schedule: Schedule,
        observables: list[PauliOperator | Hamiltonian],
        initial_state: QTensor,
        nshots: int = 1000,
        store_intermediate_results: bool = False,
    ) -> None:
        """
        Initialize the TimeEvolution simulation.

        Args:
            backend (AnalogBackend): The backend to use for simulating the dynamics.
            schedule (Schedule): The evolution schedule defining the time-dependent Hamiltonian.
            observables (list[PauliOperator | Hamiltonian]): A list of observables to measure at the end of the evolution.
            initial_state (QTensor): The initial quantum state from which the simulation starts.
            n_shots (int, optional): The number of simulation repetitions (shots). Defaults to 1000.
        """
        super().__init__()
        self.initial_state = initial_state
        self.schedule = schedule
        self.observables = observables
        self.nshots = nshots
        self.store_intermediate_results = store_intermediate_results

    @property
    def nparameters(self) -> int:
        return self.schedule.nparameters

    def get_parameters(self) -> dict[str, RealNumber]:
        return self.schedule.get_parameters()

    def set_parameters(self, parameters: dict[str, RealNumber]) -> None:
        self.schedule.set_parameters(parameters)

    def get_parameter_names(self) -> list[str]:
        return self.schedule.get_parameter_names()

    def get_parameter_values(self) -> list[RealNumber]:
        return self.schedule.get_parameter_values()

    def set_parameter_values(self, values: list[float]) -> None:
        self.schedule.set_parameter_values(values)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self.schedule.get_parameter_bounds()

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        self.schedule.set_parameter_bounds(ranges)
