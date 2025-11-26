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
from qilisdk.core.qtensor import QTensor
from qilisdk.core.variables import ComparisonTerm, RealNumber
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.yaml import yaml


@yaml.register_class
class TimeEvolution(PrimitiveFunctional[TimeEvolutionResult]):
    """
    Simulate the dynamics induced by a time-dependent Hamiltonian schedule.

    Example:
        .. code-block:: python

            from qilisdk.analog import Schedule, Hamiltonian, Z
            from qilisdk.core import ket
            from qilisdk.functionals.time_evolution import TimeEvolution

            h0 = Z(0)
            schedule = Schedule(hamiltonians={"h0": h0}, total_time=10.0)
            functional = TimeEvolution(schedule, observables=[Z(0), X(0)], initial_state=ket(0))
    """

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
        Args:
            schedule (Schedule): Annealing or control schedule describing the Hamiltonian evolution.
            observables (list[PauliOperator | Hamiltonian]): Observables measured at the end of the evolution.
            initial_state (QTensor): Quantum state used as the simulation starting point.
            nshots (int, optional): Number of executions for statistical estimation. Defaults to 1000.
            store_intermediate_results (bool, optional): Keep intermediate states if produced by the backend. Defaults to False.
        """
        super().__init__()
        self.initial_state = initial_state
        self.schedule = schedule
        self.observables = observables
        self.nshots = nshots
        self.store_intermediate_results = store_intermediate_results

    @property
    def nparameters(self) -> int:
        """Return the number of schedule parameters."""
        return self.schedule.nparameters

    def get_parameters(self) -> dict[str, RealNumber]:
        """Return the schedule parameters and their current value."""
        return self.schedule.get_parameters()

    def set_parameters(self, parameters: dict[str, RealNumber]) -> None:
        """Update a subset of schedule parameters."""
        self.schedule.set_parameters(parameters)

    def get_parameter_names(self) -> list[str]:
        """Return order-stable parameter labels from the schedule."""
        return self.schedule.get_parameter_names()

    def get_parameter_values(self) -> list[RealNumber]:
        """Return parameter values in the order provided by ``get_parameter_names``."""
        return self.schedule.get_parameter_values()

    def set_parameter_values(self, values: list[float]) -> None:
        """Assign all schedule parameters according to ``get_parameter_names`` order."""
        self.schedule.set_parameter_values(values)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return current bounds for schedule parameters."""
        return self.schedule.get_parameter_bounds()

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Update bounds for selected schedule parameters."""
        self.schedule.set_parameter_bounds(ranges)

    def get_constraints(self) -> list[ComparisonTerm]:
        """Return the parameter constraints defined within the underlying schedule."""
        return self.schedule.get_constraints()
