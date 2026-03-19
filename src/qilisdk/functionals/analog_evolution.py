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
from typing import ClassVar, Iterator

from qilisdk.analog.schedule import Schedule
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.qtensor import QTensor
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class AnalogEvolution(PrimitiveFunctional):
    """
    Simulate the dynamics induced by a time-dependent Hamiltonian schedule.

    Example:
        .. code-block:: python

            from qilisdk.analog import Schedule, Z
            from qilisdk.core import ket
            from qilisdk.functionals import AnalogEvolution
            from qilisdk.readout import StateTomographyReadout

            h0 = Z(0)
            schedule = Schedule(hamiltonians={"h0": h0}, total_time=10.0)
            functional = AnalogEvolution(schedule, initial_state=ket(0))
            result = backend.execute(functional, readout=[StateTomographyReadout()])
    """

    result_type: ClassVar[type[FunctionalResult]] = FunctionalResult

    def __init__(
        self,
        schedule: Schedule,
        initial_state: QTensor,
        store_intermediate_results: bool = False,
    ) -> None:
        """
        Args:
            schedule (Schedule): Annealing or control schedule describing the Hamiltonian evolution.
            initial_state (QTensor): Quantum state used as the simulation starting point.
            store_intermediate_results (bool, optional): Keep intermediate states if produced by the backend. Defaults to False.

        Raises:
            ValueError: if the number of qubits of the initial state doesn't match the number of qubits in the schedule.
        """
        super().__init__()
        self.initial_state = initial_state
        self.schedule = schedule
        self.store_intermediate_results = store_intermediate_results

        if initial_state.nqubits != schedule.nqubits:
            raise ValueError(
                f"The initial state provided acts on {initial_state.nqubits} qubits while the schedule acts on {schedule.nqubits} qubits"
            )

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        yield self.schedule

    def __repr__(self) -> str:
        lines = [
            f"{type(self).__qualname__}(",
            f"  schedule={self.schedule!r},",
            f"  initial_state={self.initial_state!r},",
            f"  store_intermediate_results={self.store_intermediate_results!r},",
            ")",
        ]
        return "\n".join(lines)
