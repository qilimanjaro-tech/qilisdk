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
from pprint import pformat

import numpy as np

from qilisdk.core.model import Model
from qilisdk.core.qtensor import QTensor
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class TimeEvolutionResult(FunctionalResult):
    """Container for expectation values and states produced by a time-evolution simulation."""

    def __init__(
        self,
        final_expected_values: np.ndarray | None = None,
        expected_values: np.ndarray | None = None,
        final_state: QTensor | None = None,
        intermediate_states: list[QTensor] | None = None,
    ) -> None:
        """
        Args:
            final_expected_values (np.ndarray | None, optional): Expectation values evaluated at the end of the evolution.
            expected_values (np.ndarray | None, optional): Time series of expectation values, if recorded.
            final_state (QTensor | None, optional): Final quantum state associated with the evolution.
            intermediate_states (list[QTensor] | None, optional): Intermediate states captured during the evolution.
        """
        super().__init__()
        self._final_expected_values = final_expected_values if final_expected_values is not None else np.array([])
        self._expected_values = expected_values if expected_values is not None else np.array([])
        self._final_state = final_state
        self._intermediate_states = intermediate_states or []

    @property
    def final_expected_values(self) -> np.ndarray:
        """Final expectation values measured at the end of the evolution."""
        return self._final_expected_values

    @property
    def expected_values(self) -> np.ndarray:
        """Expectation values tracked over the course of the evolution."""
        return self._expected_values

    @property
    def final_state(self) -> QTensor | None:
        """Final quantum state if provided by the backend."""
        return self._final_state

    @property
    def intermediate_states(self) -> list[QTensor]:
        """List of intermediate quantum states captured during the simulation."""
        return self._intermediate_states

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n"
            + f"  final_expected_values={pformat(self.final_expected_values)},\n"
            + (f"  expected_values={pformat(self.expected_values)}\n" if len(self.expected_values) > 0 else "")
            + (f"  final_state={pformat(self.final_state)}\n" if self.final_state is not None else "")
            + (
                f"  intermediate_states={pformat(self.intermediate_states)}\n"
                if len(self.intermediate_states) > 0
                else ""
            )
            + ")"
        )

    def compute_cost(self, cost_model: Model) -> float:
        raise NotImplementedError
