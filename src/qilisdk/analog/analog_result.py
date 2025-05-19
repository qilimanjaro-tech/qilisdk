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

from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.common.result import Result
from qilisdk.yaml import yaml


@yaml.register_class
class AnalogResult(Result):
    """
    Encapsulates the outcome of an analog quantum simulation.

    This result class stores key outputs from the simulation, including the
    final expected measurement values, the complete time-evolution of expectation
    values (if available), the final quantum state, and any intermediate quantum states.

    Attributes:
        _final_expected_values (np.ndarray): The final expectation values measured at the end of the simulation.
        _expected_values (np.ndarray): The time-series of expectation values recorded during the simulation.
        _final_state (QuantumObject | None): The final quantum state resulting from the simulation.
        _intermediate_states (list[QuantumObject] | None): A list of intermediate quantum states obtained during evolution.
    """

    def __init__(
        self,
        final_expected_values: np.ndarray | None = None,
        expected_values: np.ndarray | None = None,
        final_state: QuantumObject | None = None,
        intermediate_states: list[QuantumObject] | None = None,
    ) -> None:
        """
        Initialize an AnalogResult instance with simulation outputs.

        Args:
            final_expected_values (np.ndarray | None, optional): An array of the final expectation
                values measured at the end of the simulation. Defaults to an empty array if None.
            expected_values (np.ndarray | None, optional): An array containing the evolution of
                expectation values during the simulation. Defaults to an empty array if None.
            final_state (QuantumObject | None, optional): The final quantum state as a QuantumObject.
                Defaults to None.
            intermediate_states (list[QuantumObject] | None, optional): A list of QuantumObjects representing
                the intermediate states during the simulation. Defaults to None.
        """
        super().__init__()
        self._final_expected_values = final_expected_values if final_expected_values is not None else np.array([])
        self._expected_values = expected_values if expected_values is not None else np.array([])
        self._final_state = final_state
        self._intermediate_states = intermediate_states

    @property
    def final_expected_values(self) -> np.ndarray:
        """
        Get the final expectation values measured at the end of the simulation.

        Returns:
            np.ndarray: An array of the final expected values.
        """
        return self._final_expected_values

    @property
    def expected_values(self) -> np.ndarray:
        """
        Get the evolution of expectation values recorded during the simulation.

        Returns:
            np.ndarray: An array of expectation values over the course of the simulation.
        """
        return self._expected_values

    @property
    def final_state(self) -> QuantumObject | None:
        """
        Get the final quantum state produced by the simulation.

        Returns:
            QuantumObject | None: The final quantum state, or None if not available.
        """
        return self._final_state

    @property
    def intermediate_states(self) -> list[QuantumObject] | None:
        """
        Get the list of intermediate quantum states recorded during the simulation.

        Returns:
            list[QuantumObject] | None: A list of intermediate quantum states, or None if not stored.
        """
        return self._intermediate_states

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n"
            f"  final_expected_values={pformat(self.final_expected_values)},\n"
            f"  expected_values={pformat(self.expected_values)}\n"
            f"  final_state={pformat(self.final_state)}\n"
            f"  intermediate_states={pformat(self.intermediate_states)}\n"
            ")"
        )
