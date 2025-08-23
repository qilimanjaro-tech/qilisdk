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
from __future__ import annotations

from abc import ABC, abstractmethod


class Parameterizable(ABC):
    @property
    @abstractmethod
    def nparameters(self) -> int:
        """
        Retrieve the total RealNumber of parameters required by all parameterized gates in the circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """

    @abstractmethod
    def get_parameter_values(self) -> list[float]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """

    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, float]:
        """
        Retrieve the parameter names and values from all parameterized gates in the circuit.

        Returns:
            dict[str, float]: A dictionary of the parameters with their current values.
        """

    @abstractmethod
    def set_parameter_values(self, values: list[float]) -> None:
        """
        Set new parameter values for all parameterized gates in the circuit.

        Args:
            values (list[float]): A list containing new parameter values to assign to the parameterized gates.

        Raises:
            ValueError: If the RealNumber of provided values does not match the expected RealNumber of parameters.
        """

    @abstractmethod
    def set_parameters(self, parameter_dict: dict[str, int | float]) -> None:
        """Set the parameter values by their label. No need to provide the full list of parameters.

        Args:
            parameter_dict (dict[str, RealNumber]): a dictionary mapping each parameter label to its value.

        Raises:
            ValueError: if the provided parameter label is not defined in the list of parameters contained in this object.
        """

    @abstractmethod
    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Returns a dictionary specifying the bounds of each parameter."""

    @abstractmethod
    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Updates the ranges for the specified parameters.

        Args:
            ranges (dict[str, tuple[RealNumber, RealNumber]]): A dictionary mapping each parameter to its new range.

        Raises:
            ValueError: if the provided parameter label is not defined in the list of parameters contained in this object.
        """
