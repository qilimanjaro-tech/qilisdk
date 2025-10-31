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
        """Number of tunable parameters defined by the object."""

    @abstractmethod
    def get_parameter_values(self) -> list[float]:
        """Return the current numerical values of the parameters."""

    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        """Return the ordered list of parameter labels."""

    @abstractmethod
    def get_parameters(self) -> dict[str, float]:
        """Return a mapping from parameter labels to their current numerical values."""

    @abstractmethod
    def set_parameter_values(self, values: list[float]) -> None:
        """
        Update all parameter values at once.

        Args:
            values (list[float]): New parameter values ordered consistently with ``get_parameter_names()``.

        Raises:
            ValueError: If ``values`` does not contain exactly ``nparameters`` entries.
        """

    @abstractmethod
    def set_parameters(self, parameters: dict[str, float]) -> None:
        """
        Update a subset of parameters by label.

        Args:
            parameters (dict[str, float]): Mapping from parameter labels to updated numeric values.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """

    @abstractmethod
    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return the ``(lower, upper)`` bounds associated with each parameter."""

    @abstractmethod
    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """
        Update the allowable ranges for the specified parameters.

        Args:
            ranges (dict[str, tuple[float, float]]): Mapping from parameter label to ``(lower, upper)`` bounds.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
