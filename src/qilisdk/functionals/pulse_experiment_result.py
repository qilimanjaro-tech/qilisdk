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

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

DataArray: TypeAlias = NDArray[np.floating]
DimensionAxisArray: TypeAlias = NDArray[np.floating | np.integer]
Measurement: TypeAlias = tuple[DataArray, list["Dimension"]]


@yaml.register_class
class Dimension:
    """Stores dimension labels and their coordinate values."""

    def __init__(self, labels: list[str], values: list[DimensionAxisArray]) -> None:
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length")

        self.labels = labels
        self.values = values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(labels={self.labels!r}, values={self.values!r})"


@yaml.register_class
class PulseExperimentResult(FunctionalResult):
    def __init__(self, measurements: list[Measurement]) -> None:
        self.measurements = measurements

    def __iter__(self) -> Iterator[Measurement]:
        """Iterate over measurements.

        Returns:
            Iterator[Measurement]: Iterator over tuples containing the data array and a list of dimension dictionaries.
        """
        return iter(self.measurements)

    def __len__(self) -> int:
        return len(self.measurements)

    def get(self, measurement: int = 0) -> Measurement:
        """Retrieves data and dimensions for a specified measurement.

        Args:
            measurement (int, optional): The index of the measurement. Defaults to 0.

        Returns:
            tuple[Measurement]: A tuple containing the data array and a list of dimension dictionaries.
        """

        return self.measurements[measurement]
