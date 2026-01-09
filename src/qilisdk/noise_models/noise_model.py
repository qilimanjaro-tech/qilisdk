# Copyright 2026 Qilimanjaro Quantum Tech
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
from enum import Enum

from qilisdk.yaml import yaml


class NoiseType(Enum):
    """
    Enum for different types of noise models.
    """
    DIGITAL = "Digital Noise"
    ANALOG = "Analog Noise"
    PARAMETER = "Parameter Noise"


@yaml.register_class
class NoiseBase(ABC):
    """
    Generic Noise Class
    """

    @property
    @abstractmethod
    def noise_type(self) -> NoiseType: ...


class NoiseModel:
    """
    Composite Noise Model consisting of multiple noise passes.
    """
    def __init__(self, noise_passes: list[NoiseBase] | None = None) -> None:
        """
        Initialize a composite noise model consisting of multiple noise passes.

        Args:
            noise_passes (list[NoiseBase] | None): List of noise passes to include in the model.
        """
        self._noise_passes: list[NoiseBase] = noise_passes or []

    @property
    def noise_passes(self) -> list[NoiseBase]:
        """
        Returns the list of noise passes in the composite noise model.
        """
        return self._noise_passes

    def noise_model_types(self) -> list[NoiseType]:
        """
        Returns a list of unique noise types present in the composite noise model.
        """
        return list({noise.noise_type for noise in self._noise_passes})

    def add(self, noise: NoiseBase) -> None:
        """
        Adds a new noise pass to the composite noise model.

        Args:
            noise (NoiseBase): The noise pass to add.
        """
        self._noise_passes.append(noise)
