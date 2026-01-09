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
    """Enumeration of supported noise model categories."""

    DIGITAL = "Digital Noise"
    ANALOG = "Analog Noise"
    PARAMETER = "Parameter Noise"


@yaml.register_class
class NoiseBase(ABC):
    """Base class for all noise model definitions."""

    @property
    @abstractmethod
    def noise_type(self) -> NoiseType:
        """Return the noise category for the concrete noise model."""


class NoiseModel:
    """Container for grouping noise passes into a single noise model."""

    def __init__(self, noise_passes: list[NoiseBase] | None = None) -> None:
        """
        Args:
            noise_passes (list[NoiseBase] | None, optional): Noise passes composing the model.
                Defaults to None.
        """
        self._noise_passes: list[NoiseBase] = noise_passes or []

    @property
    def noise_passes(self) -> list[NoiseBase]:
        """Return the list of noise passes registered in the model."""
        return self._noise_passes

    def noise_model_types(self) -> list[NoiseType]:
        """Return the unique noise categories present in the model."""
        return list({noise.noise_type for noise in self._noise_passes})

    def add(self, noise: NoiseBase) -> None:
        """Append a noise pass to the model."""
        self._noise_passes.append(noise)
