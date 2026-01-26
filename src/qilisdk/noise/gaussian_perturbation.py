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

import numpy as np

from .parameter_perturbation import ParameterPerturbation


class GaussianPerturbation(ParameterPerturbation):
    """Gaussian-distributed parameter perturbation.

    Each call to ``perturb`` adds a random offset drawn from N(mean, std^2).
    """

    def __init__(
        self,
        *,
        mean: float = 0.0,
        std: float,
        seed: int = 42,
    ) -> None:
        """Args:
            mean (float): Mean of the Gaussian offset.
            std (float): Standard deviation of the Gaussian offset (must be >= 0).
            seed (int): Seed for the random number generator.

        Raises:
            ValueError: If std is negative.
        """
        if std < 0:
            raise ValueError("std must be >= 0")
        self._mean = float(mean)
        self._std = float(std)
        self._rng = np.random.default_rng(seed)

    @property
    def mean(self) -> float:
        """Return the mean of the Gaussian offset.

        Returns:
            The mean value.
        """
        return self._mean

    @property
    def std(self) -> float:
        """Return the standard deviation of the Gaussian offset.

        Returns:
            The standard deviation value.
        """
        return self._std

    def perturb(self, value: float) -> float:
        delta = self._rng.normal(self._mean, self._std)
        return value + delta
