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

import numpy as np

from .parameter_pertubation import ParameterPerturbation


class GaussianParameterPerturbation(ParameterPerturbation):
    """Gaussian-distributed parameter noise.

    Perturbs a parameter as:
        value -> value + N(mean, std^2)

    Each call to `perturb` samples a new realization.
    """

    def __init__(
        self,
        *,
        mean: float = 0.0,
        std: float,
    ) -> None:
        if std < 0:
            raise ValueError("std must be >= 0")
        self._mean = float(mean)
        self._std = float(std)
        self._rng = np.random.default_rng()

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    def perturb(self, value: float) -> float:
        delta = self._rng.normal(self._mean, self._std)
        return value + delta
