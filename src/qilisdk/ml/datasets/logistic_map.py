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

from qilisdk.ml.datasets.dataset import Dataset, DatasetSample, build_prediction_sample


class LogisticMap(Dataset):
    r"""Logistic map, the simplest one-dimensional route to chaos.

    The logistic map is the recurrence

    .. math::

        x_{n+1} = r\, x_n (1 - x_n),

    which becomes chaotic as the growth rate :math:`r` approaches 4 (the
    default :math:`r = 3.9` sits well inside the chaotic regime).
    :meth:`generate` returns a ``horizon``-step-ahead prediction task, so
    ``inputs`` and ``targets`` are shaped ``(npoints, 1)``.
    """

    def __init__(
        self,
        *,
        r: float = 3.9,
        x0: float = 0.5,
        washout: int = 100,
        horizon: int = 1,
        seed: int | None = None,
    ) -> None:
        """Configure a logistic map generator.

        Args:
            r (float): Growth-rate parameter :math:`r`. Defaults to ``3.9``.
            x0 (float): Initial value in ``[0, 1]``. Defaults to ``0.5``.
            washout (int): Number of initial iterations discarded as transient.
                Defaults to ``100``.
            horizon (int): Prediction horizon in steps. Defaults to ``1``.
            seed (int | None): Unused; the system is deterministic. Defaults to
                ``None``.

        Raises:
            ValueError: If ``x0`` is not in the closed interval ``[0, 1]``.
        """
        super().__init__(seed=seed)
        if not 0.0 <= x0 <= 1.0:
            raise ValueError(f"x0 must lie in [0, 1], got {x0}.")
        self._r = r
        self._x0 = x0
        self._washout = washout
        self._horizon = horizon

    def generate(self, npoints: int) -> DatasetSample:
        """Iterate the logistic map and build a prediction sample.

        Args:
            npoints (int): Number of time steps to produce.

        Returns:
            DatasetSample: A ``horizon``-step-ahead prediction pair, both arrays
            shaped ``(npoints, 1)``.

        Raises:
            ValueError: If ``npoints`` is not positive.
        """
        if npoints < 1:
            raise ValueError(f"npoints must be a positive integer, got {npoints}.")

        needed = npoints + self._horizon
        total = self._washout + needed
        series = np.empty(total, dtype=np.float64)
        series[0] = self._x0
        r = self._r
        for i in range(total - 1):
            xi = series[i]
            series[i + 1] = r * xi * (1.0 - xi)

        sampled = series[self._washout :].reshape(-1, 1)
        return build_prediction_sample(sampled, self._horizon)
