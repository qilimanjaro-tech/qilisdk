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


class HenonMap(Dataset):
    r"""Hénon map, a two-dimensional discrete chaotic system.

    The Hénon map is the recurrence

    .. math::

        x_{n+1} = 1 - a\, x_n^2 + y_n, \qquad
        y_{n+1} = b\, x_n,

    which is chaotic for the canonical parameters :math:`a = 1.4`,
    :math:`b = 0.3`. :meth:`generate` returns a ``horizon``-step-ahead
    prediction task over the two-dimensional state, so ``inputs`` and
    ``targets`` are shaped ``(npoints, 2)``.
    """

    def __init__(
        self,
        *,
        a: float = 1.4,
        b: float = 0.3,
        state0: tuple[float, float] = (0.0, 0.0),
        washout: int = 100,
        horizon: int = 1,
        seed: int | None = None,
    ) -> None:
        """Configure a Hénon map generator.

        Args:
            a (float): Nonlinearity parameter :math:`a`. Defaults to ``1.4``.
            b (float): Contraction parameter :math:`b`. Defaults to ``0.3``.
            state0 (tuple[float, float]): Initial ``(x, y)`` state. Defaults to
                ``(0.0, 0.0)``.
            washout (int): Number of initial iterations discarded as transient.
                Defaults to ``100``.
            horizon (int): Prediction horizon in steps. Defaults to ``1``.
            seed (int | None): Unused; the system is deterministic. Defaults to
                ``None``.
        """
        super().__init__(seed=seed)
        self._a = a
        self._b = b
        self._state0 = state0
        self._washout = washout
        self._horizon = horizon

    def generate(self, npoints: int) -> DatasetSample:
        """Iterate the Hénon map and build a prediction sample.

        Args:
            npoints (int): Number of time steps to produce.

        Returns:
            DatasetSample: A ``horizon``-step-ahead prediction pair, both arrays
            shaped ``(npoints, 2)``.

        Raises:
            ValueError: If ``npoints`` is not positive.
        """
        if npoints < 1:
            raise ValueError(f"npoints must be a positive integer, got {npoints}.")

        needed = npoints + self._horizon
        total = self._washout + needed
        traj = np.empty((total, 2), dtype=np.float64)
        traj[0] = np.asarray(self._state0, dtype=np.float64)
        for i in range(total - 1):
            x, y = traj[i]
            traj[i + 1] = (1.0 - self._a * x * x + y, self._b * x)

        sampled = traj[self._washout :]
        return build_prediction_sample(sampled, self._horizon)
