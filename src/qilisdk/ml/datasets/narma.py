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

from qilisdk.ml.datasets.dataset import Dataset, DatasetSample

# Above this order the plain recurrence can diverge, so the update is squashed
# with a ``tanh`` to keep the series bounded (standard practice in the literature).
_SQUASH_ORDER_THRESHOLD = 10


class NARMA(Dataset):
    r"""
    Nonlinear Auto-Regressive Moving Average (NARMA) benchmark.

    NARMA is a classic system-identification task. A
    random input stream :math:`u(t) \sim \mathcal{U}(0, 0.5)` drives an
    order-:math:`n` nonlinear recurrence whose output :math:`y(t)` must be
    predicted from :math:`u`:

    .. math::

        y(t+1) = \alpha\, y(t)
                 + \beta\, y(t) \sum_{i=0}^{n-1} y(t-i)
                 + \gamma\, u(t-n+1)\, u(t)
                 + \delta.

    The default coefficients :math:`(\alpha, \beta, \gamma, \delta) =
    (0.3, 0.05, 1.5, 0.1)` correspond to the ubiquitous ``NARMA10`` task
    (``order=10``).
    """

    def __init__(
        self,
        order: int = 10,
        *,
        alpha: float = 0.3,
        beta: float = 0.05,
        gamma: float = 1.5,
        delta: float = 0.1,
        input_range: tuple[float, float] = (0.0, 0.5),
        seed: int | None = None,
    ) -> None:
        """Configure a NARMA generator.

        Args:
            order (int): Memory order :math:`n` of the recurrence. Defaults to ``10``.
            alpha (float): Linear self-feedback coefficient. Defaults to ``0.3``.
            beta (float): Nonlinear memory coefficient. Defaults to ``0.05``.
            gamma (float): Input-coupling coefficient. Defaults to ``1.5``.
            delta (float): Constant bias term. Defaults to ``0.1``.
            input_range (tuple[float, float]): Bounds of the driving signal. Defaults to ``(0.0, 0.5)``.
            seed (int | None): Seed for the driving signal. Defaults to ``None``.

        Raises:
            ValueError: If ``order`` is less than 1.
        """
        super().__init__(seed=seed)
        if order < 1:
            raise ValueError(f"order must be a positive integer, got {order}.")
        self._order = order
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._input_range = input_range

    @property
    def order(self) -> int:
        """Return the memory order of the recurrence.

        Returns:
            int: The configured order :math:`n`.
        """
        return self._order

    def generate(self, npoints: int) -> DatasetSample:
        """Generate a NARMA input/output stream.

        Args:
            npoints (int): Number of time steps to produce.

        Returns:
            DatasetSample: The DataSetSample containing the inputs and targets.

        Raises:
            ValueError: If ``npoints`` is not positive.
        """
        if npoints < 1:
            raise ValueError(f"npoints must be a positive integer, got {npoints}.")

        rng = self._rng()
        low, high = self._input_range
        u = rng.uniform(low, high, size=npoints)
        y = np.zeros(npoints, dtype=np.float64)
        n = self._order
        squash = n > _SQUASH_ORDER_THRESHOLD

        for t in range(n - 1, npoints - 1):
            memory = float(np.sum(y[t - n + 1 : t + 1]))
            update = self._alpha * y[t] + self._beta * y[t] * memory + self._gamma * u[t - n + 1] * u[t] + self._delta
            y[t + 1] = np.tanh(update) if squash else update

        return DatasetSample(inputs=u.reshape(-1, 1), targets=y.reshape(-1, 1))
