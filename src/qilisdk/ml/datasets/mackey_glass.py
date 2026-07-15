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


class MackeyGlass(Dataset):
    r"""Mackey--Glass chaotic time series.

    The Mackey--Glass system is a nonlinear delay differential equation that,
    for suitable delays, produces a well-known chaotic attractor widely used as
    a reservoir-computing prediction benchmark:

    .. math::

        \frac{dx}{dt} = \beta\, \frac{x(t - \tau)}{1 + x(t - \tau)^{n}}
                        - \gamma\, x(t).

    With the standard parameters :math:`\beta = 0.2`, :math:`\gamma = 0.1`,
    :math:`n = 10`, the behaviour is set by the delay :math:`\tau`: the series
    is periodic for small :math:`\tau`, mildly chaotic at :math:`\tau = 17`, and
    increasingly chaotic beyond (:math:`\tau = 30` is also common).

    The equation is integrated with a fixed-step RK4 scheme at resolution ``dt``
    and then sub-sampled every ``sample_every`` steps. :meth:`generate` returns a
    ``horizon``-step-ahead prediction task: ``inputs`` is the series and
    ``targets`` is the same series shifted forward by ``horizon``, both shaped
    ``(npoints, 1)``.
    """

    def __init__(
        self,
        *,
        tau: float = 17.0,
        beta: float = 0.2,
        gamma: float = 0.1,
        n: float = 10.0,
        x0: float = 1.2,
        dt: float = 0.1,
        sample_every: int = 10,
        washout: int = 1000,
        horizon: int = 1,
        seed: int | None = None,
    ) -> None:
        """Configure a Mackey--Glass generator.

        Args:
            tau (float): Delay :math:`\\tau`. Defaults to ``17.0``.
            beta (float): Production coefficient :math:`\\beta`. Defaults to ``0.2``.
            gamma (float): Decay coefficient :math:`\\gamma`. Defaults to ``0.1``.
            n (float): Nonlinearity exponent :math:`n`. Defaults to ``10.0``.
            x0 (float): Constant initial-history value. Defaults to ``1.2``.
            dt (float): Internal integration step. Defaults to ``0.1``.
            sample_every (int): Sub-sampling stride applied to the integrated
                trajectory. Defaults to ``10`` (i.e. an effective step of 1.0).
            washout (int): Number of integration steps discarded as transient
                before sampling begins. Defaults to ``1000``.
            horizon (int): Prediction horizon in sampled steps. Defaults to ``1``.
            seed (int | None): Unused; the system is deterministic. Kept for a
                uniform interface. Defaults to ``None``.

        Raises:
            ValueError: If ``tau``, ``dt`` or ``sample_every`` is not positive.
        """
        super().__init__(seed=seed)
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}.")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if sample_every < 1:
            raise ValueError(f"sample_every must be a positive integer, got {sample_every}.")
        self._tau = tau
        self._beta = beta
        self._gamma = gamma
        self._n = n
        self._x0 = x0
        self._dt = dt
        self._sample_every = sample_every
        self._washout = washout
        self._horizon = horizon

    def generate(self, npoints: int) -> DatasetSample:
        """Integrate the Mackey--Glass equation and build a prediction sample.

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
        tau_steps = max(1, round(self._tau / self._dt))
        n_steps = self._washout + needed * self._sample_every

        beta, gamma, n, dt = self._beta, self._gamma, self._n, self._dt

        def deriv(x: float, x_delayed: float) -> float:
            return beta * x_delayed / (1.0 + x_delayed**n) - gamma * x

        traj = np.empty(tau_steps + n_steps + 1, dtype=np.float64)
        traj[: tau_steps + 1] = self._x0

        for i in range(tau_steps, tau_steps + n_steps):
            xd = traj[i - tau_steps]
            xi = traj[i]
            k1 = deriv(xi, xd)
            k2 = deriv(xi + 0.5 * dt * k1, xd)
            k3 = deriv(xi + 0.5 * dt * k2, xd)
            k4 = deriv(xi + dt * k3, xd)
            traj[i + 1] = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        start = tau_steps + self._washout
        sampled = traj[start :: self._sample_every][:needed]
        return build_prediction_sample(sampled.reshape(-1, 1), self._horizon)
