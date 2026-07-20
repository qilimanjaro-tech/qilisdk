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

from qilisdk.ml.datasets.dataset import Dataset, DatasetSample, FloatArray, build_prediction_sample, rk4_step


def integrate_lorenz(
    *,
    sigma: float,
    rho: float,
    beta: float,
    initial_state: tuple[float, float, float],
    dt: float,
    n_steps: int,
) -> FloatArray:
    r"""
    Integrate the Lorenz system with a fixed-step RK4 scheme.

    The Lorenz equations are

    .. math::

        \dot{x} = \sigma (y - x), \quad
        \dot{y} = x (\rho - z) - y, \quad
        \dot{z} = x y - \beta z.

    Args:
        sigma (float): Prandtl number :math:`\sigma`.
        rho (float): Rayleigh number :math:`\rho`.
        beta (float): Geometric factor :math:`\beta`.
        initial_state (tuple[float, float, float]): Initial ``(x, y, z)`` state.
        dt (float): Integration step.
        n_steps (int): Number of RK4 steps to take.

    Returns:
        FloatArray: The trajectory, shaped ``(n_steps + 1, 3)``.
    """

    def deriv(state: FloatArray) -> FloatArray:
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=np.float64)

    traj = np.empty((n_steps + 1, 3), dtype=np.float64)
    traj[0] = np.asarray(initial_state, dtype=np.float64)
    for i in range(n_steps):
        traj[i + 1] = rk4_step(traj[i], dt, deriv)
    return traj


class Lorenz(Dataset):
    r"""
    Lorenz attractor, a chaotic dynamical system.
    """

    def __init__(
        self,
        *,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0),
        dt: float = 0.01,
        sample_every: int = 5,
        washout: int = 1000,
        horizon: int = 1,
        seed: int | None = None,
    ) -> None:
        """Configure a Lorenz generator.

        Args:
            sigma (float): Prandtl number :math:`\\sigma`. Defaults to ``10.0``.
            rho (float): Rayleigh number :math:`\\rho`. Defaults to ``28.0``.
            beta (float): Geometric factor :math:`\\beta`. Defaults to ``8/3``.
            initial_state (tuple[float, float, float]): Initial ``(x, y, z)`` state. Defaults to ``(1.0, 1.0, 1.0)``.
            dt (float): Internal integration step. Defaults to ``0.01``.
            sample_every (int): Sub-sampling stride. Defaults to ``5``.
            washout (int): Integration steps discarded as transient. Defaults to ``1000``.
            horizon (int): Prediction horizon in sampled steps. Defaults to ``1``.
            seed (int | None): Unused; the system is deterministic. Defaults to ``None``.

        Raises:
            ValueError: If ``dt`` or ``sample_every`` is not positive.
        """
        super().__init__(seed=seed)
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if sample_every < 1:
            raise ValueError(f"sample_every must be a positive integer, got {sample_every}.")
        self._sigma = sigma
        self._rho = rho
        self._beta = beta
        self._initial_state = initial_state
        self._dt = dt
        self._sample_every = sample_every
        self._washout = washout
        self._horizon = horizon

    def generate(self, npoints: int) -> DatasetSample:
        """
        Integrate the Lorenz system and build a prediction sample.

        Args:
            npoints (int): Number of time steps to produce.

        Returns:

        Raises:
            ValueError: If ``npoints`` is not positive.
        """
        if npoints < 1:
            raise ValueError(f"npoints must be a positive integer, got {npoints}.")

        needed = npoints + self._horizon
        n_steps = self._washout + needed * self._sample_every
        traj = integrate_lorenz(
            sigma=self._sigma,
            rho=self._rho,
            beta=self._beta,
            initial_state=self._initial_state,
            dt=self._dt,
            n_steps=n_steps,
        )
        sampled = traj[self._washout :: self._sample_every][:needed]
        return build_prediction_sample(sampled, self._horizon)
