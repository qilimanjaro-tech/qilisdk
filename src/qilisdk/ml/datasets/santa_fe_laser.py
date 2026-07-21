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

from qilisdk.ml.datasets.dataset import Dataset, DatasetSample, build_prediction_sample
from qilisdk.ml.datasets.lorenz import integrate_lorenz


class SantaFeLaser(Dataset):
    r"""
    Santa Fe laser time series (equation-based version).

    The original *Santa Fe Time Series Competition* Data Set A is a recording
    of the chaotic intensity pulsations of a far-infrared :math:`\mathrm{NH_3}`
    laser. To instead generate this data on the fly rather than just using points
    points, this class reproduces the same qualitative dynamics from
    first principles using the single-mode **Lorenz--Haken** laser equations:

    .. math::

        \dot{E} = \sigma (P - E), \quad
        \dot{P} = E (\rho - N) - P, \quad
        \dot{N} = E P - \beta N,

    where :math:`E` is the field amplitude, :math:`P` the polarization and
    :math:`N` the population inversion. The measured quantity is the laser
    **intensity** :math:`I \propto E^2`, which reproduces the behaviour
    of the Santa Fe recording.
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
        """Configure a Santa Fe laser generator.

        Args:
            sigma (float): Field relaxation rate :math:`\\sigma`. Defaults to ``10.0``.
            rho (float): Pump parameter :math:`\\rho`. Defaults to ``28.0``.
            beta (float): Inversion relaxation rate :math:`\\beta`. Defaults to ``8/3``.
            initial_state (tuple[float, float, float]): Initial ``(E, P, N)`` state. Defaults to ``(1.0, 1.0, 1.0)``.
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
        Integrate the laser equations and build an intensity prediction sample.

        Args:
            npoints (int): Number of time steps to produce.

        Returns:
            DatasetSample: The DataSetSample containing the inputs and targets.

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
        field = traj[self._washout :: self._sample_every, 0][:needed]
        intensity = (field**2).reshape(-1, 1)
        return build_prediction_sample(intensity, self._horizon)
