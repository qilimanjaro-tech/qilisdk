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

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from qilisdk.ml.datasets.dataset import Dataset, DatasetSample, build_prediction_sample, rk4_step

if TYPE_CHECKING:
    from qilisdk.ml.datasets.dataset import FloatArray
    from qilisdk.utils.visualization.dataset_renderers import Transform
    from qilisdk.utils.visualization.style import DatasetStyle

_DEFAULT_ATTRACTOR_DELAY = 17


class MackeyGlass(Dataset):
    r"""
    Mackey--Glass chaotic time series.

    The Mackey--Glass system is a nonlinear delay differential equation that
    produces a well-known chaotic attractor:

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

    _DEFAULT_DRAW_STYLE = "1d"
    _DRAW_COMPONENT_LABELS = ("x",)

    _DRAW_STYLE_DEFAULTS: ClassVar[dict[str, dict[str, Any]]] = {
        "2d": {"title": "Mackey-Glass attractor"},
        "3d": {"title": "Mackey-Glass attractor"},
    }

    @staticmethod
    def _attractor_transform(mode: str, delay: int) -> Transform | None:
        """
        Build the delay-embedding transform for a plot mode.

        The classic Mackey--Glass attractor is the plot of ``(P(t), P(t - tau))``,
        extended in three dimensions with ``P(t - 2 tau)``.

        Args:
            mode (str): Plot mode, one of ``"1d"``, ``"2d"`` or ``"3d"``.
            delay (int): Embedding delay :math:`\\tau`, in sampled steps.

        Returns:
            Transform | None: The transform for ``mode``, or ``None`` if the mode
            needs no delay embedding.
        """
        if mode == "2d":
            return lambda d: [
                ("P(t)", d[delay:, 0]),
                (f"P(t - {delay})", d[:-delay, 0]),
            ]
        if mode == "3d":
            return lambda d: [
                ("P(t)", d[2 * delay :, 0]),
                (f"P(t - {delay})", d[delay:-delay, 0]),
                (f"P(t - {2 * delay})", d[: -2 * delay, 0]),
            ]
        return None

    @classmethod
    def draw(
        cls,
        sample: DatasetSample | FloatArray,
        style: str | None = None,
        *,
        config: DatasetStyle | None = None,
        transform: Transform | None = None,
        filepath: str | None = None,
        attractor_delay: int = _DEFAULT_ATTRACTOR_DELAY,
    ) -> None:
        """
        Render a generated :class:`DatasetSample`, or a raw series, with matplotlib.

        Overrides :meth:`Dataset.draw`, except allows an extra arg: ``attractor_delay``.

        Args:
            sample (DatasetSample | FloatArray): A sample produced by
                :meth:`generate`, of which the ``inputs`` are plotted, or an array
                holding the series to plot directly.
            style (str | None): Plot mode, one of ``"1d"``, ``"2d"`` or ``"3d"``.
                Defaults to the dataset's natural mode.
            config (DatasetStyle | None): Visual style configuration. Defaults to
                :class:`DatasetStyle`, merged over the dataset's per-mode defaults.
            transform (Transform | None): Callable mapping the series to the
                coordinates to plot, overriding the delay embedding entirely.
            filepath (str | None): If given, the figure is saved to this path
                (format inferred from the extension) instead of being shown.
            attractor_delay (int): Embedding delay, in sampled steps, used by the
                default ``"2d"``/``"3d"`` transforms. Defaults to ``17``.

        Raises:
            ValueError: If ``attractor_delay`` is not positive.
        """
        if attractor_delay < 1:
            raise ValueError(f"attractor_delay must be a positive integer, got {attractor_delay}.")
        mode = style or cls._DEFAULT_DRAW_STYLE
        super().draw(
            sample,
            mode,
            config=config,
            transform=transform or cls._attractor_transform(mode, attractor_delay),
            filepath=filepath,
        )

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
        """
        Configure a Mackey--Glass generator.

        Args:
            tau (float): Delay :math:`\\tau`. Defaults to ``17.0``.
            beta (float): Production coefficient :math:`\\beta`. Defaults to ``0.2``.
            gamma (float): Decay coefficient :math:`\\gamma`. Defaults to ``0.1``.
            n (float): Nonlinearity exponent :math:`n`. Defaults to ``10.0``.
            x0 (float): Constant initial-history value. Defaults to ``1.2``.
            dt (float): Internal integration step. Defaults to ``0.1``.
            sample_every (int): Sub-sampling stride. Defaults to ``10``.
            washout (int): Number of integration steps discarded before sampling begins. Defaults to ``1000``.
            horizon (int): Prediction horizon in sampled steps. Defaults to ``1``.
            seed (int | None): Seed for the random number generator. Defaults to ``None``.

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
        """
        Integrate the Mackey--Glass equation and build a prediction sample.

        This produces a single time series of length ``npoints + horizon``, discarding
        the first ``washout`` steps, and then sub-sampling every ``sample_every` steps.
        The resulting series is split into ``inputs`` and ``targets``, where
        ``targets`` is the same series shifted forward by ``horizon``.

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
            traj[i + 1] = rk4_step(traj[i], dt, lambda x, xd=xd: deriv(x, xd))

        start = tau_steps + self._washout
        sampled = traj[start :: self._sample_every][:needed]
        return build_prediction_sample(sampled.reshape(-1, 1), self._horizon)
