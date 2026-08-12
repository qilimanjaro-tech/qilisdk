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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, TypeAlias, TypeVar, cast

import numpy as np

from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from qilisdk.utils.visualization.dataset_renderers import Transform
    from qilisdk.utils.visualization.style import DatasetStyle

if get_settings().complex_precision == "COMPLEX_64":
    FloatArray: TypeAlias = "NDArray[np.float32]"
else:
    FloatArray: TypeAlias = "NDArray[np.float64]"


# A single integration state: either a scalar (1-D system) or a vector of states.
State = TypeVar("State", float, "FloatArray")


def rk4_step(state: State, dt: float, deriv: Callable[[State], State]) -> State:
    """Advance a state by one fixed-step classic Runge--Kutta (RK4) step.

    Works for both scalar (``float``) and vector (:data:`FloatArray`) states,
    since only NumPy-broadcastable arithmetic is used. For systems whose
    derivative depends on more than the current state (e.g. a delayed value),
    close the extra arguments into ``deriv`` so they stay fixed across the four
    stages.

    Args:
        state (State): Current state ``y_i``.
        dt (float): Integration step.
        deriv (Callable[[State], State]): Function returning ``dy/dt`` for a
            given state.

    Returns:
        State: The state advanced by one step, ``y_{i+1}``.
    """
    k1 = deriv(state)
    k2 = deriv(cast("State", state + 0.5 * dt * k1))
    k3 = deriv(cast("State", state + 0.5 * dt * k2))
    k4 = deriv(cast("State", state + dt * k3))
    return cast("State", state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


@dataclass(frozen=True)
class DatasetSample:
    """
    A generated batch of samples produced by a :class:`Dataset`.
    A sample is an ``(inputs, targets)`` pair.
    """

    inputs: FloatArray
    targets: FloatArray

    def __iter__(self) -> Iterator[FloatArray]:
        """
        Handy iterator over the inputs and targets, so the sample unpacks as
        ``inputs, targets = sample``.

        Yields:
            FloatArray: The inputs array, then the targets array.
        """
        yield self.inputs
        yield self.targets

    def __len__(self) -> int:
        """Return the number of time steps in the sample.

        Returns:
            int: The length of the leading axis of :attr:`inputs`.
        """
        return len(self.inputs)


def build_prediction_sample(series: FloatArray, horizon: int) -> DatasetSample:
    """
    Turn a time series into a prediction sample.

    Given a series of length ``npoints + horizon``, the inputs are the first
    ``npoints`` steps and the targets are the latter ``horizon`` steps.

    Args:
        series (FloatArray): The raw series
        horizon (int): Number of steps ahead to predict. Must be positive.

    Returns:
        DatasetSample: The aligned ``(inputs, targets)`` pair

    Raises:
        ValueError: If ``horizon`` is not positive.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}.")
    return DatasetSample(inputs=series[:-horizon], targets=series[horizon:])


class Dataset(ABC):
    """
    Abstract base class for ML datasets
    """

    # Default plot mode used by :meth:`draw` when none is supplied
    _DEFAULT_DRAW_STYLE: ClassVar[str] = "1d"

    # Labels for the components of the series
    _DRAW_COMPONENT_LABELS: ClassVar[tuple[str, ...] | None] = None

    # Per-mode :class:`DatasetStyle` field defaults
    _DRAW_STYLE_DEFAULTS: ClassVar[dict[str, dict[str, Any]]] = {}

    # Per-mode transforms deciding what each axis shows
    _DRAW_TRANSFORMS: ClassVar[dict[str, Transform]] = {}

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialise the dataset.

        Args:
            seed (int | None): Seed for the random number generator
        """
        self._seed = seed

    @property
    def seed(self) -> int | None:
        """Return the configured random seed.

        Returns:
            int | None: The seed passed at construction time.
        """
        return self._seed

    def _rng(self) -> np.random.Generator:
        """Build a fresh random generator from the configured seed.

        Returns:
            numpy.random.Generator: A seeded (or OS-seeded) generator.
        """
        return np.random.default_rng(self._seed)

    @abstractmethod
    def generate(self, npoints: int) -> DatasetSample:
        """Generate ``npoints`` samples from the dataset.

        Args:
            npoints (int): Number of time steps to produce.

        Returns:
            DatasetSample: The generated ``(inputs, targets)`` pair.
        """
        ...

    @classmethod
    def draw(
        cls,
        sample: DatasetSample | FloatArray,
        style: str | None = None,
        *,
        config: DatasetStyle | None = None,
        transform: Transform | None = None,
        filepath: str | None = None,
    ) -> None:
        """Render a generated :class:`DatasetSample`, or a raw series, with matplotlib.

        The kind of plot is selected by ``style``:

        * ``"1d"`` -- every component of the series against the sample index.
        * ``"2d"`` -- a phase portrait (two coordinates).
        * ``"3d"`` -- a three-dimensional phase portrait (three coordinates).

        What each axis shows is decided by a *transform*: a callable mapping the
        full ``(n_points, n_components)`` series to the coordinates to plot. It may
        reshape, slice or delay-embed the data, not merely select columns. If none
        is given, the dataset's per-mode transform (:attr:`_DRAW_TRANSFORMS`) is
        used, and failing that a dimension-based default (the first components, or
        a delay embedding of a one-dimensional series).

        The plot's *appearance* (theme, fonts, colours, grid, ...) is controlled
        independently via ``config``, mirroring how :class:`ScheduleStyle` and
        :class:`CircuitStyle` customise schedule and circuit plots. Each dataset
        may tailor the defaults of a given mode via :attr:`_DRAW_STYLE_DEFAULTS`;
        any field you set explicitly on ``config`` overrides those defaults.

        Args:
            sample (DatasetSample | FloatArray): A sample produced by
                :meth:`generate`, of which the ``inputs`` are plotted, or an array
                holding the series to plot directly -- so that just one half of a
                sample can be drawn::

                    inputs, targets = MackeyGlass(tau=17.0).generate(2000)
                    MackeyGlass.draw(inputs, style="1d")

                The array is either one-dimensional or shaped
                ``(n_points, n_components)``.
            style (str | None): Plot mode, one of ``"1d"``, ``"2d"`` or ``"3d"``.
                Defaults to the dataset's natural mode.
            config (DatasetStyle | None): Visual style configuration. Defaults to
                :class:`DatasetStyle`, merged over the dataset's per-mode defaults.
            transform (Transform | None): Callable mapping the series to the
                coordinates to plot, overriding the dataset's default view. Returns
                two arrays for ``"2d"``, three for ``"3d"``, or any number of lines
                for ``"1d"``, each optionally paired with an axis label::

                    Lorenz.draw(sample, style="2d", transform=lambda d: [("x", d[:, 0]), ("z", d[:, 2])])
            filepath (str | None): If given, the figure is saved to this path
                (format inferred from the extension) instead of being shown.
        """
        from qilisdk.utils.visualization.dataset_renderers import (  # ruff: ignore[import-outside-top-level]
            MatplotlibDatasetRenderer,
        )

        mode = style or cls._DEFAULT_DRAW_STYLE
        renderer = MatplotlibDatasetRenderer(
            sample,
            mode,
            config=cls._resolve_draw_style(mode, config),
            labels=cls._DRAW_COMPONENT_LABELS,
            title=cls.__name__,
            transform=transform or cls._DRAW_TRANSFORMS.get(mode),
        )
        renderer.plot()
        if filepath:
            renderer.save(filepath)
        else:
            renderer.show()

    @classmethod
    def _resolve_draw_style(cls, mode: str, config: DatasetStyle | None) -> DatasetStyle:
        """Merge the dataset's per-mode style defaults with a user ``config``.

        Fields the user set explicitly on ``config`` always take precedence; any
        field left at its default is filled from :attr:`_DRAW_STYLE_DEFAULTS` for
        the requested ``mode``, falling back to the :class:`DatasetStyle` defaults.

        Args:
            mode (str): The plot mode being drawn.
            config (DatasetStyle | None): The user-supplied style, if any.

        Returns:
            DatasetStyle: The effective style to render with.
        """
        from qilisdk.utils.visualization.style import DatasetStyle  # ruff: ignore[import-outside-top-level]

        defaults = cls._DRAW_STYLE_DEFAULTS.get(mode, {})
        if not defaults:
            return config or DatasetStyle()
        if config is None:
            return DatasetStyle(**defaults)
        user_set = {name: getattr(config, name) for name in config.model_fields_set}
        return DatasetStyle(**{**defaults, **user_set})
