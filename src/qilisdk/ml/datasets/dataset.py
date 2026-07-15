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
from typing import TYPE_CHECKING, Iterator, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

FloatArray: TypeAlias = "NDArray[np.float64]"
"""Alias for a NumPy array of 64-bit floats, used throughout the datasets module."""


@dataclass(frozen=True)
class DatasetSample:
    """A generated batch of samples produced by a :class:`Dataset`.

    A sample is an ``(inputs, targets)`` pair laid out for supervised
    time-series learning. Both arrays share their leading axis (one entry per
    time step); trailing axes hold the feature dimensions of the system (for
    example, three columns for the Lorenz attractor).

    The container supports tuple unpacking, so both of the following work::

        sample = dataset.generate(1000)
        inputs, targets = dataset.generate(1000)
    """

    inputs: FloatArray
    """Model inputs, shaped ``(npoints, ...)``."""

    targets: FloatArray
    """Prediction targets aligned with :attr:`inputs`, shaped ``(npoints, ...)``."""

    def __iter__(self) -> Iterator[FloatArray]:
        """Iterate over ``(inputs, targets)`` to support tuple unpacking.

        Yields:
            FloatArray: First :attr:`inputs`, then :attr:`targets`.
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
    """Turn a time series into a one-to-``horizon``-step-ahead prediction sample.

    Given a series of length ``npoints + horizon``, the inputs are the first
    ``npoints`` steps and the targets are the same steps shifted forward by
    ``horizon``, so that ``targets[t] == series[t + horizon]``.

    Args:
        series (FloatArray): The raw series, shaped ``(npoints + horizon, ...)``.
        horizon (int): Number of steps ahead to predict. Must be positive.

    Returns:
        DatasetSample: The aligned ``(inputs, targets)`` pair, each shaped
        ``(npoints, ...)``.

    Raises:
        ValueError: If ``horizon`` is not positive.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}.")
    return DatasetSample(inputs=series[:-horizon], targets=series[horizon:])


class Dataset(ABC):
    """Abstract base class for equation-based dataset generators.

    Concrete datasets synthesise their data on the fly from a governing
    equation (a map, an ordinary/delay differential equation, or a driven
    recurrence) rather than loading pre-recorded points. Every dataset is
    reproducible: fixing ``seed`` makes repeated :meth:`generate` calls return
    identical output.

    Subclasses implement :meth:`generate`, which returns a
    :class:`DatasetSample`.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialise the dataset.

        Args:
            seed (int | None): Seed for the random number generator used by
                stochastic datasets. Deterministic datasets ignore it. Defaults
                to ``None``, which draws fresh entropy from the operating system.
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

        A new generator is created on every call so that, for a fixed seed,
        successive :meth:`generate` calls reproduce the same sequence.

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
