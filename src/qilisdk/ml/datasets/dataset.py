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
from typing import TYPE_CHECKING, Callable, Iterator, TypeAlias, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

FloatArray: TypeAlias = "NDArray[np.float64]"
"""Alias for a NumPy array of 64-bit floats, used throughout the datasets module."""

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

    def __iter__(self) -> Iterator[tuple[FloatArray, FloatArray]]:
        """
        Handy iterator over the inputs and targets

        Yields:
            tuple[FloatArray, FloatArray]: Yield each input target tuple, in order
        """
        for i in range(len(self.inputs)):
            yield self.inputs[i], self.targets[i]

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
