# Copyright 2025 Qilimanjaro Quantum Tech
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

from abc import ABC
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from qilisdk.functionals.functional import Functional
from qilisdk.speqtrum.experiments.experiment_result import (
    ExperimentResult,
    RabiExperimentResult,
    T1ExperimentResult,
)
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    import numpy as np

TResult_co = TypeVar("TResult_co", bound=ExperimentResult, covariant=True)


@yaml.register_class
class ExperimentFunctional(Functional, ABC, Generic[TResult_co]):
    """Abstract base class for single-qubit experiment functionals.

    This class serves as a generic interface for defining quantum
    characterization experiments such as Rabi or T1. Each subclass
    specifies a concrete `ExperimentResult` type and the corresponding
    sweep parameters.
    """

    def __init__(self, qubit: int) -> None:
        """Initialize the experiment functional.

        Args:
            qubit (int): The physical qubit index on which the experiment is performed.
        """
        self._qubit = qubit

    @property
    def qubit(self) -> int:
        """The physical qubit index on which the experiment is performed.

        Returns:
            int: Index of the qubit.
        """
        return self._qubit


@yaml.register_class
class RabiExperiment(ExperimentFunctional[RabiExperimentResult]):
    """Rabi experiment functional for a single qubit.

    This functional defines a standard Rabi oscillation experiment where
    the drive pulse duration is swept to measure the oscillatory response
    of the qubit under continuous driving.
    """

    result_type: ClassVar[type[RabiExperimentResult]] = RabiExperimentResult
    """Result type returned by this functional."""

    def __init__(self, qubit: int, drive_duration_values: np.ndarray) -> None:
        """Initialize a Rabi experiment functional.

        Args:
            qubit (int): The physical qubit index on which the experiment is performed.
            drive_duration_values (np.ndarray): Array of drive pulse durations (in nanoseconds)
                used to sweep the experiment.
        """
        super().__init__(qubit=qubit)
        self._drive_duration_values = drive_duration_values

    @property
    def drive_duration_values(self) -> np.ndarray:
        """Drive pulse duration sweep values.

        Returns:
            np.ndarray: The set of drive durations (in nanoseconds) used in the Rabi experiment.
        """
        return self._drive_duration_values


@yaml.register_class
class T1Experiment(ExperimentFunctional[T1ExperimentResult]):
    """T1 relaxation experiment functional for a single qubit.

    This functional defines a standard T1 (energy relaxation) experiment,
    where the delay between excitation and measurement is varied to extract
    the relaxation time constant of the qubit.
    """

    result_type: ClassVar[type[T1ExperimentResult]] = T1ExperimentResult
    """Result type returned by this functional."""

    def __init__(self, qubit: int, wait_duration_values: np.ndarray) -> None:
        """Initialize a T1 experiment functional.

        Args:
            qubit (int): The physical qubit index on which the experiment is performed.
            wait_duration_values (np.ndarray): Array of waiting times (in nanoseconds)
                between excitation and measurement.
        """
        super().__init__(qubit=qubit)
        self._wait_duration_values: np.ndarray = wait_duration_values

    @property
    def wait_duration_values(self) -> np.ndarray:
        """Waiting time sweep values.

        Returns:
            np.ndarray: The set of delay durations (in nanoseconds) used in the T1 experiment.
        """
        return self._wait_duration_values
