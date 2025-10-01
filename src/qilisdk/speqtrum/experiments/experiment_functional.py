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
from qilisdk.speqtrum.experiments.experiment_functional_results import (
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
    def __init__(self, qubit: int) -> None:
        """
        Args:
            qubit (int): The id of the qubit on which the Rabi experiment is performed.
        """
        self._qubit = qubit

    @property
    def qubit(self) -> int:
        """The id of the qubit on which the Rabi experiment is performed."""
        return self._qubit



@yaml.register_class
class RabiExperiment(ExperimentFunctional[RabiExperimentResult]):
    """A Rabi experiment on a single qubit."""

    result_type: ClassVar[type[RabiExperimentResult]] = RabiExperimentResult

    def __init__(self, qubit: int, drive_duration_values: np.ndarray) -> None:
        """
        Args:
            qubit (int): The id of the qubit on which the Rabi experiment is performed.
        """
        super().__init__(qubit=qubit)
        self._drive_duration_values = drive_duration_values

    @property
    def drive_duration_values(self) -> np.ndarray:
        """The values for the drive duration."""
        return self._drive_duration_values


@yaml.register_class
class T1Experiment(ExperimentFunctional[T1ExperimentResult]):
    """A T1 experiment on a single qubit."""

    result_type: ClassVar[type[T1ExperimentResult]] = T1ExperimentResult
