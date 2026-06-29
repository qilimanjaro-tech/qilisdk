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
from typing import ClassVar, Generic, TypeVar

from qilisdk.experiments.experiment_result import ExperimentResult
from qilisdk.functionals.functional import Functional
from qilisdk.yaml import yaml

TResult_co = TypeVar("TResult_co", bound=ExperimentResult, covariant=True)


@yaml.register_class
class ExperimentFunctional(Functional, ABC, Generic[TResult_co]):
    result_type: ClassVar[type[ExperimentResult]]
    """Concrete :class:`~qilisdk.experiments.experiment_result.ExperimentResult` subclass returned."""

    """Abstract base class for single-qubit experiment functionals.

    This class serves as a generic interface for defining quantum
    characterization experiments such as Rabi or T1. Each subclass
    specifies a concrete `ExperimentResult` type and the corresponding
    sweep parameters.

    Concrete experiment functionals are provided by the ``qili-experiments``
    plugin library, not by qilisdk.
    """

    def __init__(self, qubit: int, averages: int) -> None:
        """Initialize the experiment functional.

        Args:
            qubit (int): The physical qubit index on which the experiment is performed.
            averages (int): Number of averages to acquire for the experiment.
        """
        self._qubit = qubit
        self._averages = averages

    @property
    def qubit(self) -> int:
        """The physical qubit index on which the experiment is performed.

        Returns:
            int: Index of the qubit.
        """
        return self._qubit

    @property
    def averages(self) -> int:
        """
        Number of averages to acquire for the experiment.

        Returns:
            int: Number of averages.
        """

        return self._averages
