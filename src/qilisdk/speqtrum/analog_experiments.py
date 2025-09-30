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
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import numpy as np

from qilisdk.functionals.functional import Functional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.speqtrum.analog_experiments_result import ExperimentResult, RabiExperimentResult, T1ExperimentResult
from qilisdk.yaml import yaml

# if TYPE_CHECKING:
#     from ruamel.yaml.nodes import ScalarNode
#     from ruamel.yaml.representer import RoundTripRepresenter

TResult_co = TypeVar("TResult_co", bound=ExperimentResult, covariant=True)


@yaml.register_class
class ExperimentFunctional(Functional, ABC, Generic[TResult_co]):
    ...


@yaml.register_class
class RabiExperiment(ExperimentFunctional[RabiExperimentResult]):
    """A Rabi experiment on a single qubit."""

    result_type: ClassVar[type[RabiExperimentResult]] = RabiExperimentResult

    def __init__(self, qubit_id: int, drive_duration_values: np.ndarray) -> None:
        """
        Args:
            qubit_id (int): The id of the qubit on which the Rabi experiment is performed.
        """
        self._qubit_id = qubit_id
        self._drive_duration_values = drive_duration_values

    @property
    def qubit_id(self) -> int:
        """The id of the qubit on which the Rabi experiment is performed."""
        return self._qubit_id

    @property
    def drive_duration_values(self) -> np.ndarray:
        """The values for the drive duration."""
        return self._drive_duration_values


@yaml.register_class
class T1Experiment(ExperimentFunctional[T1ExperimentResult]):
    ...
#     """A T1 experiment on a single qubit."""

#     result_type: ClassVar[type[T1ExperimentResult]] = T1ExperimentResult

#     def __init__(self, qubit_id: int, pulse_duration: int, delay_between_pulses: int) -> None:
#         """
#         Args:
#             qubit_id (int): The id of the qubit on which the T1 experiment is performed.
#             pulse_duration (int): The duration of the pulse used in the T1 experiment.
#             delay_between_pulses (int): The delay between the pulse and the measurement pulse in the T1 experiment.
#         """
#         self._qubit_id = qubit_id
#         self._pulse_duration = pulse_duration
#         self._delay_between_pulses = delay_between_pulses

#     @property
#     def qubit_id(self) -> int:
#         """The id of the qubit on which the T1 experiment is performed."""
#         return self._qubit_id

#     @property
#     def pulse_duration(self) -> int:
#         """The duration of the pulse used in the T1 experiment."""
#         return self._pulse_duration

#     @property
#     def delay_between_pulses(self) -> int:
#         """The delay between the pulse and the measurement pulse in the T1 experiment."""
#         return self._delay_between_pulses
