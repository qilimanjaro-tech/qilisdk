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

from qilisdk.functionals.functional import Functional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.speqtrum.analog_experiments_result import RabiExperimentResult, T1ExperimentResult
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from ruamel.yaml.nodes import ScalarNode
    from ruamel.yaml.representer import RoundTripRepresenter

TResult_co = TypeVar("TResult_co", bound=FunctionalResult, covariant=True)


@yaml.register_class
class ExperimentType(str, Enum):
    Rabi = "rabi_experiment"
    T1 = "t1_experiment"

    @classmethod
    def to_yaml(cls, representer: RoundTripRepresenter, node: ExperimentType) -> ScalarNode:
        """
        Method to be called automatically during YAML serialization.

        Returns:
            ScalarNode: The YAML scalar node representing the Operation.
        """
        return representer.represent_scalar("!Operation", f"{node.value}")

    @classmethod
    def from_yaml(cls, _, node: ScalarNode) -> ExperimentType:
        """
        Method to be called automatically during YAML deserialization.

        Returns:
            Operation: The Operation instance created from the YAML node value.
        """
        return cls(node.value)


@yaml.register_class
class ExperimentFunctional(Functional, ABC, Generic[TResult_co]):
    experiment_type: ExperimentType


@yaml.register_class
class RabiExperiment(ExperimentFunctional[RabiExperimentResult]):
    """A Rabi experiment on a single qubit."""

    result_type: ClassVar[type[RabiExperimentResult]] = RabiExperimentResult
    experiment_type = ExperimentType.Rabi

    def __init__(self, qubit_id: int) -> None:
        """
        Args:
            qubit_id (int): The id of the qubit on which the Rabi experiment is performed.
        """
        self._qubit_id = qubit_id

    @property
    def qubit_id(self) -> int:
        """The id of the qubit on which the Rabi experiment is performed."""
        return self._qubit_id


@yaml.register_class
class T1Experiment(ExperimentFunctional[T1ExperimentResult]):
    """A T1 experiment on a single qubit."""

    result_type: ClassVar[type[T1ExperimentResult]] = T1ExperimentResult
    experiment_type = ExperimentType.T1

    def __init__(self, qubit_id: int, pulse_duration: int, delay_between_pulses: int) -> None:
        """
        Args:
            qubit_id (int): The id of the qubit on which the T1 experiment is performed.
            pulse_duration (int): The duration of the pulse used in the T1 experiment.
            delay_between_pulses (int): The delay between the pulse and the measurement pulse in the T1 experiment.
        """
        self._qubit_id = qubit_id
        self._pulse_duration = pulse_duration
        self._delay_between_pulses = delay_between_pulses

    @property
    def qubit_id(self) -> int:
        """The id of the qubit on which the T1 experiment is performed."""
        return self._qubit_id

    @property
    def pulse_duration(self) -> int:
        """The duration of the pulse used in the T1 experiment."""
        return self._pulse_duration

    @property
    def delay_between_pulses(self) -> int:
        """The delay between the pulse and the measurement pulse in the T1 experiment."""
        return self._delay_between_pulses
