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

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class ExperimentResult(FunctionalResult): ...


@yaml.register_class
class RabiExperimentResult(ExperimentResult):
    """Result of a Rabi experiment."""

    def __init__(self, s21: list[complex], drive_pulse_duration: list[int]) -> None:
        self._s21 = s21
        self._drive_pulse_duration = drive_pulse_duration

    @property
    def s21(self) -> list[complex]:
        """The measured complex S21 values."""
        return self._s21

    @property
    def drive_pulse_duration(self) -> list[int]:
        """The drive pulse durations used in the experiment."""
        return self._drive_pulse_duration

    def __repr__(self) -> str:
        return f"RabiExperimentResult(s21={self.s21}, drive_pulse_duration={self.drive_pulse_duration})"


@yaml.register_class
class T1ExperimentResult(ExperimentResult):
    """Result of a T1 experiment."""

    def __init__(self, s21: list[complex], wait_times: list[int]) -> None:
        self._s21 = s21
        self._wait_times = wait_times

    @property
    def s21(self) -> list[complex]:
        """The measured complex S21 values."""
        return self._s21

    @property
    def wait_times(self) -> list[int]:
        """The wait times used in the experiment."""
        return self._wait_times

    def __repr__(self) -> str:
        return f"T1ExperimentResult(s21={self.s21}, wait_times={self.wait_times})"
