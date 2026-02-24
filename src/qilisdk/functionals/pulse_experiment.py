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

from typing import ClassVar

from qilisdk.functionals.functional import Functional
from qilisdk.functionals.pulse_experiment_result import PulseExperimentResult
from qilisdk.qprogram import QProgram
from qilisdk.yaml import yaml


@yaml.register_class
class PulseExperiment(Functional):
    result_type: ClassVar[type[PulseExperimentResult]] = PulseExperimentResult

    def __init__(self, qprogram: QProgram, bus_mapping: dict[str, str] | None = None) -> None:
        self.qprogram = qprogram
        self.bus_mapping = bus_mapping
