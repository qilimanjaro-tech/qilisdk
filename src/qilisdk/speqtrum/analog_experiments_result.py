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
import numpy as np

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class Dimension:
    def __init__(self, labels: list[str], values: list[np.ndarray]) -> None:
        self.labels = labels
        self.values = values


@yaml.register_class
class ExperimentResult(FunctionalResult):
    def __init__(self, data: np.ndarray, dims: list[Dimension]) -> None:
        self.data = data
        self.dims = dims


@yaml.register_class
class RabiExperimentResult(ExperimentResult):
    ...

@yaml.register_class
class T1ExperimentResult(ExperimentResult):
    ...
