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

import numpy as np

from qilisdk.functionals import TimeEvolutionResult
from qilisdk.settings import get_settings
from qilisdk.yaml import yaml


def _complex_dtype() -> np.dtype:
    return get_settings().complex_precision.dtype


@yaml.register_class
class QuantumReservoirResult(TimeEvolutionResult):
    """Container for expectation values and states produced by a reservoir simulation."""
