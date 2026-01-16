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

from .parameter_pertubation import ParameterPerturbation


class StaticOffsetPerturbation(ParameterPerturbation):
    """Adds a constant offset to a parameter value.

    Parameter:
    - offset: additive bias

    Useful for systematic over/under-rotations or miscalibrated coefficients.
    """

    def __init__(self, *, offset: float) -> None:
        self._offset = float(offset)

    def perturb(self, value: float) -> float:
        return value + self._offset
