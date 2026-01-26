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

from .parameter_perturbation import ParameterPerturbation


class OffsetPerturbation(ParameterPerturbation):
    """Parameter perturbation that adds a constant offset."""

    def __init__(self, *, offset: float) -> None:
        """Args:
        offset (float): Additive bias applied to the parameter value.
        """
        self._offset = float(offset)

    @property
    def offset(self) -> float:
        """Return the constant offset applied to parameter values.

        Returns:
            float: The additive offset.
        """
        return self._offset

    def perturb(self, value: float) -> float:
        return value + self._offset
