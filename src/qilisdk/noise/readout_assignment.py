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
from __future__ import annotations

from .noise_abc import NoiseABC
from .utils import _check_probability


class ReadoutAssignment(NoiseABC):
    """Classical readout assignment error model for measurement outcomes."""

    def __init__(self, *, p01: float, p10: float) -> None:
        """Args:
            p01 (float): Probability to report "1" when the state is |0>.
            p10 (float): Probability to report "0" when the state is |1>.

        Raises:
            ValueError: If any probability is outside [0, 1].
        """
        self._p01 = _check_probability(p01, "p01")
        self._p10 = _check_probability(p10, "p10")

    @property
    def p01(self) -> float:
        """Return the probability of reporting "1" for a |0> state.

        Returns:
            The p01 probability.
        """
        return self._p01

    @property
    def p10(self) -> float:
        """Return the probability of reporting "0" for a |1> state.

        Returns:
            The p10 probability.
        """
        return self._p10
