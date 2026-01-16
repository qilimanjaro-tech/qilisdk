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

from typing import TYPE_CHECKING

from .noise_abc import NoiseABC
from .protocols import AttachmentScope, HasAllowedScopes
from .utils import _check_probability

if TYPE_CHECKING:
    from .representations import KrausChannel


class ReadoutAssignment(NoiseABC):
    """Classical readout assignment error."""

    def __init__(self, *, p01: float, p10: float) -> None:
        """
        Args:
            p01 (float): probability to report '1' when the state is |0>
            p10 (float): probability to report '0' when the state is |1>
        """
        self._p01 = _check_probability(p01, "p01")
        self._p10 = _check_probability(p10, "p10")

    @property
    def p01(self) -> float:
        return self._p01
    
    @property
    def p10(self) -> float:
        return self._p10
