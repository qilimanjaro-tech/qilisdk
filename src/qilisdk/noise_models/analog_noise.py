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

from typing import TYPE_CHECKING

from .noise_models import NoiseBase, NoiseType

if TYPE_CHECKING:
    from qilisdk.core.qtensor import QTensor


class DissipationNoise(NoiseBase):

    def __init__(self, jump_operators: list[QTensor]) -> None:
        self._jump_operators: list[QTensor] = jump_operators or []

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.ANALOG

    @property
    def jump_operators(self) -> list[QTensor]:
        return self._jump_operators
