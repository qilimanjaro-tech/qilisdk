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

from .noise_model import NoiseBase, NoiseType


class ParameterNoise(NoiseBase):

    def __init__(self, affected_parameters: list[str] | None = None, noise_std: float = 0.1) -> None:
        self._affected_parameters: list[str] = affected_parameters or []
        self._noise_std = noise_std

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.PARAMETER

    @property
    def affected_parameters(self) -> list[str]:
        return self._affected_parameters

    @property
    def noise_std(self) -> float:
        return self._noise_std
