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
    from qilisdk.digital.gates import Gate


class KrausNoise(NoiseBase):

    def __init__(
        self,
        kraus_operators: list[QTensor],
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        self._kraus_operators: list[QTensor] = kraus_operators or []
        self._affected_qubits: list[int] = affected_qubits or []
        self._affected_gates: list[type[Gate]] = affected_gates or []

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.DIGITAL

    @property
    def kraus_operators(self) -> list[QTensor]:
        return self._kraus_operators

    @property
    def affected_qubits(self) -> list[int]:
        return self._affected_qubits

    @property
    def affected_gates(self) -> list[type[Gate]]:
        return self._affected_gates
