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

import numpy as np

from qilisdk.core.qtensor import QTensor

from .noise_model import NoiseBase, NoiseType


class DissipationNoise(NoiseBase):
    def __init__(
        self, jump_operators: list[QTensor], affected_qubits: list[int] = [], rate: float | None = None
    ) -> None:
        """
        Initialize a dissipation noise model.
        This is defined by a set of jump operators (as per the Lindblad master equation).
        If the list of affected qubits is empty:
            - if the jump operators are the same size as the system, they act on the whole system
            - if the jump operators are smaller than the full size, they act on all qubits individually
        Args:
            jump_operators (list[QTensor]): List of jump operators defining the noise channel.
            affected_qubits (list[int]): List of qubit indices the noise affects.
            rate (float | None): Optional scaling factor for the jump operators. If None, no scaling is applied.
        Raises:
            ValueError: If any affected qubit index is invalid.
        """
        self._jump_operators = jump_operators
        self._affected_qubits = affected_qubits

        # Apply the rate
        if rate is not None:
            self._jump_operators = [op * rate for op in self._jump_operators]

        # Validate affected qubits
        for q in self._affected_qubits:
            if q < 0:
                raise ValueError(f"Invalid qubit index: {q}")

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.ANALOG

    @property
    def jump_operators(self) -> list[QTensor]:
        return self._jump_operators

    @property
    def affected_qubits(self) -> list[int]:
        return self._affected_qubits


class AnalogDepolarizingNoise(DissipationNoise):
    def __init__(self, gamma: float, affected_qubits: list[int]) -> None:
        """
        Analog depolarizing noise model using jump operators.
        Args:
            gamma (float): Depolarizing rate.
            affected_qubits (list[int]): List of qubit indices the noise affects.
        """
        jump_operators = [
            (gamma / 4) ** 0.5 * QTensor(np.array([[0, 1], [1, 0]])),  # X
            (gamma / 4) ** 0.5 * QTensor(np.array([[0, -1j], [1j, 0]])),  # Y
            (gamma / 4) ** 0.5 * QTensor(np.array([[1, 0], [0, -1]])),  # Z
        ]
        super().__init__(jump_operators=jump_operators, affected_qubits=affected_qubits)


class AnalogDephasingNoise(DissipationNoise):
    def __init__(self, gamma: float, affected_qubits: list[int]) -> None:
        """
        Analog dephasing noise model using jump operators.
        Args:
            gamma (float): Dephasing rate.
            affected_qubits (list[int]): List of qubit indices the noise affects.
        """
        op = (gamma / 2) ** 0.5 * QTensor(np.array([[1, 0], [0, -1]]))  # Z
        super().__init__(jump_operators=[op], affected_qubits=affected_qubits)


class AnalogAmplitudeDampingNoise(DissipationNoise):
    def __init__(self, gamma: float, affected_qubits: list[int]) -> None:
        """
        Analog amplitude damping noise model using jump operators.
        Args:
            gamma (float): Amplitude damping rate.
            affected_qubits (list[int]): List of qubit indices the noise affects.
        """
        op = (gamma) ** 0.5 * QTensor(np.array([[0, 1], [0, 0]]))  # Lowering operator
        super().__init__(jump_operators=[op], affected_qubits=affected_qubits)
