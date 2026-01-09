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

import numpy as np

from qilisdk.core.qtensor import QTensor

from .noise_model import NoiseBase, NoiseType

if TYPE_CHECKING:
    from qilisdk.digital.gates import Gate


class KrausNoise(NoiseBase):
    """
    Generic noise model represented by Kraus operators
    """
    def __init__(
        self,
        kraus_operators: list[QTensor],
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        """
        Initialize a Kraus noise model.
        If the list of affected qubits is empty:
            - if the Kraus operators are the same size as the system, they act on the whole system
            - if the Kraus operators are smaller than the full size, they act on all qubits individually
        If the list of affected gates is empty, the noise affects all gates.

        Args:
            kraus_operators (list[QTensor]): List of Kraus operators defining the noise channel.
            affected_qubits (list[int] | None): List of qubit indices the noise affects.
            affected_gates (list[type[Gate]] | None): List of gate types the noise affects.

        Raises:
            ValueError: If the Kraus operators do not satisfy the completeness relation.
            ValueError: If the kraus_operators list is empty.
            ValueError: If the Kraus operators do not have consistent dimensions.
            ValueError: If any affected qubit index is invalid.
        """

        self._kraus_operators: list[QTensor] = kraus_operators or []
        self._affected_qubits: list[int] = affected_qubits or []
        self._affected_gates: list[type[Gate]] = affected_gates or []

        # Validate kraus operators
        if not self._kraus_operators:
            raise ValueError("Kraus operators list cannot be empty.")
        dim = self._kraus_operators[0].shape[0]
        for K in self._kraus_operators:
            if K.shape[0] != dim or K.shape[1] != dim:
                raise ValueError("All Kraus operators must have the same dimensions.")
        identity = sum(K.adjoint() @ K for K in self._kraus_operators)
        if type(identity) is QTensor and not np.allclose(identity.dense(), np.eye(dim)):
            raise ValueError("Kraus operators do not satisfy the completeness relation.")

        # Make sure the affected qubits are all valid
        for q in self._affected_qubits:
            if q < 0:
                raise ValueError(f"Invalid qubit index: {q}")

    @property
    def noise_type(self) -> NoiseType:
        """
        Returns the type of noise.
        """
        return NoiseType.DIGITAL

    @property
    def kraus_operators(self) -> list[QTensor]:
        """
        Returns the list of Kraus operators defining the noise channel.
        """
        return self._kraus_operators

    @property
    def affected_qubits(self) -> list[int]:
        """
        Returns the list of qubit indices the noise affects.
        """
        return self._affected_qubits

    @property
    def affected_gates(self) -> list[type[Gate]]:
        """
        Returns the list of gate types the noise affects.
        """
        return self._affected_gates


class DigitalBitFlipNoise(KrausNoise):
    """
    Noise model representing a bit flip channel.
    """
    def __init__(
        self,
        probability: float,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        """
        Initialize a Bit Flip noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of undergoing a bit flip (X gate).

        Args:
            probability (float): Probability of a bit flip occurring (0 <= p <= 1).
            affected_qubits (list[int] | None): List of qubit indices the noise affects. If None, affects all qubits.
            affected_gates (list[type[Gate]] | None): List of gate types the noise affects. If None, affects all gates.

        Raises:
            ValueError: If probability is not in the range [0, 1].
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError("The probability must be in the range [0, 1].")

        K0 = np.sqrt(1 - probability) * np.array([[1, 0], [0, 1]])
        K1 = np.sqrt(probability) * np.array([[0, 1], [1, 0]])
        kraus_operators = [K0, K1]

        super().__init__(
            kraus_operators=[QTensor(K) for K in kraus_operators],
            affected_qubits=affected_qubits,
            affected_gates=affected_gates,
        )


class DigitalDepolarizingNoise(KrausNoise):
    """
    Noise model representing a depolarizing channel.
    """
    def __init__(
        self,
        probability: float,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        """
        Initialize a Depolarizing noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of being replaced by the maximally mixed state.

        Args:
            probability (float): Probability of depolarization occurring (0 <= p <= 1).
            affected_qubits (list[int] | None): List of qubit indices the noise affects. If None, affects all qubits.
            affected_gates (list[type[Gate]] | None): List of gate types the noise affects. If None, affects all gates.

        Raises:
            ValueError: If probability is not in the range [0, 1].
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError("The probability must be in the range [0, 1].")

        K0 = np.sqrt(1 - 3 * probability / 4) * np.array([[1, 0], [0, 1]])
        K1 = np.sqrt(probability / 4) * np.array([[0, 1], [1, 0]])
        K2 = np.sqrt(probability / 4) * np.array([[0, -1j], [1j, 0]])
        K3 = np.sqrt(probability / 4) * np.array([[1, 0], [0, -1]])
        kraus_operators = [K0, K1, K2, K3]

        super().__init__(
            kraus_operators=[QTensor(K) for K in kraus_operators],
            affected_qubits=affected_qubits,
            affected_gates=affected_gates,
        )


class DigitalDephasingNoise(KrausNoise):
    """
    Noise model representing a dephasing channel.
    """
    def __init__(
        self,
        probability: float,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        """
        Initialize a Dephasing noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of undergoing dephasing.

        Args:
            probability (float): Probability of dephasing occurring (0 <= p <= 1).
            affected_qubits (list[int] | None): List of qubit indices the noise affects. If None, affects all qubits.
            affected_gates (list[type[Gate]] | None): List of gate types the noise affects. If None, affects all gates.

        Raises:
            ValueError: If probability is not in the range [0, 1].
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError("The probability must be in the range [0, 1].")

        K0 = np.sqrt(1 - probability) * np.array([[1, 0], [0, 1]])
        K1 = np.sqrt(probability) * np.array([[1, 0], [0, -1]])
        kraus_operators = [K0, K1]

        super().__init__(
            kraus_operators=[QTensor(K) for K in kraus_operators],
            affected_qubits=affected_qubits,
            affected_gates=affected_gates,
        )


class DigitalAmplitudeDampingNoise(KrausNoise):
    """
    Noise model representing an amplitude damping channel.
    """
    def __init__(
        self,
        gamma: float,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        """
        Initialize an Amplitude Damping noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of losing energy to the environment.

        Args:
            gamma (float): Probability of amplitude damping occurring (0 <= gamma <= 1).
            affected_qubits (list[int] | None): List of qubit indices the noise affects. If None, affects all qubits.
            affected_gates (list[type[Gate]] | None): List of gate types the noise affects. If None, affects all gates.

        Raises:
            ValueError: If gamma is not in the range [0, 1].
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("The gamma must be in the range [0, 1].")

        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        kraus_operators = [K0, K1]

        super().__init__(
            kraus_operators=[QTensor(K) for K in kraus_operators],
            affected_qubits=affected_qubits,
            affected_gates=affected_gates,
        )
