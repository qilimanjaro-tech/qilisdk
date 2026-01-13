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

from qilisdk.analog.hamiltonian import PauliI, PauliOperator, PauliX, PauliY, PauliZ
from qilisdk.core.qtensor import QTensor
from qilisdk.yaml import yaml

from .noise_model import DigitalNoise


@yaml.register_class
class KrausChannelNoise(DigitalNoise):
    """
    Generic noise model represented by Kraus operators
    """

    def __init__(self, kraus_operators: list[QTensor]) -> None:
        """
        Initialize a Kraus noise model.
        If the list of affected qubits is empty:
            - if the Kraus operators are the same size as the system, they act on the whole system
            - if the Kraus operators are smaller than the full size, they act on all qubits individually
        If the list of affected gates is empty, the noise affects all gates.

        Args:
            kraus_operators (list[QTensor]): List of Kraus operators defining the noise channel.

        Raises:
            ValueError: If the Kraus operators do not satisfy the completeness relation.
            ValueError: If the kraus_operators list is empty.
            ValueError: If the Kraus operators do not have consistent dimensions.
            ValueError: If any affected qubit index is invalid.
        """

        self._kraus_operators: list[QTensor] = kraus_operators or []

        # Validate kraus operators
        if not self._kraus_operators:
            raise ValueError("Kraus operators list cannot be empty.")
        dim = self._kraus_operators[0].shape[0]
        for K in self._kraus_operators:
            if K.shape[0] != dim or K.shape[1] != dim:
                raise ValueError("All Kraus operators must have the same dimensions.")
        identity = sum(K.adjoint() @ K for K in self._kraus_operators)
        if isinstance(identity, QTensor) and not np.allclose(identity.dense(), np.eye(dim)):
            raise ValueError("Kraus operators do not satisfy the completeness relation.")

    @property
    def name(self) -> str:
        return "Kraus Channel Noise"

    def get_kraus_operators(self) -> list[QTensor]:
        """
        Returns:
            list[QTensor] : the list of Kraus operators defining the noise channel.
        """
        return self._kraus_operators


@yaml.register_class
class PauliChannelNoise(DigitalNoise):
    """
    Pauli noise channel defined by probabilities over Pauli strings.
    """

    def __init__(
        self,
        pauli_probabilities: dict[tuple[PauliOperator], float],
        num_qubits: int | None = None,
    ) -> None:
        """
        Initialize a Pauli noise channel.

        Args:
            pauli_probabilities (dict[tuple[PauliOperator, ...], float]): Mapping of Pauli strings
                to their probabilities. Each Pauli string is a tuple of single-qubit Pauli operators.
            num_qubits (int | None): Total number of qubits. If None, inferred from the strings.

        Raises:
            ValueError: If probabilities are invalid or qubit indices are inconsistent.
        """

        if not pauli_probabilities:
            raise ValueError("Pauli probabilities cannot be empty.")

        for probability in pauli_probabilities.values():
            if probability < 0:
                raise ValueError("Pauli probabilities must be non-negative.")

        inferred_max = None
        for term in pauli_probabilities:
            for op in term:
                inferred_max = op.qubit if inferred_max is None else max(inferred_max, op.qubit)

        if num_qubits is None:
            if inferred_max is None:
                raise ValueError("num_qubits is required when no Pauli operators are provided.")
            num_qubits = inferred_max + 1

        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive.")

        for term in pauli_probabilities:
            seen_qubits: set[int] = set()
            for op in term:
                if op.qubit in seen_qubits:
                    raise ValueError(f"Duplicate Pauli operators for qubit {op.qubit}.")
                if op.qubit < 0 or op.qubit >= num_qubits:
                    raise ValueError(f"Invalid qubit index: {op.qubit}")
                seen_qubits.add(op.qubit)

        total_probability = sum(pauli_probabilities.values())
        if not np.isclose(total_probability, 1.0):
            raise ValueError("Pauli probabilities must sum to 1.")

        self._pauli_probabilities = dict(pauli_probabilities)
        self._num_qubits = num_qubits
        self._kraus_operators: list[QTensor] | None = None

    @property
    def pauli_probabilities(self) -> dict[tuple[PauliOperator], float]:
        return dict(self._pauli_probabilities)

    @property
    def name(self) -> str:
        return "Pauli Channel Noise"

    def get_kraus_operators(self) -> list[QTensor]:
        """
        Returns:
            list[QTensor]: Kraus operators for the Pauli channel.
        """
        if self._kraus_operators is None:
            identity = PauliI(0).matrix
            kraus_ops: list[QTensor] = []
            for term, probability in self._pauli_probabilities.items():
                if np.isclose(probability, 0.0):
                    continue
                op_by_qubit = {op.qubit: op for op in term}
                full_matrix = None
                for q in range(self._num_qubits):
                    op = op_by_qubit.get(q)
                    factor = op.matrix if op is not None else identity
                    full_matrix = factor if full_matrix is None else np.kron(full_matrix, factor)
                if full_matrix is None:
                    full_matrix = np.array([[1.0]])
                kraus_ops.append(QTensor(np.sqrt(probability) * full_matrix))
            self._kraus_operators = kraus_ops

        return list(self._kraus_operators)


@yaml.register_class
class DigitalBitFlipNoise(PauliChannelNoise):
    """
    Noise model representing a bit flip channel.
    """

    def __init__(self, probability: float) -> None:
        """
        Initialize a Bit Flip noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of undergoing a bit flip (X gate).

        Args:
            probability (float): Probability of a bit flip occurring (0 <= p <= 1).

        Raises:
            ValueError: If probability is not in the range [0, 1].
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError("The probability must be in the range [0, 1].")

        pauli_probabilities = {
            (PauliI(0),): 1.0 - probability,
            (PauliX(0),): probability,
        }

        super().__init__(pauli_probabilities=pauli_probabilities, num_qubits=1)

    @property
    def name(self) -> str:
        return "Digital Bit-flip Noise"


@yaml.register_class
class DigitalDepolarizingNoise(PauliChannelNoise):
    """
    Noise model representing a depolarizing channel.
    """

    def __init__(self, probability: float) -> None:
        """
        Initialize a Depolarizing noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of being replaced by the maximally mixed state.

        Args:
            probability (float): Probability of depolarization occurring (0 <= p <= 1).

        Raises:
            ValueError: If probability is not in the range [0, 1].
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError("The probability must be in the range [0, 1].")

        pauli_probabilities = {
            (PauliI(0),): 1.0 - 3.0 * probability / 4.0,
            (PauliX(0),): probability / 4.0,
            (PauliY(0),): probability / 4.0,
            (PauliZ(0),): probability / 4.0,
        }

        super().__init__(pauli_probabilities=pauli_probabilities, num_qubits=1)

    @property
    def name(self) -> str:
        return "Digital Depolarizing Noise"


@yaml.register_class
class DigitalDephasingNoise(PauliChannelNoise):
    """
    Noise model representing a dephasing channel.
    """

    def __init__(self, probability: float) -> None:
        """
        Initialize a Dephasing noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of undergoing dephasing.

        Args:
            probability (float): Probability of dephasing occurring (0 <= p <= 1).

        Raises:
            ValueError: If probability is not in the range [0, 1].
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError("The probability must be in the range [0, 1].")

        pauli_probabilities = {
            (PauliI(0),): 1.0 - probability,
            (PauliZ(0),): probability,
        }

        super().__init__(
            pauli_probabilities=pauli_probabilities,
            num_qubits=1,
        )

    @property
    def name(self) -> str:
        return "Digital Dephasing Noise"


@yaml.register_class
class DigitalAmplitudeDampingNoise(KrausChannelNoise):
    """
    Noise model representing an amplitude damping channel.
    """

    def __init__(
        self,
        gamma: float,
    ) -> None:
        """
        Initialize an Amplitude Damping noise model.
        This model represents a quantum noise channel where each qubit has a certain probability of losing energy to the environment.

        Args:
            gamma (float): Probability of amplitude damping occurring (0 <= gamma <= 1).

        Raises:
            ValueError: If gamma is not in the range [0, 1].
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("The gamma must be in the range [0, 1].")

        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        kraus_operators = [K0, K1]

        super().__init__(kraus_operators=[QTensor(K) for K in kraus_operators])

    @property
    def name(self) -> str:
        return "Digital Amplitude Damping Noise"
