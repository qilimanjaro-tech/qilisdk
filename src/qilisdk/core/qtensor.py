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

from typing import TYPE_CHECKING, Literal

import numpy as np
from qtensor_module import QTensorCpp  # ty:ignore
from scipy.sparse import csr_matrix, sparray, spmatrix

from qilisdk.settings import get_settings
from qilisdk.utils.hashing import hash as qili_hash
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from qilisdk.core.types import Number

Complex = int | float | complex
NormTypes = Literal["auto", "frobenius", "trace", "l1", "l2", "inf", "nuclear"] | int
QTensorType = Literal["ket", "bra", "operator"]


@yaml.register_class
class QTensor:
    """
    Lightweight wrapper around sparse matrices representing quantum states or operators.

    The QTensor class is a wrapper around sparse matrices (or NumPy arrays,
    which are converted to sparse matrices) that represent quantum states (kets, bras, or density matrices)
    or operators. It provides utility methods for common quantum operations such as
    taking the adjoint (dagger), computing tensor products, partial traces, and norms.

    The internal data is stored as an Eigen CSR (Compressed Sparse Row) matrix for
    efficient arithmetic and manipulation. The expected shapes for the data are:
    - (2**N, 2**N) for operators or density matrices (or scalars),
    - (2**N, 1) for ket states,
    - (1, 2**N) for bra states.

    Example:
        .. code-block:: python

            import numpy as np
            from qilisdk.core import QTensor

            ket = QTensor(np.array([[1.0], [0.0]]))
            density = ket * ket.adjoint()
    """

    def __init__(self, other: np.ndarray | sparray | spmatrix | list[list[Number]] | QTensor | QTensorCpp) -> None:
        """
        Args:
            other (np.ndarray | sparray | spmatrix | list[list[Number]] | QTensor | QTensorCpp): Dense or sparse matrix defining the quantum object. Expected
                shapes are ``(2**N, 2**N)`` for operators, ``(2**N, 1)`` for kets, ``(1, 2**N)`` for bras, or ``(1, 1)`` for scalars.

        Raises:
            ValueError: If ``data`` is not 2-D or does not correspond to valid qubit dimensions.
        """
        if isinstance(other, np.ndarray):
            self._qtensor_cpp = QTensorCpp(np.array(other, dtype=np.complex128))
        else:
            self._qtensor_cpp = QTensorCpp(other)

    @property
    def data(self) -> csr_matrix:
        """
        Get the internal sparse matrix representation of the QTensor.

        Returns:
            csc_matrix: The internal representation as a CSR matrix.
        """
        return self._qtensor_cpp.as_scipy()

    @property
    def nqubits(self) -> int:
        """
        Compute the number of qubits represented by the QTensor.

        Returns:
            int: The number of qubits if determinable; otherwise, -1.
        """
        return self._qtensor_cpp.get_nqubits()

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the QTensor's internal matrix.

        Returns:
            tuple[int, ...]: The shape of the internal matrix.
        """
        return self._qtensor_cpp.get_shape()

    def dense(self) -> np.ndarray:
        """
        Get the dense (NumPy array) representation of the QTensor.

        Returns:
            np.ndarray: The dense array representation.
        """
        return self._qtensor_cpp.as_numpy()

    def is_ket(self) -> bool:
        """
        Check if the QTensor represents a ket state.

        Returns:
            bool: True if the QTensor is a ket state, False otherwise.
        """
        return self._qtensor_cpp.is_ket()

    def is_bra(self) -> bool:
        """
        Check if the QTensor represents a bra state.

        Returns:
            bool: True if the QTensor is a bra state, False otherwise.
        """
        return self._qtensor_cpp.is_bra()

    def is_scalar(self) -> bool:
        """
        Check if the QTensor represents a scalar.

        Returns:
            bool: True if the QTensor is a scalar, False otherwise.
        """
        return self._qtensor_cpp.is_scalar()

    def is_operator(self) -> bool:
        """
        Check if the QTensor represents an operator (square matrix).

        Returns:
            bool: True if the QTensor is an operator, False otherwise.
        """
        return self._qtensor_cpp.is_operator()

    def is_density_matrix(self, tol: float = get_settings().atol) -> bool:
        """
        Determine if the QTensor is a valid density matrix.

        A valid density matrix must be square, Hermitian, positive semi-definite, and have a trace equal to 1.

        Returns:
            bool: True if the QTensor is a valid density matrix, False otherwise.
        """
        return self._qtensor_cpp.is_density_matrix(tol)

    def is_hermitian(self) -> bool:
        """
        Check if the QTensor is Hermitian.

        Returns:
            bool: True if the QTensor is Hermitian, False otherwise.
        """
        return self._qtensor_cpp.is_hermitian(get_settings().atol)

    def adjoint(self) -> QTensor:
        """
        Compute the adjoint (conjugate transpose) of the QTensor.

        Returns:
            QTensor: A new QTensor that is the adjoint of this object.
        """
        return QTensor(self._qtensor_cpp.adjoint())

    def conjugate(self) -> QTensor:
        """
        Compute the complex conjugate of the QTensor.

        Returns:
            QTensor: A new QTensor that is the complex conjugate of this object.
        """
        return QTensor(self._qtensor_cpp.conjugate())

    def transpose(self) -> QTensor:
        """
        Compute the transpose of the QTensor.

        Returns:
            QTensor: A new QTensor that is the transpose of this object.
        """
        return QTensor(self._qtensor_cpp.transpose())

    def dagger(self) -> QTensor:
        """
        Alias for adjoint() to provide a more familiar method name for quantum objects.

        Returns:
            QTensor: A new QTensor that is the adjoint of this object.
        """
        return self.adjoint()

    def trace(self) -> complex:
        """
        Compute the trace of the QTensor.

        Returns:
            complex: The trace of the QTensor.
        """
        return self._qtensor_cpp.trace()

    def dot(self, other: QTensor) -> Number:
        """
        Compute the dot product (inner product) between this QTensor and another.

        For state vectors, this corresponds to the inner product. For operators, it corresponds to the Hilbert-Schmidt inner product.

        Args:
            other (QTensor): The other QTensor to compute the dot product with.

        Returns:
            Number: The computed dot product, which may be a complex number.
        """
        return self._qtensor_cpp.dot_python(other)

    def ptrace(self, keep: list[int]) -> QTensor:
        """
        Wrapper for backwards compatibility: see partial_trace().

        Args:
            keep (list[int]): A list of indices corresponding to the subsystems to retain.

        Returns:
            QTensor: A new QTensor representing the reduced density matrix for the subsystems specified in 'keep'.

        Raises:
            ValueError: If there are duplicate indices in the 'keep' list.
        """
        as_set = set(keep)
        if len(as_set) != len(keep):
            raise ValueError("Duplicate indices in keep list")
        return self.partial_trace(set(keep))

    def partial_trace(self, keep: set[int]) -> QTensor:
        """
        Compute the partial trace over subsystems not in 'keep'.

        This method calculates the reduced density matrix by tracing out
        the subsystems that are not specified in the 'keep' parameter.

        Args:
            keep (set[int]): A set of indices corresponding to the subsystems to retain.

        Raises:
            ValueError: If the indices in 'keep' are out of range.

        Returns:
            QTensor: A new QTensor representing the reduced density matrix.
        """
        return QTensor(self._qtensor_cpp.partial_trace_python(list(set(keep))))

    def norm(self, order: NormTypes = "auto") -> float:
        """
        Compute the norm of the QTensor.

        If the order is left as "auto", the method will choose an appropriate norm based on the type of QTensor
        (i.e., trace for operators, frobenius for state vectors).

        Integers can also be supplied, where 1 corresponds to the L1 norm and 2 corresponds to the L2 (Frobenius) norm etc.

        Args:
            order (Literal["auto", "frobenius", "trace", "l1", "l2", "inf", "nuclear"], optional): The order of the norm. Defaults to "auto".

        Returns:
            float: The computed norm of the QTensor.

        Raises:
            ValueError: If an unsupported norm order is specified.
        """
        if isinstance(order, int):
            if order <= 0:
                raise ValueError("Norm order must be a positive integer")
            return self._qtensor_cpp.norm("l" + str(order))
        if order == "tr":
            return self._qtensor_cpp.norm("nuclear")
        return self._qtensor_cpp.norm(order)

    def eig(self) -> tuple[list[complex], list[QTensor]]:
        """
        Compute the eigendecomposition of the QTensor and return the eigenvalues and eigenvectors.
        The eigendecomposition is only computed once and cached for future use.

        Returns:
            tuple[list[complex], list[QTensor]]: A tuple containing a list of eigenvalues and a list of corresponding eigenvectors as QTensors.

        Raises:
            ValueError: If the QTensor is not a square matrix (i.e., not an operator).
        """
        self._qtensor_cpp.compute_eigendecomposition()
        return self.eigenvalues, self.eigenvectors

    def unit(self, order: NormTypes = "auto") -> QTensor:
        """
        Wrapper for backwards compatibility: see normalized().

        Args:
            order (NormTypes, optional): The order of the norm to use for normalization.

        Returns:
            QTensor: A new QTensor that is the normalized version of this object.
        """
        return self.normalized(order)

    def normalized(self, order: NormTypes = "auto") -> QTensor:
        """
        Normalize the QTensor.

        Scales the QTensor so that its norm becomes 1, according to the specified norm order.
        See the norm() method for details on the supported norm orders and their behavior.

        Args:
            order (NormTypes, optional): The order of the norm to use for normalization.

        Raises:
            ValueError: If the norm of the QTensor is 0, making normalization impossible.

        Returns:
            QTensor: A new QTensor that is the normalized version of this object.
        """
        return QTensor(self._qtensor_cpp.normalized(order))

    def expm(self) -> QTensor:
        """
        Wrapper for backwards compatibility: see exponential().

        Returns:
            QTensor: A new QTensor representing the matrix exponential.
        """
        return self.exp()

    def exp(self) -> QTensor:
        """
        Compute the matrix exponential of the QTensor.

        Returns:
            QTensor: A new QTensor representing the matrix exponential.
        """
        return QTensor(self._qtensor_cpp.exp())

    def inverse(self) -> QTensor:
        """
        Compute the matrix inverse of the QTensor.

        Returns:
            QTensor: A new QTensor representing the matrix inverse.

        Raises:
            ValueError: If the QTensor is not invertible.
        """
        return QTensor(self._qtensor_cpp.inverse())

    def fidelity(self, other: QTensor) -> float:
        """
        Compute the fidelity between this QTensor and another QTensor.

        The fidelity is a measure of similarity between two quantum states or operators.
        For state vectors, it is defined as F(psi, phi) = |⟨psi|phi⟩|^2.
        For density matrices, it is defined as F(rho, sigma) = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2.

        Args:
            other (QTensor): The other QTensor to compute the fidelity with.

        Returns:
            float: The fidelity between this QTensor and the other QTensor, ranging from 0 to 1.
        """
        return self._qtensor_cpp.fidelity_python(other)

    def to_density_matrix(self, max_relative_correction: float = 0.1) -> QTensor:
        """
        Wrapper for backwards compatibility: see as_density_matrix().

        Returns:
            QTensor: A new QTensor representing the density matrix.
        """
        return self.as_density_matrix(max_relative_correction=max_relative_correction)

    def as_density_matrix(self, max_relative_correction: float = 0.1) -> QTensor:
        """
        Convert the QTensor to a density matrix.

        If the QTensor represents a state vector (ket or bra), this method
        calculates the corresponding density matrix by taking the outer product.
        If the QTensor is already a density matrix, it is returned unchanged.
        The resulting density matrix is normalized.

        Raises:
            ValueError: If the QTensor is a scalar, as a density matrix cannot be derived.
            ValueError: If the QTensor is an operator that is not a density matrix.

        Returns:
            QTensor: A new QTensor representing the density matrix.
        """
        return QTensor(self._qtensor_cpp.as_density_matrix(get_settings().atol, max_relative_correction))

    def __add__(self, other: QTensor | Number) -> QTensor:
        return QTensor(self._qtensor_cpp.add_python(other))

    def __radd__(self, other: QTensor | Number) -> QTensor:
        return self.__add__(other)

    def __sub__(self, other: QTensor) -> QTensor:
        return QTensor(self._qtensor_cpp.sub_python(other))

    def __mul__(self, other: QTensor | Number) -> QTensor:
        return QTensor(self._qtensor_cpp.mul_python(other))

    def __matmul__(self, other: QTensor) -> QTensor:
        return QTensor(self._qtensor_cpp.matmul_python(other))

    def __rmul__(self, other: QTensor | Number) -> QTensor:
        return self.__mul__(other)

    def __repr__(self) -> str:
        return self._qtensor_cpp.as_string()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QTensor):
            return NotImplemented
        return self._qtensor_cpp.equals_python(other)

    def __truediv__(self, other: Number) -> QTensor:
        if isinstance(other, (int, float, complex)):
            return QTensor(self._qtensor_cpp.div(other))
        return NotImplemented

    def __hash__(self) -> int:
        return qili_hash(self.data, self.shape, self.nqubits)

    def __getitem__(self, index: tuple[int, int]) -> complex:
        return self._qtensor_cpp.coeff(index[0], index[1])

    @classmethod
    def tensor_product(cls, others: list[QTensor]) -> QTensor:
        """
        Compute the tensor product of a list of QTensor.

        Args:
            others (list[QTensor]): A list of QTensors to tensor product.

        Returns:
            QTensor: A new QTensor representing the tensor product.
        """
        return QTensor(QTensorCpp.tensor_product_python(QTensorCpp(), others))

    @classmethod
    def ket(cls, *state: int) -> QTensor:
        """
        Create a ket state vector for a given basis state.

        Args:
            state (int): A variable number of integers representing the state of each qubit (e.g., 0, 1, 0 for a 3-qubit state).

        Returns:
            QTensor: A new QTensor representing the ket state.
        """
        return QTensor(QTensorCpp.ket_python(state))

    @classmethod
    def bra(cls, *state: int) -> QTensor:
        """
        Create a bra state vector for a given basis state.

        Args:
            state (int): A variable number of integers representing the state of each qubit (e.g., 0, 1, 0 for a 3-qubit state).

        Returns:
            QTensor: A new QTensor representing the bra state.
        """
        return QTensor(QTensorCpp.bra_python(state))

    @classmethod
    def zero(cls, nqubits: int, qtensor_type: QTensorType = "operator") -> QTensor:
        """
        Create a QTensor representing the zero matrix of the specified shape.

        Args:
            nqubits (int): The number of qubits, which determines the size of the zero matrix (2^nqubits x 2^nqubits).
            qtensor_type (QTensorType): The type of QTensor to create ("ket", "bra", or "operator"), which determines the shape of the zero matrix.

        Returns:
            QTensor: A new QTensor representing the zero matrix.
        """
        return QTensor(QTensorCpp.zero(nqubits, qtensor_type))

    @classmethod
    def identity(cls, nqubits: int) -> QTensor:
        """
        Create an identity of the specified dimension.

        Args:
            nqubits (int): The number of qubits, which determines the size of the identity matrix (2^nqubits x 2^nqubits).

        Returns:
            QTensor: A new QTensor representing the identity.
        """
        return QTensor(QTensorCpp.identity(nqubits))

    @classmethod
    def ghz(cls, nqubits: int) -> QTensor:
        """
        Create a GHZ state for the specified number of qubits.

        Args:
            nqubits (int): The number of qubits in the GHZ state.

        Returns:
            QTensor: A new QTensor representing the GHZ state.
        """
        return QTensor(QTensorCpp.ghz(nqubits))

    def reset_qubits(self, qubits: set[int]) -> QTensor:
        """
        Reset the specified qubits to the |0⟩ state.

        This method applies a reset operation to the specified qubits, effectively tracing out those qubits and replacing them with |0⟩ states.

        Args:
            qubits (set[int]): A set of indices corresponding to the qubits to reset.

        Returns:
            QTensor: A new QTensor representing the state after resetting the specified qubits.
        """
        return QTensor(self._qtensor_cpp.reset_qubits_python(qubits))

    def probabilities(self) -> list[float]:
        """
        Compute the probabilities of measuring each basis state.

        This method calculates the probabilities of obtaining each basis state upon measurement, which is given by the squared magnitudes of the coefficients in the state vector (for kets) or the diagonal elements (for density matrices).

        Returns:
            np.ndarray: A 1D array containing the probabilities of measuring each basis state.
        """
        return self._qtensor_cpp.probabilities_python()

    def entropy_von_neumann(self) -> float:
        """
        Compute the von Neumann entropy of the QTensor.

        This method calculates the von Neumann entropy, which is defined as S(rho) = -Tr(rho log(rho)), where rho is the density matrix represented by the QTensor.

        Returns:
            float: The von Neumann entropy of the QTensor.
        """
        return self._qtensor_cpp.entropy_von_neumann()

    def entropy_renyi(self, alpha: float) -> float:
        """
        Compute the Rényi entropy of the QTensor for a given order alpha.

        This method calculates the Rényi entropy, which is defined as S_alpha(rho) = (1/(1-alpha)) log(Tr(rho^alpha)), where rho is the density matrix represented by the QTensor and alpha is the order of the entropy.

        Args:
            alpha (float): The order of the Rényi entropy. Must be greater than 0 and not equal to 1.

        Returns:
            float: The Rényi entropy of the QTensor for the specified order alpha.
        """
        return self._qtensor_cpp.entropy_renyi(alpha)

    def commutator(self, other: QTensor) -> QTensor:
        """
        Compute the commutator [A, B] = AB - BA with another QTensor.

        Args:
            other (QTensor): The other QTensor to compute the commutator with.

        Returns:
            QTensor: A new QTensor representing the commutator.
        """
        return QTensor(self._qtensor_cpp.commutator_python(other))

    def anticommutator(self, other: QTensor) -> QTensor:
        """
        Compute the anticommutator {A, B} = AB + BA with another QTensor.

        Args:
            other (QTensor): The other QTensor to compute the anticommutator with.

        Returns:
            QTensor: A new QTensor representing the anticommutator.
        """
        return QTensor(self._qtensor_cpp.anticommutator_python(other))

    def pow(self, exponent: int) -> QTensor:
        """
        Compute the integer power of the QTensor.

        Args:
            exponent (int): The exponent to raise the QTensor to.

        Returns:
            QTensor: A new QTensor representing the result of raising this QTensor to the given power.
        """
        return QTensor(self._qtensor_cpp.pow(exponent))

    def sqrt(self) -> QTensor:
        """
        Compute the matrix square root of the QTensor.

        Returns:
            QTensor: A new QTensor representing the matrix square root.
        """
        return QTensor(self._qtensor_cpp.sqrt())

    def rank(self) -> int:
        """
        Compute the rank of the QTensor.

        Returns:
            int: The rank of the QTensor.
        """
        return self._qtensor_cpp.rank()

    def log(self) -> QTensor:
        """
        Compute the matrix logarithm of the QTensor.

        Returns:
            QTensor: A new QTensor representing the matrix logarithm.
        """
        return QTensor(self._qtensor_cpp.log())

    @property
    def eigenvalues(self) -> list[Number]:
        """
        Get the eigenvalues of the QTensor.
        Note that this computes the eigendecomposition if it has not already been computed and cached.

        Returns:
            list[Number]: A list of eigenvalues of the QTensor.
        """
        self._qtensor_cpp.compute_eigendecomposition()
        return self._qtensor_cpp.get_eigenvalues_python()

    @property
    def eigenvectors(self) -> list[QTensor]:
        """
        Get the eigenvectors of the QTensor.
        Note that this computes the eigendecomposition if it has not already been computed and cached.

        Returns:
            list[QTensor]: A list of QTensor objects representing the eigenvectors of the QTensor.
        """
        self._qtensor_cpp.compute_eigendecomposition()
        return [QTensor(vec) for vec in self._qtensor_cpp.get_eigenvectors_python()]

    def expectation_value(self, other: QTensor, nshots: int = 0) -> complex:
        """
        Compute the expectation value of another QTensor with respect to this QTensor.

        For state vectors, this corresponds to ⟨psi|O|psi⟩. For operators, it corresponds to Tr(rho O).

        Args:
            other (QTensor): The other QTensor to compute the expectation value with.
            nshots (int, optional): The number of shots to use for a stochastic estimation of the expectation value. If 0 or negative, the exact expectation value is computed using the trace formula.

        Returns:
            complex: The computed expectation value, which may be a complex number.
        """
        return self._qtensor_cpp.expectation_value_python(other, nshots)

    def __getstate__(self) -> dict[str, csr_matrix]:
        """
        Get the state of the QTensor for pickling.

        Returns:
            dict: A dictionary containing the state of the QTensor for serialization.
        """
        return {"data": self.data}

    def __setstate__(self, state: dict[str, csr_matrix]) -> None:
        """
        Set the state of the QTensor from a pickled state.

        Args:
            state (dict): A dictionary containing the state of the QTensor for deserialization.
        """
        self.__init__(state["data"])


def ket(*state: int) -> QTensor:
    """
    Wrapper for backwards compatibility: see QTensor.ket().

    Args:
        state (int): A sequence of integers representing the state of each qubit (e.g., 0, 1, 0, 1 for a 4-qubit state).

    Returns:
        QTensor: A new QTensor representing the ket state.
    """
    return QTensor.ket(*state)


def bra(*state: int) -> QTensor:
    """
    Wrapper for backwards compatibility: see QTensor.bra().

    Args:
        state (int): A sequence of integers representing the state of each qubit (e.g., 0, 1, 0, 1 for a 4-qubit state).

    Returns:
        QTensor: A new QTensor representing the bra state.
    """
    return QTensor.bra(*state)


def expect_val(operator: QTensor, state: QTensor) -> complex:
    """
    Wrapper for backwards compatibility: see QTensor.expectation_value().

    Args:
        state (QTensor): The state vector or density matrix to compute the expectation value with respect to.
        operator (QTensor): The operator for which to compute the expectation value.

    Returns:
        complex: The computed expectation value, which may be a complex number.
    """
    return state.expectation_value(operator)


def tensor_prod(states: list[QTensor]) -> QTensor:
    """
    Wrapper for backwards compatibility: see QTensor.tensor_product().

    Args:
        states (list[QTensor]): A list of QTensors to tensor product.

    Returns:
        QTensor: A new QTensor representing the tensor product.
    """
    return QTensor.tensor_product(states)


def reset_qubits(state: QTensor, qubits: list[int]) -> QTensor:
    """
    Wrapper for backwards compatibility: see QTensor.reset_qubits().

    Args:
        state (QTensor): The quantum state to reset qubits in.
        qubits (list[int]): A list of indices corresponding to the qubits to reset.

    Returns:
        QTensor: A new QTensor representing the state after resetting the specified qubits.
    """
    return state.reset_qubits(set(qubits))


def zero(nqubits: int, qtensor_type: QTensorType = "operator") -> QTensor:
    """
    Wrapper for backwards compatibility: see QTensor.zero().

    Args:
        nqubits (int): The number of qubits, which determines the size of the zero matrix (2^nqubits x 2^nqubits).
        qtensor_type (QTensorType): The type of QTensor to create ("ket", "bra", or "operator"), which determines the shape of the zero matrix.

    Returns:
        QTensor: A new QTensor representing the zero matrix.
    """
    return QTensor.zero(nqubits, qtensor_type)


def basis_state(n: int, N: int) -> QTensor:
    r"""
    Generate the n'th basis vector representation, on a N-size Hilbert space (N=2**num_qubits).

    This function creates a column vector (ket) representing the Fock state \|n⟩ in a Hilbert space of dimension N.

    Args:
        n (int): The desired number state (from 0 to N-1).
        N (int): The dimension of the Hilbert space, has a value 2**num_qubits.

    Raises:
        ValueError: If n >= N.

    Returns:
        QTensor: A QTensor representing the \|n⟩'th basis state on a N-size Hilbert space (N=2**num_qubits).
    """
    if not (0 <= n < N):
        raise ValueError(f"n must be in [0, {N - 1}]")
    # one nonzero at (row=n, col=0), value=1.0
    mat = csr_matrix(([1.0], ([n], [0])), shape=(N, 1))
    return QTensor(mat)


def identity(nqubits: int) -> QTensor:
    """
    Wrapper for QTensor.identity().

    Args:
        nqubits (int): The number of qubits in the identity operator.

    Returns:
        QTensor: A QTensor representing the identity operator of the specified dimension.
    """
    return QTensor(QTensorCpp.identity(nqubits))


def ghz(nqubits: int) -> QTensor:
    """
    Wrapper for QTensor.ghz().

    Args:
        nqubits (int): The number of qubits in the GHZ state.

    Returns:
        QTensor: A QTensor representing the GHZ state for the specified number of qubits.
    """
    return QTensor(QTensorCpp.ghz(nqubits))
