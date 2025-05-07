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

import string
from typing import Literal

import numpy as np
from scipy.sparse import csc_array, csr_matrix, issparse, kron, sparray, spmatrix
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import norm as scipy_norm

from qilisdk.yaml import yaml

Complex = int | float | complex


###############################################################################
# Main Class Definition
###############################################################################


@yaml.register_class
class QuantumObject:
    """
    Represents a quantum state or operator using a sparse matrix representation.

    The QuantumObject class is a wrapper around sparse matrices (or NumPy arrays,
    which are converted to sparse matrices) that represent quantum states (kets, bras)
    or operators. It provides utility methods for common quantum operations such as
    taking the adjoint (dagger), computing tensor products, partial traces, and norms.

    The internal data is stored as a SciPy CSR (Compressed Sparse Row) matrix for
    efficient arithmetic and manipulation. The expected shapes for the data are:
      - (2**N, 2**N) for operators or density matrices (or scalars),
      - (2**N, 1) for ket states,
      - (1, 2**N) or (2**N,) for bra states.
    """

    def __init__(self, data: np.ndarray | sparray | spmatrix) -> None:
        """
        Initialize a QuantumObject with the given data.

        Converts a NumPy array to a CSR matrix if needed and validates the shape of the input.
        The input must represent a valid quantum state or operator with appropriate dimensions.
        Notice that 1D arrays of shape (2N,) are considered/transformed to bras with shape (1, 2N).

        Args:
            data (np.ndarray | sparray | spmatrix): A dense NumPy array or a SciPy sparse matrix
                representing a quantum state or operator. Should be of shape: (2**N, 2**N) for operators
                (1, 2**N) for ket states, (2**N, 1) or (2**N,) for bra states, or (1, 1) for scalars.

        Raises:
            ValueError: If the input data is not a NumPy array or a SciPy sparse matrix,
                or if the data's shape does not correspond to a valid quantum state/operator.
        """
        if isinstance(data, np.ndarray):
            self._data = csr_matrix(data)
        elif issparse(data):
            self._data = data.tocsr()
        else:
            raise ValueError("Input must be a NumPy array or a SciPy sparse matrix")

        # Valid shapes are operators = (2**N, 2**N) (scalars included), bra's = (1, 2**N) / (2**N,), or ket's =(2**N, 1):
        valid_shape = self.is_operator() or self.is_ket() or self.is_bra()

        if len(self._data.shape) != 2 or not valid_shape:  # noqa: PLR2004
            raise ValueError(
                "Dimension of data is wrong. expected data to have shape similar to (2**N, 2**N), (1, 2**N), (2**N, 1)",
                f"but received {self._data.shape}",
            )

    # ------------- Properties --------------

    @property
    def data(self) -> csr_matrix:
        """
        Get the internal sparse matrix representation of the QuantumObject.

        Returns:
            csr_matrix: The internal representation as a CSR matrix.
        """
        return self._data

    @property
    def dense(self) -> np.ndarray:
        """
        Get the dense (NumPy array) representation of the QuantumObject.

        Returns:
            np.ndarray: The dense array representation.
        """
        return self._data.toarray()

    @property
    def nqubits(self) -> int:
        """
        Compute the number of qubits represented by the QuantumObject.

        Returns:
            int: The number of qubits if determinable; otherwise, -1.
        """
        if self._data.shape[0] == self._data.shape[1]:
            return int(np.log2(self._data.shape[0]))
        if self._data.shape[0] == 1:
            return int(np.log2(self._data.shape[1]))
        if self._data.shape[1] == 1:
            return int(np.log2(self._data.shape[0]))
        return -1

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the QuantumObject's internal matrix.

        Returns:
            tuple[int, ...]: The shape of the internal matrix.
        """
        return self._data.shape

    # ----------- Matrix Logic Operations ------------

    def adjoint(self) -> QuantumObject:
        """
        Compute the adjoint (conjugate transpose) of the QuantumObject.

        Returns:
            QuantumObject: A new QuantumObject that is the adjoint of this object.
        """
        out = QuantumObject(self._data.conj().T)
        return out

    def ptrace(self, dims: list[int], keep: list[int]) -> "QuantumObject":
        """
        Compute the partial trace over subsystems not in 'keep'.

        This method calculates the reduced density matrix by tracing out
        the subsystems that are not specified in the 'keep' parameter.
        The input 'dims' represents the dimensions of each subsystem,
        and 'keep' indicates the indices of the subsystems to be retained.

        Args:
            dims (list[int]): A list specifying the dimensions of each subsystem.
            keep (list[int]): A list of indices corresponding to the subsystems to retain.

        Raises:
            ValueError: If the product of the dimensions in dims does not match the
                shape of the QuantumObject's dense representation.

        Returns:
            QuantumObject: A new QuantumObject representing the reduced density matrix
                for the subsystems specified in 'keep'.
        """
        rho = self.dense
        total_dim = np.prod(dims)
        if rho.shape != (total_dim, total_dim):
            raise ValueError("Dimension mismatch between provided dims and QuantumObject shape")

        # Use letters from the ASCII alphabet (both cases) for einsum indices.
        # For each subsystem, assign two letters: one for the row index and one for the column index.
        row_letters, col_letters = [], []
        out_row, out_col = [], []  # Letters that will remain in the output for the row part and for the column part.
        letters = iter(string.ascii_letters)

        for i in range(len(dims)):
            if i in keep:
                # For a subsystem we want to keep, use two different letters (r, c)
                r, c = next(letters), next(letters)
                row_letters.append(r)
                col_letters.append(c)
                out_row.append(r)
                out_col.append(c)
            else:
                # For subsystems to be traced out, assign the same letter (r, r) so that those indices are summed.
                r = next(letters)
                row_letters.append(r)
                col_letters.append(r)

        # Create the einsum subscript strings.
        # The input tensor has 2*n indices (first n for rows, next n for columns).
        input_subscript = "".join(row_letters + col_letters)
        # The output will only contain the indices corresponding to the subsystems we keep.
        output_subscript = "".join(out_row + out_col)

        # Reshape rho into a tensor with shape dims + dims.
        reshaped = rho.reshape(dims + dims)
        # Use einsum to sum over the indices that appear twice (i.e. those being traced out).
        reduced_tensor = np.einsum(f"{input_subscript}->{output_subscript}", reshaped)

        # The resulting tensor has separate indices for each subsystem kept.
        # Reshape it into a matrix (i.e. combine the row indices and column indices).
        dims_keep = [dims[i] for i in keep]
        new_dim = np.prod(dims_keep)
        reduced_matrix = reduced_tensor.reshape(new_dim, new_dim)

        return QuantumObject(reduced_matrix)

    def norm(self, order: int | Literal["fro", "tr"] = 1) -> float:
        """
        Compute the norm of the QuantumObject.

        For density matrices, the norm order can be specified. For state vectors, the norm is computed accordingly.

        Args:
            order (int or {"fro", "tr"}, optional): The order of the norm.
                Only applies if the QuantumObject represents a density matrix. Other than all the
                orders accepted by scipy, it also accepts 'tr' for the trace norm. Defaults to 1.

        Raises:
            ValueError: If the QuantumObject is not a valid density matrix or state vector,

        Returns:
            float: The computed norm of the QuantumObject.
        """
        if self.is_scalar():
            return self.dense[0][0]

        if self.is_density_matrix() or self.shape[0] == self.shape[1]:
            if order == "tr":
                return np.sum(np.abs(np.linalg.eigvalsh(self.dense)))
            return scipy_norm(self._data, ord=order)

        if self.is_bra():
            return np.sqrt(self._data @ self._data.conj().T).toarray()[0, 0]

        if self.is_ket():
            return np.sqrt(self._data.conj().T @ self._data).toarray()[0, 0]

        raise ValueError("The QuantumObject is not a valid density matrix or state vector. Cannot compute the norm.")

    def unit(self, order: int | Literal["fro", "tr"] = "tr") -> QuantumObject:
        """
        Normalize the QuantumObject.

        Scales the QuantumObject so that its norm becomes 1, according to the specified norm order.

        Args:
            order (int or {"fro", "tr"}, optional): The order of the norm to use for normalization.
                Only applies if the QuantumObject represents a density matrix. Other than all the
                orders accepted by scipy, it also accepts 'tr' for the trace norm. Defaults to "tr".

        Raises:
            ValueError: If the norm of the QuantumObject is 0, making normalization impossible.

        Returns:
            QuantumObject: A new QuantumObject that is the normalized version of this object.
        """
        norm = self.norm(order=order)
        if norm == 0:
            raise ValueError("Cannot normalize a zero-norm Quantum Object")

        return QuantumObject(self._data / norm)

    def expm(self) -> QuantumObject:
        """
        Compute the matrix exponential of the QuantumObject.

        Returns:
            QuantumObject: A new QuantumObject representing the matrix exponential.
        """
        return QuantumObject(expm(self._data))

    def to_density_matrix(self) -> QuantumObject:
        """
        Convert the QuantumObject to a density matrix.

        If the QuantumObject represents a state vector (ket or bra), this method
        calculates the corresponding density matrix by taking the outer product.
        If the QuantumObject is already a density matrix, it is returned unchanged.
        The resulting density matrix is normalized.

        Raises:
            ValueError: If the QuantumObject is a scalar, as a density matrix cannot be derived.

        Returns:
            QuantumObject: A new QuantumObject representing the density matrix.
        """
        if self.is_scalar():
            raise ValueError("Cannot make a density matrix from scalar.")

        if self.is_density_matrix():
            return self

        if self.is_bra():
            return (self.adjoint() @ self).unit()

        if self.is_ket():
            return (self @ self.adjoint()).unit()

        raise ValueError(
            "Cannot make a density matrix from this QuantumObject. "
            "It must be either a ket, a bra or already a density matrix."
        )

    # ----------- Checks for Matrices ------------

    def is_ket(self) -> bool:
        """
        Check if the QuantumObject represents a ket (column vector) state.

        Returns:
            bool: True if the QuantumObject is a ket state, False otherwise.
        """
        return self.shape[1] == 1 and self.shape[0].bit_count() == 1

    def is_bra(self) -> bool:
        """
        Check if the QuantumObject represents a bra (row vector) state.

        Returns:
            bool: True if the QuantumObject is a bra state, False otherwise.
        """
        return self.shape[0] == 1 and self.shape[1].bit_count() == 1

    def is_scalar(self) -> bool:
        """
        Check if the QuantumObject is a scalar (1x1 matrix).

        Returns:
            bool: True if the QuantumObject is a scalar, False otherwise.
        """
        return self.shape == (1, 1)

    def is_operator(self) -> bool:
        """
        Check if the QuantumObject is an operator (square matrix).

        Returns:
            bool: True if the QuantumObject is an operator, False otherwise.
        """
        return self._data.shape[1] == self._data.shape[0] and self._data.shape[0].bit_count() == 1

    def is_density_matrix(self, tol: float = 1e-8) -> bool:
        """
        Determine if the QuantumObject is a valid density matrix.

        A valid density matrix must be square, Hermitian, positive semi-definite, and have a trace equal to 1.

        Args:
            tol (float, optional): The numerical tolerance for verifying Hermiticity,
                eigenvalue non-negativity, and trace. Defaults to 1e-8.

        Returns:
            bool: True if the QuantumObject is a valid density matrix, False otherwise.
        """
        # Check if rho is a square matrix
        if not self.is_operator():
            return False

        # Check Hermitian condition: rho should be equal to its conjugate transpose
        if not self.is_hermitian(tol=tol):
            return False

        # Check if eigenvalues are non-negative (positive semi-definite)
        eigenvalues = np.linalg.eigvalsh(self.dense)  # More stable for Hermitian matrices
        if np.any(eigenvalues < -tol):  # Allow small numerical errors
            return False

        # Check if the trace is 1
        return np.isclose(self._data.trace(), 1, atol=tol)

    def is_hermitian(self, tol: float = 1e-8) -> bool:
        """
        Check if the QuantumObject is Hermitian.

        Args:
            tol (float, optional): The numerical tolerance for verifying Hermiticity.
                Defaults to 1e-8.

        Returns:
            bool: True if the QuantumObject is Hermitian, False otherwise.
        """
        return np.allclose(self.dense, self._data.conj().T.toarray(), atol=tol)

    # ----------- Basic Arithmetic Operators ------------

    def __add__(self, other: QuantumObject | Complex) -> QuantumObject:
        if isinstance(other, QuantumObject):
            return QuantumObject(self._data + other._data)
        if isinstance(other, Complex) and other == 0:
            return self

        raise TypeError("Addition is only supported between QuantumState instances")

    def __sub__(self, other: QuantumObject) -> QuantumObject:
        if isinstance(other, QuantumObject):
            return QuantumObject(self._data - other._data)

        raise TypeError("Subtraction is only supported between QuantumState instances")

    def __mul__(self, other: QuantumObject | Complex) -> QuantumObject:
        if isinstance(other, (int, float, complex)):
            return QuantumObject(self._data * other)
        if isinstance(other, QuantumObject):
            return QuantumObject(self._data * other._data)

        raise TypeError("Unsupported multiplication type")

    def __matmul__(self, other: QuantumObject) -> QuantumObject:
        if isinstance(other, QuantumObject):
            return QuantumObject(self._data @ other._data)

        raise TypeError("Dot product is only supported between QuantumState instances")

    def __rmul__(self, other: QuantumObject | Complex) -> QuantumObject:
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f"{self.dense}"


###############################################################################
# Outside class Function Definitions
###############################################################################


def basis_state(n: int, N: int) -> QuantumObject:
    """
    Generate the n'th basis vector representation, on a N-size Hilbert space (N=2**num_qubits).

    This function creates a column vector (ket) representing the Fock state |n⟩ in a Hilbert space of dimension N.

    Args:
        n (int): The desired number state (from 0 to N-1).
        N (int): The dimension of the Hilbert space, has a value 2**num_qubits.

    Returns:
        QuantumObject: A QuantumObject representing the |n⟩'th basis state on a N-size Hilbert space (N=2**num_qubits).
    """
    return QuantumObject(csc_array(([1], ([n], [0])), shape=(N, 1)))


def ket(*state: int) -> QuantumObject:
    """
    Generate a ket state for a multi-qubit system.

    This function creates a tensor product of individual qubit states (kets) based on the input values.
    Each input must be either 0 or 1. For example, ket(0, 1) creates a two-qubit ket state |0⟩ ⊗ |1⟩.

    Args:
        *state (int): A sequence of integers representing the state of each qubit (0 or 1).

    Raises:
        ValueError: If any of the provided qubit states is not 0 or 1.

    Returns:
        QuantumObject: A QuantumObject representing the multi-qubit ket state.
    """
    if any(s not in {0, 1} for s in state):
        raise ValueError(f"the state can only contain 1s or 0s. But received: {state}")

    return tensor_prod([QuantumObject(csc_array(([1], ([s], [0])), shape=(2, 1))) for s in state])


def bra(*state: int) -> QuantumObject:
    """
    Generate a bra state for a multi-qubit system.

    This function creates a tensor product of individual qubit states (bras) based on the input values.
    Each input must be either 0 or 1. For example, bra(0, 1) creates a two-qubit bra state ⟨0| ⊗ ⟨1|.

    Args:
        *state (int): A sequence of integers representing the state of each qubit (0 or 1).

    Raises:
        ValueError: If any of the provided qubit states is not 0 or 1.

    Returns:
        QuantumObject: A QuantumObject representing the multi-qubit bra state.
    """
    if any(s not in {0, 1} for s in state):
        raise ValueError(f"the state can only contain 1s or 0s. But received:: {state}")

    return tensor_prod([QuantumObject(csc_array(([1], ([0], [s])), shape=(1, 2))) for s in state])


def tensor_prod(operators: list[QuantumObject]) -> QuantumObject:
    """
    Calculate the tensor product of a list of QuantumObjects.

    This function computes the tensor (Kronecker) product of all input QuantumObjects,
    resulting in a composite QuantumObject that represents the combined state or operator.

    Args:
        operators (list[QuantumObject]): A list of QuantumObjects to be combined via tensor product.

    Returns:
        QuantumObject: A new QuantumObject representing the tensor product of the inputs.
    """
    out = operators[0].data
    if len(operators) > 1:
        for i in range(1, len(operators)):
            out = kron(out, operators[i].data)

    return QuantumObject(out)


def expect_val(operator: QuantumObject, state: QuantumObject) -> Complex:
    """
    Calculate the expectation value of an operator with respect to a quantum state.

    Computes the expectation value ⟨state| operator |state⟩. The function handles both
    pure state vectors and density matrices appropriately.

    Args:
        operator (QuantumObject): The quantum operator represented as a QuantumObject.
        state (QuantumObject): The quantum state or density matrix represented as a QuantumObject.

    Raises:
        ValueError: If the operator is not a square matrix.

    Returns:
        Complex: The expectation value. The result is guaranteed to be real if the operator
                 is Hermitian, and may be complex otherwise.
    """
    if not operator.is_operator():
        raise ValueError("The operator must be a square matrix.")

    if state.data.shape[1] == state.data.shape[0]:
        return (operator @ state).dense.trace()

    return (state.adjoint() @ operator @ state).dense[0, 0]
