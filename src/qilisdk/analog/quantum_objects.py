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

Complex = int | float | complex
TWO = 2


class QuantumObject:
    def __init__(self, data: np.ndarray | sparray | spmatrix) -> None:
        if isinstance(data, np.ndarray):
            self._data = csr_matrix(data)
        elif issparse(data):
            self._data = data.tocsr()
        else:
            raise ValueError("Input must be a NumPy array or a SciPy sparse matrix")
        invalid_shape = (
            len(self._data.shape) > TWO
            or (self._data.shape[0] == 1 and self._data.shape[1] != 1 and self._data.shape[1] % 2 != 0)
            or (self._data.shape[1] == 1 and self._data.shape[0] != 1 and self._data.shape[0] % 2 != 0)
            or (self._data.shape[0] != self._data.shape[1] and self._data.shape[0] != 1 and self._data.shape[1] != 1)
            or (
                self._data.shape[1] == self._data.shape[0] and self._data.shape[0] % 2 != 0 and self._data.shape[0] != 1
            )
        )
        if invalid_shape:
            raise ValueError(
                "Dimension of data is wrong. expected data to have shape similar to (2**N, 2**N), (1, 2**N), (2**N, 1)",
                f"but received {self._data.shape}",
            )

    @property
    def data(self) -> csr_matrix:
        return self._data

    @property
    def dense(self) -> np.ndarray:
        return self._data.toarray()

    @property
    def nqubits(self) -> int:
        if self._data.shape[0] == self._data.shape[1]:
            return int(np.log2(self._data.shape[0]))
        if self._data.shape[0] == 1:
            return int(np.log2(self._data.shape[1]))
        if self._data.shape[1] == 1:
            return int(np.log2(self._data.shape[0]))
        return -1

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def dag(self) -> QuantumObject:
        """Computes the Adjoint (dagger) of Quantum Object.

        Returns:
            QuantumObject: the adjoin of the Quantum Object.
        """
        out = QuantumObject(self._data.conj().T)
        return out

    def ptrace(self, dims: list[int], keep: list[int]) -> "QuantumObject":
        """
        Perform a partial trace over subsystems not in `keep`.

        Parameters:
            dims (list of int): Dimensions of each subsystem.
            keep (list of int): Indices of subsystems to retain.

        Raises:
            ValueError: if the dimensions provided in dims don't match the shape of the quantum object.

        Returns:
            QuantumState: A new QuantumState representing the reduced density matrix.
        """
        rho = self.dense
        total_dim = np.prod(dims)
        if rho.shape != (total_dim, total_dim):
            raise ValueError("Dimension mismatch between provided dims and QuantumObject shape")

        n = len(dims)
        # Use letters from the ASCII alphabet (both cases) for einsum indices.
        # For each subsystem, assign two letters: one for the row index and one for the column index.
        row_letters = []
        col_letters = []
        out_row = []  # Letters that will remain in the output for the row part.
        out_col = []  # Letters for the column part.
        letters = iter(string.ascii_letters)

        for i in range(n):
            if i in keep:
                # For a subsystem we want to keep, use two different letters
                r = next(letters)
                c = next(letters)
                row_letters.append(r)
                col_letters.append(c)
                out_row.append(r)
                out_col.append(c)
            else:
                # For subsystems to be traced out, assign the same letter so that those indices are summed.
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

    def norm(self, order: int | Literal["fro", "tr"] = 1) -> float:
        """Compute the norm of the Quantum Object

        Args:
            order (int, "fro", "tr"): Order of the norm. Only used if the Quantum Object is a density matrix.

        Returns:
            float: the norm.
        """
        if self.is_scalar():
            return self.dense[0][0]
        if self.is_dm() or self.shape[0] == self.shape[1]:
            if order == "tr":
                return self._data.trace()
            return scipy_norm(self._data, ord=order)
        if self.is_bra():
            return np.sqrt(self._data @ self._data.conj().T).toarray()[0, 0]
        return np.sqrt(self._data.conj().T @ self._data).toarray()[0, 0]

    def unit(self, order: int | Literal["fro", "tr"] = "tr") -> QuantumObject:
        """Normalizes the quantum Object

        Args:
            order (int, "fro", "tr"): Order of the norm. Only used if the Quantum Object is a density matrix.

        Raises:
            ValueError: If the norm of the Quantum Object is 0

        Returns:
            QuantumObject: The normalized Quantum Object.
        """
        norm = self.norm(order=order)
        if norm == 0:
            raise ValueError("Cannot normalize a zero-norm Quantum Object")
        return QuantumObject(self._data / norm)

    def __repr__(self) -> str:
        return f"{self.dense}"

    def expm(self) -> QuantumObject:
        """Computes the matrix exponentiation of the Quantum Object

        Returns:
            QuantumObject: the matrix exponentiation of the Quantum Object.
        """
        return QuantumObject(expm(self._data))

    def is_ket(self) -> bool:
        return self.shape[0] % 2 == 0 and self.shape[1] == 1

    def is_bra(self) -> bool:
        return self.shape[1] % 2 == 0 and self.shape[0] == 1

    def is_scalar(self) -> bool:
        return self.data.shape == (1, 1)

    def is_dm(self, tol: float = 1e-8) -> bool:
        """
        Checks if a given matrix is a valid density matrix.

        Parameters:
            rho (numpy.ndarray): The matrix to check.
            tol (float): Numerical tolerance for checking conditions.

        Returns:
            bool: True if rho is a valid density matrix, False otherwise.
        """
        # Check if rho is a square matrix
        if self.shape[0] != self.shape[1]:
            return False

        # Check Hermitian condition: rho should be equal to its conjugate transpose
        if not np.allclose(self.dense, self._data.conj().T.toarray(), atol=tol):
            return False

        # Check if eigenvalues are non-negative (positive semi-definite)
        eigenvalues = np.linalg.eigvalsh(self.dense)  # More stable for Hermitian matrices
        if np.any(eigenvalues < -tol):  # Allow small numerical errors
            return False

        # Check if the trace is 1
        return np.isclose(self._data.trace(), 1, atol=tol)

    def is_herm(self, tol: float = 1e-8) -> bool:
        return np.allclose(self.dense, self._data.conj().T.toarray(), atol=tol)

    def to_dm(self) -> QuantumObject:
        if self.is_scalar():
            raise ValueError("Cannot make a density matrix from scalar.")
        if self.is_dm():
            return self
        if self.is_bra():
            return (self.dag() @ self).unit()
        return (self @ self.dag()).unit()


def basis(N: int, n: int) -> QuantumObject:
    """Generates the vector representation of a Fock state.

    Args:
        N (int): Number of Fock states in Hilbert space
        n (int): Integer corresponding to desired number state, defaults to 0 if omitted.

    Returns:
        QuantumObject: A QuantumObject representing the requested number state ``|n>``
    """
    return QuantumObject(csc_array(([1], ([n], [0])), shape=(N, 1)))


def ket(*state: int) -> QuantumObject:
    """Generate a ket state for a set of qubits.

    Raises:
        ValueError: if the qubit states provided are not 0 or 1.

    Returns:
        QuantumObject: a Quantum Object representing the state.
    """
    if any(s not in {0, 1} for s in state):
        raise ValueError("the state can only be 1 or 0.")
    return tensor([QuantumObject(csc_array(([1], ([s], [0])), shape=(2, 1))) for s in state])


def bra(*state: int) -> QuantumObject:
    """Generate a bra state for a set of qubits.

    Raises:
        ValueError: if the qubit states provided are not 0 or 1.

    Returns:
        QuantumObject: a Quantum Object representing the state.
    """
    if any(s not in {0, 1} for s in state):
        raise ValueError("the state can only be 1 or 0.")
    return tensor([QuantumObject(csc_array(([1], ([0], [s])), shape=(1, 2))) for s in state])


def tensor(operators: list[QuantumObject]) -> QuantumObject:
    """Calculates the tensor product of input operators or states.

    Args:
        operators (list[QuantumObject]): a list of Quantum Objects for tensor product.

    Returns:
        QuantumObject: A composite Quantum Object.
    """
    out = operators[0].data
    if len(operators) > 1:
        for i in range(1, len(operators)):
            out = kron(out, operators[i].data)
    return QuantumObject(out)


def expect(operator: QuantumObject, state: QuantumObject) -> Complex:
    """Calculates the expectation value for operator(s) and state(s).

    Args:
        operator (QuantumObject): operator for expectation value.
        state (QuantumObject): quantum state or density matrix.

    Returns:
        float/int/complex : The expectation value. The output is real if the operator is Hermitian, complex otherwise.
    """
    if state.data.shape[1] == state.data.shape[0]:
        return (operator @ state).dense.trace()
    return (state.dag() @ operator @ state).dense[0, 0]
