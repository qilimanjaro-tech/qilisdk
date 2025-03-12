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

import numpy as np
from scipy.sparse import csc_array, csr_matrix, issparse, kron, sparray, spmatrix
from scipy.sparse.linalg import expm

Complex = int | float | complex
TWO = 2


class QuantumObject:

    def __init__(self, data: np.ndarray | sparray | spmatrix) -> None:
        if isinstance(data, np.ndarray):
            self._data = csr_matrix(data)
        elif issparse(data):
            self._data = data
        else:
            raise ValueError("Input must be a NumPy array or a SciPy sparse matrix")
        invalid_shape = (
            len(self._data.shape) > TWO
            or (self._data.shape[0] == 1 and self._data.shape[1] % 2 != 0)
            or (self._data.shape[1] == 1 and self._data.shape[0] % 2 != 0)
            or (self._data.shape[0] != self._data.shape[1] and self._data.shape[0] != 1 and self._data.shape[1] != 1)
            or (self._data.shape[1] == self._data.shape[0] and self._data.shape[0] % 2 != 0)
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
    def shape(self) -> np.ndarray:
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

    def norm(self) -> float:
        """Compute the norm of the Quantum Object

        Returns:
            float: the norm.
        """
        return np.sqrt(self._data.conj().T @ self._data).toarray()[0, 0]

    def unit(self) -> QuantumObject:
        """Normalizes the quantum Object

        Raises:
            ValueError: If the norm of the Quantum Object is 0

        Returns:
            QuantumObject: The normalized Quantum Object.
        """
        norm = self.norm()
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


def basis(N: int, n: int) -> QuantumObject:
    """Generates the vector representation of a Fock state.

    Args:
        N (int): Number of Fock states in Hilbert space
        n (int): Integer corresponding to desired number state, defaults to 0 if omitted.

    Returns:
        QuantumObject: A QuantumObject representing the requested number state ``|n>``
    """
    return QuantumObject(csc_array(([1], ([n], [0])), shape=(N, 1)))


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
