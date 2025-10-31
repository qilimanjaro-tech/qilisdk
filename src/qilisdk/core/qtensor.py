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

from collections import defaultdict
from typing import Iterable, Literal

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse, kron, sparray, spmatrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh, expm
from scipy.sparse.linalg import norm as scipy_norm

from qilisdk.yaml import yaml

Complex = int | float | complex


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _prod(xs: Iterable[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


###############################################################################
# Main Class Definition
###############################################################################


@yaml.register_class
class QTensor:
    """
    Lightweight wrapper around sparse matrices representing quantum states or operators.

    The QTensor class is a wrapper around sparse matrices (or NumPy arrays,
    which are converted to sparse matrices) that represent quantum states (kets, bras, or density matrices)
    or operators. It provides utility methods for common quantum operations such as
    taking the adjoint (dagger), computing tensor products, partial traces, and norms.

    The internal data is stored as a SciPy CSR (Compressed Sparse Row) matrix for
    efficient arithmetic and manipulation. The expected shapes for the data are:
    - (2**N, 2**N) for operators or density matrices (or scalars),
    - (2**N, 1) for ket states,
    - (1, 2**N) for bra states.

    Converts a NumPy array to a CSR matrix if needed and validates the shape of the input.
    The input must represent a valid quantum state or operator with appropriate dimensions.


    Example:
        .. code-block:: python

            import numpy as np
            from qilisdk.core import QTensor

            ket = QTensor(np.array([[1.0], [0.0]]))
            density = ket * ket.adjoint()
    """

    def __init__(self, data: np.ndarray | sparray | spmatrix) -> None:
        """
        Args:
            data (np.ndarray | sparray | spmatrix): Dense or sparse matrix defining the quantum object. Expected
                shapes are ``(2**N, 2**N)`` for operators, ``(2**N, 1)`` for kets, ``(1, 2**N)`` for bras, or ``(1, 1)`` for scalars.

        Raises:
            ValueError: If ``data`` is not 2-D or does not correspond to valid qubit dimensions.
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 2:  # noqa: PLR2004
                raise ValueError("Input ndarray must be 2D")
            self._data = csr_matrix(data)
        elif issparse(data):
            self._data = data.tocsr()
        else:
            raise ValueError("Input must be a NumPy array or a SciPy sparse matrix")

        r, c = self._data.shape

        # Validate "qubit-like" shapes
        valid = (
            (r == c and _is_pow2(r))  # operator/density
            or (c == 1 and _is_pow2(r))  # ket
            or (r == 1 and _is_pow2(c))  # bra
            or (r == c == 1)  # scalar
        )
        if not valid:
            raise ValueError(
                f"Data must have shape (2**N, 2**N), (2**N, 1), (1, 2**N), or (1,1). Got {self._data.shape}."
            )
        # Cache nqubits (immutable once constructed since we never resize in-place)
        if r == c:
            self._nqubits = int(np.log2(r))
        elif r == 1:
            self._nqubits = int(np.log2(c))
        else:
            self._nqubits = int(np.log2(r))

    # ------------- Properties --------------

    @property
    def data(self) -> csr_matrix:
        """
        Get the internal sparse matrix representation of the QTensor.

        Returns:
            csc_matrix: The internal representation as a CSR matrix.
        """
        return self._data

    @property
    def nqubits(self) -> int:
        """
        Compute the number of qubits represented by the QTensor.

        Returns:
            int: The number of qubits if determinable; otherwise, -1.
        """
        return self._nqubits

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the QTensor's internal matrix.

        Returns:
            tuple[int, ...]: The shape of the internal matrix.
        """
        return self._data.shape

    @property
    def dense(self) -> np.ndarray:
        """
        Get the dense (NumPy array) representation of the QTensor.

        Returns:
            np.ndarray: The dense array representation.
        """
        return self._data.toarray()

    # ------------- Basic structural tests -------------

    def is_ket(self) -> bool:
        r, c = self.shape
        return c == 1 and _is_pow2(r)

    def is_bra(self) -> bool:
        r, c = self.shape
        return r == 1 and _is_pow2(c)

    def is_scalar(self) -> bool:
        return self.shape == (1, 1)

    def is_operator(self) -> bool:
        r, c = self.shape
        return r == c and _is_pow2(r)

    # ----------- Matrix Operations ------------

    def adjoint(self) -> QTensor:
        """
        Compute the adjoint (conjugate transpose) of the QTensor.

        Returns:
            QTensor: A new QTensor that is the adjoint of this object.
        """
        return QTensor(self._data.getH())

    def trace(self) -> complex:
        # diagonal() returns dense 1D array; summing it is cheap
        return complex(self._data.diagonal().sum())

    def ptrace(self, keep: list[int], dims: list[int] | None = None) -> QTensor:
        """
        Compute the partial trace over subsystems not in 'keep'.

        This method calculates the reduced density matrix by tracing out
        the subsystems that are not specified in the 'keep' parameter.
        The input 'dims' represents the dimensions of each subsystem (optional),
        and 'keep' indicates the indices of those subsystems to be retained.

        If the QTensor is a ket or bra, it will first be converted to a density matrix.

        Args:
            keep (list[int]): A list of indices corresponding to the subsystems to retain.
                The order of the indices in 'keep' is not important, since dimensions will
                be returned in the tensor original order, but the indices must be unique.
            dims (list[int], optional): A list specifying the dimensions of each subsystem.
                If not specified, a density matrix of qubit states is assumed, and the
                dimensions are inferred accordingly (i.e. we split the state in dim 2 states).

        Raises:
            ValueError: If the product of the dimensions in dims does not match the
                shape of the QTensor's dense representation or if any dimension is non-positive.
            ValueError: If the indices in 'keep' are not unique or are out of range.
            ValueError: If the QTensor is not a valid density matrix or state vector.
            ValueError: If the number of subsystems exceeds the available ASCII letters.

        Returns:
            QTensor: A new QTensor representing the reduced density matrix
                for the subsystems specified in 'keep'.
        """
        if dims is None:
            dims = [2] * self.nqubits
        if any(d <= 0 for d in dims):
            raise ValueError("All subsystem dimensions must be positive")

        dim_total = self.shape[0] if self.is_operator() else (self.shape[0] if self.is_ket() else self.shape[1])
        if _prod(dims) != dim_total:
            raise ValueError(
                f"Product of dims {dims} = {_prod(dims)} does not match Hilbert space dimension {dim_total}."
            )

        nsub = len(dims)
        keep_set = set(keep)
        if len(keep_set) != len(keep):
            raise ValueError("Duplicate indices in keep")
        if any(i < 0 or i >= nsub for i in keep_set):
            raise ValueError("keep indices out of range")

        keep_idx = sorted(keep_set)  # return order is “original” ordering
        drop_idx = [i for i in range(nsub) if i not in keep_set]
        dims_keep = [dims[i] for i in keep_idx]
        dims_drop = [dims[i] for i in drop_idx]
        Kdim = _prod(dims_keep) if dims_keep else 1

        # Pure-state path ψ ⇒ p_keep = M @ M† (no NxN density matrix created)
        if self.is_ket() or self.is_bra():
            psi = self._data
            if self.is_bra():
                psi = psi.T.conj()  # make it a column vector
            # Decide whether to process as dense vector or sparse-vector path
            N = _prod(dims)
            density = psi.nnz / N
            # Dense vector path is faster once the vector is reasonably filled
            if density >= 0.05 or N <= (1 << 20):  # noqa: PLR2004
                psi1d = np.asarray(psi.toarray().reshape(-1))  # only vector, not matrix
                psi_nd = psi1d.reshape(dims)
                perm = keep_idx + drop_idx  # bring keep first
                psi_perm = np.transpose(psi_nd, perm)
                M = psi_perm.reshape(Kdim, -1)
                rho_keep = M @ M.conj().T
                return QTensor(csr_matrix(rho_keep))
            # Truly sparse ψ: build M implicitly by grouping by the traced index
            coo = psi.tocoo()
            nz_idx = coo.row
            nz_val = coo.data
            # unravel all non-zero positions once
            digits = np.vstack(np.unravel_index(nz_idx, dims))  # (nsub, nnz)
            k_digits = digits[keep_idx, :] if keep_idx else np.zeros((0, nz_val.size), dtype=int)
            t_digits = digits[drop_idx, :] if drop_idx else np.zeros((0, nz_val.size), dtype=int)
            k_lin = np.ravel_multi_index(k_digits, dims_keep) if keep_idx else np.zeros(nz_val.size, dtype=int)  # type: ignore[call-overload]
            t_lin = np.ravel_multi_index(t_digits, dims_drop) if drop_idx else np.zeros(nz_val.size, dtype=int)  # type: ignore[call-overload]

            # For each traced index t, accumulate outer products of the K-dimensional slice
            buckets: dict[int, list[tuple[int, complex]]] = defaultdict(list)
            for kl, tl, v in zip(k_lin, t_lin, nz_val):
                buckets[int(tl)].append((int(kl), v))

            data, row, col = [], [], []
            for _, items in buckets.items():
                # x is (Kdim,) sparse vector represented by indices & values
                ks = np.fromiter((i for i, _ in items), dtype=int)
                vs = np.fromiter((v for _, v in items), dtype=complex)
                # Outer product of this slice: accumulate into COO lists
                # Note: number of pairs is len(items)^2 which is fine for very sparse ψ.
                r = np.repeat(ks, ks.size)
                c = np.tile(ks, ks.size)
                d = (vs[:, None] * np.conj(vs[None, :])).ravel()
                row.append(r)
                col.append(c)
                data.append(d)

            if data:
                row = np.concatenate(row)
                col = np.concatenate(col)
                data = np.concatenate(data)
                out = coo_matrix((data, (row, col)), shape=(Kdim, Kdim))
                out.sum_duplicates()
                return QTensor(out.tocsr())
            return QTensor(csr_matrix((Kdim, Kdim)))

        # Operator/density-matrix path: COO remapping with traced-equal mask
        if self.is_operator():
            rho = self._data.tocoo()
            # unravel rows/cols to multi-indices
            r_multi = np.vstack(np.unravel_index(rho.row, dims))  # (nsub, nnz)
            c_multi = np.vstack(np.unravel_index(rho.col, dims))
            if drop_idx:
                mask = np.all(r_multi[drop_idx, :] == c_multi[drop_idx, :], axis=0)
            else:
                mask = np.ones(rho.nnz, dtype=bool)

            if keep_idx:
                rK = r_multi[keep_idx, :][:, mask]
                cK = c_multi[keep_idx, :][:, mask]
                new_r = np.ravel_multi_index(rK, dims_keep)  # type: ignore[call-overload]
                new_c = np.ravel_multi_index(cK, dims_keep)  # type: ignore[call-overload]
                data = rho.data[mask]
            else:
                # keep nothing → scalar
                new_r = np.zeros(mask.sum(), dtype=int)
                new_c = np.zeros(mask.sum(), dtype=int)
                data = rho.data[mask]

            out = coo_matrix((data, (new_r, new_c)), shape=(Kdim, Kdim))
            out.sum_duplicates()
            return QTensor(out.tocsr())

        raise ValueError("The QTensor is not a valid state or operator for ptrace().")

    def norm(self, order: int | Literal["fro", "tr"] = 1) -> float:
        """
        Compute the norm of the QTensor.

        For density matrices, the norm order can be specified. For state vectors, the norm is computed accordingly.

        Args:
            order (int or {"fro", "tr"}, optional): The order of the norm.
                Only applies if the QTensor represents a density matrix. Other than all the
                orders accepted by scipy, it also accepts 'tr' for the trace norm. Defaults to 1.

        Raises:
            ValueError: If the QTensor is not a valid density matrix or state vector,

        Returns:
            float: The computed norm of the QTensor.
        """
        if self.is_scalar():
            return float(abs(self._data.toarray()[0, 0]))

        if self.is_operator():
            if order == "tr":
                if self.is_density_matrix():
                    return 1.0
                # Only correct for Hermitian; otherwise, nuclear norm requires SVD
                if self.is_hermitian():
                    r, _ = self.shape
                    if r <= 1024:  # noqa: PLR2004
                        w = np.linalg.eigvalsh(self._data.toarray())
                        return float(np.sum(np.abs(w)))
                    raise ValueError("Trace norm for large Hermitian operators is not implemented without densifying.")
                r, _ = self.shape
                if r <= 1024:  # noqa: PLR2004
                    s = np.linalg.svd(self._data.toarray(), compute_uv=False)
                    return float(np.sum(s))
                raise ValueError(
                    "Trace (nuclear) norm for large non-Hermitian operators requires SVD; not implemented."
                )
            # Delegate other norms to SciPy; supported: 'fro', 1, np.inf, etc.
            return float(scipy_norm(self._data, ord=order))

        # kets/bras: 2-norm of vector
        v = self._data
        if self.is_bra():
            v = v.T.conj()
        return float(np.sqrt(np.real(v.conj().multiply(v).sum())))

    def unit(self, order: int | Literal["fro", "tr"] = "tr") -> QTensor:
        """
        Normalize the QTensor.

        Scales the QTensor so that its norm becomes 1, according to the specified norm order.

        Args:
            order (int or {"fro", "tr"}, optional): The order of the norm to use for normalization.
                Only applies if the QTensor represents a density matrix. Other than all the
                orders accepted by scipy, it also accepts 'tr' for the trace norm. Defaults to "tr".

        Raises:
            ValueError: If the norm of the QTensor is 0, making normalization impossible.

        Returns:
            QTensor: A new QTensor that is the normalized version of this object.
        """
        norm = self.norm(order=order)
        if norm == 0:
            raise ValueError("Cannot normalize a zero-norm Quantum Object")

        return QTensor(self._data / norm)

    def expm(self) -> QTensor:
        """
        Compute the matrix exponential of the QTensor.

        Returns:
            QTensor: A new QTensor representing the matrix exponential.
        """
        return QTensor(expm(self._data.tocsc()))

    def to_density_matrix(self) -> QTensor:
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
        if self.is_scalar():
            raise ValueError("Cannot make a density matrix from a scalar.")
        if self.is_density_matrix():
            return self
        if self.is_ket():
            rho = self @ self.adjoint()
        elif self.is_bra():
            rho = self.adjoint() @ self
        elif self.is_operator():
            raise ValueError("Operator is not a density matrix (trace≠1 or not Hermitian).")
        else:
            raise ValueError("Invalid object for density matrix conversion.")

        tr = float(np.real(rho.trace()))
        if tr == 0.0:
            raise ValueError("Cannot normalize density matrix with zero trace.")
        return QTensor(rho.data / tr)  # keep it sparse

    def is_density_matrix(self, tol: float = 1e-8) -> bool:
        """
        Determine if the QTensor is a valid density matrix.

        A valid density matrix must be square, Hermitian, positive semi-definite, and have a trace equal to 1.

        Args:
            tol (float, optional): The numerical tolerance for verifying Hermiticity,
                eigenvalue non-negativity, and trace. Defaults to 1e-8.

        Returns:
            bool: True if the QTensor is a valid density matrix, False otherwise.
        """
        # Check if rho is a square matrix
        if not self.is_operator():
            return False
        if abs(self.trace() - 1.0) > tol:
            return False
        if not self.is_hermitian(tol=tol):
            return False
        # PSD check via smallest eigenvalue of Hermitian matrix
        try:
            vals = eigsh(self._data, k=1, which="SA", return_eigenvectors=False, tol=1e-6)
            lam_min = float(np.real(vals[0]))
        except ArpackNoConvergence:
            # If ARPACK fails, fall back to dense only if small
            r, _ = self.shape
            if r <= 2048:  # noqa: PLR2004
                lam_min = float(np.linalg.eigvalsh(self._data.toarray()).min())
            else:
                # Conservative fallback: don't claim it's a DM if we can't certify PSD
                return False
        return lam_min >= -tol

    def is_hermitian(self, tol: float = 1e-8) -> bool:
        """
        Check if the QTensor is Hermitian.

        Args:
            tol (float, optional): The numerical tolerance for verifying Hermiticity.
                Defaults to 1e-8.

        Returns:
            bool: True if the QTensor is Hermitian, False otherwise.
        """
        if not self.is_operator():
            return False
        diff = self._data - self._data.getH()
        if diff.nnz == 0:
            return True
        return float(scipy_norm(diff, ord="fro")) <= tol

    # ----------- Basic Arithmetic Operators ------------

    def __add__(self, other: QTensor | Complex) -> QTensor:
        if isinstance(other, QTensor):
            return QTensor(self._data + other._data)
        if other == 0:
            return self
        return NotImplemented

    def __radd__(self, other: QTensor | Complex) -> QTensor:
        return self.__add__(other)

    def __sub__(self, other: QTensor) -> QTensor:
        if isinstance(other, QTensor):
            return QTensor(self._data - other._data)
        return NotImplemented

    def __mul__(self, other: QTensor | Complex) -> QTensor:
        if isinstance(other, (int, float, complex)):
            return QTensor(self._data * other)
        if isinstance(other, QTensor):
            return QTensor(self._data * other._data)
        return NotImplemented

    def __matmul__(self, other: QTensor) -> QTensor:
        if isinstance(other, QTensor):
            return QTensor(self._data @ other._data)
        return NotImplemented

    def __rmul__(self, other: QTensor | Complex) -> QTensor:
        return self.__mul__(other)

    def __repr__(self) -> str:
        r, c = self.shape
        nnz = self._data.nnz
        s = f"QTensor(shape={r}x{c}, nnz={nnz}, format='csr')"
        if r * c <= 64:  # noqa: PLR2004
            s += f"\n{self._data.toarray()}"
        return s


###############################################################################
# Outside class Function Definitions
###############################################################################


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


def ket(*state: int) -> QTensor:
    r"""
    Generate a ket state for a multi-qubit system.

    This function creates a tensor product of individual qubit states (kets) based on the input values.
    Each input must be either 0 or 1. For example, ket(0, 1) creates a two-qubit ket state \|0⟩ ⊗ \|1⟩.

    Args:
        *state (int): A sequence of integers representing the state of each qubit (0 or 1).

    Raises:
        ValueError: If any of the provided qubit states is not 0 or 1.

    Returns:
        QTensor: A QTensor representing the multi-qubit ket state.
    """
    if not state:
        raise ValueError("ket() requires at least one qubit (0/1).")
    if any(s not in {0, 1} for s in state):
        raise ValueError(f"state must contain only 0s/1s, got {state}")

    # Number of qubits
    n = len(state)
    N = 1 << n  # 2**n

    # Big-endian linear index: kron(|s0>, |s1>, ..., |s_{n-1}>) -> index int(s0...s_{n-1}, base=2)
    idx = 0
    for s in state:
        idx = (idx << 1) | s

    # Reuse existing basis_state creator (sparse, single 1 at (idx, 0))
    return basis_state(idx, N)


def bra(*state: int) -> QTensor:
    r"""
    Generate a bra state for a multi-qubit system.

    This function creates a tensor product of individual qubit states (bras) based on the input values.
    Each input must be either 0 or 1. For example, bra(0, 1) creates a two-qubit bra state ⟨0\| ⊗ ⟨1\|.

    Args:
        *state (int): A sequence of integers representing the state of each qubit (0 or 1).

    Returns:
        QTensor: A QTensor representing the multi-qubit bra state.
    """
    return ket(*state).adjoint()


def tensor_prod(operators: list[QTensor]) -> QTensor:
    """
    Calculate the tensor product of a list of QTensors.

    This function computes the tensor (Kronecker) product of all input QTensors,
    resulting in a composite QTensor that represents the combined state or operator.

    Args:
        operators (list[QTensor]): A list of QTensors to be combined via tensor product.

    Returns:
        QTensor: A new QTensor representing the tensor product of the inputs.

    Raises:
        ValueError: If operators list is empty.
    """
    if not operators:
        raise ValueError("tensor_prod requires at least one operator/state")
    out = operators[0].data
    for op in operators[1:]:
        # Sparse kron returns same sparse type; keep CSR at the end
        out = kron(out, op.data).tocsr()
    return QTensor(out)


def expect_val(operator: QTensor, state: QTensor) -> Complex:
    r"""
    Calculate the expectation value of an operator with respect to a quantum state.

    Computes the expectation value ⟨state\| operator \|state⟩. The function handles both
    pure state vectors and density matrices appropriately.

    Args:
        operator (QTensor): The quantum operator represented as a QTensor.
        state (QTensor): The quantum state or density matrix represented as a QTensor.

    Raises:
        ValueError: If the operator is not a square matrix.
        ValueError: If the state provided is not a valid quantum state.

    Returns:
        Complex: The expectation value. The result is guaranteed to be real if the operator is Hermitian, and may be complex otherwise.
    """
    if not operator.is_operator():
        raise ValueError("operator must be square")
    # p case: tr(O p) = sum((O.T) ⊙ p)
    if state.is_density_matrix():
        return complex((operator.data.T.multiply(state.data)).sum())
    # |ψ⟩ case: ⟨ψ| O |ψ⟩ = (ψ† (O ψ))
    if state.is_ket():
        v = operator.data @ state.data  # (N,1)
        return complex((state.data.getH() @ v).toarray()[0, 0])
    if state.is_bra():
        v = state.data @ operator.data  # (1,N)
        return complex((v @ state.data.getH()).toarray()[0, 0])
    raise ValueError("state is invalid for expect_val")
