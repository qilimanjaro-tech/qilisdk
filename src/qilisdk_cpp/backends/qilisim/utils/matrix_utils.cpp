// Copyright 2025 Qilimanjaro Quantum Tech
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "matrix_utils.h"
#include "../libs/pybind.h"

SparseMatrix exp_mat_action(const SparseMatrix& H, std::complex<double> dt, const SparseMatrix& e1) {
    /*
    Compute the action of the matrix exponential exp(H*dt) acting on a vector e1.

    Args:
        H (SparseMatrix): The upper Hessenberg matrix.
        dt (std::complex<double>): The time step. Can be complex if needed.
        e1 (SparseMatrix): The vector to apply the exponential to.

    Returns:
        SparseMatrix: The result of exp(H*dt) * e1.
    */
    DenseMatrix H_dense = dt * DenseMatrix(H);
    DenseMatrix exp_H = H_dense.exp() * e1;
    SparseMatrix exp_H_sparse = exp_H.sparseView();
    return exp_H_sparse;
}

SparseMatrix exp_mat(const SparseMatrix& H, std::complex<double> dt) {
    /*
    Compute the matrix exponential exp(H*dt).

    Args:
        H (SparseMatrix): The matrix to exponentiate.
        dt (std::complex<double>): The time step. Can be complex if needed.

    Returns:
        SparseMatrix: The matrix exponential exp(H*dt).
    */
    DenseMatrix H_dense = dt * DenseMatrix(H);
    DenseMatrix exp_H = H_dense.exp();
    SparseMatrix exp_H_sparse = exp_H.sparseView();
    return exp_H_sparse;
}

std::complex<double> dot(const SparseMatrix& v1, const SparseMatrix& v2) {
    /*
    Compute the inner product between two sparse matrices.
    Note that the first matrix is conjugated.

    Args:
        v1 (SparseMatrix): The first matrix.
        v2 (SparseMatrix): The second matrix.

    Returns:
        std::complex<double>: The inner product result.
    */
    return v1.conjugate().cwiseProduct(v2).sum();
}

std::complex<double> dot(const DenseMatrix& v1, const DenseMatrix& v2) {
    /*
    Compute the inner product between two dense matrices.
    Note that the first matrix is conjugated.

    Args:
        v1 (DenseMatrix): The first matrix.
        v2 (DenseMatrix): The second matrix.

    Returns:
        std::complex<double>: The inner product result.
    */
    return v1.conjugate().cwiseProduct(v2).sum();
}

std::complex<double> trace(const SparseMatrix& matrix) {
    /*
    Compute the trace of a square matrix.

    Args:
        matrix (SparseMatrix): The input square matrix.

    Returns:
        std::complex<double>: The trace of the matrix.
    */
    if (matrix.rows() != matrix.cols()) {
        throw py::value_error("Matrix must be square to compute trace.");
    }
    std::complex<double> tr = 0.0;
    for (int i = 0; i < matrix.rows(); ++i) {
        tr += matrix.coeff(i, i);
    }
    return tr;
}

SparseMatrix expand_operator(int qubit, int nqubits, const SparseMatrix& op) {
    /*
    Expand a single-qubit operator to act on the full n-qubit system.

    Args:
        qubit (int): The target qubit index.
        nqubits (int): The total number of qubits.
        op (SparseMatrix): The single-qubit operator.

    Returns:
        SparseMatrix: The expanded operator acting on the full system.
    */
    SparseMatrix result(1, 1);
    result.insert(0, 0) = 1.0;
    result.makeCompressed();
    for (int q = 0; q < nqubits; ++q) {
        if (q == qubit) {
            result = Eigen::kroneckerProduct(result, op).eval();
        } else {
            result = Eigen::kroneckerProduct(result, I).eval();
        }
    }
    return result;
}

SparseMatrix expand_operator(int nqubits, const SparseMatrix& op) {
    /*
    Expand an operator to act on the full n-qubit system.

    Args:
        nqubits (int): The total number of qubits.
        op (SparseMatrix): The operator to expand.

    Returns:
        SparseMatrix: The expanded operator acting on the full system.
        
    Raises:
        py::value_error: If the operator size is not compatible with the number of qubits.
    */
    int current_qubits = static_cast<int>(std::log2(op.rows()));
    if (current_qubits == nqubits) {
        return op;
    }
    SparseMatrix result(1, 1);
    result.insert(0, 0) = 1.0;
    result.makeCompressed();
    if (nqubits % current_qubits != 0) {
        throw py::value_error("The operator size is not compatible with the number of qubits.");
    }
    int repeats_needed = nqubits / current_qubits;
    for (int q = 0; q < repeats_needed; ++q) {
        result = Eigen::kroneckerProduct(result, op).eval();
    }
    return result;
}

SparseMatrix expand_operator(const std::vector<int>& target_qubits, int nqubits, const SparseMatrix& op) {
    /*
    Expand a multi-qubit operator to act on the full n-qubit system.

    Args:
        target_qubits (std::vector<int>): The list of target qubit indices.
        nqubits (int): The total number of qubits.
        op (SparseMatrix): The multi-qubit operator.

    Returns:
        SparseMatrix: The expanded operator acting on the full system.
        
    Raises:
        py::value_error: If the operator size is not compatible with the number of target qubits.
    */
    int current_qubits = static_cast<int>(std::log2(op.rows()));
    if (current_qubits == nqubits) {
        return op;
    }
    SparseMatrix result(1, 1);
    result.insert(0, 0) = 1.0;
    result.makeCompressed();
    for (int q = 0; q < nqubits; ++q) {
        if (std::find(target_qubits.begin(), target_qubits.end(), q) != target_qubits.end()) {
            result = Eigen::kroneckerProduct(result, op).eval();
        } else {
            result = Eigen::kroneckerProduct(result, I).eval();
        }
    }
    return result;
}

SparseMatrix vectorize(const SparseMatrix& matrix, double atol) {
    /*
    Vectorize a matrix by stacking its columns.

    Args:
        matrix (SparseMatrix): The input matrix.
        atol (double): Absolute tolerance for considering values as non-zero.

    Returns:
        SparseMatrix: The vectorized matrix as a column vector.
    */
    int rows = int(matrix.rows());
    int cols = int(matrix.cols());
    Triplets vec_entries;
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            std::complex<double> val = matrix.coeff(r, c);
            if (std::abs(val) > atol) {
                vec_entries.emplace_back(Triplet(r + c * rows, 0, val));
            }
        }
    }
    SparseMatrix vec_matrix(long(rows * cols), 1);
    vec_matrix.setFromTriplets(vec_entries.begin(), vec_entries.end());
    return vec_matrix;
}

SparseMatrix devectorize(const SparseMatrix& vec_matrix, double atol) {
    /*
    Devectorize a column vector back into a square matrix.

    Args:
        vec_matrix (SparseMatrix): The input vectorized matrix.
        atol (double): Absolute tolerance for considering values as non-zero.

    Returns:
        SparseMatrix: The devectorized square matrix.
    */
    long dim = static_cast<long>(std::sqrt(vec_matrix.rows()));
    Triplets mat_entries;
    for (int c = 0; c < dim; ++c) {
        for (int r = 0; r < dim; ++r) {
            std::complex<double> val = vec_matrix.coeff(r + c * dim, 0);
            if (std::abs(val) > atol) {
                mat_entries.emplace_back(Triplet(r, c, val));
            }
        }
    }
    SparseMatrix mat(dim, dim);
    mat.setFromTriplets(mat_entries.begin(), mat_entries.end());
    return mat;
}