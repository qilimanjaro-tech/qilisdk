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
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

// GCOV_EXCL_BR_START

// Our real and complex types
#ifdef SINGLE_PRECISION
typedef float Real;
#else
typedef double Real;
#endif
typedef std::complex<Real> Complex;

// Eigen specfic type defs (all derived from Complex/Real so they follow the toggle)
typedef Eigen::SparseMatrix<Complex, Eigen::RowMajor> SparseMatrix;
typedef Eigen::SparseMatrix<Complex, Eigen::ColMajor> SparseMatrixCol;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> DenseVector;
typedef Eigen::Matrix<Complex, 1, Eigen::Dynamic> DenseRowVector;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> RealMatrix;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RealVector;
typedef Eigen::Triplet<Complex> Triplet;
typedef std::vector<Triplet> Triplets;

inline void nan_error() {
    /*
    Raise an error indicating that the state has become invalid (NaN or Inf).

    Note: pybind11 translates std::invalid_argument into a Python ValueError.
    */
    throw std::invalid_argument("State became invalid during evolution (NaN or Inf). Consider increasing the atol or adaptive_tol parameters.");
}

inline void check_state_diverged(const DenseMatrix& matrix) {
    /*
    Check if the matrix has diverged to a non-finite state, errors if so.

    Args:
        matrix (DenseMatrix): The matrix to check.
    */
    if (!matrix.allFinite()) {
        nan_error();
    }
}

inline void check_valid_divisor(std::complex<double> divisor) {
    /*
    Check if the divisor is valid, errors if not.

    Args:
        divisor (std::complex<double>): The divisor to check.
    */
    if (!(std::isfinite(divisor.real()) && std::isfinite(divisor.imag()) && divisor != std::complex<double>(0.0, 0.0))) {
        nan_error();
    }
}

inline void check_valid_divisor(double divisor) {
    /*
    Check if the divisor is valid, errors if not.

    Args:
        divisor (double): The divisor to check.
    */
    if (!(std::isfinite(divisor) && divisor != 0.0)) {
        nan_error();
    }
}

// Identity matrix constant
const SparseMatrix I = []() {
    Triplets entries;
    entries.emplace_back(Triplet(0, 0, 1.0));
    entries.emplace_back(Triplet(1, 1, 1.0));
    SparseMatrix I_mat(2, 2);
    I_mat.setFromTriplets(entries.begin(), entries.end());
    return I_mat;
}();

// Single-qubit Pauli matrices, as dense 2x2 matrices
const DenseMatrix PAULI_I = (DenseMatrix(2, 2) << Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)).finished();
const DenseMatrix PAULI_X = (DenseMatrix(2, 2) << Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)).finished();
const DenseMatrix PAULI_Y = (DenseMatrix(2, 2) << Complex(0.0, 0.0), Complex(0.0, -1.0), Complex(0.0, 1.0), Complex(0.0, 0.0)).finished();
const DenseMatrix PAULI_Z = (DenseMatrix(2, 2) << Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(-1.0, 0.0)).finished();

// Two-qubit SWAP gate matrix, which exchanges the two qubits
const DenseMatrix SWAP = (DenseMatrix(4, 4) << Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)).finished();

inline const DenseMatrix& pauli_matrix(bool x, bool z) {
    /*
    Select the single-qubit Pauli matrix corresponding to an (x, z) mask pair,
    following the PauliString convention where both bits set means Y (with no extra phase).

    Args:
        x (bool): Whether to include the X component.
        z (bool): Whether to include the Z component.

    Returns:
        DenseMatrix&: The 2x2 Pauli matrix.
    */
    if (x && z) {
        return PAULI_Y;
    }
    if (x) {
        return PAULI_X;
    }
    if (z) {
        return PAULI_Z;
    }
    return PAULI_I;
}

// GCOV_EXCL_BR_STOP
