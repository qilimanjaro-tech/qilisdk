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

// A NaN complex scalar, used to mark a diverged / numerically-invalid state so it can never be
// mistaken for a valid one (see set_nan / mark_nan_if_diverged below).
inline Complex nan_complex() {
    return Complex(std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN());
}

// Overwrite a matrix with NaN to explicitly mark it as a diverged / invalid state. Callers use
// this at the point they detect an unstable normalization (e.g. a trace whose magnitude has
// overflowed to inf, which would otherwise silently collapse the matrix to all zeros).
inline void set_nan(DenseMatrix& matrix) {
    matrix.setConstant(nan_complex());
}

// If a matrix has already become non-finite (NaN/Inf anywhere), replace it with an explicit NaN
// state and return true; otherwise leave it untouched and return false. Lets a caller detect
// divergence once, after a step, and react (warn, stop stepping) uniformly.
inline bool mark_nan_if_diverged(DenseMatrix& matrix) {
    if (matrix.allFinite()) {
        return false;
    }
    set_nan(matrix);
    return true;
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

// GCOV_EXCL_BR_STOP
