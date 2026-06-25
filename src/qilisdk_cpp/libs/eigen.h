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

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

// GCOV_EXCL_BR_START

// Compile with -Ccmake.define.QILISIM_SINGLE_PRECISION=ON to enable single precision

// Our real and complex types
#ifdef QILISIM_SINGLE_PRECISION
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
