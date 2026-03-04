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
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>
#include <unsupported/Eigen/MatrixFunctions>

// Eigen specfic type defs
typedef Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> SparseMatrix;
typedef Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor> SparseMatrixCol;
typedef Eigen::MatrixXcd DenseMatrix;
typedef Eigen::Triplet<std::complex<double>> Triplet;
typedef std::vector<Eigen::Triplet<std::complex<double>>> Triplets;

// Identity matrix constant
const SparseMatrix I = []() {
    Triplets entries;
    entries.emplace_back(Triplet(0, 0, 1.0));
    entries.emplace_back(Triplet(1, 1, 1.0));
    SparseMatrix I_mat(2, 2);
    I_mat.setFromTriplets(entries.begin(), entries.end());
    return I_mat;
}();
