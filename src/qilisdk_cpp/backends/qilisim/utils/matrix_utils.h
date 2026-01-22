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

#include <complex>
#include <vector>
#include "../libs/eigen.h"
#include "../config/qilisim_config.h"

SparseMatrix exp_mat_action(const SparseMatrix& H, std::complex<double> dt, const SparseMatrix& e1);
SparseMatrix exp_mat(const SparseMatrix& H, std::complex<double> dt);
std::complex<double> dot(const SparseMatrix& v1, const SparseMatrix& v2);
std::complex<double> dot(const DenseMatrix& v1, const DenseMatrix& v2);
std::complex<double> trace(const SparseMatrix& matrix);
SparseMatrix vectorize(const SparseMatrix& matrix, double atol);
SparseMatrix devectorize(const SparseMatrix& vec_matrix, double atol);
