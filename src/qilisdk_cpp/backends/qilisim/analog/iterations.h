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

#include <vector>
#include "../../../libs/eigen.h"
#include "../representations/exponential_ansatz.h"
#include "../representations/matrix_free_hamiltonian.h"

// GCOV_EXCL_BR_START

void arnoldi(const SparseMatrix& L, const DenseMatrix& v0, int m, std::vector<DenseMatrix>& V, SparseMatrix& H, double atol);
void arnoldi_mat(const SparseMatrix& Hsys, const DenseMatrix& rho0, int m, std::vector<DenseMatrix>& V, SparseMatrix& Hk, double atol);

DenseMatrix iter_direct(const DenseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector);
DenseMatrix iter_arnoldi(const DenseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, int arnoldi_dim, int num_substeps, bool is_unitary_on_statevector, double atol, bool normalize = true);
DenseMatrix iter_rk4_matrix(const DenseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector, bool normalize = true);
void iter_rk4(DenseMatrix& rho_t, double t, double dt, const std::vector<double>& step_list, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector, bool normalize = true);
double iter_rk45(DenseMatrix& rho_t, double t, double& dt, const std::vector<double>& step_list, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector, double tol, DenseMatrix& k1, bool normalize = true);
void iter_rk4(ExponentialAnsatz& rho_t, double t, double dt, const std::vector<double>& step_list, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list);

// GCOV_EXCL_BR_STOP