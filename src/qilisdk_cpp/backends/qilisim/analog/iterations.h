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
#include "../libs/eigen.h"

void arnoldi(const SparseMatrix& L, const SparseMatrix& v0, int m, std::vector<SparseMatrix>& V, SparseMatrix& H);
void arnoldi_mat(const SparseMatrix& Hsys, const SparseMatrix& rho0, int m, std::vector<SparseMatrix>& V, SparseMatrix& Hk);

SparseMatrix iter_arnoldi(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, int arnoldi_dim, int num_substeps, bool is_unitary_on_statevector, double atol);
SparseMatrix iter_integrate(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, int num_substeps, bool is_unitary_on_statevector);
SparseMatrix iter_direct(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector, double atol);