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

SparseMatrix create_superoperator(const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators);
void lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const SparseMatrix& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector);
