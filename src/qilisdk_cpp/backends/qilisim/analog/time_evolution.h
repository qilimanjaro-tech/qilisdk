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
#include "../config/qilisim_config.h"
#include "../noise/noise_model.h"
#include "../representations/matrix_free_hamiltonian.h"
#include "../representations/exponential_ansatz.h"

// GCOV_EXCL_BR_START

void time_evolution(SparseMatrix rho_0, const std::vector<SparseMatrix>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, NoiseModelCpp& noise_model_cpp, QiliSimConfig& config, DenseMatrix& rho_t, std::vector<DenseMatrix>& intermediate_rhos);

void time_evolution_matrix_free(SparseMatrix rho_0, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, NoiseModelCpp& noise_model_cpp, QiliSimConfig& config, DenseMatrix& rho_t, std::vector<DenseMatrix>& intermediate_rhos);

void time_evolution_truncated_polynomial_expansion(MatrixFreeHamiltonian& rho_t_as_h, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, QiliSimConfig& config);

void time_evolution_variational_exponential(ExponentialAnsatz& rho_t_as_h, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, QiliSimConfig& config);

// GCOV_EXCL_BR_STOP
