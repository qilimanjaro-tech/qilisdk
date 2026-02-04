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

#include "../config/qilisim_config.h"
#include "../libs/eigen.h"
#include "../noise/noise_model.h"

void time_evolution(SparseMatrix rho_0, 
                    const std::vector<SparseMatrix>& hamiltonians, 
                    const std::vector<std::vector<double>>& parameters_list, 
                    const std::vector<double>& step_list, 
                    NoiseModelCpp& noise_model_cpp, 
                    const std::vector<SparseMatrix>& observable_matrices, 
                    QiliSimConfig& config, 
                    SparseMatrix& rho_t, 
                    std::vector<SparseMatrix>& intermediate_rhos, 
                    std::vector<double>& expectation_values, 
                    std::vector<std::vector<double>>& intermediate_expectation_values);