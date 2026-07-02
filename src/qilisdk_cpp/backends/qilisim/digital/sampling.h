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

#include <map>
#include <string>
#include <vector>
#include "../../../libs/eigen.h"
#include "../../../libs/pybind.h"
#include "../config/qilisim_config.h"
#include "../digital/gate.h"
#include "../noise/noise_model.h"
#include "../representations/matrix_free_operator.h"
#include "../representations/stabilizer_state.h"

// Helper functions for sampling.cpp
void sampling(const std::vector<Gate>& gates, int n_qubits, const SparseMatrix& initial_state, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout);
void sampling_matrix_free(const std::vector<Gate>& gates, int n_qubits, const SparseMatrix& initial_state, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout);
void sampling_stabilizer(const std::vector<Gate>& gates, int n_qubits, const StabilizerStateSum& initial_state, NoiseModelCpp& noise_model_cpp, StabilizerStateSum& state, const QiliSimConfig& config, const py::object& readout);

// GCOV_EXCL_BR_STOP