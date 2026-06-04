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

#include "qilisim_config.h"
#include "../../../libs/pybind.h"

// GCOV_EXCL_BR_START

void QiliSimConfig::validate() const {
    /*
    Validate the QiliSim configuration.

    Raises:
        py::value_error: If any configuration parameter is invalid.
    */

    if (arnoldi_dim <= 0) {
        throw py::value_error("Arnoldi dimension must be positive.");
    }
    if (num_arnoldi_substeps <= 0) {
        throw py::value_error("Number of Arnoldi substeps must be positive.");
    }
    const std::string valid_evolution_methods = "'direct', 'arnoldi', 'variational_exponential', 'integrate_rk4', 'integrate_rk45_matrix_free', or 'integrate_rk4_matrix_free'";
    if (valid_evolution_methods.find(time_evolution_method) == std::string::npos) {
        throw py::value_error("Time evolution method must be one of " + valid_evolution_methods);
    }
    const std::string valid_sampling_methods = "'statevector', 'statevector_matrix_free', 'stabilizer'";
    if (valid_sampling_methods.find(sampling_method) == std::string::npos) {
        throw py::value_error("Sampling method must be one of " + valid_sampling_methods);
    }
    if (monte_carlo && num_monte_carlo_trajectories <= 0) {
        throw py::value_error("Number of Monte Carlo trajectories must be positive.");
    }
    if (num_threads <= 0) {
        throw py::value_error("Number of threads must be positive.");
    }
    if (this->atol <= 0) {
        throw py::value_error("Absolute tolerance must be positive.");
    }
    if (max_cache_size <= 0) {
        throw py::value_error("Max cache size must be positive.");
    }
    if (adaptive_tol <= 0) {
        throw py::value_error("Adaptive tolerance must be positive.");
    }
    if (order <= 0) {
        throw py::value_error("Order must be positive.");
    }
    if (shots <= 0) {
        throw py::value_error("Shots must be positive.");
    }
    if (warmups < 0) {
        throw py::value_error("Warmups cannot be negative.");
    }
}

// GCOV_EXCL_BR_STOP
