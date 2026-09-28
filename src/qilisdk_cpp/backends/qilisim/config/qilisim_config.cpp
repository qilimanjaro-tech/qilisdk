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
#include <set>
#include "../../../libs/pybind.h"

// GCOV_EXCL_BR_START

namespace {

void validate_choice(const std::string& value, const std::set<std::string>& valid, const std::string& name) {
    /*
    Check that a value is exactly one of a set of valid names.

    Args:
        value (std::string): The value to check.
        valid (std::set<std::string>): The accepted names.
        name (std::string): The name of the setting, used in the error message.

    Raises:
        py::value_error: If the value is not one of the valid names.
    */
    if (valid.count(value) > 0) {
        return;
    }
    std::string options;
    for (const auto& option : valid) {
        options += (options.empty() ? "'" : ", '") + option + "'";
    }
    throw py::value_error(name + " must be one of " + options + ", got '" + value + "'");
}

}  // namespace

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
    static const std::set<std::string> valid_evolution_methods = {"direct", "arnoldi", "arnoldi_matrix_free", "variational_exponential", "integrate_rk4", "integrate_rk45_matrix_free", "integrate_rk4_matrix_free"};
    validate_choice(time_evolution_method, valid_evolution_methods, "Time evolution method");
    static const std::set<std::string> valid_digital_methods = {"statevector", "statevector_matrix_free", "stabilizer"};
    validate_choice(digital_method, valid_digital_methods, "Digital method");
    if (monte_carlo && num_monte_carlo_trajectories <= 0) {
        throw py::value_error("Number of Monte Carlo trajectories must be positive.");
    }
    if (num_threads <= 0) {
        throw py::value_error("Number of threads must be positive.");
    }
    if (this->atol <= 0) {
        throw py::value_error("Absolute tolerance must be positive.");
    }
    if (max_cache_size < 0) {
        throw py::value_error("Max cache size must be non-negative.");
    }
    if (max_fused_qubits < 0) {
        throw py::value_error("Max fused qubits must be non-negative (0 selects an automatic depth based on the qubit count).");
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
    if (order > 4) {
        throw py::value_error("Order greater than 4 not supported yet.");
    }
}

int QiliSimConfig::next_seed() const {
    /*
    Advance the seed and return the next one, based on the root seed
    and how many times we've advanced the seed.

    Returns:
        int: A non-negative seed, distinct per call.
    */
    uint64_t z = static_cast<uint64_t>(static_cast<uint32_t>(seed)) + 0x9e3779b97f4a7c15ULL * (++seed_stream);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return static_cast<int>(z & 0x7fffffffULL);
}

// GCOV_EXCL_BR_STOP
