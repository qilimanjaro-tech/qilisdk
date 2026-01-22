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
#include "../libs/pybind.h"

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
    if (num_integrate_substeps <= 0) {
        throw py::value_error("Number of integration substeps must be positive.");
    }
    if (method != "arnoldi" && method != "integrate" && method != "direct") {
        throw py::value_error("Evolution method must be one of 'arnoldi', 'integrate', or 'direct'.");
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
}
