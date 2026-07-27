// Copyright 2026 Qilimanjaro Quantum Tech
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

#include "../libs/logging.h"
#include "../libs/pybind.h"
#include "parsers.h"

// GCOV_EXCL_BR_START
// GCOVR_EXCL_START

PYBIND11_MODULE(solvers_module, m) {
    initialize_external_pybind_types();
    m.def("_refresh_log_level", &qilisdk::refresh_log_level);
    m.add_object("_qilisdk_cleanup", py::capsule(&finalize_all_pybind_types));
    m.def("solve_with_simulated_annealing", &solve_with_simulated_annealing, "qubo"_a, "num_reads"_a, "num_sweeps"_a, "beta_min"_a = 0.0, "beta_max"_a = 0.0, "seed"_a = 0, "num_threads"_a = 0);
}

// GCOVR_EXCL_STOP
// GCOV_EXCL_BR_STOP
