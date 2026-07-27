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
#pragma once

#include <string>
#include <utility>
#include <vector>
#include "../libs/pybind.h"

// GCOV_EXCL_BR_START

// The C++ form of a QUBO model
struct ParsedCostCpp {
    int num_variables;
    std::vector<std::pair<std::vector<int>, double>> monomials;
    double offset;
    std::vector<std::string> labels;
};

ParsedCostCpp parse_qubo(const py::object& qubo);
py::object solve_with_simulated_annealing(const py::object& qubo, int num_reads, int num_sweeps, double beta_min, double beta_max, int seed, int num_threads);

// GCOV_EXCL_BR_STOP
