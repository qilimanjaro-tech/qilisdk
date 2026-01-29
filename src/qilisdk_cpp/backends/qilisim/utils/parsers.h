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

#include "../digital/gate.h"
#include "../qilisim.h"

std::vector<SparseMatrix> parse_hamiltonians(const py::object& Hs, double atol);
std::vector<SparseMatrix> parse_jump_operators(const py::object& jumps, double atol);
std::vector<SparseMatrix> parse_observables(const py::object& observables, long nqubits, double atol);
std::vector<std::vector<double>> parse_parameters(const py::object& coeffs);
std::vector<double> parse_time_steps(const py::object& steps);
SparseMatrix parse_initial_state(const py::object& initial_state, double atol);
std::vector<Gate> parse_gates(const py::object& circuit, double atol);
std::vector<bool> parse_measurements(const py::object& circuit);
