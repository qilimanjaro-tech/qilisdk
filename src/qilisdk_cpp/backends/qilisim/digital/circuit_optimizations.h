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

#include "gate.h"

// GCOV_EXCL_BR_START

std::vector<Gate> combine_single_qubit_gates(const std::vector<Gate>& gates);

// Fuse runs of adjacent gates acting on a small set of qubits into a single
// dense multi-qubit gate, reducing the number of passes over the statevector.
// Blocks are limited to at most `max_fused_qubits` qubits. Measurements act as
// barriers, and gates that already span more than `max_fused_qubits` qubits are
// passed through unchanged.
std::vector<Gate> fuse_gates(const std::vector<Gate>& gates, int max_fused_qubits);

// GCOV_EXCL_BR_STOP