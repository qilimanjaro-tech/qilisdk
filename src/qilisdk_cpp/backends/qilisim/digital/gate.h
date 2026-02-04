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

#include <complex>
#include <string>
#include <vector>

#include "../libs/eigen.h"

class Gate {
   private:
    std::string gate_type;
    SparseMatrix base_matrix;
    std::vector<int> control_qubits;
    std::vector<int> target_qubits;
    std::vector<std::pair<std::string, double>> parameters;

    // gate.cpp
    int permute_bits(int index, const std::vector<int>& perm) const;
    Triplets tensor_product(Triplets& A, Triplets& B, int B_width) const;
    SparseMatrix base_to_full(const SparseMatrix& base_gate, int num_qubits, const std::vector<int>& control_qubits, const std::vector<int>& target_qubits) const;

   public:
    Gate(const std::string& gate_type_, const SparseMatrix& base_matrix_, const std::vector<int>& controls_, const std::vector<int>& targets_, const std::vector<std::pair<std::string, double>>& parameters_) : gate_type(gate_type_), base_matrix(base_matrix_), control_qubits(controls_), target_qubits(targets_), parameters(parameters_) {}

    // gate.cpp
    int get_nqubits() const;
    std::vector<int> get_target_qubits() const;
    std::vector<int> get_control_qubits() const;
    std::vector<std::pair<std::string, double>> get_parameters() const;
    std::string get_name() const;
    std::string get_id() const;
    SparseMatrix get_base_matrix() const;
    SparseMatrix get_full_matrix(int num_qubits) const;
};

std::ostream& operator<<(std::ostream& os, const std::vector<Gate>& gates);
