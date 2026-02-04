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

#include "../libs/eigen.h"
#include <vector>
#include <map>

class NoiseModelCpp {
    private:
        bool has_something = false;

        std::vector<SparseMatrix> cached_jump_operators;
        std::vector<std::vector<SparseMatrix>> cached_kraus_operators_global;
        std::map<int, std::vector<std::vector<SparseMatrix>>> cached_kraus_operators_per_qubit;
        std::map<std::string, std::vector<std::vector<SparseMatrix>>> cached_kraus_operators_per_gate;
        std::map<std::pair<std::string, int>, std::vector<std::vector<SparseMatrix>>> cached_kraus_operators_per_gate_qubit;

        std::pair<double, double> readout_error_global = {0.0, 0.0};
        std::map<int, std::pair<double, double>> readout_error_per_qubit;

    public:

        bool is_empty() const;

        void add_jump_operator(const SparseMatrix& L);
        void add_kraus_operators_global(const std::vector<SparseMatrix>& Ks);
        void add_kraus_operators_per_qubit(int qubit, const std::vector<SparseMatrix>& Ks);
        void add_kraus_operators_per_gate(const std::string& gate_name, const std::vector<SparseMatrix>& Ks);
        void add_kraus_operators_per_gate_qubit(const std::string& gate_name, int qubit, const std::vector<SparseMatrix>& Ks);
        void add_readout_error_global(double p01, double p10);
        void add_readout_error_per_qubit(int qubit, double p01, double p10);

        std::vector<SparseMatrix>& get_jump_operators();
        std::vector<std::vector<SparseMatrix>>& get_kraus_operators_global();
        std::map<int, std::vector<std::vector<SparseMatrix>>>& get_kraus_operators_per_qubit();
        std::map<std::string, std::vector<std::vector<SparseMatrix>>>& get_kraus_operators_per_gate();
        std::map<std::pair<std::string, int>, std::vector<std::vector<SparseMatrix>>>& get_kraus_operators_per_gate_qubit();
        std::pair<double, double> get_relevant_readout_error(int qubit);
        std::vector<std::vector<SparseMatrix>> get_relevant_kraus_operators(const std::string& gate_name, const std::vector<int>& target_qubits, int nqubits);
};