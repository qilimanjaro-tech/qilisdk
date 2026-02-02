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

enum class NoiseModelType {
    STATIC_KRAUS,
    STATIC_JUMP,
    TIME_DEPENDENT_KRAUS,
    TIME_DEPENDENT_JUMP,
    PERTURBATION,
};

class NoisePassCpp {
    private:
        NoiseModelType type;
        double probability;
        std::vector<SparseMatrix> operator_matrices;
    public:
        NoisePassCpp(NoiseModelType type_, double probability_, const std::vector<SparseMatrix>& operator_matrices_)
            : type(type_), probability(probability_), operator_matrices(operator_matrices_) {}
        double get_probability() const;
        const std::vector<SparseMatrix>& get_operator_matrices() const;
};

class NoiseModelCpp {
    private:
        std::vector<NoisePassCpp> noise_passes_global;
        std::map<int, std::vector<NoisePassCpp>> noise_passes_per_qubit;
        std::map<std::string, std::vector<NoisePassCpp>> noise_passes_per_gate;
        std::map<std::pair<std::string, int>, std::vector<NoisePassCpp>> noise_passes_per_gate_qubit;

        std::vector<SparseMatrix> cached_jump_operators;
        std::vector<SparseMatrix> cached_kraus_operators_global;
        std::map<int, std::vector<SparseMatrix>> cached_kraus_operators_per_qubit;
        std::map<std::string, std::vector<SparseMatrix>> cached_kraus_operators_per_gate;
        std::map<std::pair<std::string, int>, std::vector<SparseMatrix>> cached_kraus_operators_per_gate_qubit;

    public:

        void add_global_noise_pass(const NoisePassCpp& noise_pass);
        void add_qubit_noise_pass(int qubit, const NoisePassCpp& noise_pass);
        void add_gate_noise_pass(const std::string& gate_name, const NoisePassCpp& noise_pass);
        void add_gate_qubit_noise_pass(const std::string& gate_name, int qubit, const NoisePassCpp& noise_pass);

        const std::vector<NoisePassCpp>& get_global_noise_passes() const;
        const std::vector<NoisePassCpp>& get_qubit_noise_passes(int qubit) const;
        const std::vector<NoisePassCpp>& get_gate_noise_passes(const std::string& gate_name) const;
        const std::vector<NoisePassCpp>& get_gate_qubit_noise_passes(const std::string& gate_name, int qubit) const;

        void cache_all_noise_passes(int nqubits);
        std::vector<SparseMatrix>& get_jump_operators();
        std::vector<SparseMatrix>& get_kraus_operators_global();
        std::vector<SparseMatrix>& get_kraus_operators_per_qubit(int qubit);
        std::vector<SparseMatrix>& get_kraus_operators_per_gate(const std::string& gate_name);
        std::vector<SparseMatrix>& get_kraus_operators_per_gate_qubit(const std::string& gate_name, int qubit);
};