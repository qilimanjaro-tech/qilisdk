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

#include "noise_model.h"
#include "../utils/matrix_utils.h"

bool NoiseModelCpp::is_empty() const {
    /*
    Check if the noise model has any cached operators.

    Returns:
        bool: True if no operators are cached, False otherwise.
    */
    return !has_something;
}

void NoiseModelCpp::add_jump_operator(const SparseMatrix& L) {
    /*
    Add a jump operator to the cached list.

    Args:
        L (SparseMatrix): The jump operator matrix to add.
    */
    has_something = true;
    cached_jump_operators.push_back(L);
}

void NoiseModelCpp::add_kraus_operators_global(const std::vector<SparseMatrix>& Ks) {
    /*
    Add a global Kraus operator to the cached list.

    Args:
        Ks (std::vector<SparseMatrix>): The global Kraus operator matrices to add.
    */
    has_something = true;
    cached_kraus_operators_global.push_back(Ks);
}

void NoiseModelCpp::add_kraus_operators_per_qubit(int qubit, const std::vector<SparseMatrix>& Ks) {
    /*
    Add a Kraus operator for a specific qubit to the cached map.

    Args:
        qubit (int): The qubit index.
        Ks (std::vector<SparseMatrix>): The Kraus operator matrices to add.
    */
    has_something = true;
    cached_kraus_operators_per_qubit[qubit].push_back(Ks);
}


void NoiseModelCpp::add_kraus_operators_per_gate(const std::string& gate_name, const std::vector<SparseMatrix>& Ks) {
    /*
    Add a Kraus operator for a specific gate to the cached map.

    Args:
        gate_name (std::string): The name of the gate.
        Ks (std::vector<SparseMatrix>): The Kraus operator matrices to add.
    */
    has_something = true;
    cached_kraus_operators_per_gate[gate_name].push_back(Ks);
}


void NoiseModelCpp::add_kraus_operators_per_gate_qubit(const std::string& gate_name, int qubit, const std::vector<SparseMatrix>& Ks) {
    /*
    Add a Kraus operator for a specific gate on a specific qubit to the cached map.

    Args:
        gate_name (std::string): The name of the gate.
        qubit (int): The qubit index.
        Ks (std::vector<SparseMatrix>): The Kraus operator matrices to add.
    */
    has_something = true;
    cached_kraus_operators_per_gate_qubit[std::make_pair(gate_name, qubit)].push_back(Ks);
}


std::vector<SparseMatrix>& NoiseModelCpp::get_jump_operators() {
    /*
    Get the cached jump operators.

    Returns:
        std::vector<SparseMatrix>&: The cached jump operators.
    */
    return cached_jump_operators;
}

std::vector<std::vector<SparseMatrix>>& NoiseModelCpp::get_kraus_operators_global() {
    /*
    Get the cached global Kraus operators.

    Returns:
        std::vector<std::vector<SparseMatrix>>&: The cached global Kraus operators.
    */
    return cached_kraus_operators_global;
}

std::map<int, std::vector<std::vector<SparseMatrix>>>& NoiseModelCpp::get_kraus_operators_per_qubit() {
    /*
    Get the cached Kraus operators for a specific qubit.

    Returns:
        std::map<int, std::vector<std::vector<SparseMatrix>>>&: The cached Kraus operators per qubit.
    */
    return cached_kraus_operators_per_qubit;
}

std::map<std::string, std::vector<std::vector<SparseMatrix>>>& NoiseModelCpp::get_kraus_operators_per_gate() {
    /*
    Get the cached Kraus operators for a specific gate.

    Returns:
        std::map<std::string, std::vector<std::vector<SparseMatrix>>>&: The cached Kraus operators per gate.
    */
    return cached_kraus_operators_per_gate;
}

std::map<std::pair<std::string, int>, std::vector<std::vector<SparseMatrix>>>& NoiseModelCpp::get_kraus_operators_per_gate_qubit() {
    /*
    Get the cached Kraus operators for a specific gate on a specific qubit.

    Returns:
        std::map<std::pair<std::string, int>, std::vector<std::vector<SparseMatrix>>>&: The cached Kraus operators per gate and qubit.
    */
    return cached_kraus_operators_per_gate_qubit;
}

std::vector<std::vector<SparseMatrix>> NoiseModelCpp::get_relevant_kraus_operators(const std::string& gate_name, const std::vector<int>& target_qubits, int nqubits) {
    /*
    Get all relevant Kraus operators for a given gate and its target qubits.

    Args:
        gate_name (std::string): The name of the gate.
        target_qubits (std::vector<int>): The list of target qubit indices.

    Returns:
        std::vector<std::vector<SparseMatrix>>: The list of relevant Kraus operators.
    */
    std::vector<std::vector<SparseMatrix>> relevant_operators;
    relevant_operators.clear();

    // Add global Kraus operators
    if (!cached_kraus_operators_global.empty()) {
        for (const auto& operator_set : cached_kraus_operators_global) {
            relevant_operators.push_back(operator_set);
        }
    }

    // Add per-gate Kraus operators
    if (cached_kraus_operators_per_gate.find(gate_name) != cached_kraus_operators_per_gate.end()) {
        for (const auto& operator_set : cached_kraus_operators_per_gate[gate_name]) {
            relevant_operators.push_back(operator_set);
        }
    }

    // Add per-qubit Kraus operators
    for (int qubit : target_qubits) {
        if (cached_kraus_operators_per_qubit.find(qubit) != cached_kraus_operators_per_qubit.end()) {
            for (const auto& operator_set : cached_kraus_operators_per_qubit[qubit]) {
                relevant_operators.push_back(operator_set);
            }
        }
    }

    // Add per-gate-per-qubit Kraus operators
    for (int qubit : target_qubits) {
        auto key = std::make_pair(gate_name, qubit);
        if (cached_kraus_operators_per_gate_qubit.find(key) != cached_kraus_operators_per_gate_qubit.end()) {
            for (const auto& operator_set : cached_kraus_operators_per_gate_qubit[key]) {
                relevant_operators.push_back(operator_set);
            }
        }
    }

    // Ensure that they are all the correct size
    for (auto& set : relevant_operators) {
        for (auto& K : set) {
            K = expand_operator(target_qubits, nqubits, K);
        }
    }

    return relevant_operators;
}

void NoiseModelCpp::add_readout_error_global(double p01, double p10) {
    /*
    Add global readout error probabilities.

    Args:
        p01 (double): Probability of measuring 0 as 1.
        p10 (double): Probability of measuring 1 as 0.
    */
    has_something = true;
    readout_error_global.first = p01;
    readout_error_global.second = p10;
}

void NoiseModelCpp::add_readout_error_per_qubit(int qubit, double p01, double p10) {
    /*
    Add readout error probabilities for a specific qubit.

    Args:
        qubit (int): The qubit index.
        p01 (double): Probability of measuring 0 as 1.
        p10 (double): Probability of measuring 1 as 0.
    */
    has_something = true;
    readout_error_per_qubit[qubit] = std::make_pair(p01, p10);
}

std::pair<double, double> NoiseModelCpp::get_relevant_readout_error(int qubit) {
    /*
    Get the relevant readout error probabilities for a given qubit.

    Args:
        qubit (int): The qubit index.

    Returns:
        std::pair<double, double>: The readout error probabilities (p01, p10).
    */
    double p01 = readout_error_global.first;
    double p10 = readout_error_global.second;
    if (readout_error_per_qubit.find(qubit) != readout_error_per_qubit.end()) {
        p01 = std::max(p01, readout_error_per_qubit[qubit].first);
        p10 = std::max(p10, readout_error_per_qubit[qubit].second);
    }
    return std::make_pair(p01, p10);
}