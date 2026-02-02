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

double NoisePassCpp::get_probability() const {
    /*
    Get the probability of this noise pass.

    Returns:
        double: The probability of the noise pass.
    */
    return probability;
}
const std::vector<SparseMatrix>& NoisePassCpp::get_operator_matrices() const {
    /*
    Get the list of operator matrices for this noise pass.

    Returns:
        const std::vector<SparseMatrix>&: The list of operator matrices.
    */
    return operator_matrices;
}

void NoiseModelCpp::add_global_noise_pass(const NoisePassCpp& noise_pass) {
    /*
    Add a global noise pass to the noise model.

    Args:
        noise_pass (NoisePassCpp): The noise pass to add.
    */
    noise_passes_global.push_back(noise_pass);
}

void NoiseModelCpp::add_qubit_noise_pass(int qubit, const NoisePassCpp& noise_pass) {
    /*
    Add a noise pass for a specific qubit.

    Args:
        qubit (int): The qubit index.
        noise_pass (NoisePassCpp): The noise pass to add.
    */
    noise_passes_per_qubit[qubit].push_back(noise_pass);
}

void NoiseModelCpp::add_gate_noise_pass(const std::string& gate_name, const NoisePassCpp& noise_pass) {
    /*
    Add a noise pass for a specific gate.

    Args:
        gate_name (std::string): The name of the gate.
        noise_pass (NoisePassCpp): The noise pass to add.
    */
    noise_passes_per_gate[gate_name].push_back(noise_pass);
}

void NoiseModelCpp::add_gate_qubit_noise_pass(const std::string& gate_name, int qubit, const NoisePassCpp& noise_pass) {
    /*
    Add a noise pass for a specific gate on a specific qubit.

    Args:
        gate_name (std::string): The name of the gate.
        qubit (int): The qubit index.
        noise_pass (NoisePassCpp): The noise pass to add.
    */
    noise_passes_per_gate_qubit[std::make_pair(gate_name, qubit)].push_back(noise_pass);
}

const std::vector<NoisePassCpp>& NoiseModelCpp::get_global_noise_passes() const {
    /*
    Get the list of global noise passes.

    Returns:
        const std::vector<NoisePassCpp>&: The list of global noise passes.
    */
    return noise_passes_global;
}

const std::vector<NoisePassCpp>& NoiseModelCpp::get_qubit_noise_passes(int qubit) const {
    /*
    Get the list of noise passes for a specific qubit.

    Args:
        qubit (int): The qubit index.

    Returns:
        const std::vector<NoisePassCpp>&: The list of noise passes for the qubit.
    */
    static const std::vector<NoisePassCpp> empty;
    auto it = noise_passes_per_qubit.find(qubit);
    return it != noise_passes_per_qubit.end() ? it->second : empty;
}

const std::vector<NoisePassCpp>& NoiseModelCpp::get_gate_noise_passes(const std::string& gate_name) const {
    /*
    Get the list of noise passes for a specific gate.

    Args:
        gate_name (std::string): The name of the gate.

    Returns:
        const std::vector<NoisePassCpp>&: The list of noise passes for the gate.
    */
    static const std::vector<NoisePassCpp> empty;
    auto it = noise_passes_per_gate.find(gate_name);
    return it != noise_passes_per_gate.end() ? it->second : empty;
}

const std::vector<NoisePassCpp>& NoiseModelCpp::get_gate_qubit_noise_passes(const std::string& gate_name, int qubit) const {
    /*
    Get the list of noise passes for a specific gate on a specific qubit.

    Args:
        gate_name (std::string): The name of the gate.
        qubit (int): The qubit index.

    Returns:
        const std::vector<NoisePassCpp>&: The list of noise passes for the gate on the qubit.
    */
    static const std::vector<NoisePassCpp> empty;
    auto it = noise_passes_per_gate_qubit.find(std::make_pair(gate_name, qubit));
    return it != noise_passes_per_gate_qubit.end() ? it->second : empty;
}

std::vector<SparseMatrix>& NoiseModelCpp::get_jump_operators() {
    /*
    Get the cached jump operators.

    Returns:
        std::vector<SparseMatrix>&: The cached jump operators.
    */
    return cached_jump_operators;
}

std::vector<SparseMatrix>& NoiseModelCpp::get_kraus_operators_global() {
    /*
    Get the cached global Kraus operators.

    Returns:
        std::vector<SparseMatrix>&: The cached global Kraus operators.
    */
    return cached_kraus_operators_global;
}

std::vector<SparseMatrix>& NoiseModelCpp::get_kraus_operators_per_qubit(int qubit) {
    /*
    Get the cached Kraus operators for a specific qubit.

    Args:
        qubit (int): The qubit index.

    Returns:
        std::vector<SparseMatrix>&: The cached Kraus operators for the qubit.
    */
    return cached_kraus_operators_per_qubit[qubit];
}

std::vector<SparseMatrix>& NoiseModelCpp::get_kraus_operators_per_gate(const std::string& gate_name) {
    /*
    Get the cached Kraus operators for a specific gate.

    Args:
        gate_name (std::string): The name of the gate.

    Returns:
        std::vector<SparseMatrix>&: The cached Kraus operators for the gate.
    */
    return cached_kraus_operators_per_gate[gate_name];
}

std::vector<SparseMatrix>& NoiseModelCpp::get_kraus_operators_per_gate_qubit(const std::string& gate_name, int qubit) {
    /*
    Get the cached Kraus operators for a specific gate on a specific qubit.

    Args:
        gate_name (std::string): The name of the gate.
        qubit (int): The qubit index.

    Returns:
        std::vector<SparseMatrix>&: The cached Kraus operators for the gate on the qubit.
    */
    return cached_kraus_operators_per_gate_qubit[std::make_pair(gate_name, qubit)];
}

void NoiseModelCpp::cache_all_noise_passes(int nqubits) {
    /*
    Cache all noise passes by precomputing their operator matrices.
    This function processes all noise passes (global, per qubit, per gate, and per gate-qubit)
    and stores their operator matrices in the corresponding cached vectors/maps for efficient access.

    Args:
        nqubits (int): The total number of qubits in the system.
    */

    // Cache global noise passes
    cached_jump_operators.clear();
    cached_kraus_operators_global.clear();
    for (const auto& pass : noise_passes_global) {
        const auto& ops = pass.get_operator_matrices();
        // TODO
    }

    // Cache per-qubit noise passes
    cached_kraus_operators_per_qubit.clear();
    for (const auto& [qubit, passes] : noise_passes_per_qubit) {
        std::vector<SparseMatrix> kraus_ops;
        for (const auto& pass : passes) {
            const auto& ops = pass.get_operator_matrices();
            // TODO
        }
        cached_kraus_operators_per_qubit[qubit] = kraus_ops;
    }

    // Cache per-gate noise passes
    cached_kraus_operators_per_gate.clear();
    for (const auto& [gate_name, passes] : noise_passes_per_gate) {
        std::vector<SparseMatrix> kraus_ops;
        for (const auto& pass : passes) {
            const auto& ops = pass.get_operator_matrices();
            // TODO
        }
        cached_kraus_operators_per_gate[gate_name] = kraus_ops;
    }

    // Cache per-gate-qubit noise passes
    cached_kraus_operators_per_gate_qubit.clear();
    for (const auto& [gate_qubit, passes] : noise_passes_per_gate_qubit) {
        std::vector<SparseMatrix> kraus_ops;
        for (const auto& pass : passes) {
            const auto& ops = pass.get_operator_matrices();
            // TODO
        }
        cached_kraus_operators_per_gate_qubit[gate_qubit] = kraus_ops;
    }

}