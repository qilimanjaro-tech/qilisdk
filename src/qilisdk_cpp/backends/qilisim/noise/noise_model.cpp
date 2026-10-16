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
#include <cmath>
#include "../../../libs/pybind.h"
#include "../utils/matrix_utils.h"

// GCOV_EXCL_BR_START

namespace {

void append_expanded_sets(std::vector<std::vector<SparseMatrix>>& out, const std::vector<std::vector<SparseMatrix>>& sets, const std::vector<int>& qubits, int nqubits, bool allow_full_register) {
    /*
    Expand each Kraus operator set onto the full register and append the results.

    A single-qubit set is applied independently to every qubit in `qubits`, producing one expanded
    set per qubit. A set that already spans the whole register is appended unchanged, but only if
    `allow_full_register` is true: for noise attached to a specific qubit, a whole-register set is
    ambiguous (it is not clear how many times it should be applied) and is rejected instead.

    Args:
        out (std::vector<std::vector<SparseMatrix>>&): The list the expanded sets are appended to.
        sets (std::vector<std::vector<SparseMatrix>>): The Kraus operator sets to expand.
        qubits (std::vector<int>): The qubits the sets are attached to.
        nqubits (int): The total number of qubits.
        allow_full_register (bool): Whether a set spanning the whole register is accepted.

    Raises:
        py::value_error: If a set acts on more than one qubit but not on the whole register, or if
            it spans the whole register when `allow_full_register` is false.
    */
    for (const auto& set : sets) {
        if (set.empty()) {
            continue;  // GCOV_EXCL_LINE
        }
        const int set_qubits = static_cast<int>(std::log2(set.front().rows()));
        if (set_qubits == nqubits && allow_full_register) {
            out.push_back(set);
            continue;
        }
        if (set_qubits != 1) {
            if (!allow_full_register) {
                throw py::value_error("Kraus operators attached to a specific qubit must act on a single qubit.");
            }
            throw py::value_error("Kraus operators must act either on a single qubit or on the whole register.");
        }
        for (int qubit : qubits) {
            std::vector<SparseMatrix> expanded;
            expanded.reserve(set.size());
            for (const auto& K : set) {
                expanded.push_back(expand_operator(qubit, nqubits, K));
            }
            out.push_back(std::move(expanded));
        }
    }
}

}  // namespace

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
    Add a jump operator with a constant (already folded) rate to the cached list.

    Args:
        L (SparseMatrix): The jump operator matrix to add (rate already folded as sqrt(rate)*L).
    */
    has_something = true;
    cached_jump_operators.push_back(L);
    // Empty series signals "constant rate already folded in; do not re-scale per step".
    cached_jump_rate_series.emplace_back();
}

void NoiseModelCpp::add_jump_operator(const SparseMatrix& base, const std::vector<double>& sqrt_rate_series) {
    /*
    Add a jump operator with a time-dependent rate to the cached list.

    Args:
        base (SparseMatrix): The base jump operator matrix (without any rate folded in).
        sqrt_rate_series (std::vector<double>): The per-step sqrt(rate(t)) multiplier, one entry per
            time step. The base operator must be scaled by this multiplier at the corresponding step.
    */
    has_something = true;
    has_time_dependent_jumps = true;
    cached_jump_operators.push_back(base);
    cached_jump_rate_series.push_back(sqrt_rate_series);
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

const std::vector<SparseMatrix>& NoiseModelCpp::get_jump_operators() const {
    /*
    Get the cached jump operators.

    Returns:
        const std::vector<SparseMatrix>&: The cached jump operators.
    */
    return cached_jump_operators;
}

const std::vector<std::vector<double>>& NoiseModelCpp::get_jump_rate_series() const {
    /*
    Get the per-step sqrt(rate(t)) multiplier series for each jump operator.

    Returns:
        const std::vector<std::vector<double>>&: Series aligned by index with the jump operators;
            an empty inner vector means the operator's constant rate is already folded in.
    */
    return cached_jump_rate_series;
}

bool NoiseModelCpp::has_time_dependent_rates() const {
    /*
    Check whether any jump operator has a time-dependent rate.

    Returns:
        bool: True if at least one jump operator carries a per-step rate series.
    */
    return has_time_dependent_jumps;
}

const std::vector<std::vector<SparseMatrix>>& NoiseModelCpp::get_kraus_operators_global() const {
    /*
    Get the cached global Kraus operators.

    Returns:
        const std::vector<std::vector<SparseMatrix>>&: The cached global Kraus operators.
    */
    return cached_kraus_operators_global;
}

const std::map<int, std::vector<std::vector<SparseMatrix>>>& NoiseModelCpp::get_kraus_operators_per_qubit() const {
    /*
    Get the cached Kraus operators for a specific qubit.

    Returns:
        const std::map<int, std::vector<std::vector<SparseMatrix>>>&: The cached Kraus operators per qubit.
    */
    return cached_kraus_operators_per_qubit;
}

const std::map<std::string, std::vector<std::vector<SparseMatrix>>>& NoiseModelCpp::get_kraus_operators_per_gate() const {
    /*
    Get the cached Kraus operators for a specific gate.

    Returns:
        const std::map<std::string, std::vector<std::vector<SparseMatrix>>>&: The cached Kraus operators per gate.
    */
    return cached_kraus_operators_per_gate;
}

const std::map<std::pair<std::string, int>, std::vector<std::vector<SparseMatrix>>>& NoiseModelCpp::get_kraus_operators_per_gate_qubit() const {
    /*
    Get the cached Kraus operators for a specific gate on a specific qubit.

    Returns:
        const std::map<std::pair<std::string, int>, std::vector<std::vector<SparseMatrix>>>&: The cached Kraus operators per gate and qubit.
    */
    return cached_kraus_operators_per_gate_qubit;
}

std::string NoiseModelCpp::make_gate_key(const std::string& base_name, int num_controls) {
    /*
    Build the per-gate noise key from a base gate name and its number of control qubits.

    Args:
        base_name (std::string): The base (internal) gate name, e.g. "Z".
        num_controls (int): The number of control qubits.

    Returns:
        std::string: The key, e.g. "Z" for a plain gate or "Z_c1" for a controlled gate.
    */
    if (num_controls <= 0) {
        return base_name;
    }
    return base_name + "_c" + std::to_string(num_controls);
}

std::vector<std::vector<SparseMatrix>> NoiseModelCpp::get_relevant_kraus_operators(const std::string& gate_name, int num_controls, const std::vector<int>& gate_qubits, int nqubits) const {
    /*
    Get all relevant Kraus operators for a given gate, expanded onto the full register.

    Global and per-gate noise is applied independently to every qubit the gate acts on, while
    per-qubit and per-gate-per-qubit noise is applied only to the qubit it was attached to.

    Args:
        gate_name (std::string): The base name of the gate.
        num_controls (int): The number of control qubits on the gate, used to distinguish
            a controlled gate (e.g. CZ) from its base gate (e.g. Z).
        gate_qubits (std::vector<int>): The qubits the gate acts on, controls included.
        nqubits (int): The total number of qubits.

    Returns:
        std::vector<std::vector<SparseMatrix>>: The list of relevant Kraus operators.
    */

    // The list of operators to fill
    std::vector<std::vector<SparseMatrix>> relevant_operators;

    // Get the key for per-gate noise lookup
    const std::string gate_key = make_gate_key(gate_name, num_controls);

    // Add global Kraus operators, applied to each qubit the gate acts on
    append_expanded_sets(relevant_operators, cached_kraus_operators_global, gate_qubits, nqubits, true);

    // Add per-gate Kraus operators, applied to each qubit the gate acts on
    auto gate_it = cached_kraus_operators_per_gate.find(gate_key);
    if (gate_it != cached_kraus_operators_per_gate.end()) {
        append_expanded_sets(relevant_operators, gate_it->second, gate_qubits, nqubits, true);
    }

    // For both types of per-qubit noise
    for (int qubit : gate_qubits) {
        // Add per-qubit Kraus operators, applied only to the qubit they are attached to
        auto qubit_it = cached_kraus_operators_per_qubit.find(qubit);
        if (qubit_it != cached_kraus_operators_per_qubit.end()) {
            append_expanded_sets(relevant_operators, qubit_it->second, {qubit}, nqubits, false);
        }

        // Add per-gate-per-qubit Kraus operators, applied only to the qubit they are attached to
        auto gate_qubit_it = cached_kraus_operators_per_gate_qubit.find(std::make_pair(gate_key, qubit));
        if (gate_qubit_it != cached_kraus_operators_per_gate_qubit.end()) {
            append_expanded_sets(relevant_operators, gate_qubit_it->second, {qubit}, nqubits, false);
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

std::pair<double, double> NoiseModelCpp::get_relevant_readout_error(int qubit) const {
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
        p01 = std::max(p01, readout_error_per_qubit.at(qubit).first);
        p10 = std::max(p10, readout_error_per_qubit.at(qubit).second);
    }
    return std::make_pair(p01, p10);
}

// GCOV_EXCL_BR_STOP