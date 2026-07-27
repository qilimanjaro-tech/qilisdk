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

#include "gate.h"

// GCOV_EXCL_BR_START

Gate::Gate(const std::string& gate_type_, const SparseMatrix& base_matrix_, const std::vector<int>& controls_, const std::vector<int>& targets_, const std::vector<std::pair<std::string, double>>& parameters_) {
    this->gate_type = gate_type_;
    this->base_matrix = base_matrix_;
    this->control_qubits = controls_;
    this->target_qubits = targets_;
    this->parameters = parameters_;

    // Convert common names to standard
    if (this->gate_type == "Toffoli" || this->gate_type == "CCX" || this->gate_type == "CNOT" || this->gate_type == "CX") {
        this->gate_type = "X";
    } else if (this->gate_type == "CCY" || this->gate_type == "CY") {
        this->gate_type = "Y";
    } else if (this->gate_type == "CCZ" || this->gate_type == "CZ") {
        this->gate_type = "Z";
    }
}

int Gate::permute_bits(int index, const std::vector<int>& perm) const {
    /*
    Permute the bits of an index according to a given permutation.
    This works by taking each bit from the original index and placing it in the new position
    specified by the permutation vector.
    Also note, the qubit versus bit ordering is reversed (i.e. {1, 0, 2} swaps the
    first bits (0 and 1), not the last/least-significant).

    Args:
        index (int): The original index.
        perm (std::vector<int>): The permutation vector.

    Returns:
        int: The permuted index.
    */
    int n = int(perm.size());
    int out = 0;
    for (int old_q = 0; old_q < n; old_q++) {
        int new_q = perm[old_q];
        int old_bit = n - 1 - old_q;
        int new_bit = n - 1 - new_q;
        int bit = (index >> old_bit) & 1;
        out |= (bit << new_bit);
    }
    return out;
}

Triplets Gate::tensor_product(Triplets& A, Triplets& B, int B_width) const {
    /*
    Compute the tensor product of two sets of (row, col, value) tuples.

    Args:
        A (Triplets&): First matrix entries.
        B (Triplets&): Second matrix entries.
        B_width (int): Width of the second matrix (assumed square).

    Returns:
        Triplets: The tensor product entries.
    */
    Triplets result;
    int row = 0;
    int col = 0;
    Complex val = 0.0;
    for (const auto& a : A) {
        for (const auto& b : B) {
            row = a.row() * B_width + b.row();
            col = a.col() * B_width + b.col();
            val = a.value() * b.value();
            result.emplace_back(row, col, val);
        }
    }
    return result;
}

SparseMatrix Gate::base_to_full(const SparseMatrix& base_gate, int num_qubits, const std::vector<int>& control_qubits, const std::vector<int>& target_qubits) const {
    /*
    Expand a base gate matrix to the full matrix on the entire qubit register.

    Args:
        base_gate (SparseMatrix): The base gate matrix.
        num_qubits (int): Total number of qubits in the register.
        control_qubits (std::vector<int>): List of control qubit indices.
        target_qubits (std::vector<int>): List of target qubit indices.

    Returns:
        SparseMatrix: The full matrix representation of the gate.
    */

    // Determine how many tensor products we need before and after
    int min_qubit = num_qubits;
    for (int q : control_qubits) {
        if (q < min_qubit)
            min_qubit = q;
    }
    for (int q : target_qubits) {
        if (q < min_qubit)
            min_qubit = q;
    }
    int base_gate_qubits = std::ceil(std::log2(base_gate.cols()));
    int total_qubits_needed = int(target_qubits.size()) + int(control_qubits.size());
    int missing_controls = total_qubits_needed - base_gate_qubits;
    int needed_before = min_qubit;
    int needed_after = num_qubits - needed_before - total_qubits_needed;

    // Do everything in tuple form for easier manipulation
    Triplets out_entries;
    for (int k = 0; k < base_gate.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(base_gate, k); it; ++it) {
            int row = int(it.row());
            int col = int(it.col());
            Complex val = it.value();
            out_entries.emplace_back(Triplet(row, col, val));
        }
    }

    // Make it controlled if needed
    // Reminder that control gates look like:
    // 1 0 0 0
    // 0 1 0 0
    // 0 0 G G
    // 0 0 G G
    for (int i = 0; i < missing_controls; ++i) {
        int delta = 1 << (base_gate_qubits + i);
        Triplets new_entries;
        for (const auto& entry : out_entries) {
            int row = int(entry.row());
            int col = int(entry.col());
            Complex val = entry.value();
            new_entries.emplace_back(Triplet(row + delta, col + delta, val));
        }
        for (int i = 0; i < delta; ++i) {
            new_entries.emplace_back(Triplet(i, i, 1.0));
        }
        out_entries = new_entries;
    }

    // Perform the tensor products with the identity
    Triplets identity_entries = {Triplet(0, 0, 1.0), Triplet(1, 1, 1.0)};
    int gate_size = 1 << (target_qubits.size() + control_qubits.size());
    for (int i = 0; i < needed_before; ++i) {
        out_entries = tensor_product(identity_entries, out_entries, gate_size);
        gate_size *= 2;
    }
    for (int i = 0; i < needed_after; ++i) {
        out_entries = tensor_product(out_entries, identity_entries, 2);
        gate_size *= 2;
    }

    // Get a list of all qubits involved
    std::vector<int> all_qubits;
    all_qubits.insert(all_qubits.end(), control_qubits.begin(), control_qubits.end());
    all_qubits.insert(all_qubits.end(), target_qubits.begin(), target_qubits.end());

    // Determine the permutation mapping each local qubit to its global qubit
    std::vector<bool> is_gate_qubit(num_qubits, false);
    for (int q : all_qubits) {
        is_gate_qubit[q] = true;
    }
    std::vector<int> perm(num_qubits, -1);
    for (int i = 0; i < int(all_qubits.size()); ++i) {
        perm[needed_before + i] = all_qubits[i];
    }
    int next_spectator = 0;
    for (int pos = 0; pos < num_qubits; ++pos) {
        if (pos >= needed_before && pos < needed_before + int(all_qubits.size())) {
            continue;
        }
        while (is_gate_qubit[next_spectator]) {
            ++next_spectator;
        }
        perm[pos] = next_spectator++;
    }

    // Apply the permutation
    for (size_t i = 0; i < out_entries.size(); ++i) {
        int old_row = out_entries[i].row();
        int old_col = out_entries[i].col();
        int new_row = permute_bits(old_row, perm);
        int new_col = permute_bits(old_col, perm);
        out_entries[i] = Triplet(new_row, new_col, out_entries[i].value());
    }

    // Make sure the entries are sorted
    std::sort(out_entries.begin(), out_entries.end(), [](const Triplet& a, const Triplet& b) {
        if (a.row() != b.row()) {
            return a.row() < b.row();
        } else {
            return a.col() < b.col();
        }
    });

    // Form the matrix and return
    int full_size = 1 << num_qubits;
    SparseMatrix out_matrix(full_size, full_size);
    out_matrix.setFromTriplets(out_entries.begin(), out_entries.end());
    return out_matrix;
}

std::string Gate::get_name() const {
    /*
    Get the name of the gate.

    Returns:
        std::string: The gate name.
    */
    return gate_type;
}

std::string Gate::get_id() const {
    /*
    Get a unique identifier for the gate based on its type and qubits.
    Returns:
        std::string: The gate identifier.
    */
    std::ostringstream oss;
    oss << get_name();
    if (!control_qubits.empty()) {
        oss << "_c_";
        for (const auto& q : control_qubits) {
            oss << q << "_";
        }
    }
    if (!target_qubits.empty()) {
        oss << "_t_";
        for (const auto& q : target_qubits) {
            oss << q << "_";
        }
    }
    if (!parameters.empty()) {
        oss << "p_";
        for (const auto& param : parameters) {
            oss << param.first << "_" << param.second << "_";
        }
    }
    return oss.str();
}

SparseMatrix Gate::get_base_matrix() const {
    /*
    Get the base matrix of the gate.

    Returns:
        SparseMatrix: The base matrix of the gate.
    */
    return base_matrix;
}

SparseMatrix Gate::get_full_matrix(int num_qubits) const {
    /*
    Get the full matrix representation of the gate on the entire qubit register.

    Args:
        num_qubits (int): Total number of qubits in the register.

    Returns:
        SparseMatrix: The full matrix representation of the gate.
    */
    if (control_qubits.empty() && num_qubits == std::ceil(std::log2(base_matrix.cols()))) {
        return base_matrix;
    } else {
        return base_to_full(base_matrix, num_qubits, control_qubits, target_qubits);
    }
}

int Gate::get_nqubits() const {
    /*
    Get the total number of qubits the gate acts on (controls + targets).

    Returns:
        int: The total number of qubits.
    */
    return int(control_qubits.size()) + int(target_qubits.size());
}

std::vector<int> Gate::get_qubits() const {
    /*
    Get the list of all qubits the gate acts on (controls + targets).

    Returns:
        std::vector<int>: The qubit indices.
    */
    std::vector<int> all_qubits;
    all_qubits.insert(all_qubits.end(), control_qubits.begin(), control_qubits.end());
    all_qubits.insert(all_qubits.end(), target_qubits.begin(), target_qubits.end());
    return all_qubits;
}

bool Gate::is_normalized() const {
    /*
    Check if the gate's base matrix is normalized (i.e., unitary).

    Returns:
        bool: True if the matrix is normalized, False otherwise.
    */
    // We can check if the matrix is normalized by verifying that U * U^dagger = I
    SparseMatrix base_matrix = get_base_matrix();
    SparseMatrix identity(base_matrix.rows(), base_matrix.cols());
    identity.setIdentity();
    SparseMatrix product = base_matrix * base_matrix.adjoint();
    return product.isApprox(identity);
}

std::vector<int> Gate::get_target_qubits() const {
    /*
    Get the list of target qubits.

    Returns:
        std::vector<int>: The target qubit indices.
    */
    return target_qubits;
}

std::vector<int> Gate::get_control_qubits() const {
    /*
    Get the list of control qubits.

    Returns:
        std::vector<int>: The control qubit indices.
    */
    return control_qubits;
}

std::vector<std::pair<std::string, double>> Gate::get_parameters() const {
    /*
    Get the list of gate parameters.

    Returns:
        std::vector<std::pair<std::string, double>>: The gate parameters.
    */
    return parameters;
}

// Overload stream output for vectors of Gates (for debugging)
std::ostream& operator<<(std::ostream& os, const std::vector<Gate>& gates) {
    os << "[\n";
    for (const auto& gate : gates) {
        os << "  " << gate.get_id() << ",\n";
    }
    os << "]";
    return os;
}

// Stream output for a single gate (for debugging)
std::ostream& operator<<(std::ostream& os, const Gate& gate) {  // GCOV_EXCL_START
    os << gate.get_id();
    return os;
}  // GCOV_EXCL_STOP

// GCOV_EXCL_BR_STOP
