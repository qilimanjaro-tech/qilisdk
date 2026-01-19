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

#include <set>
#include "../qilisim.h"

void QiliSimCpp::combine_single_qubit_gates(std::vector<Gate>& gates) const {
    /*
    Combine consecutive single-qubit gates acting on the same qubit.

    Args:
        gates (std::vector<Gate>): The list of Gate objects.
    */

    // Loop over the gates
    std::vector<Gate> new_gates;
    std::vector<bool> gate_used(gates.size(), false);
    for (int i = 0; i < int(gates.size()); ++i) {
        if (gate_used[i]) {
            continue;
        }

        // Start with the current gate
        Gate new_gate = gates[i];

        // Check if the current gate is single-qubit
        if (new_gate.get_nqubits() == 1) {
            std::vector<Gate> combined_gates = {new_gate};
            int this_qubit = new_gate.get_target_qubits()[0];

            // Look ahead for consecutive single-qubit gates on the same qubit
            for (int j = i + 1; j < int(gates.size()); ++j) {
                // If we hit a multi-qubit gate on either of these qubits, stop looking ahead
                if (gates[j].get_nqubits() != 1) {
                    bool conflict = false;
                    for (int cq : gates[j].get_control_qubits()) {
                        if (cq == this_qubit) {
                            conflict = true;
                            break;
                        }
                    }
                    for (int tq : gates[j].get_target_qubits()) {
                        if (tq == this_qubit) {
                            conflict = true;
                            break;
                        }
                    }
                    if (conflict) {
                        break;
                    }
                }

                // If it's on the same qubit, add it to the list to combine
                if (this_qubit == gates[j].get_target_qubits()[0] && !gate_used[j]) {
                    combined_gates.push_back(gates[j]);
                    gate_used[j] = true;
                }
            }

            // If we found more than the original gate, create the new combined gate
            if (combined_gates.size() > 1) {
                // Combine the matrices
                SparseMatrix combined_matrix = I;
                for (const auto& g : combined_gates) {
                    combined_matrix = g.get_full_matrix(1) * combined_matrix;
                }
                // Combine the names
                std::string combined_name = "";
                for (size_t idx = 0; idx < combined_gates.size(); ++idx) {
                    combined_name += combined_gates[idx].get_name();
                    if (idx < combined_gates.size() - 1) {
                        combined_name += "_";
                    }
                }
                // Combine parameters
                std::vector<std::pair<std::string, double>> combined_params;
                for (const auto& g : combined_gates) {
                    std::vector<std::pair<std::string, double>> g_params = g.get_parameters();
                    combined_params.insert(combined_params.end(), g_params.begin(), g_params.end());
                }
                // Create a new Gate with the combined matrix
                new_gate = Gate(combined_name, combined_matrix, {}, {this_qubit}, combined_params);
            }
        }

        // Add this new gate to the list
        new_gates.push_back(new_gate);
        gate_used[i] = true;
    }

    // Replace the original gates with the new combined gates
    gates = new_gates;
}

std::vector<std::vector<Gate>> QiliSimCpp::compress_gate_layers(std::vector<Gate>& gates) const {
    /*
    Compress gate layers by combining gates that act on different sets of qubits.

    Args:
        gate_layers (std::vector<std::vector<Gate>>&): The list of gate layers.
    */

    std::vector<std::vector<Gate>> compressed_layers;
    std::vector<std::set<int>> occupied_qubits_layers;
    std::vector<bool> layer_has_multi_qubit;
    for (const auto& gate : gates) {
        bool placed = false;

        // Get the list of qubits this gate acts on
        std::set<int> gate_qubits;
        for (int tq : gate.get_target_qubits()) {
            gate_qubits.insert(tq);
        }
        for (int cq : gate.get_control_qubits()) {
            gate_qubits.insert(cq);
        }

        // Try to place the gate in an existing layer (for now only single-qubit gates)
        if (gate_qubits.size() == 1) {
            for (int layer_idx = int(compressed_layers.size()) - 1; layer_idx > 0; --layer_idx) {
                // Check if there's a conflict between the qubit lists
                bool conflict = false;
                for (int q : gate_qubits) {
                    if (occupied_qubits_layers[layer_idx].count(q) > 0) {
                        conflict = true;
                        break;
                    }
                }

                // If no conflict and the layer has no multi-qubit gates, place it here
                if (!conflict && !layer_has_multi_qubit[layer_idx]) {
                    compressed_layers[layer_idx].push_back(gate);
                    occupied_qubits_layers[layer_idx].insert(gate_qubits.begin(), gate_qubits.end());
                    placed = true;
                    break;
                }

                // If at any point we hit a conflict, we have to stop
                if (conflict) {
                    break;
                }
            }
        }

        // If not placed, create a new layer
        if (!placed) {
            compressed_layers.push_back({gate});
            occupied_qubits_layers.push_back(gate_qubits);
            layer_has_multi_qubit.push_back(gate_qubits.size() > 1);
        }
    }

    return compressed_layers;
}

SparseMatrix QiliSimCpp::layer_to_matrix(const std::vector<Gate>& gate_layer, int n_qubits) const {
    /*
    Convert a layer of gates into a full sparse matrix.
    For some cases this can be more efficient than multiplying individual gate matrices.

    Args:
        gate_layer (std::vector<Gate>): The layer of gates to convert.
        n_qubits (int): The total number of qubits in the circuit.

    Returns:
        SparseMatrix: The full sparse matrix representing the gate layer.
    */

    // If we have only one gate, just return its full matrix
    if (gate_layer.size() == 1) {
        return gate_layer[0].get_full_matrix(n_qubits);
    } else {
        // Sort the gates by target qubit (assuming single-qubit gates for simplicity)
        std::vector<Gate> sorted_gates(n_qubits, Gate("I", I, {}, {0}, {}));
        for (const auto& gate : gate_layer) {
            if (gate.get_nqubits() != 1) {
                throw std::runtime_error("layer_to_matrix currently only supports single-qubit gates.");
            }
            int target_qubit = gate.get_target_qubits()[0];
            sorted_gates[target_qubit] = gate;
        }

        // Tensor product the gates in the layer
        SparseMatrix full_matrix(1, 1);
        full_matrix.coeffRef(0, 0) = 1.0;
        full_matrix.makeCompressed();
        for (const auto& gate : sorted_gates) {
            SparseMatrix gate_base_matrix = gate.get_full_matrix(1);
            full_matrix = Eigen::kroneckerProduct(full_matrix, gate_base_matrix).eval();
        }

        return full_matrix;
    }
}