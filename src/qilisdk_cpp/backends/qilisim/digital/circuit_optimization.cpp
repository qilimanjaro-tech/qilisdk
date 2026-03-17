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

#include "circuit_optimizations.h"

#include <iostream>  // TODO(luke) remove

std::vector<Gate> combine_single_qubit_gates(const std::vector<Gate>& gates) {
    /*
    Combine consecutive single-qubit gates on the same qubit into a single gate to speed up simulation.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.

    Returns:
        std::vector<Gate>: A new list of gates where consecutive single-qubit gates on the same qubit have been combined into a single gate.
    */
    std::vector<Gate> combined_gates;
    std::vector<bool> already_used(gates.size(), false);
    for (size_t i = 0; i < gates.size(); ++i) {
        if (already_used[i]) {
            continue;
        }
        auto& gate = gates[i];
        already_used[i] = true;

        // Find a single-qubit gate
        if (gate.get_qubits().size() == 1) {
            // Start with the current gate
            int qubit = gate.get_qubits()[0];
            std::vector<Gate> gates_to_combine = {gate};

            // Look ahead to see if we have more single-qubit gates on the same target qubit that we can combine
            for (size_t j = i + 1; j < gates.size(); ++j) {
                const auto& next_gate = gates[j];
                std::vector<int> next_gate_qubits = next_gate.get_qubits();
                if (!already_used[j] && next_gate_qubits.size() == 1 && next_gate_qubits[0] == qubit) {
                    gates_to_combine.push_back(next_gate);
                    already_used[j] = true;

                    // Stop if there's a gate that acts on the same qubit but we can't combine
                } else if (next_gate_qubits.size() > 1 && std::find(next_gate_qubits.begin(), next_gate_qubits.end(), qubit) != next_gate_qubits.end()) {
                    break;
                }
            }

            // If we have things to combine
            if (gates_to_combine.size() > 1) {
                // Combine the gates into one gate
                SparseMatrix combined_matrix = gates_to_combine[0].get_base_matrix();
                for (size_t j = 1; j < gates_to_combine.size(); ++j) {
                    combined_matrix = gates_to_combine[j].get_base_matrix() * combined_matrix;
                }
                std::string combined_name = "";
                for (size_t j = 0; j < gates_to_combine.size(); ++j) {
                    combined_name += gates_to_combine[j].get_id();
                }
                combined_gates.push_back(Gate(combined_name, combined_matrix, {}, gate.get_qubits(), {}));

                // Otherwise just add it normally
            } else {
                combined_gates.push_back(gate);
            }

            // Otherwise just add it normally
        } else {
            combined_gates.push_back(gate);
        }
    }

    // std::cout << "Gates before:" << std::endl;
    // for (const auto& gate : gates) {
    //     std::cout << gate.get_id() << std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << "Gates after:" << std::endl;
    // for (const auto& gate : combined_gates) {
    //     std::cout << gate.get_id() << std::endl;
    // }

    return combined_gates;
}
