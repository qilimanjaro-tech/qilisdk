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

#include <algorithm>
#include <map>
#include <set>
#if defined(__linux__)
#include <unistd.h>
#endif

// GCOV_EXCL_BR_START

int auto_max_fused_qubits(int n_qubits) {
    /*
    Choose a fusion depth automatically from the qubit count.

    If the number of qubits is small enough that the entire state vector fits in the last-level cache, 
    we can afford to fuse fewer qubits (4) because the state vector is already cache-resident. 
    If the number of qubits is larger, we fuse more qubits (7) to reduce the number of matrix-vector multiplications and memory accesses.

    Args:
        n_qubits (int): The number of qubits in the circuit.

    Returns:
        int: The chosen maximum number of qubits a fused block may span.
    */

    // A statevector amplitude is one complex number. We only need an order-of-magnitude
    // size estimate to decide cache- vs DRAM-resident; assuming 16 bytes (complex<double>)
    // is fine, single precision merely shifts the crossover by one qubit.
    constexpr long long bytes_per_amplitude = 16;

    // Estimate the last-level cache size, falling back to 16 MiB where it can't be queried.
    long long llc_bytes = 16LL * 1024 * 1024;
#if defined(_SC_LEVEL3_CACHE_SIZE)
    long detected = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (detected > 0) {
        llc_bytes = static_cast<long long>(detected);
    }
#endif
    const long long llc_amplitudes = llc_bytes / bytes_per_amplitude;

    // Check if the state vector fits in the last-level cache
    const bool cache_resident = (n_qubits < 62) && ((1LL << n_qubits) <= llc_amplitudes);
    int k = cache_resident ? 4 : 7;

    // Never fuse more qubits than exist, and cap the dense block size
    k = std::min(k, n_qubits);
    k = std::min(k, 8);
    if (k < 1) {
        k = 1;
    }
    return k;
}

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
        if (gate.get_qubits().size() == 1 && gate.get_name() != "M") {
            // Start with the current gate
            int qubit = gate.get_qubits()[0];
            std::vector<Gate> gates_to_combine = {gate};

            // Look ahead to see if we have more single-qubit gates on the same target qubit that we can combine
            for (size_t j = i + 1; j < gates.size(); ++j) {
                const auto& next_gate = gates[j];
                std::vector<int> next_gate_qubits = next_gate.get_qubits();
                if (!already_used[j] && next_gate_qubits.size() == 1 && next_gate_qubits[0] == qubit && next_gate.get_name() != "M") {
                    gates_to_combine.push_back(next_gate);
                    already_used[j] = true;

                    // Stop if there's a gate that acts on the same qubit but we can't combine
                } else if (std::find(next_gate_qubits.begin(), next_gate_qubits.end(), qubit) != next_gate_qubits.end()) {
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

    return combined_gates;
}

namespace {

// A block of gates being accumulated for fusion
struct FusionBlock {
    std::set<int> qubits;
    std::vector<int> gate_indices;
};

Gate build_fused_gate(const FusionBlock& block, const std::vector<Gate>& gates) {
    /*
    Build a fused gate from a block of gates.

    Args:
        block (FusionBlock): The block of gates to fuse.
        gates (std::vector<Gate>&): The list of gates in the circuit.

    Returns:
        Gate: The fused gate representing the block, or the original gate if the block has only one gate.
    */

    // Make sure we have at least one gate
    if (block.gate_indices.size() == 1) {
        return gates[block.gate_indices[0]];
    }

    // Map the block's qubits to a contiguous local space
    std::vector<int> block_qubits(block.qubits.begin(), block.qubits.end());
    int k = int(block_qubits.size());
    std::map<int, int> to_local;
    for (int li = 0; li < k; ++li) {
        to_local[block_qubits[li]] = li;
    }

    // Multiply the matrices together to get the new gate
    SparseMatrix combined;
    bool first = true;
    for (int gi : block.gate_indices) {
        const Gate& g = gates[gi];
        std::vector<int> local_targets;
        std::vector<int> local_controls;
        for (int q : g.get_target_qubits()) {
            local_targets.push_back(to_local[q]);
        }
        for (int q : g.get_control_qubits()) {
            local_controls.push_back(to_local[q]);
        }
        Gate local_gate(g.get_name(), g.get_base_matrix(), local_controls, local_targets, g.get_parameters());
        SparseMatrix local_full = local_gate.get_full_matrix(k);
        if (first) {
            combined = local_full;
            first = false;
        } else {
            combined = local_full * combined;
        }
    }
    combined.makeCompressed();

    return Gate("FUSED", combined, {}, block_qubits, {});
}

}  // namespace

std::vector<Gate> fuse_gates(const std::vector<Gate>& gates, int max_fused_qubits) {
    /*
    Fuse runs of adjacent gates acting on a small set of qubits into a single
    dense multi-qubit gate.

    The pass walks the circuit in order, greedily growing "blocks" of gates that
    share qubits. A gate joins the blocks it overlaps as long as the resulting
    qubit set stays within `max_fused_qubits`; otherwise the overlapping blocks
    are closed (emitted) and the gate starts a fresh block. Because open blocks
    always have mutually disjoint qubit sets, gates in different blocks commute,
    so emitting blocks independently preserves the circuit's semantics.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        max_fused_qubits (int): Maximum number of qubits a fused block may span.

    Returns:
        std::vector<Gate>: The fused gate list.
    */
    std::vector<Gate> result;
    std::vector<FusionBlock> open_blocks;

    // Output the blocks in circuit order (by their earliest gate index)
    auto emit_blocks = [&](std::vector<FusionBlock>& blocks) {
        std::sort(blocks.begin(), blocks.end(), [](const FusionBlock& a, const FusionBlock& b) { return a.gate_indices.front() < b.gate_indices.front(); });
        for (const auto& block : blocks) {
            result.push_back(build_fused_gate(block, gates));
        }
    };

    // Output all open blocks and clear the list
    auto flush_all = [&]() {
        emit_blocks(open_blocks);
        open_blocks.clear();
    };

    // Go through the circuit
    for (int i = 0; i < int(gates.size()); ++i) {
        const Gate& gate = gates[i];
        std::vector<int> gate_qubits = gate.get_qubits();

        // Measurements act as a barrier
        if (gate.get_name() == "M") {
            flush_all();
            result.push_back(gate);
            continue;
        }

        // Find all currently-open blocks that share a qubit with this gate
        std::set<int> gate_qubit_set(gate_qubits.begin(), gate_qubits.end());
        std::vector<int> overlapping;
        std::set<int> candidate_qubits = gate_qubit_set;
        for (int b = 0; b < int(open_blocks.size()); ++b) {
            bool overlaps = false;
            for (int q : open_blocks[b].qubits) {
                if (gate_qubit_set.count(q)) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps) {
                overlapping.push_back(b);
                candidate_qubits.insert(open_blocks[b].qubits.begin(), open_blocks[b].qubits.end());
            }
        }

        // Check if the gate and the merged block fit within the max qubit limit
        bool gate_fits = int(gate_qubit_set.size()) <= max_fused_qubits;
        bool merged_fits = int(candidate_qubits.size()) <= max_fused_qubits;

        // If so, merge it
        if (gate_fits && merged_fits) {
            FusionBlock merged;
            merged.qubits = candidate_qubits;
            for (int b : overlapping) {
                merged.gate_indices.insert(merged.gate_indices.end(), open_blocks[b].gate_indices.begin(), open_blocks[b].gate_indices.end());
            }
            merged.gate_indices.push_back(i);
            std::sort(merged.gate_indices.begin(), merged.gate_indices.end());

            // Remove the merged blocks (erase from the back to keep indices valid).
            std::sort(overlapping.begin(), overlapping.end());
            for (auto it = overlapping.rbegin(); it != overlapping.rend(); ++it) {
                open_blocks.erase(open_blocks.begin() + *it);
            }
            open_blocks.push_back(std::move(merged));
        } else {
            // The gate cannot join the blocks it touches, so close them.
            std::vector<FusionBlock> to_emit;
            std::sort(overlapping.begin(), overlapping.end());
            for (auto it = overlapping.rbegin(); it != overlapping.rend(); ++it) {
                to_emit.push_back(std::move(open_blocks[*it]));
                open_blocks.erase(open_blocks.begin() + *it);
            }
            emit_blocks(to_emit);

            // Then start a new block with this gate if it fits, otherwise just emit it as-is
            if (gate_fits) {
                FusionBlock block;
                block.qubits = gate_qubit_set;
                block.gate_indices.push_back(i);
                open_blocks.push_back(std::move(block));
            } else {
                result.push_back(gate);
            }
        }
    }

    // Flush any remaining open blocks
    flush_all();
    return result;
}

// GCOV_EXCL_BR_STOP
