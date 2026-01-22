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

#include <iomanip>
#include <sstream>

#include "sampling.h"
#include "../utils/random.h"
#include "../libs/pybind.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

void sampling(std::vector<Gate>& gates,
                    const std::vector<bool>& qubits_to_measure,
                    int n_qubits,
                    int n_shots,
                    QiliSimConfig& config,
                    std::map<std::string, int>& counts) {
    /*
    Execute a sampling functional using a simple statevector simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        qubits_to_measure (std::vector<bool>&): A list indicating which qubits to measure.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        config (QiliSimConfig): The simulation configuration.
        counts (std::map<std::string, int>&): A map to store the measurement counts.

    Returns:
        SamplingResult: A result object containing the measurement samples and computed probabilities.

    Raises:
        py::value_error: If functional is not a Sampling instance.
        py::value_error: If nqubits is non-positive.
        py::value_error: If shots is non-positive.
    */

    // Set the number of threads
    if (config.num_threads <= 0) {
        config.num_threads = 1;
    }
#if defined(_OPENMP)
    omp_set_num_threads(config.num_threads);
#endif

    // Start with the zero state
    long dim = 1L << n_qubits;
    DenseMatrix state = DenseMatrix::Zero(dim, 1);
    state(0, 0) = 1.0;

    // Determine the start/end use of each gate
    std::map<std::string, std::pair<int, int>> gate_first_last_use;
    for (int i = 0; i < int(gates.size()); ++i) {
        std::string gate_id = gates[i].get_id();
        if (gate_first_last_use.find(gate_id) == gate_first_last_use.end()) {
            gate_first_last_use[gate_id] = std::make_pair(i, i);
        } else {
            gate_first_last_use[gate_id].second = i;
        }
    }

    // Pre-cache up to the limit
    std::map<std::string, SparseMatrix> gate_cache;
    int initial_cache_size = std::min(int(gates.size()), config.max_cache_size);
    for (int i = 0; i < initial_cache_size; ++i) {
        const auto& gate = gates[i];
        std::string gate_id = gate.get_id();
        gate_cache[gate_id] = SparseMatrix();
    }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < initial_cache_size; ++i) {
        const auto& gate = gates[i];
        std::string gate_id = gate.get_id();
        SparseMatrix gate_matrix = gate.get_full_matrix(n_qubits);
        gate_cache[gate_id] = gate_matrix;
    }

    // Apply each gate (caching as needed)
    SparseMatrix gate_matrix;
    std::string gate_id = "";
    int gate_count = 0;
    for (const auto& gate : gates) {
        // Get the id of the gate
        gate_id = gate.get_id();

        // If we already have it in the cache, use it, otherwise generate it
        if (gate_cache.find(gate_id) != gate_cache.end()) {
            gate_matrix = gate_cache[gate_id];
        } else {
            gate_matrix = gate.get_full_matrix(n_qubits);
        }

        // If it will be used again later and we have space, cache it
        if (gate_first_last_use[gate_id].second > gate_count && int(gate_cache.size()) < config.max_cache_size) {
            gate_cache[gate_id] = gate_matrix;
        }

        // Apply the gate (Sparse-Dense multiplication, OpenMP parallel if enabled)
        state = gate_matrix * state;

        // Renormalize the state
        state /= state.norm();

        // Clear the gate from the cache if this was its last use
        if (gate_first_last_use[gate_id].second <= gate_count) {
            gate_cache.erase(gate_id);
        }
        gate_count++;
    }

    // Get the probabilities
    std::vector<double> probabilities(state.rows());
    double total_prob = 0.0;
    for (int row = 0; row < state.rows(); ++row) {
        std::complex<double> amp = state(row, 0);
        double prob = std::norm(amp);
        total_prob += prob;
        probabilities[row] = prob;
    }

    // Make sure probabilities sum to 1
    if (std::abs(total_prob - 1.0) > config.atol) {
        std::stringstream ss;
        ss << std::setprecision(15) << total_prob;
        throw py::value_error("Probabilities do not sum to 1 (sum = " + ss.str() + ")");
    }

    // Sample from these probabilities
    counts = sample_from_probabilities(probabilities, n_qubits, n_shots, config.seed);

    // Only keep measured qubits in the counts
    std::map<std::string, int> filtered_counts;
    for (const auto& pair : counts) {
        std::string bitstring = pair.first;
        std::string filtered_bitstring = "";
        for (int i = 0; i < n_qubits; ++i) {
            if (qubits_to_measure[i]) {
                filtered_bitstring += bitstring[i];
            }
        }
        filtered_counts[filtered_bitstring] += pair.second;
    }
    counts = filtered_counts;

}
