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

#include "../libs/pybind.h"
#include "../utils/random.h"
#include "sampling.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

void sampling(const std::vector<Gate>& gates, 
              const std::vector<bool>& qubits_to_measure, 
              int n_qubits, 
              int n_shots, 
              const SparseMatrix& initial_state,
              DenseMatrix& state,
              std::map<std::string, int>& counts, 
              const QiliSimConfig& config) {
    /*
    Execute a sampling functional using a simple statevector simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        qubits_to_measure (std::vector<bool>&): A list indicating which qubits to measure.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        initial_state (SparseMatrix&): The initial state of the system (statevector or density matrix).
        state (DenseMatrix&): The final state after applying all gates (statevector or density matrix).
        counts (std::map<std::string, int>&): A map to store the measurement counts.
        config (QiliSimConfig): The simulation configuration.

    Returns:
        SamplingResult: A result object containing the measurement samples and computed probabilities.

    Raises:
        py::value_error: If functional is not a Sampling instance.
        py::value_error: If nqubits is non-positive.
        py::value_error: If shots is non-positive.
    */

    // Set the number of threads
#if defined(_OPENMP)
    omp_set_num_threads(config.num_threads);
    Eigen::setNbThreads(config.num_threads);
#endif

    // Start with the zero state
    long dim = 1L << n_qubits;
    state = initial_state;
    bool is_statevector = (state.cols() == 1 && state.rows() == dim);
    bool initially_was_statevector = is_statevector;

    // If it's a density matrix, check if it's pure
    if (!is_statevector) {
        DenseMatrix state_squared = state.adjoint().cwiseProduct(state);
        double trace_squared = state_squared.trace().real();
        if (std::abs(trace_squared - 1.0) < config.atol) {
            state = get_vector_from_density_matrix(state.sparseView());
            is_statevector = true;
        }
    }

    // Whether we should do monte-carlo sampling (only for density matrices)
    bool monte_carlo = (!is_statevector && config.monte_carlo);
    if (monte_carlo) {
        state = sample_from_density_matrix(state.sparseView(), config.num_monte_carlo_trajectories, config.seed);
    }

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

    // Pre-cache up to the limit (in parallel)
    int initial_cache_size = std::min(int(gates.size()), config.max_cache_size);
    std::vector<std::pair<std::string, SparseMatrix>> precomputed_gates(initial_cache_size);
#if defined(_OPENMP)
Eigen::setNbThreads(1);
omp_set_num_threads(config.num_threads);
#pragma omp parallel for
#endif
    for (int i = 0; i < initial_cache_size; ++i) {
        const auto& gate = gates[i];
        std::string gate_id = gate.get_id();
        SparseMatrix gate_matrix = gate.get_full_matrix(n_qubits);
        precomputed_gates[i] = std::make_pair(gate_id, gate_matrix);
    }
#if defined(_OPENMP)
Eigen::setNbThreads(config.num_threads);
omp_set_num_threads(config.num_threads);
#endif

    // Convert pre-cached gates to a map for easier access (in serial)
    std::map<std::string, SparseMatrix> gate_cache;
    for (int i = 0; i < initial_cache_size; ++i) {
        const auto& pair = precomputed_gates[i];
        gate_cache[pair.first] = pair.second;
        precomputed_gates[i] = std::make_pair("", SparseMatrix());
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
        if (is_statevector || monte_carlo) {
            state = gate_matrix * state;
        } else {
            state = gate_matrix * state * gate_matrix.adjoint();
        }

        // Renormalize the state
        if (monte_carlo) {
            for (int col = 0; col < state.cols(); ++col) {
                DenseMatrix traj = state.col(col);
                traj /= traj.norm();
                state.col(col) = traj;
            }
        } else if (is_statevector) {
            state /= state.norm();
        } else {
            double trace = state.trace().real();
            state /= trace;
        }

        // Clear the gate from the cache if this was its last use
        if (gate_first_last_use[gate_id].second <= gate_count) {
            gate_cache.erase(gate_id);
        }
        gate_count++;
    }

    // If we have statevector/s but we should return a density matrix
    if (monte_carlo) {
        state = trajectories_to_density_matrix(state.sparseView());
    }

    // Get the probabilities
    std::vector<double> probabilities(state.rows());
    double total_prob = 0.0;
    double prob = 0.0;
    for (int row = 0; row < state.rows(); ++row) {
        if (is_statevector) {
            prob = std::norm(state(row, 0));
        } else {
            prob = state.coeff(row, row).real();
        }
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

    // If we started with a density matrix and ended with a statevector, convert back
    if (!initially_was_statevector && is_statevector) {
        state = state * state.adjoint();
    }

}
