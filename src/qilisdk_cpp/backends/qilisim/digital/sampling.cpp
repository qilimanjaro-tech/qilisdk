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
#include <random>
#include <sstream>

#include "../digital/circuit_optimizations.h"
#include "../../../libs/pybind.h"
#include "../noise/noise_model.h"
#include "../utils/matrix_utils.h"
#include "../utils/random.h"
#include "sampling.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

std::map<std::string, int> apply_readout_error(const std::map<std::string, int>& counts, const NoiseModelCpp& noise_model_cpp, int n_qubits) {
    /*
    Apply readout error to the measurement counts.

    Args:
        counts (std::map<std::string, int>&): The original measurement counts without readout error.
        noise_model_cpp (NoiseModelCpp&): The noise model containing the readout error probabilities.
        n_qubits (int): The number of qubits in the circuit.

    Returns:
        std::map<std::string, int>: The new measurement counts after applying readout error
    */
    std::default_random_engine generator;
    generator.seed(42);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::map<std::string, int> noisy_counts;
    for (const auto& pair : counts) {
        std::string bitstring = pair.first;
        int count = pair.second;
        // For each shot in the count, apply readout error
        for (int shot = 0; shot < count; ++shot) {
            std::string noisy_bitstring = "";
            for (int q = 0; q < n_qubits; ++q) {
                char bit = bitstring[q];
                auto readout_error = noise_model_cpp.get_relevant_readout_error(q);
                double p01 = readout_error.first;
                double p10 = readout_error.second;
                double rand_val = distribution(generator);
                if (bit == '0') {
                    // 0 -> 1 with probability p01
                    if (rand_val < p01) {
                        noisy_bitstring += '1';
                    } else {
                        noisy_bitstring += '0';
                    }
                } else {
                    // 1 -> 0 with probability p10
                    if (rand_val < p10) {
                        noisy_bitstring += '0';
                    } else {
                        noisy_bitstring += '1';
                    }
                }
            }
            noisy_counts[noisy_bitstring] += 1;
        }
    }
    return noisy_counts;
}

std::map<std::string, int> filter_counts(const std::map<std::string, int>& counts, const std::vector<bool>& qubits_to_measure) {
    /*
    Filter the measurement counts to only include the measured qubits.

    Args:
        counts (std::map<std::string, int>&): The original measurement counts for all qubits.
        qubits_to_measure (std::vector<bool>&): A list indicating which qubits were measured.

    Returns:
        std::map<std::string, int>: The filtered measurement counts only for the measured qubits.
    */
    std::map<std::string, int> filtered_counts;
    for (const auto& pair : counts) {
        std::string bitstring = pair.first;
        int count = pair.second;
        std::string filtered_bitstring = "";
        for (size_t q = 0; q < qubits_to_measure.size(); ++q) {
            if (qubits_to_measure[q]) {
                filtered_bitstring += bitstring[q];
            }
        }
        filtered_counts[filtered_bitstring] += count;
    }
    return filtered_counts;
}

void sampling(const std::vector<Gate>& gates, const std::vector<bool>& qubits_to_measure, int n_qubits, int n_shots, const SparseMatrix& initial_state, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::map<std::string, int>& counts, const QiliSimConfig& config) {
    /*
    Execute a sampling functional using a simple statevector simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        qubits_to_measure (std::vector<bool>&): A list indicating which qubits to measure.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        initial_state (SparseMatrix&): The initial state of the system (statevector or density matrix).
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during simulation.
        state (DenseMatrix&): The final state after applying all gates (statevector or density matrix).
        counts (std::map<std::string, int>&): A map to store the measurement counts.
        config (QiliSimConfig): The simulation configuration.

    Returns:
        SamplingResult: A result object containing the measurement samples and computed probabilities.

    Raises:
        py::value_error: If functional is not a Sampling instance.
        py::value_error: If n_qubits is non-positive.
        py::value_error: If shots is non-positive.
    */

    // Set the number of threads
#if defined(_OPENMP)
    omp_set_num_threads(config.get_num_threads());
    Eigen::setNbThreads(config.get_num_threads());
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
        if (std::abs(trace_squared - 1.0) < config.get_atol()) {
            state = get_vector_from_density_matrix(state);
            is_statevector = true;
        }
    }

    // Check if we have noise
    bool has_noise = !noise_model_cpp.is_empty();

    // If we have noise but start with a statevector, convert to density matrix
    if (has_noise && is_statevector) {
        state = state * state.adjoint();
        is_statevector = false;
    }

    // Whether we should do monte-carlo sampling (only for density matrices)
    bool monte_carlo = (!is_statevector && config.get_monte_carlo());
    if (monte_carlo) {
        state = sample_from_density_matrix(state, config.get_num_monte_carlo_trajectories(), config.get_seed());
    }

    // Combine single-qubit gates for speed if not using noise models
    std::vector<Gate> optimized_gates = gates;
    if (!has_noise && config.get_combine_single_qubit_gates()) {
        optimized_gates = combine_single_qubit_gates(optimized_gates);
    }

    // Determine the start/end use of each gate
    std::map<std::string, std::pair<int, int>> gate_first_last_use;
    for (int i = 0; i < int(optimized_gates.size()); ++i) {
        std::string gate_id = optimized_gates[i].get_id();
        if (gate_first_last_use.find(gate_id) == gate_first_last_use.end()) {
            gate_first_last_use[gate_id] = std::make_pair(i, i);
        } else {
            gate_first_last_use[gate_id].second = i;
        }
    }

    // Pre-cache up to the limit (in parallel)
    int initial_cache_size = std::min(int(optimized_gates.size()), config.get_max_cache_size());
    std::vector<std::pair<std::string, SparseMatrix>> precomputed_gates(initial_cache_size);
#if defined(_OPENMP)
    Eigen::setNbThreads(1);
    omp_set_num_threads(config.get_num_threads());
#pragma omp parallel for
#endif
    for (int i = 0; i < initial_cache_size; ++i) {
        const auto& gate = optimized_gates[i];
        std::string gate_id = gate.get_id();
        SparseMatrix gate_matrix = gate.get_full_matrix(n_qubits);
        precomputed_gates[i] = std::make_pair(gate_id, gate_matrix);
    }
#if defined(_OPENMP)
    Eigen::setNbThreads(config.get_num_threads());
    omp_set_num_threads(config.get_num_threads());
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
    bool gate_is_new = false;
    for (const auto& gate : optimized_gates) {
        // Get the id of the gate
        gate_id = gate.get_id();

        // If we already have it in the cache, use it, otherwise generate it
        if (gate_cache.find(gate_id) != gate_cache.end()) {
            gate_is_new = false;
            gate_matrix = gate_cache[gate_id];
        } else {
            gate_is_new = true;
            gate_matrix = gate.get_full_matrix(n_qubits);
        }

        // If it will be used again later and we have space, cache it
        if (gate_is_new && gate_first_last_use[gate_id].second > gate_count && int(gate_cache.size()) < config.get_max_cache_size()) {
            gate_cache[gate_id] = gate_matrix;
        }

        // Apply the gate (Sparse-Dense multiplication, already OpenMP parallel if enabled)
        if (is_statevector || monte_carlo) {
            state = gate_matrix * state;
        } else {
            state = gate_matrix * state * gate_matrix.adjoint();
        }

        // Apply any relevant Kraus operators
        if (has_noise) {
            for (const auto& operator_set : noise_model_cpp.get_relevant_kraus_operators(gate.get_name(), gate.get_target_qubits(), n_qubits)) {
                DenseMatrix new_state(state.rows(), state.cols());
                new_state.setZero();
                for (const auto& K : operator_set) {
                    new_state += K * state * K.adjoint();
                }
                state = new_state;
            }
        }

        // Renormalize the state
        if (config.get_normalize_after_gate()) {
            normalize_state(state, is_statevector, monte_carlo);
        }

        // Clear the gate from the cache if this was its last use
        if (gate_first_last_use[gate_id].second <= gate_count) {
            gate_cache.erase(gate_id);
        }
        gate_count++;
    }

    // If we have statevector/s but we should return a density matrix
    if (monte_carlo) {
        state = trajectories_to_density_matrix(state);
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
    if (std::abs(total_prob - 1.0) > config.get_atol()) {
        std::stringstream ss;
        ss << std::setprecision(15) << total_prob;
        throw py::value_error("Probabilities do not sum to 1 (sum = " + ss.str() + ")");
    }

    // Sample from these probabilities
    counts = sample_from_probabilities(probabilities, n_qubits, n_shots, config.get_seed());

    // Apply readout error to counts
    if (has_noise) {
        counts = apply_readout_error(counts, noise_model_cpp, n_qubits);
    }

    // Only keep measured qubits in the counts
    counts = filter_counts(counts, qubits_to_measure);

    // If we started with a density matrix and ended with a statevector, convert back
    if (!initially_was_statevector && is_statevector) {
        state = state * state.adjoint();
    }
}

void sampling_matrix_free(const std::vector<Gate>& gates, const std::vector<bool>& qubits_to_measure, int n_qubits, int n_shots, const SparseMatrix& initial_state, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::map<std::string, int>& counts, const QiliSimConfig& config) {
    /*
    Execute a sampling functional using a matrix-free simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        qubits_to_measure (std::vector<bool>&): A list indicating which qubits to measure.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        initial_state (SparseMatrix&): The initial state of the system (statevector or density matrix).
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during simulation.
        state (DenseMatrix&): The final state after applying all gates (statevector or density matrix).
        counts (std::map<std::string, int>&): A map to store the measurement counts.
        config (QiliSimConfig): The simulation configuration.

    Returns:
        SamplingResult: A result object containing the measurement samples and computed probabilities.

    Raises:
        py::value_error: If functional is not a Sampling instance.
        py::value_error: If n_qubits is non-positive.
        py::value_error: If shots is non-positive.
    */

    // Set the number of threads
#if defined(_OPENMP)
    omp_set_num_threads(config.get_num_threads());
    Eigen::setNbThreads(config.get_num_threads());
#endif

    // Start with the zero state
    state = initial_state;
    bool is_statevector = (state.cols() == 1 && state.rows() == (1L << n_qubits));
    bool initially_was_statevector = is_statevector;

    // Check if we have noise
    bool has_noise = !noise_model_cpp.is_empty();

    // If we have noise but start with a statevector, convert to density matrix
    if (has_noise && is_statevector) {
        state = state * state.adjoint();
        is_statevector = false;
    }

    // Whether we should do monte-carlo sampling (only for density matrices)
    bool monte_carlo = (!is_statevector && config.get_monte_carlo());
    if (monte_carlo) {
        state = sample_from_density_matrix(state, config.get_num_monte_carlo_trajectories(), config.get_seed());
    }

    // Combine single-qubit gates for speed if not using noise models
    std::vector<Gate> optimized_gates = gates;
    if (!has_noise && config.get_combine_single_qubit_gates()) {
        optimized_gates = combine_single_qubit_gates(optimized_gates);
    }

    // Apply each gate
    for (const auto& gate : optimized_gates) {
        // Convert gate to a stabilizer operator
        MatrixFreeOperator op(gate);

        // Apply the gate
        if (monte_carlo) {
            for (int col = 0; col < state.cols(); ++col) {
                DenseMatrix traj = state.col(col);
                op.apply(traj, MatrixFreeApplicationType::Left);
                state.col(col) = traj;
            }
        } else if (is_statevector) {
            op.apply(state, MatrixFreeApplicationType::Left);
        } else {
            op.apply(state, MatrixFreeApplicationType::LeftAndRight);
        }

        // Apply noise if we have it
        if (!noise_model_cpp.is_empty()) {
            for (const auto& operator_set : noise_model_cpp.get_relevant_kraus_operators(gate.get_name(), gate.get_target_qubits(), n_qubits)) {
                DenseMatrix new_state(state.rows(), state.cols());
                new_state.setZero();
                for (const auto& K : operator_set) {
                    new_state += K * state * K.adjoint();
                }
                state = new_state;
            }
        }

        // Renormalize the state
        if (config.get_normalize_after_gate()) {
            normalize_state(state, is_statevector, monte_carlo);
        }
    }

    // If we have statevector/s but we should return a density matrix
    if (monte_carlo) {
        state = trajectories_to_density_matrix(state);
    }

    // Get the probabilities
    std::vector<double> probabilities(state.rows());
    double total_prob = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_prob) schedule(static)
#endif
    for (int row = 0; row < state.rows(); ++row) {
        double prob = 0.0;
        if (is_statevector) {
            prob = std::norm(state(row, 0));
        } else {
            prob = state.coeff(row, row).real();
        }
        total_prob += prob;
        probabilities[row] = prob;
    }

    // Make sure probabilities sum to 1
    if (std::abs(total_prob - 1.0) > config.get_atol()) {
        std::stringstream ss;
        ss << std::setprecision(15) << total_prob;
        throw py::value_error("Probabilities do not sum to 1 (sum = " + ss.str() + ")");
    }

    // Sample from these probabilities
    counts = sample_from_probabilities(probabilities, n_qubits, n_shots, config.get_seed());

    // Apply readout error to counts
    if (has_noise) {
        counts = apply_readout_error(counts, noise_model_cpp, n_qubits);
    }

    // Only keep measured qubits in the counts
    counts = filter_counts(counts, qubits_to_measure);

    // If we started with a density matrix and ended with a statevector, convert back
    if (!initially_was_statevector && is_statevector) {
        state = state * state.adjoint();
    }
}