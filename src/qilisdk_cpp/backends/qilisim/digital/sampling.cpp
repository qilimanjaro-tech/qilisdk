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

#include "../../../libs/pybind.h"
#include "../digital/circuit_optimizations.h"
#include "../noise/noise_model.h"
#include "../utils/matrix_utils.h"
#include "../utils/parsers.h"
#include "../utils/random.h"
#include "sampling.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

DenseMatrix collapse_state(const DenseMatrix& state, const std::vector<bool>& qubits_measured) {
    /*
    Collapse the state based on the measurement result.

    Args:
        state (DenseMatrix&): the state to be collapsed, either as a statevector or density matrix.
        qubits_measured (std::vector<bool>): which qubits were measured.

    Returns:
        DenseMatrix: the collapsed state as a density matrix
    */

    // Determine if we are working with a statevector or density matrix
    bool is_statevector = (state.cols() == 1);

    // If it's a statevector, convert to density matrix
    DenseMatrix density_matrix;
    if (is_statevector) {
        density_matrix = state * state.adjoint();
    } else {
        density_matrix = state;
    }

    // For each measured qubit, apply the non-selective measurement:
    // rho -> P0 * rho * P0 + P1 * rho * P1
    // Since P0 and P1 are diagonal projectors, this is equivalent to zeroing all entries
    // where the row and column indices differ in that qubit's bit position.
    long dim = density_matrix.rows();
    int nqubits = static_cast<int>(std::ceil(std::log2(dim)));
    for (int qubit = 0; qubit < int(qubits_measured.size()); ++qubit) {
        if (!qubits_measured[qubit]) {
            continue;
        }
        long mask = 1L << (nqubits - 1 - qubit);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long j = 0; j < dim; ++j) {
            long opposite_start = (j & mask) ? 0 : mask;
            for (long base = 0; base < dim; base += 2 * mask) {
                density_matrix.col(j).segment(base + opposite_start, mask).setZero();
            }
        }
    }

    return density_matrix;
}

void sampling(const std::vector<Gate>& gates, int n_qubits, const SparseMatrix& initial_state, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout) {
    /*
    Execute a sampling functional using a simple statevector simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        initial_state (SparseMatrix&): The initial state of the system (statevector or density matrix).
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during simulation.
        state (DenseMatrix&): The final state after applying all gates (statevector or density matrix).
        intermediate_results (std::vector<py::object>&): A vector to store the intermediate results after each measurement.
        config (QiliSimConfig): The simulation configuration.
        readout (py::object): A list with readout information to determine when and what to measure.

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
    for (int i = 0; i < int(optimized_gates.size()); ++i) {
        const auto& gate = optimized_gates[i];

        // If it's a measurement
        if (gate.get_name() == "M") {
            // Check for adjacent measurements and if we have any, do them all at once
            int num_measurements = 0;
            bool are_final_measurements = false;
            std::vector<bool> qubits_to_measure_after_gate(n_qubits, false);
            for (int j = i; j < int(optimized_gates.size()); ++j) {
                if (optimized_gates[j].get_name() == "M") {
                    for (int target : optimized_gates[j].get_target_qubits()) {
                        qubits_to_measure_after_gate[target] = true;
                    }
                    num_measurements++;
                    if (j == int(optimized_gates.size()) - 1) {
                        are_final_measurements = true;
                    }
                } else {
                    break;
                }
            }

            // Construct the result
            if (!are_final_measurements) {
                py::object result = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure_after_gate);
                intermediate_results.push_back(result);

                // If we have measurement_collapse enabled, apply the measurement and collapse the state
                if (config.get_measurement_collapse()) {
                    state = collapse_state(state, qubits_to_measure_after_gate);
                    is_statevector = false;
                }
            }

            // Skip the next measurements since we did them all at once
            i += (num_measurements - 1);
            continue;
        }

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

    // If we started with a density matrix and ended with a statevector, convert back
    if (!initially_was_statevector && is_statevector) {
        state = state * state.adjoint();
    }
}

void sampling_matrix_free(const std::vector<Gate>& gates, int n_qubits, const SparseMatrix& initial_state, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout) {
    /*
    Execute a sampling functional using a matrix-free simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        initial_state (SparseMatrix&): The initial state of the system (statevector or density matrix).
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during simulation.
        state (DenseMatrix&): The final state after applying all gates (statevector or density matrix).
        intermediate_results (std::vector<py::object>&): A vector to store intermediate results after each set of measurements.
        config (QiliSimConfig): The simulation configuration.
        readout (py::object): A list with readout information to determine when and what to measure.

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

    // If it's a density matrix, check if it's pure
    if (!is_statevector) {
        DenseMatrix state_squared = state.adjoint().cwiseProduct(state);
        double trace_squared = state_squared.trace().real();
        if (std::abs(trace_squared - 1.0) < config.get_atol()) {
            state = get_vector_from_density_matrix(state);
            is_statevector = true;
        }
    }

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
    for (int i = 0; i < int(optimized_gates.size()); ++i) {
        const auto& gate = optimized_gates[i];

        // Convert gate to a matrix-free operator
        MatrixFreeOperator op(gate);

        // If it's a measurement
        if (gate.get_name() == "M") {
            // Check for adjacent measurements and if we have any, do them all at once
            int num_measurements = 0;
            bool are_final_measurements = false;
            std::vector<bool> qubits_to_measure_after_gate(n_qubits, false);
            for (int j = i; j < int(optimized_gates.size()); ++j) {
                if (optimized_gates[j].get_name() == "M") {
                    for (int target : optimized_gates[j].get_target_qubits()) {
                        qubits_to_measure_after_gate[target] = true;
                    }
                    num_measurements++;
                    if (j == int(optimized_gates.size()) - 1) {
                        are_final_measurements = true;
                    }
                } else {
                    break;
                }
            }

            // Construct the result
            if (!are_final_measurements) {
                py::object result = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure_after_gate);
                intermediate_results.push_back(result);

                // If we have measurement_collapse enabled, apply the measurement and collapse the state
                if (config.get_measurement_collapse()) {
                    state = collapse_state(state, qubits_to_measure_after_gate);
                    is_statevector = false;
                }
            }

            // Skip the next measurements since we did them all at once
            i += (num_measurements - 1);
            continue;
        }

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

    // If we started with a density matrix and ended with a statevector, convert back
    if (!initially_was_statevector && is_statevector) {
        state = state * state.adjoint();
    }
}

#include <iostream>

void sampling_stabilizer(const std::vector<Gate>& gates, int n_qubits, const StabilizerStateSum& initial_state, NoiseModelCpp& noise_model_cpp, StabilizerStateSum& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout) {
    /*
    Execute a sampling functional using a stabilizer simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        n_qubits (int): The number of qubits in the circuit.
        initial_state (StabilizerStateSum&): The initial state of the system as a sum of stabilizer states.
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during simulation.
        state (StabilizerStateSum&): The final state after applying all gates as a sum of stabilizer states.
        intermediate_results (std::vector<py::object>&): A vector to store intermediate results after each set of measurements.
        config (QiliSimConfig): The simulation configuration.
        readout (py::object): A list with readout information to determine when and what to measure.
    */

    // We start with the initial state
    state = initial_state;
    state.set_max_terms(config.get_stabilizer_max_states());

    // Then we apply each gate
    std::vector<bool> qubits_measured(n_qubits, false);
    for (int i = 0; i < int(gates.size()); ++i) {
        const auto& gate = gates[i];
        state.apply_gate(gate);
    }
}

// GCOV_EXCL_BR_STOP
