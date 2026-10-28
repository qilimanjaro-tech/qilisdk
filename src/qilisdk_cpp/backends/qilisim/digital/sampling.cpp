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

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

#include "../../../libs/logging.h"
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

    // For each measured qubit, apply the non-selective measurement: rho -> P0 * rho * P0 + P1 * rho * P1
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

static int group_measurements(const std::vector<Gate>& gates, int start, std::vector<bool>& qubits_to_measure) {
    /*
    Group the run of adjacent measurement gates starting at a given index, so they can be done all at once.

    Args:
        gates (std::vector<Gate>&): The gates of the circuit.
        start (int): The index of the first measurement gate in the run.
        qubits_to_measure (std::vector<bool>&): Set to true for every qubit measured in the run.

    Returns:
        int: The number of adjacent measurement gates in the run.
    */
    int num_measurements = 0;
    for (int j = start; j < int(gates.size()) && gates[j].get_name() == "M"; ++j) {
        for (int target : gates[j].get_target_qubits()) {
            qubits_to_measure[target] = true;
        }
        num_measurements++;
    }
    return num_measurements;
}

void sampling(const std::vector<Gate>& gates, int n_qubits, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout, bool* output_is_trajectories) {
    /*
    Execute a sampling functional using a simple statevector simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during simulation.
        state (DenseMatrix&): On entry the initial state, on exit the final state after applying all gates (statevector or density matrix).
        intermediate_results (std::vector<py::object>&): A vector to store the intermediate results after each measurement.
        config (QiliSimConfig): The simulation configuration.
        readout (py::object): A list with readout information to determine when and what to measure.
        output_is_trajectories (bool*): Optional output parameter. If given, a Monte Carlo ensemble is
            returned as a batch of state vector columns instead of being averaged into a density
            matrix, and the flag is set to true when that is what `state` holds.

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

    long dim = 1L << n_qubits;
    bool is_statevector = (state.cols() == 1 && state.rows() == dim);
    bool initially_was_statevector = is_statevector;
    qilisdk::log_debug("[Sampling, C++] Preparing initial state for " + std::to_string(n_qubits) + " qubits (" + (is_statevector ? "statevector" : "density matrix") + ")");

    // Check if the input is a bunch of monte carlo trajectories
    bool input_is_trajectories = (state.cols() > 1 && state.rows() != state.cols());

    // If it's a density matrix, check if it's pure
    if (!is_statevector && !input_is_trajectories) {
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
        qilisdk::log_debug("[Sampling, C++] Noise model present, promoting statevector to density matrix");
        state = state * state.adjoint();
        is_statevector = false;
    }

    // Whether we should do monte-carlo sampling
    bool monte_carlo = input_is_trajectories || (!is_statevector && config.get_monte_carlo());
    if (monte_carlo && !input_is_trajectories) {
        qilisdk::log_debug("[Sampling, C++] Monte-Carlo sampling with " + std::to_string(config.get_num_monte_carlo_trajectories()) + " trajectories");
        state = sample_from_density_matrix(state, config.get_num_monte_carlo_trajectories(), config.next_seed());
    }

    // Whether the columns of `state` are currently Monte Carlo trajectories
    bool state_is_trajectories = monte_carlo;
    bool keep_trajectories = (output_is_trajectories != nullptr);
    if (keep_trajectories) {
        *output_is_trajectories = false;
    }

    // Combine single-qubit gates for speed if not using noise models
    std::vector<Gate> optimized_gates = gates;
    if (!has_noise && config.get_combine_single_qubit_gates()) {
        optimized_gates = combine_single_qubit_gates(optimized_gates);
        qilisdk::log_debug("[Sampling, C++] Combined single-qubit gates: " + std::to_string(gates.size()) + " -> " + std::to_string(optimized_gates.size()) + " gates");
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
    qilisdk::log_debug("[Sampling, C++] Pre-caching " + std::to_string(initial_cache_size) + " of " + std::to_string(optimized_gates.size()) + " gate matrices");
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

        // Let Ctrl-C interrupt the circuit between gates
        check_signals();

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

            qilisdk::log_trace("[Sampling, C++] Measuring " + std::to_string(num_measurements) + " qubits at gate index " + std::to_string(i));

            // Construct the result
            if (!are_final_measurements) {
                py::object result = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure_after_gate, state_is_trajectories);
                intermediate_results.push_back(result);

                // If we have measurement_collapse enabled, apply the measurement and collapse the state
                if (config.get_measurement_collapse()) {
                    // If we have trajectories, convert to a density matrix first
                    if (state_is_trajectories) {
                        state = trajectories_to_density_matrix(state);
                        state_is_trajectories = false;
                    }

                    // Collapse the state based on the measurement result
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

        qilisdk::log_trace("[Sampling, C++] Applying gate " + gate.get_id() + (gate_is_new ? " (cache miss)" : " (cache hit)"));

        // If it will be used again later and we have space, cache it
        if (gate_is_new && gate_first_last_use[gate_id].second > gate_count && int(gate_cache.size()) < config.get_max_cache_size()) {
            gate_cache[gate_id] = gate_matrix;
        }

        // Apply the gate (Sparse-Dense multiplication, already OpenMP parallel if enabled)
        if (is_statevector || state_is_trajectories) {
            state = gate_matrix * state;
        } else {
            state = gate_matrix * state * gate_matrix.adjoint();
        }

        // Apply any relevant Kraus operators
        if (has_noise) {
            for (const auto& operator_set : noise_model_cpp.get_relevant_kraus_operators(gate.get_name(), static_cast<int>(gate.get_control_qubits().size()), gate.get_qubits(), n_qubits)) {
                DenseMatrix new_state(state.rows(), state.cols());
                new_state.setZero();
                for (const auto& K : operator_set) {
                    new_state += K * state * K.adjoint();
                }
                state = new_state;
            }
        }

        // Renormalize the state
        if (config.get_normalize_state() && config.get_normalize_after_gate()) {
            normalize_state(state, is_statevector, state_is_trajectories);
        }

        // Clear the gate from the cache if this was its last use
        if (gate_first_last_use[gate_id].second <= gate_count) {
            gate_cache.erase(gate_id);
        }
        gate_count++;
    }

    // If we have statevectors but we should return a density matrix, unless we are keeping the trajectories
    if (state_is_trajectories && keep_trajectories) {
        *output_is_trajectories = true;
    } else if (state_is_trajectories && !input_is_trajectories) {
        state = trajectories_to_density_matrix(state);
    }

    // If we started with a density matrix and ended with a statevector, convert back
    if (!initially_was_statevector && is_statevector) {
        state = state * state.adjoint();
    }

    // Make sure we haven't diverged
    check_state_diverged(state);

    qilisdk::log_debug("[Sampling, C++] Applied " + std::to_string(gate_count) + " gates, circuit sampling complete");
}

void sampling_matrix_free(const std::vector<Gate>& gates, int n_qubits, NoiseModelCpp& noise_model_cpp, DenseMatrix& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout, bool* output_is_trajectories) {
    /*
    Execute a sampling functional using a matrix-free simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        n_qubits (int): The number of qubits in the circuit.
        n_shots (int): The number of shots to sample.
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during simulation.
        state (DenseMatrix&): On entry the initial state, on exit the final state after applying all gates (statevector or density matrix).
        intermediate_results (std::vector<py::object>&): A vector to store intermediate results after each set of measurements.
        config (QiliSimConfig): The simulation configuration.
        readout (py::object): A list with readout information to determine when and what to measure.
        output_is_trajectories (bool*): Optional output parameter. If given, a Monte Carlo ensemble is
            returned as a batch of state vector columns instead of being averaged into a density
            matrix, and the flag is set to true when that is what `state` holds.

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

    bool is_statevector = (state.cols() == 1 && state.rows() == (1L << n_qubits));
    bool initially_was_statevector = is_statevector;
    qilisdk::log_debug("[Sampling, C++] Preparing initial state for " + std::to_string(n_qubits) + " qubits (matrix-free, " + (is_statevector ? "statevector" : "density matrix") + ")");

    // Check if we have noise
    bool has_noise = !noise_model_cpp.is_empty();

    // Check if the input is a bunch of monte carlo trajectories
    bool input_is_trajectories = (state.cols() > 1 && state.rows() != state.cols());

    // If it's a density matrix, check if it's pure
    if (!is_statevector && !input_is_trajectories) {
        DenseMatrix state_squared = state.adjoint().cwiseProduct(state);
        double trace_squared = state_squared.trace().real();
        if (std::abs(trace_squared - 1.0) < config.get_atol()) {
            state = get_vector_from_density_matrix(state);
            is_statevector = true;
        }
    }

    // If we have noise but start with a statevector, convert to density matrix
    if (has_noise && is_statevector) {
        qilisdk::log_debug("[Sampling, C++] Noise model present, promoting statevector to density matrix");
        state = state * state.adjoint();
        is_statevector = false;
    }

    // Whether we should do monte-carlo sampling
    bool monte_carlo = input_is_trajectories || (!is_statevector && config.get_monte_carlo());
    if (monte_carlo && !input_is_trajectories) {
        qilisdk::log_debug("[Sampling, C++] Monte-Carlo sampling with " + std::to_string(config.get_num_monte_carlo_trajectories()) + " trajectories");
        state = sample_from_density_matrix(state, config.get_num_monte_carlo_trajectories(), config.next_seed());
    }

    // Whether the columns of `state` are currently Monte Carlo trajectories
    bool state_is_trajectories = monte_carlo;
    bool keep_trajectories = (output_is_trajectories != nullptr);
    if (keep_trajectories) {
        *output_is_trajectories = false;
    }

    // Fuse gates for speed if not using noise models, either in groups of single-qubit gates or multi-qubit gates (up to a limit)
    const int min_threads_for_fusion = 4;
    bool use_fusion = !has_noise && config.get_fuse_gates() && (is_statevector || monte_carlo) && config.get_num_threads() >= min_threads_for_fusion;
    std::vector<Gate> optimized_gates = gates;
    if (use_fusion) {
        // A configured value of <= 0 means auto pick the depth from the qubit count
        int max_fused = config.get_max_fused_qubits();
        if (max_fused <= 0) {
            max_fused = auto_max_fused_qubits(n_qubits);
        }
        // Pre-combine consecutive single-qubit gates on the same qubit first
        if (config.get_combine_single_qubit_gates()) {
            optimized_gates = fuse_gates(combine_single_qubit_gates(gates), max_fused);
        } else {
            optimized_gates = fuse_gates(gates, max_fused);
        }
    } else if (!has_noise && config.get_combine_single_qubit_gates()) {
        optimized_gates = combine_single_qubit_gates(optimized_gates);
        qilisdk::log_debug("[Sampling, C++] Combined single-qubit gates: " + std::to_string(gates.size()) + " -> " + std::to_string(optimized_gates.size()) + " gates");
    }

    // Apply each gate
    for (int i = 0; i < int(optimized_gates.size()); ++i) {
        const auto& gate = optimized_gates[i];

        // Let Ctrl-C interrupt the circuit between gates
        check_signals();

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
                py::object result = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure_after_gate, state_is_trajectories);
                intermediate_results.push_back(result);

                // If we have measurement_collapse enabled, apply the measurement and collapse the state
                if (config.get_measurement_collapse()) {
                    // If we have trajectories, convert to a density matrix first
                    if (state_is_trajectories) {
                        state = trajectories_to_density_matrix(state);
                        state_is_trajectories = false;
                    }

                    // Collapse the state based on the measurement result
                    state = collapse_state(state, qubits_to_measure_after_gate);
                    is_statevector = false;
                }
            }

            qilisdk::log_trace("[Sampling, C++] Measuring " + std::to_string(num_measurements) + " qubits at gate index " + std::to_string(i));

            // Skip the next measurements since we did them all at once
            i += (num_measurements - 1);
            continue;
        }

        qilisdk::log_trace("[Sampling, C++] Applying gate " + gate.get_id() + " (matrix-free)");

        // Apply the gate
        if (state_is_trajectories) {
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
            for (const auto& operator_set : noise_model_cpp.get_relevant_kraus_operators(gate.get_name(), static_cast<int>(gate.get_control_qubits().size()), gate.get_qubits(), n_qubits)) {
                DenseMatrix new_state(state.rows(), state.cols());
                new_state.setZero();
                for (const auto& K : operator_set) {
                    new_state += K * state * K.adjoint();
                }
                state = new_state;
            }
        }

        // Renormalize the state
        if (config.get_normalize_state() && config.get_normalize_after_gate()) {
            normalize_state(state, is_statevector, state_is_trajectories);
        }
    }

    // If we have statevectors but we should return a density matrix, unless we are keeping the trajectories
    if (state_is_trajectories && keep_trajectories) {
        *output_is_trajectories = true;
    } else if (state_is_trajectories && !input_is_trajectories) {
        state = trajectories_to_density_matrix(state);
    }

    // If we started with a density matrix and ended with a statevector, convert back
    if (!initially_was_statevector && is_statevector) {
        state = state * state.adjoint();
    }

    // Make sure we haven't diverged
    check_state_diverged(state);

    qilisdk::log_debug("[Sampling, C++] Applied " + std::to_string(optimized_gates.size()) + " gates, matrix-free circuit sampling complete");
}

#include <iostream>

void sampling_stabilizer(const std::vector<Gate>& gates, int n_qubits, const StabilizerStateSum& initial_state, NoiseModelCpp& noise_model_cpp, StabilizerStateSum& state, const QiliSimConfig& config, const py::object& readout) {
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

    // Set the number of threads
#if defined(_OPENMP)
    omp_set_num_threads(config.get_num_threads());
    Eigen::setNbThreads(config.get_num_threads());
#endif

    // Throw a warning if we have noise, since the stabilizer simulator does not support noise models
    if (!noise_model_cpp.is_empty()) {
        warning("Noise models are not supported in the stabilizer simulator. Noise will be ignored.");
    }

    // Throw a warning if we are using monte-carlo sampling, since the stabilizer simulator does not support it
    if (config.get_monte_carlo()) {
        warning("Monte Carlo sampling is not supported in the stabilizer simulator. Monte Carlo will be ignored.");
    }

    // We start with the initial state
    state = initial_state;
    state.set_max_terms(config.get_stabilizer_max_states());

    // Then we apply each gate
    std::vector<bool> qubits_measured(n_qubits, false);
    for (int i = 0; i < int(gates.size()); ++i) {
        const auto& gate = gates[i];

        // Let Ctrl-C interrupt the circuit between gates
        check_signals();

        state.apply_gate(gate);
    }
}

void sampling_mps(const std::vector<Gate>& gates, int n_qubits, const MPSState& initial_state, NoiseModelCpp& noise_model_cpp, MPSState& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout) {
    /*
    Execute a sampling functional using a matrix product state simulator.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        n_qubits (int): The number of qubits in the circuit.
        initial_state (MPSState&): The initial state of the system as an MPS.
        noise_model_cpp (NoiseModelCpp&): The noise model, of which only readout error is supported.
        state (MPSState&): The final state after applying all gates.
        intermediate_results (std::vector<py::object>&): A vector to store intermediate results after each set of measurements.
        config (QiliSimConfig): The simulation configuration.
        readout (py::object): A list with readout information to determine when and what to measure.

    Raises:
        py::value_error: If the noise model has Kraus or Lindblad terms, or if the state diverges to NaN during the simulation.
    */

    // Set the number of threads
#if defined(_OPENMP)
    omp_set_num_threads(config.get_num_threads());
    Eigen::setNbThreads(config.get_num_threads());
#endif

    // Sanity checks
    if (initial_state.get_nqubits() != n_qubits) {
        throw py::value_error("The initial MPS has " + std::to_string(initial_state.get_nqubits()) + " qubits but the circuit has " + std::to_string(n_qubits) + ".");
    }

    // Kraus and Lindblad noise would make the state mixed, which an MPS cannot hold
    bool has_kraus = !noise_model_cpp.get_kraus_operators_global().empty() || !noise_model_cpp.get_kraus_operators_per_qubit().empty() || !noise_model_cpp.get_kraus_operators_per_gate().empty() || !noise_model_cpp.get_kraus_operators_per_gate_qubit().empty();
    if (has_kraus || !noise_model_cpp.get_jump_operators().empty()) {
        throw py::value_error("The MPS simulator only supports readout error and parameter perturbation noise, not Kraus or Lindblad noise. Use a statevector method instead.");
    }

    // Monte Carlo is not supported
    if (config.get_monte_carlo()) {
        warning("Monte Carlo sampling is not supported in the MPS simulator. Monte Carlo will be ignored.");
    }

    // Throw a warning if we are using measurement collapse, since an MPS cannot represent the resulting mixed state
    if (config.get_measurement_collapse()) {
        warning("Measurement collapse is not supported in the MPS simulator. Measurement collapse will be ignored.");
    }

    // We start with the initial state, with the settings from the config
    state = initial_state;
    state.set_max_bond_dimension(config.get_mps_max_bond_dimension());
    state.set_truncation_cutoff(config.get_mps_truncation_cutoff());

    // Some small circuit optimizations, since we don't have noise
    std::vector<Gate> optimized_gates = gates;
    if (config.get_combine_single_qubit_gates()) {
        optimized_gates = combine_single_qubit_gates(optimized_gates);
        qilisdk::log_debug("[Sampling, C++] Combined single-qubit gates: " + std::to_string(gates.size()) + " -> " + std::to_string(optimized_gates.size()) + " gates");
    }

    // Fused blocks are capped at two qubits, since apply_gate only takes one- and two-qubit gates
    const int mps_max_fused_qubits = 2;
    if (config.get_fuse_gates()) {
        size_t before_fusion = optimized_gates.size();
        optimized_gates = fuse_gates(optimized_gates, mps_max_fused_qubits);
        qilisdk::log_debug("[Sampling, C++] Fused gates: " + std::to_string(before_fusion) + " -> " + std::to_string(optimized_gates.size()) + " gates");
    }

    // Then we apply each gate
    for (int i = 0; i < int(optimized_gates.size()); ++i) {
        const auto& gate = optimized_gates[i];

        // Let Ctrl-C interrupt the circuit between gates
        check_signals();

        // If it's a measurement
        if (gate.get_name() == "M") {
            // Check for adjacent measurements and if we have any, do them all at once
            std::vector<bool> qubits_to_measure_after_gate(n_qubits, false);
            int num_measurements = group_measurements(optimized_gates, i, qubits_to_measure_after_gate);
            bool are_final_measurements = (i + num_measurements == int(optimized_gates.size()));

            // Construct the result
            if (!are_final_measurements) {
                py::object result = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure_after_gate);
                intermediate_results.push_back(result);
            }

            qilisdk::log_trace("[Sampling, C++] Measuring " + std::to_string(num_measurements) + " qubits at gate index " + std::to_string(i));

            // Skip the next measurements since we did them all at once
            i += (num_measurements - 1);
            continue;
        }

        // Apply the gate
        state.apply_gate(gate);

        // Check for nan anywhere the gate touched, before normalizing would hide it
        std::vector<int> qubits = gate.get_qubits();
        int first = *std::min_element(qubits.begin(), qubits.end());
        int last = *std::max_element(qubits.begin(), qubits.end());
        for (int q = first; q <= last; ++q) {
            if (state.get_site_tensor(q).has_nan()) {
                throw py::value_error("State has become NaN while applying gate " + gate.get_id() + ".");
            }
        }

        // Normalize
        if (config.get_normalize_state() && config.get_normalize_after_gate()) {
            state.normalize();
        }
    }

    // Make sure we're normalized at the end
    if (config.get_normalize_state()) {
        state.normalize();
    }

    // Log some debug info
    qilisdk::log_debug("[QiliSim, C++] MPS simulation complete: max bond dimension " + std::to_string(state.get_max_bond_dimension_used()) + " of " + std::to_string(config.get_mps_max_bond_dimension()) + ", accumulated truncation error " + std::to_string(state.get_truncation_error()));
    if (state.get_truncation_error() > MPSState::TRUNCATION_WARNING_THRESHOLD) {
        warning("The MPS simulation discarded a total weight of " + std::to_string(state.get_truncation_error()) + " when truncating, increase mps_max_bond_dimension for a more accurate result.");
    }
}

void sampling_bp(const std::vector<Gate>& gates, int n_qubits, const BPState& initial_state, NoiseModelCpp& noise_model_cpp, BPState& state, std::vector<py::object>& intermediate_results, const QiliSimConfig& config, const py::object& readout) {
    /*
    Execute a digital functional using a tensor network shaped like the circuit's interaction graph,
    gauged and contracted with belief propagation.

    Args:
        gates (std::vector<Gate>&): The list of gates in the circuit.
        n_qubits (int): The number of qubits in the circuit.
        initial_state (BPState&): The initial state of the system as a tensor network.
        noise_model_cpp (NoiseModelCpp&): The noise model, of which only parameter perturbations are supported.
        state (BPState&): The final state after applying all gates.
        intermediate_results (std::vector<py::object>&): A vector to store intermediate results after each set of measurements.
        config (QiliSimConfig): The simulation configuration.
        readout (py::object): A list with readout information to determine when and what to measure.

    Raises:
        py::value_error: If the noise model has Kraus or Lindblad terms, or if the state diverges to NaN during the simulation.
    */

    // Set the number of threads
#if defined(_OPENMP)
    omp_set_num_threads(config.get_num_threads());
    Eigen::setNbThreads(config.get_num_threads());
#endif

    // Sanity checks
    if (initial_state.get_nqubits() != n_qubits) {
        throw py::value_error("The initial tensor network has " + std::to_string(initial_state.get_nqubits()) + " qubits but the circuit has " + std::to_string(n_qubits) + ".");
    }

    // Kraus and Lindblad noise would make the state mixed, which a pure tensor network cannot hold
    bool has_kraus = !noise_model_cpp.get_kraus_operators_global().empty() || !noise_model_cpp.get_kraus_operators_per_qubit().empty() || !noise_model_cpp.get_kraus_operators_per_gate().empty() || !noise_model_cpp.get_kraus_operators_per_gate_qubit().empty();
    if (has_kraus || !noise_model_cpp.get_jump_operators().empty()) {
        throw py::value_error("The BP simulator only supports parameter perturbation noise, not Kraus or Lindblad noise. Use a statevector method instead.");
    }

    // Monte Carlo is not supported
    if (config.get_monte_carlo()) {
        warning("Monte Carlo sampling is not supported in the BP simulator. Monte Carlo will be ignored.");
    }

    // Measurement collapse is not supported either
    if (config.get_measurement_collapse()) {
        warning("Measurement collapse is not supported in the BP simulator. Measurement collapse will be ignored.");
    }

    // We start with the initial state, with the settings from the config
    state = initial_state;
    state.set_max_bond_dimension(config.get_bp_max_bond_dimension());
    state.set_truncation_cutoff(config.get_bp_truncation_cutoff());
    state.set_max_iterations(config.get_bp_max_iterations());
    state.set_tolerance(config.get_bp_tolerance());

    // Some small circuit optimizations, since we don't have noise
    std::vector<Gate> optimized_gates = gates;
    if (config.get_combine_single_qubit_gates()) {
        optimized_gates = combine_single_qubit_gates(optimized_gates);
        qilisdk::log_debug("[Sampling, C++] Combined single-qubit gates: " + std::to_string(gates.size()) + " -> " + std::to_string(optimized_gates.size()) + " gates");
    }

    // Fused blocks are capped at two qubits, since apply_gate only takes one- and two-qubit gates
    const int bp_max_fused_qubits = 2;
    if (config.get_fuse_gates()) {
        size_t before_fusion = optimized_gates.size();
        optimized_gates = fuse_gates(optimized_gates, bp_max_fused_qubits);
        qilisdk::log_debug("[Sampling, C++] Fused gates: " + std::to_string(before_fusion) + " -> " + std::to_string(optimized_gates.size()) + " gates");
    }

    // Then we apply each gate
    for (int i = 0; i < int(optimized_gates.size()); ++i) {
        const auto& gate = optimized_gates[i];

        // Let Ctrl-C interrupt the circuit between gates
        check_signals();

        // If it's a measurement
        if (gate.get_name() == "M") {
            // Check for adjacent measurements and if we have any, do them all at once
            std::vector<bool> qubits_to_measure_after_gate(n_qubits, false);
            int num_measurements = group_measurements(optimized_gates, i, qubits_to_measure_after_gate);
            bool are_final_measurements = (i + num_measurements == int(optimized_gates.size()));

            // Construct the result
            if (!are_final_measurements) {
                state.run_belief_propagation();
                py::object result = construct_result_object(state, readout, n_qubits);
                intermediate_results.push_back(result);
            }

            // Skip the next measurements since we did them all at once
            i += (num_measurements - 1);
            continue;
        }

        // Apply the gate and check for nan on the qubits it touched
        state.apply_gate(gate);
        for (int q : gate.get_qubits()) {
            if (state.get_site_tensor(q).has_nan()) {
                throw py::value_error("State has become NaN while applying gate " + gate.get_id() + ".");
            }
        }
    }

    // The readout estimates rely on converged messages
    state.run_belief_propagation();

    // Log some debug info
    qilisdk::log_debug("[QiliSim, C++] BP simulation complete: " + std::to_string(state.get_num_edges()) + " edges, max bond dimension " + std::to_string(state.get_max_bond_dimension_used()) + " of " + std::to_string(config.get_bp_max_bond_dimension()) + ", accumulated truncation error " + std::to_string(state.get_truncation_error()) + ", " + std::to_string(state.get_bp_runs()) + " BP runs taking " + std::to_string(state.get_bp_iterations()) + " sweeps");
    if (state.get_truncation_error() > BPState::TRUNCATION_WARNING_THRESHOLD) {
        warning("The BP simulation discarded a total weight of " + std::to_string(state.get_truncation_error()) + " when truncating, increase bp_max_bond_dimension for a more accurate result.");
    }
    if (!state.get_bp_converged()) {
        warning("Belief propagation did not always converge within bp_max_iterations, so the results may be inaccurate.");
    }
}

// GCOV_EXCL_BR_STOP
