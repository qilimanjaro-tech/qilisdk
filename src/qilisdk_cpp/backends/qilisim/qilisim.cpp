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

#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>

#include "../../libs/numpy.h"
#include "analog/time_evolution.h"
#include "config/qilisim_config.h"
#include "digital/gate.h"
#include "digital/sampling.h"
#include "noise/noise_model.h"
#include "qilisim.h"
#include "representations/matrix_free_hamiltonian.h"
#include "utils/parsers.h"
#include "utils/random.h"
#include "utils/sample.h"

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

py::object construct_result_object(const DenseMatrix& state_dense, const py::object& readout, NoiseModelCpp& noise_model_cpp, int n_qubits, const QiliSimConfig& config, const std::vector<bool>& qubits_to_measure) {
    py::list results;
    py::array final_state_numpy = to_numpy(state_dense);

    for (py::handle ro_handle : readout) {
        py::object ro = py::reinterpret_borrow<py::object>(ro_handle);

        if (py::isinstance(ro, StateTomographyReadout)) {
            // Check state_tomography_method != "exact"
            std::string method = ro.attr("method").cast<std::string>();
            if (method != "exact") {
                throw py::value_error("State Tomography methods that are not exact are not supported yet.");
            }
            results.append(StateTomographyReadoutResult("state"_a = QTensor(final_state_numpy)));

        } else if (py::isinstance(ro, ExpectationReadout)) {
            results.append(ExpectationReadoutResult.attr("from_state")("expectation_readout"_a = py::module_::import("copy").attr("copy")(ro), "state"_a = QTensor(final_state_numpy)));

        } else if (py::isinstance(ro, SamplingReadout)) {
            int n_shots = ro.attr("nshots").cast<int>();
            std::map<std::string, int> counts = construct_samples(state_dense, n_qubits, n_shots, noise_model_cpp, config, qubits_to_measure);
            py::dict samples;
            for (const auto& pair : counts) {
                samples[py::cast(pair.first)] = py::cast(pair.second);
            }
            results.append(SamplingReadoutResult.attr("from_samples")("samples"_a = samples));

        } else {
            // Replicate: raise ValueError(f"Unsupported Readout Method provided: {ro}")
            std::string ro_repr = py::repr(ro).cast<std::string>();
            throw py::value_error("Unsupported Readout Method provided: " + ro_repr);
        }
    }

    return ReadoutCompositeResults.attr("from_list")(results);
}

// The public execute_sampling
py::object QiliSimCpp::execute_digital_propagation(const py::object& functional, const py::object& readout, const py::object& noise_model, const py::object& initial_state, const py::dict& solver_params) {
    /*
    Execute a sampling functional using a simple statevector simulator.
    Note that this is just the wrapper mapping the Python objects to C++ objects.
    For the actual implementation, see the method execute_sampling_internal.

    Args:
        functional (py::object): The Sampling functional to execute.
        readout (py::object): A list with readout
        noise_model (py::object): The noise model to apply during simulation.
        initial_state (py::object): The initial state as a QTensor or none.
        solver_params (py::dict): Solver parameters, including 'max_cache_size'.

    Returns:
        SamplingResult: A result object containing the measurement samples and computed probabilities.

    Raises:
        py::value_error: If functional is not a Sampling instance.
        py::value_error: If nqubits is non-positive.
        py::value_error: If shots is non-positive.
    */

    // Ensure that the functional is of the correct type
    if (!py::isinstance(functional, DigitalPropagation)) {
        throw py::value_error("The provided functional is not a DigitalPropagation instance");
    }

    // Get info from the functional
    int n_qubits = functional.attr("circuit").attr("nqubits").cast<int>();

    // Get parameters
    QiliSimConfig config = parse_solver_params(solver_params);

    // Sanity checks
    if (n_qubits <= 0) {
        throw py::value_error("nqubits must be positive.");
    }

    // Parse the Python objects into C++ objects
    std::vector<bool> qubits_to_measure = parse_measurements(functional.attr("circuit"));
    NoiseModelCpp noise_model_cpp = parse_noise_model(noise_model, n_qubits, config.get_atol());
    std::vector<Gate> gates = parse_gates(functional.attr("circuit"), config.get_atol(), noise_model);

    // If we have any exponential gates, we need to force renormalization
    for (const auto& gate : gates) {
        if (!gate.is_normalized()) {
            config.set_normalize_after_gate(true);
            break;
        }
    }

    // Pass everything to the internal implementation
    std::map<std::string, int> counts;
    DenseMatrix state_dense;
    SparseMatrix initial_state_cpp;
    if (initial_state.is_none()) {
        long dim = 1L << n_qubits;
        initial_state_cpp = SparseMatrix(dim, 1);
        initial_state_cpp.coeffRef(0, 0) = 1.0;
        initial_state_cpp.makeCompressed();
    } else {
        initial_state_cpp = parse_initial_state(initial_state, config.get_atol());
    }
    if (config.get_sampling_method() == "statevector_matrix_free") {
        sampling_matrix_free(gates, n_qubits, initial_state_cpp, noise_model_cpp, state_dense, config);
    } else {
        sampling(gates, n_qubits, initial_state_cpp, noise_model_cpp, state_dense, config);
    }

    // Construct the result object
    py::object result = construct_result_object(state_dense, readout, noise_model_cpp, n_qubits, config, qubits_to_measure);

    return FunctionalResult("readout_results"_a = result);
}

// The public execute_time_evolution
py::object QiliSimCpp::execute_analog_evolution(const py::object& functional, const py::object& readout, const py::object& noise_model, const py::dict& solver_params) {
    /*
    Execute a time evolution functional.
    Note that this is just the wrapper mapping the Python objects to C++ objects.

    Args:
        functional (py::object): The TimeEvolution functional to execute.
        noise_model (py::object): The noise model to apply during evolution.
        params (py::dict): Additional parameters for the method. See the Python wrapper for details.

    Returns:
        TimeEvolutionResult: The results of the evolution.

    Raises:
        py::value_error: If the functional is not a TimeEvolution instance.
        py::value_error: If no Hamiltonians are provided.
        py::value_error: If no time steps are provided.
        py::value_error: If number of parameters for any Hamiltonian does not match number of time steps.
        py::value_error: If an unknown time evolution method is specified.
    */

    // Ensure that the functional is of the correct type
    if (!py::isinstance(functional, AnalogEvolution)) {
        throw py::value_error("The provided functional is not a AnalogEvolution instance");
    }

    // Check if we need to perturb the parameters
    py::object schedule = functional.attr("schedule");
    if (!noise_model.is_none()) {
        py::dict schedule_parameters = schedule.attr("get_parameters")();
        py::dict global_noise_map = noise_model.attr("global_perturbations");
        for (auto item : global_noise_map) {
            py::handle param_name = item.first;
            if (schedule_parameters.contains(param_name)) {
                for (auto perturbation : global_noise_map[param_name]) {
                    double original_value = schedule_parameters[param_name].cast<double>();
                    double new_value = perturbation.attr("perturb")(original_value).cast<double>();
                    schedule_parameters[param_name] = new_value;
                }
            }
        }
        schedule.attr("set_parameters")(schedule_parameters);
        schedule_parameters = schedule.attr("get_parameters")();
    }

    // Pre-process the Python objects
    py::object initial_state = functional.attr("initial_state");
    py::object hamiltonians_full = functional.attr("schedule").attr("hamiltonians");
    py::list hamiltonians_keys = hamiltonians_full.attr("keys")();
    py::list hamiltonians_values = hamiltonians_full.attr("values")();
    py::object steps = functional.attr("schedule").attr("tlist");

    // Get parameters
    QiliSimConfig config = parse_solver_params(solver_params);
    if (functional.attr("store_intermediate_results").cast<bool>()) {
        config.set_store_intermediate_results(true);
    }

    // Common between methods
    SparseMatrix rho_0 = parse_initial_state(initial_state, config.get_atol());
    int nqubits = static_cast<int>(std::log2(rho_0.rows()));
    NoiseModelCpp noise_model_cpp = parse_noise_model(noise_model, nqubits, config.get_atol());
    std::vector<std::vector<double>> parameters_list = parse_coefficients(schedule, hamiltonians_keys, steps);
    std::vector<double> step_list = parse_time_steps(steps);

    // Depending on the method, call the internal implementation
    std::vector<DenseMatrix> intermediate_rhos;
    DenseMatrix rho_t;
    std::vector<double> expectation_values;
    if (config.get_time_evolution_method() == "integrate_rk4_matrix_free" || config.get_time_evolution_method() == "integrate_rk45_matrix_free") {
        // Parse the Hamiltonians
        std::vector<MatrixFreeHamiltonian> hamiltonians = parse_hamiltonians_matrix_free(hamiltonians_values);
        if (hamiltonians.size() != parameters_list.size()) {
            throw py::value_error("Number of Hamiltonians does not match number of parameter lists");
        }

        // Call the implementation
        time_evolution_matrix_free(rho_0, hamiltonians, parameters_list, step_list, noise_model_cpp, config, rho_t, intermediate_rhos);

    } else {
        // Parse the Hamiltonians
        std::vector<SparseMatrix> hamiltonians = parse_hamiltonians(hamiltonians_values, config.get_atol());
        if (hamiltonians.size() != parameters_list.size()) {
            throw py::value_error("Number of Hamiltonians does not match number of parameter lists");
        }

        // Call the implementation
        time_evolution(rho_0, hamiltonians, parameters_list, step_list, noise_model_cpp, config, rho_t, intermediate_rhos);
    }

    // Construct the result object
    int n_qubits = functional.attr("schedule").attr("nqubits").cast<int>();
    std::vector<bool> qubits_to_measure(n_qubits, true);
    py::object result = construct_result_object(rho_t, readout, noise_model_cpp, n_qubits, config, qubits_to_measure);

    bool store_intermediate_results = functional.attr("store_intermediate_results").cast<bool>();
    // If we have intermediates, process them too
    if (store_intermediate_results) {
        py::list inter_results;
        for (size_t step = 0; step < intermediate_rhos.size(); ++step) {
            auto& rho_intermediate = intermediate_rhos[step];
            inter_results.append(construct_result_object(rho_intermediate, readout, noise_model_cpp, n_qubits, config, qubits_to_measure));
        }
        return FunctionalResult("readout_results"_a = result, "intermediate_results"_a = inter_results);
    }
    return FunctionalResult("readout_results"_a = result);
}

// The public execute_sampling
py::object QiliSimCpp::execute_quantum_reservoir(const py::object& functional, const py::object& readout, const py::object& noise_model, const py::dict& solver_params) {
    /*
    Execute a sampling functional using a simple statevector simulator.
    Note that this is just the wrapper mapping the Python objects to C++ objects.
    For the actual implementation, see the method execute_sampling_internal.

    Args:
        functional (py::object): The Sampling functional to execute.
        readout (py::object): A list with readout
        noise_model (py::object): The noise model to apply during simulation.
        initial_state (py::object): The initial state as a QTensor or none.
        solver_params (py::dict): Solver parameters, including 'max_cache_size'.

    Returns:
        SamplingResult: A result object containing the measurement samples and computed probabilities.

    Raises:
        py::value_error: If functional is not a Sampling instance.
        py::value_error: If nqubits is non-positive.
        py::value_error: If shots is non-positive.
    */

    // Ensure that the functional is of the correct type
    if (!py::isinstance(functional, QuantumReservoir)) {
        throw py::value_error("The provided functional is not a QuantumReservoir instance");
    }

    // Get info from the functional
    int n_qubits = functional.attr("nqubits").cast<int>();

    // Get parameters
    QiliSimConfig config = parse_solver_params(solver_params);

    // Sanity checks
    if (n_qubits <= 0) {
        throw py::value_error("nqubits must be positive.");
    }

    // Parse the Python objects into C++ objects
    std::vector<bool> qubits_to_measure(n_qubits, true);
    NoiseModelCpp noise_model_cpp = parse_noise_model(noise_model, n_qubits, config.get_atol());

    py::object initial_state = functional.attr("initial_state");
    SparseMatrix rho_0 = parse_initial_state(initial_state, config.get_atol());
    // Ensure state is always a density matrix (matching Python's to_density_matrix())
    DenseMatrix state;
    if (rho_0.cols() == 1) {
        DenseMatrix ket = DenseMatrix(rho_0);
        state = ket * ket.adjoint();
    } else {
        state = DenseMatrix(rho_0);
    }

    py::list inter_results;
    for (py::handle input_handler : functional.attr("input_per_layer")) {
        py::object input_dict = py::reinterpret_borrow<py::object>(input_handler);
        functional.attr("reservoir_layer").attr("set_parameters")(input_dict);
        for (py::handle step_handler : functional.attr("reservoir_layer")) {
            py::object step = py::reinterpret_borrow<py::object>(step_handler);

            if (py::isinstance(step, Circuit)) {
                std::vector<Gate> gates = parse_gates(step, config.get_atol(), noise_model);
                // If we have any exponential gates, we need to force renormalization
                for (const auto& gate : gates) {
                    if (!gate.is_normalized()) {
                        config.set_normalize_after_gate(true);
                        break;
                    }
                }

                // Pass everything to the internal implementation
                std::map<std::string, int> counts;
                if (config.get_sampling_method() == "statevector_matrix_free") {
                    sampling_matrix_free(gates, n_qubits, state.sparseView(), noise_model_cpp, state, config);
                } else {
                    sampling(gates, n_qubits, state.sparseView(), noise_model_cpp, state, config);
                }
            } else if (py::isinstance(step, Schedule)) {
                // Check if we need to perturb the parameters
                if (!noise_model.is_none()) {
                    py::dict schedule_parameters = step.attr("get_parameters")();
                    py::dict global_noise_map = noise_model.attr("global_perturbations");
                    for (auto item : global_noise_map) {
                        py::handle param_name = item.first;
                        if (schedule_parameters.contains(param_name)) {
                            for (auto perturbation : global_noise_map[param_name]) {
                                double original_value = schedule_parameters[param_name].cast<double>();
                                double new_value = perturbation.attr("perturb")(original_value).cast<double>();
                                schedule_parameters[param_name] = new_value;
                            }
                        }
                    }
                    step.attr("set_parameters")(schedule_parameters);
                    schedule_parameters = step.attr("get_parameters")();
                }

                py::object hamiltonians_full = step.attr("hamiltonians");
                py::list hamiltonians_keys = hamiltonians_full.attr("keys")();
                py::list hamiltonians_values = hamiltonians_full.attr("values")();
                py::object steps = step.attr("tlist");

                std::vector<std::vector<double>> parameters_list = parse_coefficients(step, hamiltonians_keys, steps);
                std::vector<double> step_list = parse_time_steps(steps);

                // Depending on the method, call the internal implementation
                std::vector<DenseMatrix> intermediate_rhos;
                if (config.get_time_evolution_method() == "integrate_rk4_matrix_free") {
                    // Parse the Hamiltonians
                    std::vector<MatrixFreeHamiltonian> hamiltonians = parse_hamiltonians_matrix_free(hamiltonians_values);
                    if (hamiltonians.size() != parameters_list.size()) {
                        throw py::value_error("Number of Hamiltonians does not match number of parameter lists");
                    }
                    for (size_t h_ind = 0; h_ind < hamiltonians.size(); ++h_ind) {
                        if (parameters_list[h_ind].size() != step_list.size()) {
                            throw py::value_error("Number of parameters for Hamiltonian " + std::to_string(h_ind) + " does not match number of time steps");
                        }
                    }

                    // Call the implementation
                    time_evolution_matrix_free(state.sparseView(), hamiltonians, parameters_list, step_list, noise_model_cpp, config, state, intermediate_rhos);

                } else {
                    // Parse the Hamiltonians
                    std::vector<SparseMatrix> hamiltonians = parse_hamiltonians(hamiltonians_values, config.get_atol());
                    if (hamiltonians.size() != parameters_list.size()) {
                        throw py::value_error("Number of Hamiltonians does not match number of parameter lists");
                    }
                    for (size_t h_ind = 0; h_ind < hamiltonians.size(); ++h_ind) {
                        if (parameters_list[h_ind].size() != step_list.size()) {
                            throw py::value_error("Number of parameters for Hamiltonian " + std::to_string(h_ind) + " does not match number of time steps");
                        }
                    }

                    // Call the implementation
                    time_evolution(state.sparseView(), hamiltonians, parameters_list, step_list, noise_model_cpp, config, state, intermediate_rhos);
                }
            }
        }
        // Ensure state is a density matrix after each layer (matching Python's to_density_matrix())
        if (state.cols() == 1) {
            state = state * state.adjoint();
        }

        inter_results.append(construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure));
        if (!functional.attr("reservoir_layer").attr("qubits_to_reset").is_none()) {
            std::set<int> qubits_set;
            for (const auto& item : functional.attr("reservoir_layer").attr("qubits_to_reset")) {
                qubits_set.insert(item.cast<int>());
            }

            if (!qubits_set.empty()) {
                uint64_t reset_mask = 0ULL;
                for (int q : qubits_set) {
                    if (q < 0 || q >= n_qubits) {
                        throw py::value_error("Invalid qubit indices in qubits_to_reset");
                    }
                    reset_mask |= (1ULL << q);
                }

                DenseMatrix reset_rho = DenseMatrix::Zero(state.rows(), state.cols());

                for (Eigen::Index row = 0; row < state.rows(); ++row) {
                    for (Eigen::Index col = 0; col < state.cols(); ++col) {
                        if (state(row, col) == std::complex<double>(0.0, 0.0)) {
                            continue;
                        }

                        const uint64_t urow = static_cast<uint64_t>(row);
                        const uint64_t ucol = static_cast<uint64_t>(col);

                        if (((urow ^ ucol) & reset_mask) != 0ULL) {
                            continue;
                        }

                        const Eigen::Index out_row = static_cast<Eigen::Index>(urow & ~reset_mask);
                        const Eigen::Index out_col = static_cast<Eigen::Index>(ucol & ~reset_mask);

                        reset_rho(out_row, out_col) += state(row, col);
                    }
                }

                state = std::move(reset_rho);
            }
        }
    }

    py::slice slice(0, -1, 1);

    return FunctionalResult(py::arg("readout_results") = inter_results[py::int_(-1)], py::arg("intermediate_results") = inter_results[slice]);
}

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
