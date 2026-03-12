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

#include "qilisim.h"
#include "../../libs/numpy.h"
#include "../../libs/pybind_types.h"
#include "analog/time_evolution.h"
#include "config/qilisim_config.h"
#include "digital/gate.h"
#include "digital/sampling.h"
#include "noise/noise_model.h"
#include "representations/matrix_free_hamiltonian.h"
#include "utils/parsers.h"

// The public execute_sampling
py::object QiliSimCpp::execute_sampling(const py::object& functional, const py::object& noise_model, const py::object& initial_state, const py::dict& solver_params) {
    /*
    Execute a sampling functional using a simple statevector simulator.
    Note that this is just the wrapper mapping the Python objects to C++ objects.
    For the actual implementation, see the method execute_sampling_internal.

    Args:
        functional (py::object): The Sampling functional to execute.
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
    if (!py::isinstance(functional, Sampling)) {
        throw py::value_error("The provided functional is not a Sampling instance");
    }

    // Get info from the functional
    int n_shots = functional.attr("nshots").cast<int>();
    int n_qubits = functional.attr("circuit").attr("nqubits").cast<int>();

    // Get parameters
    QiliSimConfig config = parse_solver_params(solver_params);

    // Sanity checks
    if (n_qubits <= 0) {
        throw py::value_error("nqubits must be positive.");
    }
    if (n_shots <= 0) {
        throw py::value_error("nshots must be positive.");
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
        sampling_matrix_free(gates, qubits_to_measure, n_qubits, n_shots, initial_state_cpp, noise_model_cpp, state_dense, counts, config);
    } else {
        sampling(gates, qubits_to_measure, n_qubits, n_shots, initial_state_cpp, noise_model_cpp, state_dense, counts, config);
    }

    // Convert counts to samples dict
    py::dict samples;
    for (const auto& pair : counts) {
        samples[py::cast(pair.first)] = py::cast(pair.second);
    }

    // Construct the result object
    py::object result = SamplingResult("nshots"_a = n_shots, "samples"_a = samples);
    py::array final_state_numpy = to_numpy(state_dense);

    return result;
}

// The public execute_time_evolution
py::object QiliSimCpp::execute_time_evolution(const py::object& functional, const py::object& noise_model, const py::dict& solver_params) {
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
    if (!py::isinstance(functional, TimeEvolution)) {
        throw py::value_error("The provided functional is not a TimeEvolution instance");
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
    py::object observables = functional.attr("observables");

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
    std::vector<std::vector<double>> intermediate_expectation_values;
    if (config.get_time_evolution_method() == "integrate_matrix_free") {
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

        // Parse the observables
        std::vector<MatrixFreeHamiltonian> observable_matrices = parse_observables_matrix_free(observables);

        // Call the implementation
        time_evolution_matrix_free(rho_0, hamiltonians, parameters_list, step_list, noise_model_cpp, observable_matrices, config, rho_t, intermediate_rhos, expectation_values, intermediate_expectation_values);

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

        // Parse the observables
        std::vector<SparseMatrix> observable_matrices = parse_observables(observables, nqubits, config.get_atol());

        // Call the implementation
        time_evolution(rho_0, hamiltonians, parameters_list, step_list, noise_model_cpp, observable_matrices, config, rho_t, intermediate_rhos, expectation_values, intermediate_expectation_values);
    }

    // Convert things to numpy arrays
    py::array_t<std::complex<double>> rho_numpy = to_numpy(rho_t);
    py::array_t<double> expect_numpy = to_numpy(expectation_values);

    // Also convert intermediates if needed
    py::list intermediate_rho_numpy;
    py::array_t<double> intermediate_expect_numpy;
    if (config.get_store_intermediate_results()) {
        for (const auto& rho_intermediate : intermediate_rhos) {
            py::array_t<std::complex<double>> rho_step_numpy = to_numpy(rho_intermediate);
            intermediate_rho_numpy.append(QTensor(rho_step_numpy));
        }
        intermediate_expect_numpy = to_numpy(intermediate_expectation_values);
    }

    // Return a TimeEvolutionResult with these
    return TimeEvolutionResult("final_state"_a = QTensor(rho_numpy), "final_expected_values"_a = expect_numpy, "intermediate_states"_a = intermediate_rho_numpy, "expected_values"_a = intermediate_expect_numpy);
}