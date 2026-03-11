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
#include "analog/time_evolution.h"
#include "config/qilisim_config.h"
#include "digital/gate.h"
#include "digital/sampling.h"
#include "utils/matrix_utils.h"
#include "utils/numpy.h"
#include "utils/parsers.h"
#include "utils/random.h"
#include "noise/noise_model.h"
#include "representations/matrix_free_hamiltonian.h"
#include <cmath>
#include <iomanip>
#include <sstream>



py::dict samples_from_state(
    const DenseMatrix& state,
    NoiseModelCpp& noise_model_cpp, 
    int n_qubits, 
    int n_shots, 
    std::map<std::string, int>& counts, 
    const std::vector<bool>& qubits_to_measure, 
    const QiliSimConfig& config
){
    long dim = 1L << n_qubits;
    bool has_noise = !noise_model_cpp.is_empty();
    bool is_statevector = (state.cols() == 1 && state.rows() == dim);

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
                    double rand_val = ((double) rand() / (RAND_MAX));
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
        counts = noisy_counts;
    }

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
    py::dict samples;
    for (const auto& pair : counts) {
        samples[py::cast(pair.first)] = py::cast(pair.second);
    }

    return samples;
}

py::object copy_readout(const py::object& readout) {
    if (py::hasattr(readout, "model_copy")) {
        return readout.attr("model_copy")();
    }
    return py::module_::import("copy").attr("copy")(readout);
}

std::vector<int> mask_to_list(std::vector<bool> qubits_to_measure ){
    std::vector<int> measured_qubits;
    
    for(unsigned int i = 0; i < qubits_to_measure.size(); ++i){
        if(qubits_to_measure[i]){
            measured_qubits.push_back(i);
        }
    }   
    return measured_qubits;
}


// Make the QiliSimCpp class available in Python, as well as the two main methods
PYBIND11_MODULE(qilisim_module, m) {
    py::class_<QiliSimCpp>(m, "QiliSimCpp")
        .def(py::init<>())
        .def("execute_sampling", &QiliSimCpp::execute_sampling)
        .def("execute_time_evolution", &QiliSimCpp::execute_time_evolution)
        .def("execute_digital_evolution", &QiliSimCpp::execute_digital_evolution)
        .def("execute_analog_evolution", &QiliSimCpp::execute_analog_evolution);
}

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
// The public execute_digital_evolution
py::object QiliSimCpp::execute_digital_evolution(const py::object& functional, const py::object& noise_model, const py::object& initial_state, const py::dict& solver_params) {
    /*
    Execute a digital evolution functional and build FunctionalResult readout objects in C++.
    */

    // Ensure that the functional is of the correct type
    if (!py::isinstance(functional, DigitalEvolution)) {
        throw py::value_error("The provided functional is not a DigitalEvolution instance");
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
    std::vector<int> measured_qubits = mask_to_list(qubits_to_measure);
    bool all_qubits_measured = measured_qubits.size() == static_cast<size_t>(n_qubits);
    NoiseModelCpp noise_model_cpp = parse_noise_model(noise_model, n_qubits, config.get_atol());
    std::vector<Gate> gates = parse_gates(functional.attr("circuit"), config.get_atol(), noise_model);
    

    // The initial state
    long dim = 1L << n_qubits;
    SparseMatrix initial_state_cpp;
    if (initial_state.is_none()) {
        initial_state_cpp = SparseMatrix(dim, 1);
        initial_state_cpp.coeffRef(0, 0) = 1.0;
        initial_state_cpp.makeCompressed();
    } else {
        initial_state_cpp = parse_initial_state(initial_state, config.get_atol());
    }

    // Simulate once to get the final state.
    std::map<std::string, int> counts;
    DenseMatrix state;
    int nshots = 1;
    sampling(gates, qubits_to_measure, n_qubits, nshots, initial_state_cpp, noise_model_cpp, state, counts, config);

    // Build final state object once.
    SparseMatrix final_state_cpp = state.sparseView();
    py::object final_state = QTensor(to_numpy(final_state_cpp));
    if (!all_qubits_measured) {
        py::list measured_qubits_py;
        for (int qubit : measured_qubits) {
            measured_qubits_py.append(py::cast(qubit));
        }
        final_state = final_state.attr("ptrace")(measured_qubits_py);
    }
    
    py::list readout_methods = functional.attr("readout").cast<py::list>();
    // Build readout results.
    py::list readout_results;
    for (auto readout_obj : readout_methods) {
        py::object readout = py::reinterpret_borrow<py::object>(readout_obj);
        if (readout.attr("is_state_tomography")().cast<bool>()) {
            if (readout.attr("readout_method").attr("state_tomography_method").cast<std::string>() != "exact") {
                throw py::value_error("State Tomography methods that are not exact are not supported yet.");
            }
            readout_results.append(StateTomographyReadoutResults("readout"_a = copy_readout(readout.attr("readout_method")), "final_state"_a = final_state));
        }
        if (readout.attr("is_expectation_values")().cast<bool>()) {
            std::vector<SparseMatrix> observable_matrices =
                parse_observables(readout.attr("readout_method").attr("observables"), n_qubits, config.get_atol());
            std::vector<double> expected_values;
            expected_values.reserve(observable_matrices.size());
            for (const auto& observable : observable_matrices) {
                if (state.cols() == 1) {
                    expected_values.push_back(std::real(dot(state, DenseMatrix(observable * state))));
                } else {
                    expected_values.push_back(std::real(dot(DenseMatrix(observable), state)));
                }
            }
            readout_results.append(
                ExpectationReadoutResults("readout"_a = copy_readout(readout.attr("readout_method")), "expected_values"_a = to_numpy(expected_values)));
        }
        if (readout.attr("is_sample")().cast<bool>()) {
            int n_shots = readout.attr("readout_method").attr("nshots").cast<int>();
            py::dict samples =
                samples_from_state(state, noise_model_cpp, n_qubits, n_shots, counts, qubits_to_measure, config);
            readout_results.append(
                SamplingReadoutResults("readout"_a = copy_readout(readout.attr("readout_method")),
                               "samples"_a = samples));
        }
    }

    return FunctionalResult("readout_results"_a = readout_results);
}

// The public execute_analog_evolution
py::object QiliSimCpp::execute_analog_evolution(const py::object& functional, const py::object& noise_model, const py::dict& solver_params) {
    /*
    Execute an analog evolution functional and build FunctionalResult readout objects in C++.
    */

    // Ensure that the functional is of the correct type
    if (!py::isinstance(functional, AnalogEvolution)) {
        throw py::value_error("The provided functional is not an AnalogEvolution instance");
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
    }

    // Pre-process the Python objects
    py::object initial_state = functional.attr("initial_state");
    py::object hamiltonians_full = schedule.attr("hamiltonians");
    py::list hamiltonians_keys = hamiltonians_full.attr("keys")();
    py::list hamiltonians_values = hamiltonians_full.attr("values")();
    py::object steps = schedule.attr("tlist");

    // Get parameters
    QiliSimConfig config = parse_solver_params(solver_params);
    if (functional.attr("store_intermediate_results").cast<bool>()) {
        config.set_store_intermediate_results(true);
    }

    // Convert to C++ objects
    std::vector<SparseMatrix> hamiltonians = parse_hamiltonians(hamiltonians_values, config.get_atol());
    if (hamiltonians.empty()) {
        throw py::value_error("At least one Hamiltonian must be provided");
    }
    int nqubits = static_cast<int>(std::log2(hamiltonians[0].rows()));
    std::vector<SparseMatrix> observable_matrices;
    std::vector<std::vector<double>> parameters_list = parse_coefficients(schedule, hamiltonians_keys, steps);
    NoiseModelCpp noise_model_cpp = parse_noise_model(noise_model, nqubits, config.get_atol());
    std::vector<double> step_list = parse_time_steps(steps);
    SparseMatrix rho_0 = parse_initial_state(initial_state, config.get_atol());

    // Sanity checks
    if (step_list.empty()) {
        throw py::value_error("At least one time step must be provided");
    }
    if (hamiltonians.size() != parameters_list.size()) {
        throw py::value_error("Number of Hamiltonians does not match number of parameter lists");
    }
    for (size_t h_ind = 0; h_ind < hamiltonians.size(); ++h_ind) {
        if (parameters_list[h_ind].size() != step_list.size()) {
            throw py::value_error("Number of parameters for Hamiltonian " + std::to_string(h_ind) + " does not match number of time steps");
        }
    }

    // Simulate once to get the final state.
    std::vector<SparseMatrix> intermediate_rhos;
    SparseMatrix rho_t;
    std::vector<double> expectation_values;
    std::vector<std::vector<double>> intermediate_expectation_values;
    time_evolution(rho_0,
                    hamiltonians,
                    parameters_list,
                    step_list,
                    noise_model_cpp,
                    observable_matrices,
                    config,
                    rho_t,
                    intermediate_rhos,
                    expectation_values,
                    intermediate_expectation_values);

    // Build final state object once.
    py::object final_state = QTensor(to_numpy(rho_t));

    py::list readout_methods = functional.attr("readout").cast<py::list>();
    // Build readout results.
    py::list readout_results;
    for (auto readout_obj : readout_methods) {
        py::object readout = py::reinterpret_borrow<py::object>(readout_obj);
        if (readout.attr("is_state_tomography")().cast<bool>()) {
            if (readout.attr("readout_method").attr("state_tomography_method").cast<std::string>() != "exact") {
                throw py::value_error("State Tomography methods that are not exact are not supported yet.");
            }
            readout_results.append(StateTomographyReadoutResults("readout"_a = copy_readout(readout.attr("readout_method")), "final_state"_a = final_state));
        }
        if (readout.attr("is_expectation_values")().cast<bool>()) {
            std::vector<SparseMatrix> observable_matrices =
                parse_observables(readout.attr("readout_method").attr("observables"), nqubits, config.get_atol());
            std::vector<double> expected_values;
            expected_values.reserve(observable_matrices.size());
            for (const auto& observable : observable_matrices) {
                if (rho_t.cols() == 1) {
                    expected_values.push_back(std::real(dot(rho_t, DenseMatrix(observable * rho_t))));
                } else {
                    expected_values.push_back(std::real(dot(DenseMatrix(observable), rho_t)));
                }
            }
            readout_results.append(
                ExpectationReadoutResults("readout"_a = copy_readout(readout.attr("readout_method")), "expected_values"_a = to_numpy(expected_values)));
        }
        if (readout.attr("is_sample")().cast<bool>()) {
            int n_shots = readout.attr("readout_method").attr("nshots").cast<int>();
            std::map<std::string, int> counts;
            std::vector<bool> qubits_to_measure(nqubits, true);
            DenseMatrix rho_t_dense = DenseMatrix(rho_t);
            py::dict samples = samples_from_state(
                rho_t_dense, noise_model_cpp, nqubits, n_shots, counts, qubits_to_measure, config);
            readout_results.append(
                SamplingReadoutResults("readout"_a = copy_readout(readout.attr("readout_method")),
                               "samples"_a = samples));
        }
    }
    return FunctionalResult("readout_results"_a = readout_results);
}
