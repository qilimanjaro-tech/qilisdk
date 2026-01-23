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
#include "utils/numpy.h"
#include "utils/parsers.h"

// Make the QiliSimCpp class available in Python, as well as the two main methods
PYBIND11_MODULE(qilisim_module, m) {
    py::class_<QiliSimCpp>(m, "QiliSimCpp").def(py::init<>()).def("execute_sampling", &QiliSimCpp::execute_sampling).def("execute_time_evolution", &QiliSimCpp::execute_time_evolution);
}

// The public execute_sampling
py::object QiliSimCpp::execute_sampling(const py::object& functional, const py::object& noise_model, const py::dict& solver_params) {
    /*
    Execute a sampling functional using a simple statevector simulator.
    Note that this is just the wrapper mapping the Python objects to C++ objects.
    For the actual implementation, see the method execute_sampling_internal.

    Args:
        functional (py::object): The Sampling functional to execute.
        noise_model (py::object): The noise model to apply during simulation.
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
    QiliSimConfig config;
    if (solver_params.contains("max_cache_size")) {
        config.max_cache_size = solver_params["max_cache_size"].cast<int>();
    }
    if (solver_params.contains("num_threads")) {
        config.num_threads = solver_params["num_threads"].cast<int>();
    }
    if (solver_params.contains("seed")) {
        config.seed = solver_params["seed"].cast<int>();
    }
    if (solver_params.contains("atol")) {
        config.atol = solver_params["atol"].cast<double>();
    }
    config.validate();

    // Sanity checks
    if (n_qubits <= 0) {
        throw py::value_error("nqubits must be positive.");
    }
    if (n_shots <= 0) {
        throw py::value_error("nshots must be positive.");
    }

    // Parse the Python objects into C++ objects
    std::vector<Gate> gates = parse_gates(functional.attr("circuit"), config.atol);
    std::vector<bool> qubits_to_measure = parse_measurements(functional.attr("circuit"));

    // Pass everything to the interal implementation
    std::map<std::string, int> counts;
    sampling(gates, qubits_to_measure, n_qubits, n_shots, config, counts);

    // Convert counts to samples dict
    py::dict samples;
    for (const auto& pair : counts) {
        samples[py::cast(pair.first)] = py::cast(pair.second);
    }

    return SamplingResult("nshots"_a = n_shots, "samples"_a = samples);
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

    // Pre-process the Python objects
    py::object initial_state = functional.attr("initial_state");
    py::object hamiltonians_full = functional.attr("schedule").attr("hamiltonians");
    py::list hamiltonians_keys = hamiltonians_full.attr("keys")();
    py::list hamiltonians_values = hamiltonians_full.attr("values")();
    py::object steps = functional.attr("schedule").attr("tlist");
    py::object coeffs_full = functional.attr("schedule").attr("coefficients");
    py::list coeffs;
    for (const auto& h_key : hamiltonians_keys) {
        py::object h_coeffs = coeffs_full[h_key];
        py::list coeffs_for_h;
        for (const auto& t : steps) {
            coeffs_for_h.append(h_coeffs[t]);
        }
        coeffs.append(coeffs_for_h);
    }
    py::object observables = functional.attr("observables");

    // Get parameters
    QiliSimConfig config;
    if (solver_params.contains("arnoldi_dim")) {
        config.arnoldi_dim = solver_params["arnoldi_dim"].cast<int>();
    }
    if (solver_params.contains("num_arnoldi_substeps")) {
        config.num_arnoldi_substeps = solver_params["num_arnoldi_substeps"].cast<int>();
    }
    if (solver_params.contains("num_integrate_substeps")) {
        config.num_integrate_substeps = solver_params["num_integrate_substeps"].cast<int>();
    }
    if (solver_params.contains("evolution_method")) {
        config.method = solver_params["evolution_method"].cast<std::string>();
    }
    if (solver_params.contains("monte_carlo")) {
        config.monte_carlo = solver_params["monte_carlo"].cast<bool>();
    }
    if (solver_params.contains("num_monte_carlo_trajectories")) {
        config.num_monte_carlo_trajectories = solver_params["num_monte_carlo_trajectories"].cast<int>();
    }
    if (solver_params.contains("num_threads")) {
        config.num_threads = solver_params["num_threads"].cast<int>();
    }
    if (solver_params.contains("seed")) {
        config.seed = solver_params["seed"].cast<int>();
    }
    if (solver_params.contains("atol")) {
        config.atol = solver_params["atol"].cast<double>();
    }
    // store_intermediate_results is in the functional
    if (functional.attr("store_intermediate_results").cast<bool>()) {
        config.store_intermediate_results = true;
    }
    config.validate();

    // Convert to C++ objects
    std::vector<SparseMatrix> hamiltonians = parse_hamiltonians(hamiltonians_values, config.atol);
    if (hamiltonians.size() == 0) {
        throw py::value_error("At least one Hamiltonian must be provided");
    }
    int nqubits = static_cast<int>(std::log2(hamiltonians[0].rows()));
    std::vector<SparseMatrix> observable_matrices = parse_observables(observables, nqubits, config.atol);
    std::vector<std::vector<double>> parameters_list = parse_parameters(coeffs);
    std::vector<SparseMatrix> jump_operators;
    std::vector<double> step_list = parse_time_steps(steps);
    SparseMatrix rho_0 = parse_initial_state(initial_state, config.atol);

    // Sanity checks
    if (step_list.size() == 0) {
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

    // Call the internal implementation
    std::vector<SparseMatrix> intermediate_rhos;
    SparseMatrix rho_t;
    std::vector<double> expectation_values;
    std::vector<std::vector<double>> intermediate_expectation_values;
    time_evolution(rho_0, hamiltonians, parameters_list, step_list, jump_operators, observable_matrices, config, rho_t, intermediate_rhos, expectation_values, intermediate_expectation_values);

    // Convert things to numpy arrays
    py::array_t<std::complex<double>> rho_numpy = to_numpy(rho_t);
    py::array_t<double> expect_numpy = to_numpy(expectation_values);

    // Also convert intermediates if needed
    py::list intermediate_rho_numpy;
    py::array_t<double> intermediate_expect_numpy;
    if (config.store_intermediate_results) {
        for (const auto& rho_intermediate : intermediate_rhos) {
            py::array_t<std::complex<double>> rho_step_numpy = to_numpy(rho_intermediate);
            intermediate_rho_numpy.append(QTensor(rho_step_numpy));
        }
        intermediate_expect_numpy = to_numpy(intermediate_expectation_values);
    }

    // Return a TimeEvolutionResult with these
    return TimeEvolutionResult("final_state"_a = QTensor(rho_numpy), "final_expected_values"_a = expect_numpy, "intermediate_states"_a = intermediate_rho_numpy, "expected_values"_a = intermediate_expect_numpy);
}