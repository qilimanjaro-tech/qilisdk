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

#include "../qilisim.h"

py::object QiliSimCpp::execute_time_evolution(const py::object& initial_state,
                                              const py::object& Hs,
                                              const py::object& coeffs,
                                              const py::object& steps,
                                              const py::object& observables,
                                              const py::object& jumps,
                                              bool store_intermediate_results,
                                              const py::dict& solver_params) {
    /*
    Execute a time evolution functional.

    Args:
        initial_state (py::object): The initial state as a QTensor.
        Hs (py::object): A list of Hamiltonians for time-dependent Hamiltonians.
        coeffs (py::object): A list of coefficients for the Hamiltonians at each time step.
        steps (py::object): A list of time steps at which to evaluate the evolution.
        observables (py::object): A list of observables to measure at each time step.
        jumps (py::object): A list of jump operators for the Lindblad equation.
        store_intermediate_results (bool): Whether to store results at each time step.
        params (py::dict): Additional parameters for the method. See the Python wrapper for details.

    Returns:
        TimeEvolutionResult: The results of the evolution.

    Raises:
        py::value_error: If no Hamiltonians are provided.
        py::value_error: If no time steps are provided.
        py::value_error: If number of parameters for any Hamiltonian does not match number of time steps.
        py::value_error: If an unknown time evolution method is specified.
    */

    // Parse the info from the python objects
    std::vector<SparseMatrix> hamiltonians = parse_hamiltonians(Hs);
    if (hamiltonians.size() == 0) {
        throw py::value_error("At least one Hamiltonian must be provided");
    }
    int nqubits = static_cast<int>(std::log2(hamiltonians[0].rows()));
    std::vector<SparseMatrix> observable_matrices = parse_observables(observables, nqubits);
    std::vector<std::vector<double>> parameters_list = parse_parameters(coeffs);
    std::vector<SparseMatrix> jump_operators = parse_jump_operators(jumps);
    std::vector<double> step_list = parse_time_steps(steps);
    SparseMatrix rho_0 = parse_initial_state(initial_state);

    // Get parameters
    int arnoldi_dim = 10;
    if (solver_params.contains("arnoldi_dim")) {
        arnoldi_dim = solver_params["arnoldi_dim"].cast<int>();
    }
    int num_arnoldi_substeps = 1;
    if (solver_params.contains("num_arnoldi_substeps")) {
        num_arnoldi_substeps = solver_params["num_arnoldi_substeps"].cast<int>();
    }
    int num_integrate_substeps = 2;
    if (solver_params.contains("num_integrate_substeps")) {
        num_integrate_substeps = solver_params["num_integrate_substeps"].cast<int>();
    }
    std::string method = "integrate";
    if (solver_params.contains("evolution_method")) {
        method = solver_params["evolution_method"].cast<std::string>();
    }
    bool monte_carlo = false;
    if (solver_params.contains("monte_carlo")) {
        monte_carlo = solver_params["monte_carlo"].cast<bool>();
    }
    int num_monte_carlo_trajectories = 100;
    if (solver_params.contains("num_monte_carlo_trajectories")) {
        num_monte_carlo_trajectories = solver_params["num_monte_carlo_trajectories"].cast<int>();
    }
    int num_threads = 1;
    if (solver_params.contains("num_threads")) {
        num_threads = solver_params["num_threads"].cast<int>();
    }
    int seed = 42;
    if (solver_params.contains("seed")) {
        seed = solver_params["seed"].cast<int>();
    }

    // Set the number of threads
    if (num_threads <= 0) {
        num_threads = 1;
    }
    Eigen::setNbThreads(num_threads);

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
    if (method != "direct" && method != "arnoldi" && method != "integrate") {
        throw py::value_error("Unknown time evolution method: " + method);
    }
    if (arnoldi_dim <= 0) {
        throw py::value_error("arnoldi_dim must be a positive integer");
    }
    if (num_arnoldi_substeps <= 0) {
        throw py::value_error("num_arnoldi_substeps must be a positive integer");
    }
    if (num_integrate_substeps <= 0) {
        throw py::value_error("num_integrate_substeps must be a positive integer");
    }
    if (num_monte_carlo_trajectories <= 0) {
        throw py::value_error("num_monte_carlo_trajectories must be a positive integer");
    }

    // Dimensions of everything
    long dim = long(hamiltonians[0].rows());

    // Check if we have unitary dynamics
    bool is_unitary_dynamics = (jump_operators.size() == 0);

    // Determine if the input was a state vector
    bool input_was_vector = false;
    if (rho_0.rows() == 1 || rho_0.cols() == 1) {
        input_was_vector = true;
    }
    if (rho_0.rows() == 1 && rho_0.cols() > 1) {
        rho_0 = rho_0.adjoint();
    }

    // Determine if should treat it as unitary evolution on a statevector
    // Note that this can change if the input was a density matrix but is actually pure
    // Or similarly if we use monte-carlo, we treat it as statevector evolution
    bool is_unitary_on_statevector = is_unitary_dynamics && input_was_vector;

    // If we have unitary dynamics and the input was a pure state, convert to state vector
    if (is_unitary_dynamics && !input_was_vector) {
        double trace_rho2 = 0.0;
        for (int k = 0; k < rho_0.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it1(rho_0, k); it1; ++it1) {
                trace_rho2 += std::pow(std::abs(it1.value()), 2);
            }
        }
        if (std::abs(trace_rho2 - 1.0) < atol_) {
            rho_0 = get_vector_from_density_matrix(rho_0);
            is_unitary_on_statevector = true;
        }
    }

    // If we were told to do monte carlo, but we already have unitary dynamics, don't bother
    if (is_unitary_on_statevector) {
        monte_carlo = false;
    }

    // If we have non-unitary dynamics and the input was a state vector, convert to density matrix
    if (!is_unitary_dynamics && input_was_vector) {
        if (rho_0.rows() == 1) {
            rho_0 = rho_0.adjoint() * rho_0;
            input_was_vector = true;
        } else if (rho_0.cols() == 1) {
            rho_0 = rho_0 * rho_0.adjoint();
            input_was_vector = true;
        }
    }

    // If monte carlo, sample from rho_0 to get initial states
    // Then rho should be a collection of state vectors as columns
    if (monte_carlo) {
        rho_0 = sample_from_density_matrix(rho_0, num_monte_carlo_trajectories, seed);
        is_unitary_on_statevector = true;
    }

    // Init rho_0
    SparseMatrix rho_t = rho_0;
    std::vector<SparseMatrix> intermediate_rhos;

    // Precalculate the sparsity pattern of the combined Hamiltonians
    SparseMatrix combinedH(dim, dim);
    for (size_t h_ind = 0; h_ind < hamiltonians.size(); ++h_ind) {
        combinedH += hamiltonians[h_ind];
    }
    combinedH.makeCompressed();
    combinedH *= 0.0;

    // For each time step
    for (size_t step_ind = 0; step_ind < step_list.size(); ++step_ind) {
        // Get the current Hamiltonian
        SparseMatrix currentH = combinedH;
        for (size_t h = 0; h < hamiltonians.size(); ++h) {
            double c = parameters_list[h][step_ind];
            for (int k = 0; k < hamiltonians[h].outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(hamiltonians[h], k); it; ++it) {
                    currentH.coeffRef(it.row(), it.col()) += c * it.value();
                }
            }
        }

        // Determine the time step
        double dt = step_list[step_ind];
        if (step_ind > 0) {
            dt = (step_list[step_ind] - step_list[step_ind - 1]);
        }

        // Perform the iteration depending on the method
        if (method == "integrate") {
            rho_t = iter_integrate(rho_t, dt, currentH, jump_operators, num_integrate_substeps, is_unitary_on_statevector);
        } else if (method == "direct") {
            rho_t = iter_direct(rho_t, dt, currentH, jump_operators, is_unitary_on_statevector);
        } else if (method == "arnoldi") {
            rho_t = iter_arnoldi(rho_t, dt, currentH, jump_operators, arnoldi_dim, num_arnoldi_substeps, is_unitary_on_statevector);
        }

        // If we should store intermediates, do it here
        if (store_intermediate_results) {
            if (monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
                intermediate_rhos.push_back(trajectories_to_density_matrix(rho_t));
            } else {
                intermediate_rhos.push_back(rho_t);
            }
        }
    }

    // If we have statevector/s but we should return a density matrix
    if (monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
        rho_t = trajectories_to_density_matrix(rho_t);
    }

    // Apply the operators using the Born rule
    std::vector<double> expectation_values;
    for (const auto& O : observable_matrices) {
        if (rho_t.cols() == 1) {
            expectation_values.push_back(std::real(dot(rho_t, O * rho_t)));
        } else {
            expectation_values.push_back(std::real(dot(O, rho_t)));
        }
    }

    // If we have intermediates, process them too
    std::vector<std::vector<double>> intermediate_expectation_values;
    if (store_intermediate_results) {
        for (const auto& rho_intermediate : intermediate_rhos) {
            std::vector<double> step_expectation_values;
            for (const auto& O : observable_matrices) {
                if (rho_intermediate.cols() == 1) {
                    DenseMatrix rho_dense(rho_intermediate);
                    step_expectation_values.push_back(std::real(dot(rho_dense, O * rho_dense)));
                } else {
                    step_expectation_values.push_back(std::real(dot(O, rho_intermediate)));
                }
            }
            intermediate_expectation_values.push_back(step_expectation_values);
        }
    }

    // Convert things to numpy arrays
    py::array_t<std::complex<double>> rho_numpy = to_numpy(rho_t);
    py::array_t<double> expect_numpy = to_numpy(expectation_values);

    // Also convert intermediates if needed
    py::list intermediate_rho_numpy;
    py::array_t<double> intermediate_expect_numpy;
    if (store_intermediate_results) {
        for (const auto& rho_intermediate : intermediate_rhos) {
            py::array_t<std::complex<double>> rho_step_numpy = to_numpy(rho_intermediate);
            intermediate_rho_numpy.append(QTensor(rho_step_numpy));
        }
        intermediate_expect_numpy = to_numpy(intermediate_expectation_values);
    }

    // Return a TimeEvolutionResult with these
    return TimeEvolutionResult("final_state"_a = QTensor(rho_numpy), "final_expected_values"_a = expect_numpy, "intermediate_states"_a = intermediate_rho_numpy,
                               "expected_values"_a = intermediate_expect_numpy);
}