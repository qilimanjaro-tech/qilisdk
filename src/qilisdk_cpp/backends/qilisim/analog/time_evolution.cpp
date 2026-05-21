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

#include "time_evolution.h"
#include "../noise/noise_model.h"
#include "../utils/matrix_utils.h"
#include "../utils/random.h"
#include "iterations.h"

#include <iostream>

// GCOV_EXCL_BR_START

void time_evolution(SparseMatrix rho_0, const std::vector<SparseMatrix>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, NoiseModelCpp& noise_model_cpp, QiliSimConfig& config, DenseMatrix& rho_t, std::vector<DenseMatrix>& intermediate_rhos) {
    /*
    Execute a time evolution functional.

    Args:
        rho_0 (SparseMatrix): The initial state (density matrix or state vector).
        hamiltonians (std::vector<SparseMatrix>): The list of Hamiltonian terms.
        parameters_list (std::vector<std::vector<double>>): The list of parameter values for each Hamiltonian term at each time step.
        step_list (std::vector<double>): The list of time steps.
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during evolution.
        observable_matrices (std::vector<SparseMatrix>): The list of observable matrices to measure.
        config (QiliSimConfig&): Configuration parameters for the time evolution.
        rho_t (DenseMatrix&): Output parameter to hold the final state after evolution.
        intermediate_rhos (std::vector<DenseMatrix>&): Output parameter to hold intermediate states if requested.
        expectation_values (std::vector<std::vector<double>>&): Output parameter to hold the expectation values of observables at final time.

    Returns:
        TimeEvolutionResult: The results of the evolution.

    Raises:
        py::value_error: If the configuration is invalid.
    */

    // Make sure the config is valid
    config.validate();

    // Set the number of threads
    Eigen::setNbThreads(config.get_num_threads());

    // Get the jump operators from the noise model
    const std::vector<SparseMatrix>& jump_operators = noise_model_cpp.get_jump_operators();

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
        if (std::abs(trace_rho2 - 1.0) < config.get_atol()) {
            rho_0 = get_vector_from_density_matrix(rho_0);
            is_unitary_on_statevector = true;
        }
    }

    // If we were told to do monte carlo, but we already have unitary dynamics, don't bother
    if (is_unitary_on_statevector) {
        config.set_monte_carlo(false);
    }

    // If we have non-unitary dynamics and the input was a state vector, convert to density matrix
    if (!is_unitary_dynamics && input_was_vector) {
        rho_0 = rho_0 * rho_0.adjoint();
    }

    // If monte carlo, sample from rho_0 to get initial states
    // Then rho should be a collection of state vectors as columns
    bool use_monte_carlo = config.get_monte_carlo();
    if (use_monte_carlo) {
        rho_0 = sample_from_density_matrix(rho_0, config.get_num_monte_carlo_trajectories(), config.get_seed());
        is_unitary_on_statevector = true;
    }

    // Init rho_0
    rho_t = rho_0;

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
        if (config.get_time_evolution_method() == "integrate_rk4") {
            rho_t = iter_rk4_matrix(rho_t, dt, currentH, jump_operators, is_unitary_on_statevector);
        } else if (config.get_time_evolution_method() == "direct") {
            rho_t = iter_direct(rho_t, dt, currentH, jump_operators, is_unitary_on_statevector);
        } else if (config.get_time_evolution_method() == "arnoldi") {
            rho_t = iter_arnoldi(rho_t, dt, currentH, jump_operators, config.get_arnoldi_dim(), config.get_num_arnoldi_substeps(), is_unitary_on_statevector, config.get_atol());
        } else {
            throw std::invalid_argument("Invalid time evolution method: " + config.get_time_evolution_method());
        }

        // If we should store intermediates, do it here
        if (config.get_store_intermediate_results()) {
            if (use_monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
                intermediate_rhos.push_back(trajectories_to_density_matrix(rho_t));
            } else {
                intermediate_rhos.push_back(rho_t);
            }
        }
    }

    // If we have statevector/s but we should return a density matrix
    if (use_monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
        rho_t = trajectories_to_density_matrix(rho_t);
    }
}

void time_evolution_matrix_free(SparseMatrix rho_0, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, NoiseModelCpp& noise_model_cpp, QiliSimConfig& config, DenseMatrix& rho_t, std::vector<DenseMatrix>& intermediate_rhos) {
    /*
    Execute a time evolution functional.

    Args:
        rho_0 (SparseMatrix): The initial state (density matrix or state vector).
        hamiltonians (std::vector<MatrixFreeHamiltonian>): The list of Hamiltonian terms.
        parameters_list (std::vector<std::vector<double>>): The list of parameter values for each Hamiltonian term at each time step.
        step_list (std::vector<double>): The list of time steps.
        noise_model_cpp (NoiseModelCpp&): The noise model to apply during evolution.
        config (QiliSimConfig&): Configuration parameters for the time evolution.
        rho_t (DenseMatrix&): Output parameter to hold the final state after evolution.
        intermediate_rhos (std::vector<DenseMatrix>&): Output parameter to hold intermediate states if requested.
        expectation_values (std::vector<std::vector<double>>&): Output parameter to hold the expectation values of observables at final time.

    Returns:
        TimeEvolutionResult: The results of the evolution.

    Raises:
        py::value_error: If the configuration is invalid.
    */

    // Make sure the config is valid
    config.validate();

// Set the number of threads
#if defined(_OPENMP)
    Eigen::setNbThreads(1);
    omp_set_num_threads(config.get_num_threads());
#endif

    // Get the jump operators from the noise model
    const std::vector<SparseMatrix>& jump_operators = noise_model_cpp.get_jump_operators();

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
        if (std::abs(trace_rho2 - 1.0) < config.get_atol()) {
            rho_0 = get_vector_from_density_matrix(rho_0);
            is_unitary_on_statevector = true;
        }
    }

    // If we were told to do monte carlo, but we already have unitary dynamics, don't bother
    bool use_monte_carlo = config.get_monte_carlo();
    if (is_unitary_on_statevector) {
        use_monte_carlo = false;
    }

    // If we have non-unitary dynamics and the input was a state vector, convert to density matrix
    if (!is_unitary_dynamics && input_was_vector) {
        rho_0 = rho_0 * rho_0.adjoint();
    }

    // If monte carlo, sample from rho_0 to get initial states
    // Then rho should be a collection of state vectors as columns
    if (use_monte_carlo) {
        rho_0 = sample_from_density_matrix(rho_0, config.get_num_monte_carlo_trajectories(), config.get_seed());
        is_unitary_on_statevector = true;
    }

    // Init rho_0
    rho_t = rho_0;

    // If doing adaptive step size with rk45
    if (config.get_time_evolution_method() == "integrate_rk45_matrix_free") {
        // Initial step size
        double dt = 1.0;
        if (step_list.size() > 1) {
            dt = step_list[1];
        }

        // Loop until we reach the max time
        double current_time = 0.0;
        size_t iters = 0;
        const size_t max_iters = 1000000;  // Just in case to prevent infinite loops
        DenseMatrix k_saved;
        while (current_time < step_list.back()) {
            // Make sure the next step doesn't go beyond the final time point
            dt = std::min(dt, step_list.back() - current_time);

            // dt is updated to the suggested next step; dt_taken is what was actually stepped0
            double dt_taken = iter_rk45(rho_t, current_time, dt, step_list, hamiltonians, parameters_list, jump_operators, is_unitary_on_statevector, config.get_adaptive_tol(), k_saved);

            // If we should store intermediates, do it here
            if (config.get_store_intermediate_results() && dt_taken > 0) {
                if (use_monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
                    intermediate_rhos.push_back(trajectories_to_density_matrix(rho_t));
                } else {
                    intermediate_rhos.push_back(rho_t);
                }
            }

            // Update the time and step index
            current_time += dt_taken;
            iters++;
            if (iters >= max_iters) {
                throw std::runtime_error("Maximum number of iterations reached in adaptive RK45 integration.");
            }
            if (dt < config.get_atol()) {
                throw std::runtime_error("Minimum step size reached in adaptive RK45 integration.");
            }
        }

        // If doing fixed step size with rk4
    } else if (config.get_time_evolution_method() == "integrate_rk4_matrix_free") {
        // For each time step
        for (size_t step_ind = 0; step_ind < step_list.size(); ++step_ind) {
            // Determine the time step and starting time
            double t_start = (step_ind > 0) ? step_list[step_ind - 1] : 0.0;
            double dt = step_list[step_ind] - t_start;

            // Perform the iteration
            iter_rk4(rho_t, t_start, dt, step_list, hamiltonians, parameters_list, jump_operators, is_unitary_on_statevector);

            // If we should store intermediates, do it here
            if (config.get_store_intermediate_results()) {
                if (use_monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
                    intermediate_rhos.push_back(trajectories_to_density_matrix(rho_t));
                } else {
                    intermediate_rhos.push_back(rho_t);
                }
            }
        }
    } else {
        throw std::invalid_argument("Invalid matrix-free time evolution method: " + config.get_time_evolution_method());
    }

    // If we have statevector/s but we should return a density matrix
    if (use_monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
        rho_t = trajectories_to_density_matrix(rho_t);
    }
}

void time_evolution_variational_exponential(ExponentialAnsatz& rho_t, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, QiliSimConfig& config) {
    /*
    Execute an approximate time evolution functional using a variational approach with an exponential ansatz.
    The state at each point is represented as the exponential of a weighted sum of Pauli strings acting on the + state.

    Args:
        rho_t (ExponentialAnsatz&): Output parameter to hold the final state after evolution, represented as an ExponentialAnsatz.
        hamiltonians (std::vector<MatrixFreeHamiltonian>): The list of Hamiltonian terms.
        parameters_list (std::vector<std::vector<double>>): The list of parameter values for each Hamiltonian term at each time step.
        step_list (std::vector<double>): The list of time steps.
        config (QiliSimConfig&): Configuration parameters for the time evolution.
    */

    // Set the number of threads
#if defined(_OPENMP)
    Eigen::setNbThreads(config.get_num_threads());
    omp_set_num_threads(config.get_num_threads());
#endif

    // Some checks
    config.validate();
    if (hamiltonians.size() <= 0) {
        throw std::invalid_argument("At least one Hamiltonian must be provided");
    }

    // Set up the ansatz
    int n_qubits = hamiltonians[0].get_nqubits();
    rho_t = ExponentialAnsatz(n_qubits, config.get_order(), config.get_shots(), config.get_warmups());
    if (std::abs(rho_t.get_order() - 1.5) < 1e-10) {
        rho_t.prune_terms_not_in_hamiltonian(hamiltonians[hamiltonians.size() - 1]);
    }
    rho_t.set_shots(config.get_num_monte_carlo_trajectories());
    std::cout << "State has " << rho_t.get_terms().size() << " terms in the ansatz" << std::endl;

    // Initial step size: match the first scheduled interval, or fall back to 1.0
    double dt = 1.0;
    if (step_list.size() > 1) {
        dt = step_list[1];
    }

    // Fixed-step RK4 loop
    for (size_t step_ind = 0; step_ind < step_list.size(); ++step_ind) {
        double t_start = (step_ind > 0) ? step_list[step_ind - 1] : 0.0;
        double dt = step_list[step_ind] - t_start;
        iter_rk4(rho_t, t_start, dt, step_list, hamiltonians, parameters_list, config.get_max_terms());
    }

}

// GCOV_EXCL_BR_STOP
