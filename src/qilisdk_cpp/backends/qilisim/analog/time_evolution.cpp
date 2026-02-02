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
#include "../utils/matrix_utils.h"
#include "../utils/random.h"
#include "iterations.h"
#include "../noise/noise_model.h"

void time_evolution(SparseMatrix rho_0, 
                    const std::vector<SparseMatrix>& hamiltonians, 
                    const std::vector<std::vector<double>>& parameters_list, 
                    const std::vector<double>& step_list, 
                    NoiseModelCpp& noise_model_cpp,
                    const std::vector<SparseMatrix>& observable_matrices, 
                    QiliSimConfig& config, 
                    SparseMatrix& rho_t, 
                    std::vector<SparseMatrix>& intermediate_rhos, 
                    std::vector<double>& expectation_values, 
                    std::vector<std::vector<double>>& intermediate_expectation_values) {
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
        rho_t (SparseMatrix&): Output parameter to hold the final state after evolution.
        intermediate_rhos (std::vector<SparseMatrix>&): Output parameter to hold intermediate states if requested.
        expectation_values (std::vector<std::vector<double>>&): Output parameter to hold the expectation values of observables at final time.
        intermediate_expectation_values (std::vector<std::vector<std::vector<double>>>&): Output parameter to hold expectation values at intermediate times.

    Returns:
        TimeEvolutionResult: The results of the evolution.

    Raises:
        py::value_error: If the configuration is invalid.
    */

    // Make sure the config is valid
    config.validate();

    // Set the number of threads
    if (config.get_num_threads() <= 0) {
        config.set_num_threads(1);
    }
    Eigen::setNbThreads(config.get_num_threads());

    // Get the jump operators from the noise model
    std::vector<SparseMatrix>& jump_operators = noise_model_cpp.get_jump_operators();

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
    if (config.get_monte_carlo()) {
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
        if (config.get_method() == "integrate") {
            rho_t = iter_integrate(rho_t, dt, currentH, jump_operators, config.get_num_integrate_substeps(), is_unitary_on_statevector);
        } else if (config.get_method() == "direct") {
            rho_t = iter_direct(rho_t, dt, currentH, jump_operators, is_unitary_on_statevector, config.get_atol());
        } else if (config.get_method() == "arnoldi") {
            rho_t = iter_arnoldi(rho_t, dt, currentH, jump_operators, config.get_arnoldi_dim(), config.get_num_arnoldi_substeps(), is_unitary_on_statevector, config.get_atol());
        }

        // If we should store intermediates, do it here
        if (config.get_store_intermediate_results()) {
            if (config.get_monte_carlo() || (!input_was_vector && rho_t.cols() == 1)) {
                intermediate_rhos.push_back(trajectories_to_density_matrix(rho_t));
            } else {
                intermediate_rhos.push_back(rho_t);
            }
        }
    }

    // If we have statevector/s but we should return a density matrix
    if (config.get_monte_carlo() || (!input_was_vector && rho_t.cols() == 1)) {
        rho_t = trajectories_to_density_matrix(rho_t);
    }

    // Apply the operators using the Born rule
    for (const auto& O : observable_matrices) {
        if (rho_t.cols() == 1) {
            expectation_values.push_back(std::real(dot(rho_t, O * rho_t)));
        } else {
            expectation_values.push_back(std::real(dot(O, rho_t)));
        }
    }

    // If we have intermediates, process them too
    if (config.get_store_intermediate_results()) {
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
}