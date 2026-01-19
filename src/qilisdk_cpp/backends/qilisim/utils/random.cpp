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

#include <chrono>
#include <random>
#include "../qilisim.h"

std::map<std::string, int> QiliSimCpp::sample_from_probabilities(const std::vector<std::tuple<int, double>>& prob_entries, int n_qubits, int n_shots, int seed) const {
    /*
    Sample measurement outcomes from a probability distribution.

    Args:
        prob_entries (std::vector<std::tuple<int, double>>): List of (state index, probability) tuples.
        n_qubits (int): Number of qubits.
        n_shots (int): Number of measurement shots.

    Returns:
        std::map<std::string, int>: A map of bitstring outcomes to their counts.
    */
    std::map<int, std::string> binary_strings;
    std::map<std::string, int> counts;
    std::string current_bitstring = "";
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int shot = 0; shot < n_shots; ++shot) {
        double random_value = distribution(generator);
        double cumulative_prob = 0.0;
        for (const auto& entry : prob_entries) {
            double prob = std::get<1>(entry);
            cumulative_prob += prob;
            if (random_value <= cumulative_prob) {
                int state_index = std::get<0>(entry);
                if (binary_strings.find(state_index) == binary_strings.end()) {
                    std::string bitstring = "";
                    for (int b = n_qubits - 1; b >= 0; --b) {
                        bitstring += ((state_index >> b) & 1) ? '1' : '0';
                    }
                    binary_strings[state_index] = bitstring;
                }
                counts[binary_strings[state_index]]++;
                break;
            }
        }
    }
    return counts;
}

std::map<std::string, int> QiliSimCpp::sample_from_probabilities(const std::vector<double>& probabilities, int n_qubits, int n_shots, int seed) const {
    /*
    Sample measurement outcomes from a probability distribution.

    Args:
        probabilities (std::vector<double>): List of probabilities for each state index.
        n_qubits (int): Number of qubits.
        n_shots (int): Number of measurement shots.

    Returns:
        std::map<std::string, int>: A map of bitstring outcomes to their counts.
    */
    std::vector<int> counts(probabilities.size(), 0);
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int shot = 0; shot < n_shots; ++shot) {
        double random_value = distribution(generator);
        double cumulative_prob = 0.0;
        for (size_t state_index = 0; state_index < probabilities.size(); ++state_index) {
            cumulative_prob += probabilities[state_index];
            if (random_value <= cumulative_prob) {
                counts[state_index]++;
                break;
            }
        }
    }

    // Go from counts by index to counts by bitstring
    std::map<std::string, int> result;
    for (size_t state_index = 0; state_index < counts.size(); ++state_index) {
        if (counts[state_index] > 0) {
            std::string bitstring = "";
            for (int b = n_qubits - 1; b >= 0; --b) {
                bitstring += ((state_index >> b) & 1) ? '1' : '0';
            }
            result[bitstring] = counts[state_index];
        }
    }
    return result;
}

SparseMatrix QiliSimCpp::get_vector_from_density_matrix(const SparseMatrix& rho_t) const {
    /*
    Extract a state vector from a pure density matrix by finding a non-zero diagonal element.

    Args:
        rho_t (SparseMatrix): The density matrix.

    Returns:
        SparseMatrix: The extracted state vector.

    Raises:
        py::value_error: If the density matrix has no non-zero diagonal elements.
    */

    // Find a non-zero diagonal element
    int non_zero_col = -1;
    for (int r = 0; r < rho_t.rows(); ++r) {
        std::complex<double> val = rho_t.coeff(r, r);
        if (std::abs(val) > atol_) {
            non_zero_col = r;
            break;
        }
    }
    if (non_zero_col == -1) {
        throw py::value_error("Final density matrix has no non-zero diagonal elements.");
    }

    // Extract the corresponding state vector
    Triplets state_vec_entries;
    for (int r = 0; r < rho_t.rows(); ++r) {
        std::complex<double> val = rho_t.coeff(r, non_zero_col);
        if (std::abs(val) > atol_) {
            state_vec_entries.emplace_back(Triplet(r, 0, val));
        }
    }
    SparseMatrix final_state_vec(rho_t.rows(), 1);
    final_state_vec.setFromTriplets(state_vec_entries.begin(), state_vec_entries.end());
    final_state_vec /= final_state_vec.norm();

    return final_state_vec;
}

// Sample from a density matrix
SparseMatrix QiliSimCpp::sample_from_density_matrix(const SparseMatrix& rho, int n_trajectories, int seed) const {
    /*
    Get statevector samples from a density matrix, using the eigendecomposition.

    Args:
        rho (SparseMatrix): The input density matrix.
        n_trajectories (int): Number of trajectories.

    Returns:
        SparseMatrix: A matrix who's columns are the sampled statevectors.
    */

    // Eigendecompose the density matrix
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(rho);
    DenseMatrix evals = es.eigenvalues();
    DenseMatrix evecs = es.eigenvectors();
    std::vector<std::tuple<int, double>> prob_entries;
    double total_prob = 0.0;
    for (int i = 0; i < evals.size(); ++i) {
        double prob = evals(i).real();
        if (prob > atol_) {
            prob_entries.emplace_back(i, prob);
            total_prob += prob;
        }
    }

    // Make sure probabilities sum to 1
    if (std::abs(total_prob - 1.0) > atol_) {
        throw py::value_error("Probabilities from state do not sum to 1 (sum = " + std::to_string(total_prob) + ")");
    }

    // Sample from these probabilities
    int n_qubits = static_cast<int>(std::log2(rho.rows()));
    std::map<std::string, int> counts = sample_from_probabilities(prob_entries, n_qubits, n_trajectories, seed);

    // Construct the sampled states matrix
    long dim = 1L << n_qubits;
    Triplets new_mat_entries;
    int traj_index = 0;
    for (const auto& pair : counts) {
        // Get the eigenvector corresponding to this bitstring
        std::string bitstring = pair.first;
        int count = pair.second;
        int eigenvec_index = std::stoi(bitstring, nullptr, 2);
        SparseMatrix state_vec = evecs.col(eigenvec_index).sparseView();

        // Normalize the state vector
        double norm = std::sqrt(state_vec.squaredNorm());
        if (norm > atol_) {
            state_vec /= norm;
        }

        // Add this state vector count times to the new matrix
        for (int i = 0; i < count; ++i) {
            for (int k = 0; k < state_vec.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(state_vec, k); it; ++it) {
                    int row = int(it.row());
                    std::complex<double> val = it.value();
                    new_mat_entries.emplace_back(Triplet(row, traj_index, val));
                }
            }
            traj_index++;
        }
    }

    // Form the matrix from the triplets
    SparseMatrix sampled_states(dim, traj_index);
    sampled_states.setFromTriplets(new_mat_entries.begin(), new_mat_entries.end());

    return sampled_states;
}

// Convert a matrix containing trajectories as columns to a density matrix
SparseMatrix QiliSimCpp::trajectories_to_density_matrix(const SparseMatrix& trajectories) const {
    /*
    Convert a matrix containing statevector trajectories as columns to a density matrix.
    If we have N trajectories |psi_i>, the density matrix is given by
    rho = 1/N sum_i |psi_i><psi_i|. Or, in matrix form, if the trajectories are columns of a matrix T,
    rho = 1/N T T^dagger.

    Args:
        trajectories (SparseMatrix): The input matrix with statevectors as columns.

    Returns:
        SparseMatrix: The corresponding density matrix.
    */
    SparseMatrix rho = trajectories * trajectories.adjoint();
    rho /= static_cast<double>(trajectories.cols());
    rho /= trace(rho);
    return rho;
}