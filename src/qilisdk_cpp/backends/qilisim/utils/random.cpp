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

#include "random.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_map>
#include "../../../libs/pybind.h"
#include "matrix_utils.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

// GCOV_EXCL_BR_START

std::map<std::string, int> sample_from_probabilities(const std::vector<std::tuple<int, double>>& prob_entries, int n_qubits, int n_shots, int seed) {
    /*
    Sample measurement outcomes from a probability distribution.

    Args:
        prob_entries (std::vector<std::tuple<int, double>>): List of (state index, probability) tuples.
        n_qubits (int): Number of qubits.
        n_shots (int): Number of measurement shots.
        seed (int): Random seed for sampling.

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

std::map<std::string, int> sample_from_probabilities(double* probabilities, std::size_t size, int n_qubits, int n_shots, int seed) {
    /*
    Sample measurement outcomes from a probability distribution.

    Args:
        probabilities (double*): Pointer to the probability of each state index (changed in-place)
        size (std::size_t): Number of entries in probabilities.
        n_qubits (int): Number of qubits.
        n_shots (int): Number of measurement shots.
        seed (int): Random seed for sampling.

    Returns:
        std::map<std::string, int>: A map of bitstring outcomes to their counts.
    */
    // Turn the probabilities into a cumulative distribution in-place
    const size_t dim = size;
    double* cdf = probabilities;
    double running = 0.0;
    for (size_t state_index = 0; state_index < dim; ++state_index) {
        running += probabilities[state_index];
        cdf[state_index] = running;
    }
    if (dim > 0) {
        cdf[dim - 1] = 1.0;
    }

    // Accumulate a sparse list of counts
    std::map<size_t, int> index_counts;
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        int thread_id = 0;
#if defined(_OPENMP)
        thread_id = omp_get_thread_num();
#endif
        std::seed_seq seq{static_cast<unsigned>(seed), static_cast<unsigned>(thread_id)};
        std::default_random_engine generator(seq);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        std::unordered_map<size_t, int> local_counts;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int shot = 0; shot < n_shots; ++shot) {
            double random_value = distribution(generator);
            // First state whose cumulative probability reaches the draw
            size_t state_index = static_cast<size_t>(std::lower_bound(cdf, cdf + dim, random_value) - cdf);
            if (state_index >= dim) {
                state_index = dim - 1;
            }
            local_counts[state_index]++;
        }

        // Merge this thread's sparse histogram into the shared result
#if defined(_OPENMP)
#pragma omp critical
#endif
        for (const auto& entry : local_counts) {
            index_counts[entry.first] += entry.second;
        }
    }

    // Go from counts by index to counts by bitstring
    std::map<std::string, int> result;
    for (const auto& entry : index_counts) {
        size_t state_index = entry.first;
        std::string bitstring(n_qubits, '0');
        for (int b = n_qubits - 1; b >= 0; --b) {
            if ((state_index >> b) & 1) {
                bitstring[n_qubits - 1 - b] = '1';
            }
        }
        result[bitstring] = entry.second;
    }
    return result;
}

DenseMatrix get_vector_from_density_matrix(const DenseMatrix& rho_t, double atol) {
    /*
    Extract a state vector from a pure density matrix by finding a non-zero diagonal element.

    Args:
        rho_t (DenseMatrix): The density matrix.
        atol (double): Absolute tolerance for considering elements as non-zero.

    Returns:
        DenseMatrix: The extracted state vector.

    Raises:
        py::value_error: If the density matrix has no non-zero diagonal elements.
    */

    // Find a non-zero diagonal element
    int non_zero_col = -1;
    for (int r = 0; r < rho_t.rows(); ++r) {
        Complex val = rho_t(r, r);
        if (std::abs(val) > atol) {
            non_zero_col = r;
            break;
        }
    }
    if (non_zero_col == -1) {
        throw py::value_error("Final density matrix has no non-zero diagonal elements.");
    }

    // Extract the corresponding state vector
    DenseMatrix state_vec(rho_t.rows(), 1);
    for (int r = 0; r < rho_t.rows(); ++r) {
        state_vec(r, 0) = rho_t(r, non_zero_col);
    }
    state_vec /= state_vec.norm();

    return state_vec;
}

SparseMatrix get_vector_from_density_matrix(const SparseMatrix& rho_t, double atol) {
    /*
    Extract a state vector from a pure density matrix by finding a non-zero diagonal element.

    Args:
        rho_t (SparseMatrix): The density matrix.
        atol (double): Absolute tolerance for considering elements as non-zero.

    Returns:
        SparseMatrix: The extracted state vector.

    Raises:
        py::value_error: If the density matrix has no non-zero diagonal elements.
    */

    // Find a non-zero diagonal element
    int non_zero_col = -1;
    for (int r = 0; r < rho_t.rows(); ++r) {
        Complex val = rho_t.coeff(r, r);
        if (std::abs(val) > atol) {
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
        Complex val = rho_t.coeff(r, non_zero_col);
        if (std::abs(val) > atol) {
            state_vec_entries.emplace_back(Triplet(r, 0, val));
        }
    }
    SparseMatrix final_state_vec(rho_t.rows(), 1);
    final_state_vec.setFromTriplets(state_vec_entries.begin(), state_vec_entries.end());
    final_state_vec /= final_state_vec.norm();

    return final_state_vec;
}

// Sample from a density matrix
DenseMatrix sample_from_density_matrix(const DenseMatrix& rho, int n_trajectories, int seed, double atol) {
    /*
    Get statevector samples from a density matrix, using the eigendecomposition.

    Args:
        rho (DenseMatrix): The input density matrix.
        n_trajectories (int): Number of trajectories.
        seed (int): Random seed for sampling.
        atol (double): Absolute tolerance for considering eigenvalues as non-zero.

    Returns:
        DenseMatrix: A matrix who's columns are the sampled statevectors.
    */

    const long dim = rho.rows();

    // Fast path: a pure state rho = |psi><psi| needs no eigendecomposition - every
    // trajectory is just |psi>. Purity tr(rho^2) == 1 detects it, and for a
    // Hermitian rho tr(rho^2) is the squared Frobenius norm (an O(dim^2) reduction
    // versus the O(dim^3), single-threaded eigensolve below). Recover |psi> from
    // its highest-norm column: rho[:,j] = psi * conj(psi_j) is proportional to psi.
    constexpr double purity_tol = 1e-9;
    if (std::abs(rho.squaredNorm() - 1.0) <= purity_tol) {
        long best_col = 0;
        double best_norm = -1.0;
        for (long j = 0; j < dim; ++j) {
            double col_norm = rho.col(j).squaredNorm();
            if (col_norm > best_norm) {
                best_norm = col_norm;
                best_col = j;
            }
        }
        DenseMatrix psi = rho.col(best_col);
        double norm = std::sqrt(psi.squaredNorm());
        if (norm > atol) {
            psi /= norm;
        }
        return psi.replicate(1, n_trajectories);
    }

    // Eigendecompose the density matrix
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(rho);
    DenseMatrix evals = es.eigenvalues();
    DenseMatrix evecs = es.eigenvectors();
    std::vector<std::tuple<int, double>> prob_entries;
    double total_prob = 0.0;
    for (int i = 0; i < evals.size(); ++i) {
        double prob = evals(i).real();
        if (prob > atol) {
            prob_entries.emplace_back(i, prob);
            total_prob += prob;
        }
    }

    // Make sure probabilities sum to 1
    if (std::abs(total_prob - 1.0) > atol) {
        throw py::value_error("Probabilities from state do not sum to 1 (sum = " + std::to_string(total_prob) + ")");
    }

    // Sample from these probabilities
    int n_qubits = static_cast<int>(std::log2(rho.rows()));
    std::map<std::string, int> counts = sample_from_probabilities(prob_entries, n_qubits, n_trajectories, seed);

    // Construct the sampled states matrix
    DenseMatrix sampled_states(dim, n_trajectories);
    int traj_index = 0;
    for (const auto& pair : counts) {
        // Get the eigenvector corresponding to this bitstring
        std::string bitstring = pair.first;
        int count = pair.second;
        int eigenvec_index = std::stoi(bitstring, nullptr, 2);
        DenseMatrix state_vec = evecs.col(eigenvec_index);

        // Normalize the state vector
        double norm = std::sqrt(state_vec.squaredNorm());
        if (norm > atol) {
            state_vec /= norm;
        }

        // Add this state vector count times to the new matrix
        for (int i = 0; i < count; ++i) {
            sampled_states.col(traj_index) = state_vec;
            traj_index++;
        }
    }

    return sampled_states;
}

// Convert a matrix containing trajectories as columns to a density matrix
DenseMatrix trajectories_to_density_matrix(const DenseMatrix& trajectories) {
    /*
    Convert a matrix containing statevector trajectories as columns to a density matrix.
    If we have N trajectories |psi_i>, the density matrix is given by
    rho = 1/N sum_i |psi_i><psi_i|. Or, in matrix form, if the trajectories are columns of a matrix T,
    rho = 1/N T T^dagger.

    Args:
        trajectories (DenseMatrix): The input matrix with statevectors as columns.

    Returns:
        DenseMatrix: The corresponding density matrix.
    */
    DenseMatrix rho = trajectories * trajectories.adjoint();
    rho /= static_cast<double>(trajectories.cols());
    rho /= trace(rho);
    return rho;
}

DenseMatrix reset_trajectories(const DenseMatrix& trajectories, unsigned long long reset_mask, int seed) {
    /*
    Apply a reset of the masked qubits to each Monte Carlo trajectory (columns are
    state vectors). This is the stochastic unravelling of the reset channel
    (trace out the reset qubits then re-initialise them to |0>): for each
    trajectory we Born-sample a joint outcome of the reset qubits, collapse the
    state onto it, renormalise, and relabel the reset qubits to |0>. Averaging the
    returned trajectories reproduces the deterministic reset channel in expectation.

    Args:
        trajectories (DenseMatrix): dim x n_trajectories batch of state vectors.
        reset_mask (unsigned long long): Bitmask of the full-state indices' bits
            corresponding to the qubits being reset.
        seed (int): Base random seed (each trajectory uses a deterministic offset).

    Returns:
        DenseMatrix: The reset trajectories (same shape).
    */
    const long dim = trajectories.rows();
    const long n_traj = trajectories.cols();
    DenseMatrix out = DenseMatrix::Zero(dim, n_traj);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long c = 0; c < n_traj; ++c) {
        // Born probabilities of each joint configuration of the reset qubits.
        // std::map keeps a deterministic iteration order for reproducible sampling.
        std::map<unsigned long long, double> config_probs;
        for (long i = 0; i < dim; ++i) {
            double w = std::norm(trajectories(i, c));
            if (w == 0.0) {
                continue;
            }
            config_probs[static_cast<unsigned long long>(i) & reset_mask] += w;
        }
        if (config_probs.empty()) {
            continue;  // zero column, leave it zero
        }

        // Per-trajectory RNG, offset by the column so parallel runs stay deterministic.
        std::mt19937_64 rng(static_cast<unsigned long long>(seed) + 0x9e3779b97f4a7c15ULL * static_cast<unsigned long long>(c));
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        const double r = unif(rng);

        unsigned long long chosen = config_probs.rbegin()->first;  // fallback: last bucket
        double cumulative = 0.0;
        for (const auto& kv : config_probs) {
            cumulative += kv.second;
            if (r <= cumulative) {
                chosen = kv.first;
                break;
            }
        }

        // Collapse onto the chosen configuration, renormalise, and clear the reset bits.
        const double norm = std::sqrt(config_probs[chosen]);
        for (long i = 0; i < dim; ++i) {
            if ((static_cast<unsigned long long>(i) & reset_mask) == chosen) {
                long target = static_cast<long>(static_cast<unsigned long long>(i) & ~reset_mask);
                out(target, c) = trajectories(i, c) / norm;
            }
        }
    }

    return out;
}

// Sample from a density matrix
SparseMatrix sample_from_density_matrix(const SparseMatrix& rho, int n_trajectories, int seed, double atol) {
    /*
    Get statevector samples from a density matrix, using the eigendecomposition.

    Args:
        rho (SparseMatrix): The input density matrix.
        n_trajectories (int): Number of trajectories.
        seed (int): Random seed for sampling.
        atol (double): Absolute tolerance for considering eigenvalues as non-zero.

    Returns:
        SparseMatrix: A matrix who's columns are the sampled statevectors.
    */

    const long dim = rho.rows();

    // Fast path: a pure state rho = |psi><psi| needs no eigendecomposition - every
    // trajectory is just |psi>. See the dense overload for the derivation; here we
    // keep everything sparse.
    constexpr double purity_tol = 1e-9;
    if (std::abs(rho.squaredNorm() - 1.0) <= purity_tol) {
        long best_col = 0;
        double best_norm = -1.0;
        for (long j = 0; j < dim; ++j) {
            double col_norm = rho.col(j).squaredNorm();
            if (col_norm > best_norm) {
                best_norm = col_norm;
                best_col = j;
            }
        }
        SparseMatrix psi = rho.col(best_col);
        double norm = std::sqrt(psi.squaredNorm());
        if (norm > atol) {
            psi /= norm;
        }
        Triplets entries;
        for (int c = 0; c < n_trajectories; ++c) {
            for (int k = 0; k < psi.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(psi, k); it; ++it) {
                    entries.emplace_back(Triplet(int(it.row()), c, it.value()));
                }
            }
        }
        SparseMatrix sampled_states(dim, n_trajectories);
        sampled_states.setFromTriplets(entries.begin(), entries.end());
        return sampled_states;
    }

    // Eigendecompose the density matrix
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(rho);
    DenseMatrix evals = es.eigenvalues();
    DenseMatrix evecs = es.eigenvectors();
    std::vector<std::tuple<int, double>> prob_entries;
    double total_prob = 0.0;
    for (int i = 0; i < evals.size(); ++i) {
        double prob = evals(i).real();
        if (prob > atol) {
            prob_entries.emplace_back(i, prob);
            total_prob += prob;
        }
    }

    // Make sure probabilities sum to 1
    if (std::abs(total_prob - 1.0) > atol) {
        throw py::value_error("Probabilities from state do not sum to 1 (sum = " + std::to_string(total_prob) + ")");
    }

    // Sample from these probabilities
    int n_qubits = static_cast<int>(std::log2(rho.rows()));
    std::map<std::string, int> counts = sample_from_probabilities(prob_entries, n_qubits, n_trajectories, seed);

    // Construct the sampled states matrix
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
        if (norm > atol) {
            state_vec /= norm;
        }

        // Add this state vector count times to the new matrix
        for (int i = 0; i < count; ++i) {
            for (int k = 0; k < state_vec.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(state_vec, k); it; ++it) {
                    int row = int(it.row());
                    Complex val = it.value();
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
SparseMatrix trajectories_to_density_matrix(const SparseMatrix& trajectories) {
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

// GCOV_EXCL_BR_STOP