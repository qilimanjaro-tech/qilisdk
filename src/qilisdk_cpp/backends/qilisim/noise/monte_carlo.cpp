// Copyright 2026 Qilimanjaro Quantum Tech
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

#include "monte_carlo.h"

// GCOV_EXCL_BR_START

namespace {

// The splitmix64 finalizer, which scrambles a counter into a well-distributed 64-bit value
inline std::uint64_t mix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// One splitmix64 step, used as each trajectory's random stream
inline double next_uniform(std::uint64_t& stream) {
    stream += 0x9e3779b97f4a7c15ULL;
    return static_cast<double>(mix64(stream) >> 11) * 0x1.0p-53;
}

std::size_t sample_index(const std::vector<double>& weights, double uniform_draw) {
    /*
    Sample an index from a set of unnormalised weights.

    Args:
        weights (std::vector<double>): The unnormalised weights to sample from.
        uniform_draw (double): A uniform draw in [0, 1).

    Returns:
        std::size_t: The index of the sampled weight, or weights.size() if the weights do not add up to something usable.
    */
    double total = 0.0;
    for (double w : weights) {
        total += w;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return weights.size();
    }
    const double target = uniform_draw * total;
    double cumulative = 0.0;
    for (std::size_t k = 0; k < weights.size(); ++k) {
        cumulative += weights[k];
        if (target <= cumulative) {
            return k;
        }
    }
    return weights.size() - 1;
}

}  // namespace

SparseMatrix jump_drift_operator(const std::vector<SparseMatrix>& jump_operators) {
    /*
    Build the drift of the Monte Carlo effective Hamiltonian, i.e. the operator
    D = (1/2) sum_k L_k^dagger L_k appearing in H_eff = H - i D. D is Hermitian and positive
    semi-definite, so the -i D part is what makes H_eff non-Hermitian and drains the norm.

    Args:
        jump_operators (std::vector<SparseMatrix>): The jump operators, with their rates already
            folded in as sqrt(rate) * L.

    Returns:
        SparseMatrix: The drift operator D.
    */
    if (jump_operators.empty()) {
        return SparseMatrix();
    }
    SparseMatrix drift(jump_operators[0].rows(), jump_operators[0].cols());
    for (const auto& L : jump_operators) {
        SparseMatrix L_dag = L.adjoint();
        SparseMatrix L_dag_L = L_dag * L;
        drift += L_dag_L * static_cast<Real>(0.5);
    }
    drift.makeCompressed();
    return drift;
}

SparseMatrix effective_hamiltonian(const SparseMatrix& hamiltonian, const SparseMatrix& drift) {
    /*
    Form the (non-Hermitian) Monte Carlo effective Hamiltonian H_eff = H - i D. Evolving a state
    vector with H_eff makes its norm decay at exactly the rate at which the trajectory is expected
    to jump, which is what apply_jumps() then tests against.

    Args:
        hamiltonian (SparseMatrix): The Hermitian system Hamiltonian H.
        drift (SparseMatrix): The drift operator D from jump_drift_operator().

    Returns:
        SparseMatrix: The effective Hamiltonian H_eff.
    */
    if (drift.rows() == 0) {
        return hamiltonian;
    }
    SparseMatrix H_eff = hamiltonian - Complex(0.0, 1.0) * drift;
    H_eff.makeCompressed();
    return H_eff;
}

double max_jump_rate_bound(const SparseMatrix& drift) {
    /*
    Upper bound on the total jump rate 2 <psi|D|psi> of any normalized state, taken as twice the
    largest absolute row sum of D (which bounds its spectral radius). Used to bound how far a step can
    go before its jump probability stops being small.

    Args:
        drift (SparseMatrix): The drift operator D from jump_drift_operator().

    Returns:
        double: The bound on the jump rate, or zero if there is no drift.
    */
    double largest_row_sum = 0.0;
    for (int row = 0; row < drift.outerSize(); ++row) {
        double row_sum = 0.0;
        for (SparseMatrix::InnerIterator it(drift, row); it; ++it) {
            row_sum += std::abs(it.value());
        }
        largest_row_sum = std::max(largest_row_sum, row_sum);
    }
    return 2.0 * largest_row_sum;
}

TrajectoryUnraveling::TrajectoryUnraveling(int seed) : base_seed(seed) {
    /*
    Create an unravelling whose per-trajectory streams all derive from one seed.

    Args:
        seed (int): The base random seed.
    */
}

void TrajectoryUnraveling::ensure_capacity(long num_trajectories) {
    /*
    Make sure a random stream exists for every trajectory, seeding any new ones from their column
    index so that a given column always sees the same sequence.

    Args:
        num_trajectories (long): The number of trajectories (columns) that will be processed.
    */
    const std::size_t needed = static_cast<std::size_t>(num_trajectories);
    while (streams.size() < needed) {
        const std::uint64_t index = static_cast<std::uint64_t>(streams.size()) + 1ULL;
        streams.push_back(mix64(static_cast<std::uint64_t>(base_seed) + 0x9e3779b97f4a7c15ULL * index));
    }
}

void TrajectoryUnraveling::apply_jumps(DenseMatrix& trajectories, const std::vector<SparseMatrix>& jump_operators) {
    /*
    Advance the quantum-jump part of a Monte Carlo step: renormalise every trajectory and let the
    norm it lost decide whether it jumps.

    A trajectory evolved with H_eff = H - i D over a step keeps a squared norm equal to its
    probability of *not* having jumped during that step, so drawing u ~ U(0, 1) and jumping when
    u > ||psi||^2 reproduces the correct jump statistics.

    Which jump fires is then drawn with probability proportional to ||L_k psi||^2.

    Args:
        trajectories (DenseMatrix&): The (dim x num_trajectories) batch of state vectors, updated in
            place and left normalised column by column.
        jump_operators (std::vector<SparseMatrix>): The jump operators used to build the drift, with
            their rates already folded in.

    Raises:
        std::invalid_argument: If any trajectory has collapsed to a zero or non-finite norm.
    */
    if (jump_operators.empty() || trajectories.cols() == 0) {
        return;
    }
    const long num_trajectories = trajectories.cols();
    ensure_capacity(num_trajectories);

    // For each trajectory
    bool diverged = false;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(|| : diverged)
#endif
    for (long c = 0; c < num_trajectories; ++c) {
        // The squared norm left after the effective-Hamiltonian step is the survival probability
        const double survival = trajectories.col(c).squaredNorm();
        if (!std::isfinite(survival) || survival <= 0.0) {
            diverged = true;
            continue;
        }
        trajectories.col(c) /= std::sqrt(survival);

        // Integration error can leave the norm marginally above one, which simply never jumps
        if (next_uniform(streams[c]) <= survival) {
            continue;
        }

        // Pick a jump channel, weighted by how much of the state it would keep
        DenseVector psi = trajectories.col(c);
        std::vector<DenseVector> jumped(jump_operators.size());
        std::vector<double> weights(jump_operators.size());
        for (std::size_t k = 0; k < jump_operators.size(); ++k) {
            jumped[k] = jump_operators[k] * psi;
            weights[k] = jumped[k].squaredNorm();
        }
        const std::size_t chosen = sample_index(weights, next_uniform(streams[c]));
        if (chosen >= jump_operators.size()) {
            continue; 
        }

        // Renormalise the jumped state and replace the trajectory with it
        const double norm = std::sqrt(weights[chosen]);
        if (!std::isfinite(norm) || norm == 0.0) {
            diverged = true;
            continue;
        }
        trajectories.col(c) = jumped[chosen] / norm;

    }

    // Check for any trajectories that have diverged to a non-finite state
    if (diverged) {
        nan_error();
    }

}

void TrajectoryUnraveling::apply_kraus(DenseMatrix& trajectories, const std::vector<SparseMatrix>& kraus_operators) {
    /*
    Apply one Kraus channel to a batch of trajectories by unravelling it stochastically: each
    trajectory picks a single Kraus operator K_k with probability ||K_k psi||^2 and is projected
    onto it. Averaging the resulting trajectories reproduces sum_k K_k rho K_k^dagger, which is what
    the deterministic density-matrix path applies.

    Args:
        trajectories (DenseMatrix&): The (dim x num_trajectories) batch of state vectors, updated in
            place and left normalised column by column.
        kraus_operators (std::vector<SparseMatrix>): The Kraus operators of a single channel.

    Raises:
        std::invalid_argument: If any trajectory has collapsed to a zero or non-finite norm.
    */
    if (kraus_operators.empty() || trajectories.cols() == 0) {
        return;
    }
    const long num_trajectories = trajectories.cols();
    ensure_capacity(num_trajectories);

    // For each trajectory
    bool diverged = false;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(|| : diverged)
#endif
    for (long c = 0; c < num_trajectories; ++c) {
        
        // Choose a Kraus operator, weighted by how much of the state it would keep
        DenseVector psi = trajectories.col(c);
        std::vector<DenseVector> applied(kraus_operators.size());
        std::vector<double> weights(kraus_operators.size());
        for (std::size_t k = 0; k < kraus_operators.size(); ++k) {
            applied[k] = kraus_operators[k] * psi;
            weights[k] = applied[k].squaredNorm();
        }
        const std::size_t chosen = sample_index(weights, next_uniform(streams[c]));
        if (chosen >= kraus_operators.size()) {
            diverged = true;
            continue;
        }

        // Renormalise the chosen Kraus outcome and replace the trajectory with it
        const double norm = std::sqrt(weights[chosen]);
        if (!std::isfinite(norm) || norm == 0.0) {
            diverged = true;
            continue;
        }
        trajectories.col(c) = applied[chosen] / norm;

    }

    // Check if anything diverged
    if (diverged) {
        nan_error();
    }
}

// GCOV_EXCL_BR_STOP
