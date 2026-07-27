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

#include "simulated_annealing.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include "../libs/logging.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

SimulatedAnnealingCpp::SimulatedAnnealingCpp(int num_variables, const std::vector<std::pair<std::vector<int>, double>>& monomials, double offset) : num_variables(num_variables), offset(offset) {
    /*
    Build the cost function from a list of monomials over binary variables.

    Args:
        num_variables (int): the number of binary variables in the cost function.
        monomials (std::vector<std::pair<std::vector<int>, double>>&): the monomials of the cost
                function, each given as a list of variable indices and the coefficient multiplying
                their product.
        offset (double): a constant added to every cost evaluation.

    Raises:
        std::invalid_argument: if the number of variables is negative, or a monomial refers to a
                variable index that is out of range.
    */
    if (num_variables < 0) {
        throw std::invalid_argument("The number of variables must not be negative.");
    }
    this->variable_monomials.resize(static_cast<std::size_t>(num_variables));
    for (const auto& entry : monomials) {
        // If it has no variables, it is just a constant
        if (entry.first.empty()) {
            this->offset += entry.second;
            continue;
        }

        // Loop over the variables to make sure they are in range
        for (int variable : entry.first) {
            if (variable < 0 || variable >= num_variables) {
                throw std::invalid_argument("Monomial refers to variable index " + std::to_string(variable) + ", which is outside of the range [0, " + std::to_string(num_variables) + ").");
            }
        }

        // Add the monomial to the cost function and record which monomials each variable is in
        const int index = static_cast<int>(this->monomials.size());
        this->monomials.push_back(MonomialCpp{entry.first, entry.second});
        for (int variable : entry.first) {
            this->variable_monomials[static_cast<std::size_t>(variable)].push_back(index);
        }
    }
}

double SimulatedAnnealingCpp::energy(const std::vector<int>& state) const {
    /*
    Evaluate the cost function for a given assignment of the binary variables.

    Args:
        state (std::vector<int>&): the value (0 or 1) of each binary variable.

    Returns:
        double: the cost of the given assignment, including the constant offset.

    Raises:
        std::invalid_argument: if the state does not have one value per variable.
    */
    if (state.size() != static_cast<std::size_t>(num_variables)) {
        throw std::invalid_argument("The state has " + std::to_string(state.size()) + " values, but the cost function has " + std::to_string(num_variables) + " variables.");
    }
    double total = offset;
    for (const auto& monomial : monomials) {
        bool all_one = true;
        for (int variable : monomial.variables) {
            if (state[static_cast<std::size_t>(variable)] == 0) {
                all_one = false;
                break;
            }
        }
        if (all_one) {
            total += monomial.coefficient;
        }
    }
    return total;
}

double SimulatedAnnealingCpp::flip_delta(const std::vector<int>& state, int variable) const {
    /*
    Compute the change in cost caused by flipping a single binary variable.

    Args:
        state (std::vector<int>&): the current value (0 or 1) of each binary variable.
        variable (int): the index of the variable to flip.

    Returns:
        double: the cost of the flipped state minus the cost of the current state.
    */
    const bool currently_one = state[static_cast<std::size_t>(variable)] == 1;
    double delta = 0.0;
    for (int index : variable_monomials[static_cast<std::size_t>(variable)]) {
        const MonomialCpp& monomial = monomials[static_cast<std::size_t>(index)];
        bool others_all_one = true;
        for (int other : monomial.variables) {
            if (other != variable && state[static_cast<std::size_t>(other)] == 0) {
                others_all_one = false;
                break;
            }
        }
        if (others_all_one) {
            delta += currently_one ? -monomial.coefficient : monomial.coefficient;
        }
    }
    return delta;
}

std::pair<double, double> SimulatedAnnealingCpp::default_beta_range() const {
    /*
    Derive a sensible inverse temperature range from the magnitudes of the coefficients.

    The hot end is set from the largest cost change any single bit flip can cause, so that even the
    worst flip is accepted with a reasonable probability at the start of the anneal. The cold end is
    set from the smallest cost change a bit flip can cause, so that by the end of the anneal even
    the finest energy difference is resolved.

    Returns:
        std::pair<double, double>: the (minimum, maximum) inverse temperature to anneal over.
    */

    // The probability of accepting the largest flip at the hot end
    constexpr double kHotAcceptanceProbability = 0.5;

    // The probability of accepting the smallest flip at the cold end
    constexpr double kColdAcceptanceProbability = 0.01;

    // Determine the largest and smallest cost change any single bit flip can cause
    double largest_delta = 0.0;
    double smallest_delta = std::numeric_limits<double>::infinity();
    for (int variable = 0; variable < num_variables; ++variable) {
        double bound = 0.0;
        for (int index : variable_monomials[static_cast<std::size_t>(variable)]) {
            const double magnitude = std::abs(monomials[static_cast<std::size_t>(index)].coefficient);
            bound += magnitude;
            if (magnitude > 0.0) {
                smallest_delta = std::min(smallest_delta, magnitude);
            }
        }
        largest_delta = std::max(largest_delta, bound);
    }

    // A cost function with no monomials has no energy differences, so the range doesn't matter
    if (largest_delta == 0.0 || !std::isfinite(smallest_delta)) {
        return {0.1, 1.0};
    }

    return {-std::log(kHotAcceptanceProbability) / largest_delta, -std::log(kColdAcceptanceProbability) / smallest_delta};
}

std::pair<std::vector<int>, double> SimulatedAnnealingCpp::single_read(int num_sweeps, double beta_min, double beta_max, unsigned long long seed) const {
    /*
    Run a single anneal from a random starting assignment.

    The inverse temperature is ramped geometrically from beta_min to beta_max, and each sweep offers
    every variable in turn a Metropolis flip.

    Args:
        num_sweeps (int): the number of sweeps over all variables to perform.
        beta_min (double): the inverse temperature to start the anneal at.
        beta_max (double): the inverse temperature to end the anneal at.
        seed (unsigned long long): the seed of this read's random number generator.

    Returns:
        std::pair<std::vector<int>, double>: the lowest cost assignment visited and its cost.
    */

    // Prep the rng
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Start from a random assignment
    std::vector<int> state(static_cast<std::size_t>(num_variables));
    for (int variable = 0; variable < num_variables; ++variable) {
        state[static_cast<std::size_t>(variable)] = uniform(generator) < 0.5 ? 0 : 1;
    }
    double current_energy = energy(state);
    std::vector<int> best_state = state;
    double best_energy = current_energy;

    // Ramp geometrically, i.e. linearly in the logarithm of the inverse temperature
    const double log_beta_min = std::log(beta_min);
    const double log_beta_step = num_sweeps > 1 ? (std::log(beta_max) - log_beta_min) / static_cast<double>(num_sweeps - 1) : 0.0;

    // Perform the sweeps
    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        // The inverse temperature
        const double beta = std::exp(log_beta_min + log_beta_step * static_cast<double>(sweep));

        // Loop over the variables in order, offering each a Metropolis flip
        for (int variable = 0; variable < num_variables; ++variable) {
            // How much the cost would change if we flipped this variable
            const double delta = flip_delta(state, variable);

            // Downhill flips are always taken, uphill ones only with the probability based on the temp
            if (delta <= 0.0 || uniform(generator) < std::exp(-beta * delta)) {
                state[static_cast<std::size_t>(variable)] ^= 1;
                current_energy += delta;
                if (current_energy < best_energy) {
                    best_energy = current_energy;
                    best_state = state;
                }
            }
        }
    }
    return {best_state, best_energy};
}

AnnealingResultCpp SimulatedAnnealingCpp::anneal(int num_reads, int num_sweeps, double beta_min, double beta_max, int seed, int num_threads) {
    /*
    Minimize the cost function with simulated annealing.

    Each read is an independent anneal from its own random starting assignment, and the reads are
    distributed over the available threads.

    Args:
        num_reads (int): the number of independent anneals to run.
        num_sweeps (int): the number of sweeps over all variables in each anneal.
        beta_min (double): the inverse temperature to start each anneal at. If this is not positive,
                a range is derived from the coefficients of the cost function instead.
        beta_max (double): the inverse temperature to end each anneal at. If this is not positive, a
                range is derived from the coefficients of the cost function instead.
        seed (int): the seed of the random number generators, each read deriving its own from it.
        num_threads (int): the number of threads to distribute the reads over, or zero to let
                OpenMP decide.

    Returns:
        AnnealingResultCpp: the best assignment found and its cost, along with the best assignment
                and cost of each individual read.

    Raises:
        std::invalid_argument: if the number of reads or sweeps is not positive, or if only one of
                the two inverse temperatures is positive, or if beta_min is above beta_max.
    */
    if (num_reads <= 0) {
        throw std::invalid_argument("The number of reads must be positive.");
    }
    if (num_sweeps <= 0) {
        throw std::invalid_argument("The number of sweeps must be positive.");
    }
    if ((beta_min <= 0.0) != (beta_max <= 0.0)) {
        throw std::invalid_argument("The inverse temperatures must either both be positive, or both be non-positive to have them derived from the cost function.");
    }
    if (beta_min > 0.0) {
        if (beta_min > beta_max) {
            throw std::invalid_argument("The initial inverse temperature must not be above the final one, since the anneal has to cool down.");
        }
    } else {
        const std::pair<double, double> range = default_beta_range();
        beta_min = range.first;
        beta_max = range.second;
        qilisdk::log_debug("[SimulatedAnnealing, C++] Derived an inverse temperature range of [" + std::to_string(beta_min) + ", " + std::to_string(beta_max) + "] from the cost function");
    }
    qilisdk::log_debug("[SimulatedAnnealing, C++] Annealing " + std::to_string(num_variables) + " variables and " + std::to_string(monomials.size()) + " monomials with " + std::to_string(num_reads) + " reads of " + std::to_string(num_sweeps) + " sweeps");

    // An empty cost function has a single (trivial) solution, so there is nothing to anneal
    if (num_variables == 0) {
        return AnnealingResultCpp{std::vector<int>(), offset, std::vector<std::vector<int>>(static_cast<std::size_t>(num_reads)), std::vector<double>(static_cast<std::size_t>(num_reads), offset)};
    }

    std::vector<std::vector<int>> states(static_cast<std::size_t>(num_reads));
    std::vector<double> energies(static_cast<std::size_t>(num_reads));

#if defined(_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#pragma omp parallel for schedule(static)
#endif
    for (int read = 0; read < num_reads; ++read) {
        // Each read gets its own generator, so that the result does not depend on the thread count
        const std::pair<std::vector<int>, double> outcome = single_read(num_sweeps, beta_min, beta_max, static_cast<unsigned long long>(seed) + 0x9e3779b97f4a7c15ULL * static_cast<unsigned long long>(read + 1));
        states[static_cast<std::size_t>(read)] = outcome.first;
        energies[static_cast<std::size_t>(read)] = outcome.second;
    }

    // Pick the best read, preferring the earliest one so that the result is deterministic
    std::size_t best_read = 0;
    for (std::size_t read = 1; read < energies.size(); ++read) {
        if (energies[read] < energies[best_read]) {
            best_read = read;
        }
    }
    qilisdk::log_debug("[SimulatedAnnealing, C++] Best cost found was " + std::to_string(energies[best_read]) + " (read " + std::to_string(best_read) + ")");
    return AnnealingResultCpp{states[best_read], energies[best_read], states, energies};
}

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
