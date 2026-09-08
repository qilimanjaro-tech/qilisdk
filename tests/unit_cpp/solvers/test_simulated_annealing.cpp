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

// GCOV_EXCL_BR_START

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../../../src/qilisdk_cpp/solvers/simulated_annealing.h"

namespace {

// A frustrated cost function over four variables whose unique minimum is 1010, costing -5
std::vector<std::pair<std::vector<int>, double>> frustrated_monomials() {
    return {{{0}, -2.0}, {{1}, -1.0}, {{2}, -3.0}, {{3}, 1.5}, {{0, 1}, 1.0}, {{1, 2}, 4.0}, {{2, 3}, -1.0}};
}

// A cost function with a degree-three monomial, so the annealer must use the general (non-quadratic)
// path. Its unique minimum is 111, costing -4.
std::vector<std::pair<std::vector<int>, double>> cubic_monomials() {
    return {{{0}, -1.0}, {{1}, -1.0}, {{2}, -1.0}, {{0, 1}, 2.0}, {{0, 1, 2}, -3.0}};
}

}  // namespace

TEST(SimulatedAnnealingTest, EnergyEvaluatesMonomialsAndOffset) {
    SimulatedAnnealingCpp annealer(2, {{{0}, 1.5}, {{1}, -2.0}, {{0, 1}, 4.0}}, 0.5);
    EXPECT_DOUBLE_EQ(annealer.energy({0, 0}), 0.5);
    EXPECT_DOUBLE_EQ(annealer.energy({1, 0}), 2.0);
    EXPECT_DOUBLE_EQ(annealer.energy({0, 1}), -1.5);
    EXPECT_DOUBLE_EQ(annealer.energy({1, 1}), 4.0);
}

TEST(SimulatedAnnealingTest, ConstantMonomialsAreFoldedIntoTheOffset) {
    SimulatedAnnealingCpp annealer(1, {{{}, 3.0}, {{0}, 1.0}}, 1.0);
    EXPECT_EQ(annealer.get_num_monomials(), 1u);
    EXPECT_DOUBLE_EQ(annealer.get_offset(), 4.0);
    EXPECT_DOUBLE_EQ(annealer.energy({0}), 4.0);
    EXPECT_DOUBLE_EQ(annealer.energy({1}), 5.0);
}

TEST(SimulatedAnnealingTest, ConstructorRejectsInvalidInput) {
    EXPECT_THROW(SimulatedAnnealingCpp(-1, {}, 0.0), std::invalid_argument);
    EXPECT_THROW(SimulatedAnnealingCpp(2, {{{2}, 1.0}}, 0.0), std::invalid_argument);
    EXPECT_THROW(SimulatedAnnealingCpp(2, {{{-1}, 1.0}}, 0.0), std::invalid_argument);
}

TEST(SimulatedAnnealingTest, EnergyRejectsMismatchedState) {
    SimulatedAnnealingCpp annealer(2, {{{0}, 1.0}}, 0.0);
    EXPECT_THROW(annealer.energy({1}), std::invalid_argument);
}

TEST(SimulatedAnnealingTest, AnnealFindsTheMinimumOfAFrustratedCostFunction) {
    SimulatedAnnealingCpp annealer(4, frustrated_monomials(), 0.0);
    const AnnealingResultCpp result = annealer.anneal(20, 200, 0.0, 0.0, 42, 1);
    EXPECT_EQ(result.state, std::vector<int>({1, 0, 1, 0}));
    EXPECT_DOUBLE_EQ(result.energy, -5.0);
    EXPECT_DOUBLE_EQ(result.energy, annealer.energy(result.state));
}

TEST(SimulatedAnnealingTest, AnnealHandlesHigherOrderMonomials) {
    // The degree-three monomial forces the general (non-quadratic) annealing path.
    SimulatedAnnealingCpp annealer(3, cubic_monomials(), 0.0);
    EXPECT_DOUBLE_EQ(annealer.energy({1, 1, 1}), -4.0);
    const AnnealingResultCpp result = annealer.anneal(20, 200, 0.0, 0.0, 42, 1);
    EXPECT_EQ(result.state, std::vector<int>({1, 1, 1}));
    EXPECT_DOUBLE_EQ(result.energy, -4.0);
    EXPECT_DOUBLE_EQ(result.energy, annealer.energy(result.state));
}

TEST(SimulatedAnnealingTest, AnnealReportsEveryRead) {
    SimulatedAnnealingCpp annealer(4, frustrated_monomials(), 0.0);
    const AnnealingResultCpp result = annealer.anneal(7, 50, 0.0, 0.0, 1, 1);
    EXPECT_EQ(result.energies.size(), 7u);
    EXPECT_EQ(result.states.size(), 7u);
    for (std::size_t read = 0; read < result.energies.size(); ++read) {
        EXPECT_DOUBLE_EQ(result.energies[read], annealer.energy(result.states[read]));
        EXPECT_GE(result.energies[read], result.energy);
    }
}

TEST(SimulatedAnnealingTest, AnnealIsDeterministicAndThreadCountIndependent) {
    SimulatedAnnealingCpp annealer(4, frustrated_monomials(), 0.0);
    const AnnealingResultCpp serial = annealer.anneal(16, 100, 0.0, 0.0, 5, 1);
    const AnnealingResultCpp parallel = annealer.anneal(16, 100, 0.0, 0.0, 5, 4);
    EXPECT_EQ(serial.energies, parallel.energies);
    EXPECT_EQ(serial.state, parallel.state);
}

TEST(SimulatedAnnealingTest, AnnealAcceptsAnExplicitBetaRangeAndASingleSweep) {
    SimulatedAnnealingCpp annealer(4, frustrated_monomials(), 0.0);
    const AnnealingResultCpp result = annealer.anneal(50, 1, 0.01, 10.0, 3, 1);
    EXPECT_DOUBLE_EQ(result.energy, annealer.energy(result.state));
    EXPECT_LE(result.energy, 0.0);
}

TEST(SimulatedAnnealingTest, AnnealRejectsInvalidSettings) {
    SimulatedAnnealingCpp annealer(4, frustrated_monomials(), 0.0);
    EXPECT_THROW(annealer.anneal(0, 10, 0.0, 0.0, 1, 1), std::invalid_argument);
    EXPECT_THROW(annealer.anneal(10, 0, 0.0, 0.0, 1, 1), std::invalid_argument);
    EXPECT_THROW(annealer.anneal(10, 10, 1.0, 0.0, 1, 1), std::invalid_argument);
    EXPECT_THROW(annealer.anneal(10, 10, 0.0, 1.0, 1, 1), std::invalid_argument);
    EXPECT_THROW(annealer.anneal(10, 10, 10.0, 1.0, 1, 1), std::invalid_argument);
}

TEST(SimulatedAnnealingTest, DefaultBetaRangeCoolsDownAndHandlesAFlatCostFunction) {
    SimulatedAnnealingCpp annealer(4, frustrated_monomials(), 0.0);
    const std::pair<double, double> range = annealer.default_beta_range();
    EXPECT_GT(range.first, 0.0);
    EXPECT_GT(range.second, range.first);

    // Nothing to derive a temperature from, so any valid range will do
    SimulatedAnnealingCpp flat(3, {{{0}, 0.0}}, 2.0);
    const std::pair<double, double> flat_range = flat.default_beta_range();
    EXPECT_GT(flat_range.first, 0.0);
    EXPECT_GT(flat_range.second, flat_range.first);
}

TEST(SimulatedAnnealingTest, AnnealingNothingReturnsTheOffset) {
    SimulatedAnnealingCpp annealer(0, {{{}, 1.5}}, 0.5);
    const AnnealingResultCpp result = annealer.anneal(3, 10, 0.0, 0.0, 1, 1);
    EXPECT_TRUE(result.state.empty());
    EXPECT_DOUBLE_EQ(result.energy, 2.0);
    EXPECT_EQ(result.energies, std::vector<double>({2.0, 2.0, 2.0}));
}

// GCOV_EXCL_BR_STOP
