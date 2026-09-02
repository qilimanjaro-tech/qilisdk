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
#include <pybind11/embed.h>

#include <cmath>
#include <complex>
#include <functional>
#include <string>
#include <vector>

#include "../../../src/qilisdk_cpp/backends/qilisim/noise/monte_carlo.h"
#include "../../../src/qilisdk_cpp/libs/logging.h"
#include "../../../src/qilisdk_cpp/libs/pybind.h"

namespace py = pybind11;

namespace {

using cx = std::complex<double>;
constexpr double kTol = 1e-9;
constexpr double kLoose = 0.03;

SparseMatrix toSparse(const DenseMatrix& dense) {
    SparseMatrix sparse(dense.rows(), dense.cols());
    sparse = dense.sparseView();
    return sparse;
}

// sqrt(rate) * sigma_minus, i.e. amplitude damping onto |0>
SparseMatrix dampingJump(double rate = 1.0) {
    DenseMatrix jump = DenseMatrix::Zero(2, 2);
    jump(0, 1) = std::sqrt(rate);
    return toSparse(jump);
}

// The dephasing jump sqrt(rate) * Z, which never moves population between levels
SparseMatrix dephasingJump(double rate = 1.0) {
    DenseMatrix jump = DenseMatrix::Zero(2, 2);
    jump(0, 0) = std::sqrt(rate);
    jump(1, 1) = -std::sqrt(rate);
    return toSparse(jump);
}

// A batch of n copies of |1>
DenseMatrix excitedTrajectories(long n) {
    DenseMatrix trajectories = DenseMatrix::Zero(2, n);
    trajectories.row(1).setOnes();
    return trajectories;
}

bool containsString(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Run `emit` with a sink attached to the logger, and return every warning it produced
py::list captureWarnings(const std::function<void()>& emit) {
    py::list records;
    logger.attr("remove")();
    auto sink = py::cpp_function([records](py::object message) mutable { records.append(py::str(message)); });
    logger.attr("add")(sink, py::arg("format") = "{message}", py::arg("level") = "WARNING");
    qilisdk::refresh_log_level();
    emit();
    logger.attr("remove")();
    qilisdk::refresh_log_level();
    return records;
}

// The Kraus operators of amplitude damping with jump probability p
std::vector<SparseMatrix> dampingKraus(double p) {
    DenseMatrix K0 = DenseMatrix::Zero(2, 2);
    K0(0, 0) = 1.0;
    K0(1, 1) = std::sqrt(1.0 - p);
    DenseMatrix K1 = DenseMatrix::Zero(2, 2);
    K1(0, 1) = std::sqrt(p);
    return {toSparse(K0), toSparse(K1)};
}

double excitedFraction(const DenseMatrix& trajectories) {
    double excited = 0.0;
    for (long c = 0; c < trajectories.cols(); ++c) {
        excited += std::norm(trajectories(1, c));
    }
    return excited / static_cast<double>(trajectories.cols());
}

}  // namespace

TEST(JumpDriftOperatorTest, EmptyJumpListGivesEmptyDrift) {
    EXPECT_EQ(jump_drift_operator({}).rows(), 0);
}

TEST(JumpDriftOperatorTest, DriftIsHalfOfLdaggerL) {
    // sigma_minus^dagger sigma_minus = |1><1|, so the drift is 0.5 |1><1|
    SparseMatrix drift = jump_drift_operator({dampingJump(2.0)});
    EXPECT_NEAR(drift.coeff(1, 1).real(), 1.0, kTol);
    EXPECT_NEAR(std::abs(drift.coeff(0, 0)), 0.0, kTol);
}

TEST(JumpDriftOperatorTest, MultipleJumpsAccumulate) {
    SparseMatrix drift = jump_drift_operator({dampingJump(1.0), dephasingJump(1.0)});
    // 0.5 * (|1><1| + Z^dagger Z) = 0.5 * (|1><1| + identity)
    EXPECT_NEAR(drift.coeff(0, 0).real(), 0.5, kTol);
    EXPECT_NEAR(drift.coeff(1, 1).real(), 1.0, kTol);
}

TEST(EffectiveHamiltonianTest, NoDriftLeavesTheHamiltonianAlone) {
    DenseMatrix H = DenseMatrix::Zero(2, 2);
    H(0, 0) = 1.0;
    H(1, 1) = -1.0;
    SparseMatrix H_eff = effective_hamiltonian(toSparse(H), SparseMatrix());
    EXPECT_TRUE(DenseMatrix(H_eff).isApprox(H, kTol));
}

TEST(EffectiveHamiltonianTest, DriftEntersAsAnImaginaryShift) {
    DenseMatrix H = DenseMatrix::Zero(2, 2);
    H(1, 1) = 3.0;
    SparseMatrix H_eff = effective_hamiltonian(toSparse(H), jump_drift_operator({dampingJump(1.0)}));
    EXPECT_NEAR(H_eff.coeff(1, 1).real(), 3.0, kTol);
    EXPECT_NEAR(H_eff.coeff(1, 1).imag(), -0.5, kTol);
}

TEST(EffectiveHamiltonianTest, IsNotHermitianSoTheNormDecays) {
    SparseMatrix H_eff = effective_hamiltonian(SparseMatrix(2, 2), jump_drift_operator({dampingJump(1.0)}));
    DenseMatrix dense(H_eff);
    EXPECT_FALSE(dense.isApprox(dense.adjoint(), kTol));
}

TEST(MaxJumpRateBoundTest, NoDriftHasNoRate) {
    EXPECT_NEAR(max_jump_rate_bound(SparseMatrix()), 0.0, kTol);
}

TEST(MaxJumpRateBoundTest, BoundsTheActualRate) {
    // For a rate-3 damping jump the drift is 1.5 |1><1|, so the excited state jumps at rate 3
    const double bound = max_jump_rate_bound(jump_drift_operator({dampingJump(3.0)}));
    EXPECT_GE(bound, 3.0 - kTol);
}

TEST(TrajectoryUnravelingJumpsTest, NoJumpOperatorsLeavesTheBatchUntouched) {
    TrajectoryUnraveling unraveling(1);
    DenseMatrix trajectories = excitedTrajectories(4) * 0.5;  // deliberately not normalized
    DenseMatrix before = trajectories;
    unraveling.apply_jumps(trajectories, {});
    EXPECT_TRUE(trajectories.isApprox(before, kTol));
}

TEST(TrajectoryUnravelingJumpsTest, EveryTrajectoryEndsUpNormalized) {
    TrajectoryUnraveling unraveling(2);
    DenseMatrix trajectories = excitedTrajectories(32) * 0.3;
    unraveling.apply_jumps(trajectories, {dampingJump()});
    for (long c = 0; c < trajectories.cols(); ++c) {
        EXPECT_NEAR(trajectories.col(c).norm(), 1.0, kTol);
    }
}

TEST(TrajectoryUnravelingJumpsTest, FullNormMeansNoJump) {
    // A trajectory that lost no norm has survival probability one and can never jump
    TrajectoryUnraveling unraveling(3);
    DenseMatrix trajectories = excitedTrajectories(64);
    unraveling.apply_jumps(trajectories, {dampingJump()});
    EXPECT_NEAR(excitedFraction(trajectories), 1.0, kTol);
}

TEST(TrajectoryUnravelingJumpsTest, LostNormBecomesTheJumpProbability) {
    // Trajectories scaled to a squared norm of 0.75 must jump a quarter of the time
    TrajectoryUnraveling unraveling(4);
    DenseMatrix trajectories = excitedTrajectories(20000) * std::sqrt(0.75);
    unraveling.apply_jumps(trajectories, {dampingJump()});
    // Amplitude damping takes |1> to |0>, so the surviving excited fraction is the survival probability
    EXPECT_NEAR(excitedFraction(trajectories), 0.75, kLoose);
}

TEST(TrajectoryUnravelingJumpsTest, JumpChannelIsChosenByItsWeight) {
    // Damping moves |1> to |0> while dephasing leaves it excited, and both are equally likely here,
    // so half of the jumping trajectories should end up relaxed
    TrajectoryUnraveling unraveling(5);
    DenseMatrix trajectories = excitedTrajectories(20000) * std::sqrt(0.5);
    unraveling.apply_jumps(trajectories, {dampingJump(), dephasingJump()});
    EXPECT_NEAR(excitedFraction(trajectories), 0.75, kLoose);
}

TEST(TrajectoryUnravelingJumpsTest, SameSeedGivesTheSameTrajectories) {
    DenseMatrix first = excitedTrajectories(128) * std::sqrt(0.5);
    DenseMatrix second = first;
    TrajectoryUnraveling one(6);
    TrajectoryUnraveling two(6);
    one.apply_jumps(first, {dampingJump()});
    two.apply_jumps(second, {dampingJump()});
    EXPECT_TRUE(first.isApprox(second, kTol));
}

TEST(TrajectoryUnravelingJumpsTest, DifferentSeedsGiveDifferentTrajectories) {
    DenseMatrix first = excitedTrajectories(128) * std::sqrt(0.5);
    DenseMatrix second = first;
    TrajectoryUnraveling one(7);
    TrajectoryUnraveling two(8);
    one.apply_jumps(first, {dampingJump()});
    two.apply_jumps(second, {dampingJump()});
    EXPECT_FALSE(first.isApprox(second, kTol));
}

TEST(TrajectoryUnravelingJumpsTest, ConsecutiveStepsAreNotCorrelated) {
    // Each trajectory keeps advancing its own stream, so two steps of the same unravelling must not
    // reproduce each other's jump pattern
    DenseMatrix trajectories = excitedTrajectories(512) * std::sqrt(0.5);
    TrajectoryUnraveling unraveling(9);
    unraveling.apply_jumps(trajectories, {dampingJump()});
    DenseMatrix after_first_step = trajectories;
    trajectories *= std::sqrt(0.5);
    unraveling.apply_jumps(trajectories, {dampingJump()});

    int relaxed_in_first = 0;
    int relaxed_in_both = 0;
    for (long c = 0; c < trajectories.cols(); ++c) {
        const bool first_relaxed = std::abs(after_first_step(0, c)) > 0.5;
        const bool second_relaxed = std::abs(trajectories(0, c)) > 0.5;
        relaxed_in_first += first_relaxed ? 1 : 0;
        relaxed_in_both += (first_relaxed && second_relaxed) ? 1 : 0;
    }
    // Relaxing twice is not the same event as relaxing once: about half of the trajectories that
    // stayed excited should relax in the second step as well
    EXPECT_GT(relaxed_in_first, 0);
    EXPECT_LT(relaxed_in_both, trajectories.cols());
}

TEST(TrajectoryUnravelingJumpsTest, CollapsedTrajectoryThrows) {
    TrajectoryUnraveling unraveling(10);
    DenseMatrix trajectories = DenseMatrix::Zero(2, 2);
    EXPECT_ANY_THROW(unraveling.apply_jumps(trajectories, {dampingJump()}));
}

TEST(TrajectoryUnravelingJumpsTest, NonFiniteTrajectoryThrows) {
    TrajectoryUnraveling unraveling(11);
    DenseMatrix trajectories = excitedTrajectories(2);
    trajectories(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_ANY_THROW(unraveling.apply_jumps(trajectories, {dampingJump()}));
}

TEST(TrajectoryUnravelingJumpsTest, StateWithNoAvailableChannelKeepsEvolving) {
    // The ground state cannot be damped any further: the jump has zero weight everywhere, so the
    // trajectory has to survive rather than be renormalized from nothing
    TrajectoryUnraveling unraveling(12);
    DenseMatrix trajectories = DenseMatrix::Zero(2, 4);
    trajectories.row(0).setConstant(std::sqrt(0.5));  // |0>, scaled so that a jump is always drawn
    unraveling.apply_jumps(trajectories, {dampingJump()});
    for (long c = 0; c < trajectories.cols(); ++c) {
        EXPECT_NEAR(std::abs(trajectories(0, c)), 1.0, kTol);
    }
}

TEST(TrajectoryUnravelingKrausTest, EmptyChannelLeavesTheBatchUntouched) {
    TrajectoryUnraveling unraveling(13);
    DenseMatrix trajectories = excitedTrajectories(4);
    DenseMatrix before = trajectories;
    unraveling.apply_kraus(trajectories, {});
    EXPECT_TRUE(trajectories.isApprox(before, kTol));
}

TEST(TrajectoryUnravelingKrausTest, ChannelIsReproducedOnAverage) {
    TrajectoryUnraveling unraveling(14);
    DenseMatrix trajectories = excitedTrajectories(20000);
    unraveling.apply_kraus(trajectories, dampingKraus(0.3));
    EXPECT_NEAR(excitedFraction(trajectories), 0.7, kLoose);
    for (long c = 0; c < trajectories.cols(); ++c) {
        EXPECT_NEAR(trajectories.col(c).norm(), 1.0, kTol);
    }
}

TEST(TrajectoryUnravelingKrausTest, ADeterministicChannelActsOnEveryTrajectory) {
    TrajectoryUnraveling unraveling(15);
    DenseMatrix trajectories = excitedTrajectories(16);
    unraveling.apply_kraus(trajectories, dampingKraus(1.0));
    EXPECT_NEAR(excitedFraction(trajectories), 0.0, kTol);
}

TEST(TrajectoryUnravelingKrausTest, SameSeedGivesTheSameTrajectories) {
    DenseMatrix first = excitedTrajectories(128);
    DenseMatrix second = first;
    TrajectoryUnraveling one(16);
    TrajectoryUnraveling two(16);
    one.apply_kraus(first, dampingKraus(0.5));
    two.apply_kraus(second, dampingKraus(0.5));
    EXPECT_TRUE(first.isApprox(second, kTol));
}

TEST(TrajectoryUnravelingKrausTest, CollapsedTrajectoryThrows) {
    TrajectoryUnraveling unraveling(17);
    DenseMatrix trajectories = DenseMatrix::Zero(2, 2);
    EXPECT_ANY_THROW(unraveling.apply_kraus(trajectories, dampingKraus(0.5)));
}

TEST(JumpResolutionWarningTest, WarnsOnceForACoarseSchedule) {
    // A drift whose jump rate makes each of these steps carry far more than one jump
    const SparseMatrix drift = jump_drift_operator({dampingJump(100.0)});
    const std::vector<double> coarse_schedule = {0.0, 1.0, 2.0};
    reset_jump_resolution_warning();

    // Repeat the evolution the way a reservoir would, and expect a single warning for the lot
    py::list records = captureWarnings([&]() {
        for (int repeat = 0; repeat < 10; ++repeat) {
            warn_if_jumps_underresolved(drift, coarse_schedule, 0.5);
        }
    });

    ASSERT_EQ(py::len(records), 1u);
    EXPECT_TRUE(containsString(records[0].cast<std::string>(), "poorly resolved in time"));
}

TEST(JumpResolutionWarningTest, StaysSilentForAFineSchedule) {
    const SparseMatrix drift = jump_drift_operator({dampingJump(1.0)});
    const std::vector<double> fine_schedule = {0.0, 1e-3, 2e-3};
    reset_jump_resolution_warning();

    py::list records = captureWarnings([&]() { warn_if_jumps_underresolved(drift, fine_schedule, 0.5); });

    EXPECT_EQ(py::len(records), 0u);
}

TEST(JumpResolutionWarningTest, ThresholdDecidesWhetherTheScheduleIsTooCoarse) {
    // Two jumps per step: 2 * (largest row sum of D = 0.5) * dt, with dt = 2
    const SparseMatrix drift = jump_drift_operator({dampingJump(1.0)});
    const std::vector<double> schedule = {0.0, 2.0};

    reset_jump_resolution_warning();
    py::list tolerant = captureWarnings([&]() { warn_if_jumps_underresolved(drift, schedule, 5.0); });
    EXPECT_EQ(py::len(tolerant), 0u);

    reset_jump_resolution_warning();
    py::list strict = captureWarnings([&]() { warn_if_jumps_underresolved(drift, schedule, 0.1); });
    EXPECT_EQ(py::len(strict), 1u);
}

TEST(JumpResolutionWarningTest, WarningStateSurvivesADifferentDrift) {
    const SparseMatrix strong = jump_drift_operator({dampingJump(100.0)});
    const SparseMatrix weak = jump_drift_operator({dampingJump(50.0)});
    const std::vector<double> coarse_schedule = {0.0, 1.0};
    reset_jump_resolution_warning();

    py::list records = captureWarnings([&]() {
        warn_if_jumps_underresolved(strong, coarse_schedule, 0.5);
        warn_if_jumps_underresolved(weak, coarse_schedule, 0.5);
    });

    EXPECT_EQ(py::len(records), 1u);
}

TEST(ScheduleStepExtremesTest, IgnoresNonPositiveGaps) {
    const std::pair<double, double> extremes = schedule_step_extremes({0.0, 1.0, 1.0, 4.0});
    EXPECT_NEAR(extremes.first, 1.0, kTol);
    EXPECT_NEAR(extremes.second, 3.0, kTol);
}

TEST(ScheduleStepExtremesTest, AnEmptyScheduleBoundsNothing) {
    const std::pair<double, double> extremes = schedule_step_extremes({});
    EXPECT_TRUE(std::isinf(extremes.first));
    EXPECT_EQ(extremes.second, 0.0);
}

// GCOV_EXCL_BR_STOP
