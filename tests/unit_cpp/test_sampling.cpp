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

#include <gtest/gtest.h>

#include "../../src/qilisdk_cpp/backends/qilisim/digital/sampling.h"

#include <gtest/gtest.h>
#include <complex>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace {

using cx = std::complex<double>;
constexpr double kTol    = 1e-6;
constexpr double kLoose  = 0.05;

SparseMatrix sparseIdentity(int dim) {
    SparseMatrix m(dim, dim);
    for (int i = 0; i < dim; ++i) m.insert(i, i) = cx(1, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix pauliX2() {
    SparseMatrix m(2, 2);
    m.insert(0, 1) = cx(1, 0);
    m.insert(1, 0) = cx(1, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix hadamard2() {
    double v = 1.0 / std::sqrt(2.0);
    SparseMatrix m(2, 2);
    m.insert(0, 0) = cx( v, 0); m.insert(0, 1) = cx( v, 0);
    m.insert(1, 0) = cx( v, 0); m.insert(1, 1) = cx(-v, 0);
    m.makeCompressed();
    return m;
}

DenseMatrix zeroStatevector(int n_qubits) {
    long dim = 1L << n_qubits;
    DenseMatrix v = DenseMatrix::Zero(dim, 1);
    v(0, 0) = cx(1, 0);
    return v;
}

// Sparse version of the zero-state column vector (used as initial_state).
SparseMatrix zeroStateSparse(int n_qubits) {
    long dim = 1L << n_qubits;
    SparseMatrix m(dim, 1);
    m.insert(0, 0) = cx(1, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix zeroStateDenseSparse(int n_qubits) {
    long dim = 1L << n_qubits;
    SparseMatrix m(dim, dim);
    m.insert(0, 0) = cx(1, 0);
    m.makeCompressed();
    return m;
}

Gate makeX(int qubit) {
    return Gate("X", pauliX2(), {}, {qubit}, {});
}

Gate makeH(int qubit) {
    return Gate("H", hadamard2(), {}, {qubit}, {});
}

int totalCounts(const std::map<std::string, int>& counts) {
    int total = 0;
    for (const auto& p : counts) total += p.second;
    return total;
}

double fractionOf(const std::map<std::string, int>& counts,
                  const std::string& key) {
    auto it = counts.find(key);
    if (it == counts.end()) return 0.0;
    return static_cast<double>(it->second) / totalCounts(counts);
}

NoiseModelCpp emptyNoise() {
    return NoiseModelCpp();
}

NoiseModelCpp symmetricReadoutNoise(int n_qubits, double p) {
    NoiseModelCpp nm;
    for (int q = 0; q < n_qubits; ++q) {
        nm.add_readout_error_per_qubit(q, p, p);
    }
    return nm;
}

QiliSimConfig defaultConfig() {
    QiliSimConfig cfg;
    cfg.set_combine_single_qubit_gates(false);
    cfg.set_normalize_after_gate(false);
    cfg.set_monte_carlo(false);
    cfg.set_seed(42);
    cfg.set_atol(1e-8);
    cfg.set_max_cache_size(64);
    cfg.set_num_threads(1);
    return cfg;
}

}

class ApplyReadoutErrorTest : public ::testing::Test {};

TEST_F(ApplyReadoutErrorTest, ZeroError_CountsUnchanged) {
    // p01 = p10 = 0 → readout is perfect, counts must be identical.
    std::map<std::string, int> counts = { {"00", 500}, {"11", 500} };
    auto nm = symmetricReadoutNoise(2, 0.0);
    auto result = apply_readout_error(counts, nm, 2);

    EXPECT_EQ(result.at("00"), 500);
    EXPECT_EQ(result.at("11"), 500);
    EXPECT_EQ(totalCounts(result), 1000);
}

TEST_F(ApplyReadoutErrorTest, TotalShotCountPreserved) {
    // Total shots must be conserved regardless of flip probabilities.
    std::map<std::string, int> counts = { {"0", 300}, {"1", 700} };
    auto nm = symmetricReadoutNoise(1, 0.1);
    auto result = apply_readout_error(counts, nm, 1);
    EXPECT_EQ(totalCounts(result), 1000);
}

TEST_F(ApplyReadoutErrorTest, PerfectFlip_AllOnesBecomesAllZeros) {
    // p10 = 1.0 means every '1' flips to '0'.
    // With all-ones input and p10=1, output must be all-zeros.
    std::map<std::string, int> counts = { {"11", 1000} };
    NoiseModelCpp nm;
    nm.add_readout_error_per_qubit(0, 0.0, 1.0);   // qubit 0: p01=0, p10=1
    nm.add_readout_error_per_qubit(1, 0.0, 1.0);   // qubit 1: p01=0, p10=1
    auto result = apply_readout_error(counts, nm, 2);

    ASSERT_EQ(result.count("00"), 1u);
    EXPECT_EQ(result.at("00"), 1000);
}

TEST_F(ApplyReadoutErrorTest, PerfectFlip_AllZerosBecomesAllOnes) {
    // p01 = 1.0 means every '0' flips to '1'.
    std::map<std::string, int> counts = { {"00", 1000} };
    NoiseModelCpp nm;
    nm.add_readout_error_global(1.0, 0.0);
    auto result = apply_readout_error(counts, nm, 2);

    ASSERT_EQ(result.count("11"), 1u);
    EXPECT_EQ(result.at("11"), 1000);
}

TEST_F(ApplyReadoutErrorTest, PartialError_ReducesDominantOutcome) {
    // With a small flip probability, the dominant outcome should still dominate
    // but with slightly reduced count.
    std::map<std::string, int> counts = { {"0", 10000} };
    auto nm = symmetricReadoutNoise(1, 0.1);
    auto result = apply_readout_error(counts, nm, 1);

    EXPECT_EQ(totalCounts(result), 10000);
    // Approx 90 % should remain as "0".
    double frac0 = fractionOf(result, "0");
    EXPECT_NEAR(frac0, 0.9, 0.05);
}

TEST_F(ApplyReadoutErrorTest, EmptyInputCounts_ReturnsEmpty) {
    std::map<std::string, int> counts;
    auto nm = symmetricReadoutNoise(2, 0.1);
    auto result = apply_readout_error(counts, nm, 2);
    EXPECT_TRUE(result.empty());
}

TEST_F(ApplyReadoutErrorTest, SingleQubit_OutputKeysAreValidBitstrings) {
    std::map<std::string, int> counts = { {"0", 500}, {"1", 500} };
    auto nm = symmetricReadoutNoise(1, 0.05);
    auto result = apply_readout_error(counts, nm, 1);
    for (const auto& p : result) {
        EXPECT_EQ(p.first.size(), 1u);
        EXPECT_TRUE(p.first == "0" || p.first == "1");
    }
}

TEST_F(ApplyReadoutErrorTest, MultiQubit_OutputKeysHaveCorrectLength) {
    std::map<std::string, int> counts = { {"000", 500}, {"111", 500} };
    auto nm = symmetricReadoutNoise(3, 0.05);
    auto result = apply_readout_error(counts, nm, 3);
    for (const auto& p : result) {
        EXPECT_EQ(p.first.size(), 3u);
    }
}

TEST_F(ApplyReadoutErrorTest, Deterministic_SameSeedSameResult) {
    // The function uses a fixed seed (42 inside), so calling it twice
    // with the same input should give the same output.
    std::map<std::string, int> counts = { {"0", 500}, {"1", 500} };
    auto nm = symmetricReadoutNoise(1, 0.2);
    auto r1 = apply_readout_error(counts, nm, 1);
    auto r2 = apply_readout_error(counts, nm, 1);
    EXPECT_EQ(r1, r2);
}

TEST_F(ApplyReadoutErrorTest, AsymmetricError_DifferentFlipRates) {
    // p01 = 0 (no 0→1), p10 = 1 (all 1→0).
    // Mixed input: "0" shots stay "0", "1" shots all become "0".
    std::map<std::string, int> counts = { {"0", 400}, {"1", 600} };
    NoiseModelCpp nm;
    nm.add_readout_error_per_qubit(0, 0.0, 1.0);
    auto result = apply_readout_error(counts, nm, 1);

    ASSERT_EQ(result.count("0"), 1u);
    EXPECT_EQ(result.at("0"), 1000);
    EXPECT_EQ(result.count("1"), 0u);
}


// ══════════════════════════════════════════════════════════════
// 2. filter_counts
// ══════════════════════════════════════════════════════════════

class FilterCountsTest : public ::testing::Test {};

TEST_F(FilterCountsTest, AllQubitsSelected_CountsUnchanged) {
    std::map<std::string, int> counts = { {"00", 300}, {"11", 700} };
    std::vector<bool> mask = {true, true};
    auto result = filter_counts(counts, mask);
    EXPECT_EQ(result, counts);
}

TEST_F(FilterCountsTest, NoQubitsSelected_SingleEmptyKey) {
    std::map<std::string, int> counts = { {"00", 300}, {"11", 700} };
    std::vector<bool> mask = {false, false};
    auto result = filter_counts(counts, mask);
    // All bitstrings collapse to "" and counts merge.
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.at(""), 1000);
}

TEST_F(FilterCountsTest, FirstQubitOnly_ProjectsCorrectly) {
    std::map<std::string, int> counts = { {"00", 200}, {"01", 300}, {"10", 100}, {"11", 400} };
    std::vector<bool> mask = {true, false};
    auto result = filter_counts(counts, mask);
    // "00","01" → "0" (500); "10","11" → "1" (500).
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.at("0"), 500);
    EXPECT_EQ(result.at("1"), 500);
}

TEST_F(FilterCountsTest, SecondQubitOnly_ProjectsCorrectly) {
    std::map<std::string, int> counts = { {"00", 200}, {"01", 300}, {"10", 100}, {"11", 400} };
    std::vector<bool> mask = {false, true};
    auto result = filter_counts(counts, mask);
    // "00","10" → "0" (300); "01","11" → "1" (700).
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.at("0"), 300);
    EXPECT_EQ(result.at("1"), 700);
}

TEST_F(FilterCountsTest, TotalCountPreserved) {
    std::map<std::string, int> counts = { {"000", 100}, {"010", 200}, {"101", 300}, {"111", 400} };
    std::vector<bool> mask = {true, false, true};
    auto result = filter_counts(counts, mask);
    EXPECT_EQ(totalCounts(result), 1000);
}

TEST_F(FilterCountsTest, MiddleQubitSelected_ThreeQubits) {
    // Keep only qubit 1 (middle).
    std::map<std::string, int> counts = { {"000", 125}, {"010", 125}, {"100", 125}, {"110", 125},
                                          {"001", 125}, {"011", 125}, {"101", 125}, {"111", 125} };
    std::vector<bool> mask = {false, true, false};
    auto result = filter_counts(counts, mask);
    // Qubit 1 is '0' for "000","100","001","101" → 500 counts.
    // Qubit 1 is '1' for "010","110","011","111" → 500 counts.
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.at("0"), 500);
    EXPECT_EQ(result.at("1"), 500);
}

TEST_F(FilterCountsTest, EmptyInput_ReturnsEmpty) {
    std::map<std::string, int> counts;
    std::vector<bool> mask = {true, true};
    auto result = filter_counts(counts, mask);
    EXPECT_TRUE(result.empty());
}

TEST_F(FilterCountsTest, SingleQubit_AllSelected) {
    std::map<std::string, int> counts = { {"0", 600}, {"1", 400} };
    std::vector<bool> mask = {true};
    auto result = filter_counts(counts, mask);
    EXPECT_EQ(result.at("0"), 600);
    EXPECT_EQ(result.at("1"), 400);
}

TEST_F(FilterCountsTest, CollapsingDistinctBitstrings_CountsMerge) {
    // "00" and "10" both project to "0" when only qubit 1 is kept.
    std::map<std::string, int> counts = { {"00", 300}, {"10", 700} };
    std::vector<bool> mask = {false, true};
    auto result = filter_counts(counts, mask);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.at("0"), 1000);
}

class SamplingTest : public ::testing::Test {
protected:
    QiliSimConfig cfg = defaultConfig();
    NoiseModelCpp noNoise = emptyNoise();
};

TEST_F(SamplingTest, ZeroState_NoGates_AllCountsAreZeroString) {
    int n = 2;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, XGateOnQubit0_AllCountsAre10) {
    int n = 2;
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("10"), 1000);
}

TEST_F(SamplingTest, DoubleXGateOnQubit0_AllCountsAre00_NoCache) {
    int n = 2;
    std::vector<Gate> gates = { makeX(0), makeX(0) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    QiliSimConfig cfgNoCache = cfg;
    cfgNoCache.set_max_cache_size(0);
    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfgNoCache);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, DoubleXGateOnQubit0_AllCountsAre00_SmallCache) {
    int n = 2;
    std::vector<Gate> gates = { makeX(0), makeX(0), makeH(0), makeH(0) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    QiliSimConfig cfgNoCache = cfg;
    cfgNoCache.set_max_cache_size(1);
    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfgNoCache);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, XOnBothQubits_AllCountsAre11) {
    int n = 2;
    std::vector<Gate> gates = { makeX(0), makeX(1) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 1000);
}

TEST_F(SamplingTest, HadamardOnSingleQubit_ApproxFiftyFifty) {
    int n = 1;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 10000;

    sampling(gates, measure, n, shots, zeroStateSparse(n), noNoise, state, counts, cfg);

    EXPECT_EQ(totalCounts(counts), shots);
    double f0 = fractionOf(counts, "0");
    double f1 = fractionOf(counts, "1");
    EXPECT_NEAR(f0, 0.5, kLoose);
    EXPECT_NEAR(f1, 0.5, kLoose);
}

TEST_F(SamplingTest, HadamardOnBothQubits_FourOutcomesEquallyLikely) {
    int n = 2;
    std::vector<Gate> gates = { makeH(0), makeH(1) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 10000;

    sampling(gates, measure, n, shots, zeroStateSparse(n), noNoise, state, counts, cfg);

    EXPECT_EQ(totalCounts(counts), shots);
    for (const std::string& key : {"00", "01", "10", "11"}) {
        EXPECT_NEAR(fractionOf(counts, key), 0.25, kLoose);
    }
}

TEST_F(SamplingTest, ShotCountAlwaysPreserved) {
    int n = 2;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure = {true, true};
    for (int shots : {1, 10, 100, 1000}) {
        DenseMatrix state;
        std::map<std::string, int> counts;
        sampling(gates, measure, n, shots, zeroStateSparse(n), noNoise, state, counts, cfg);
        EXPECT_EQ(totalCounts(counts), shots) << "shots=" << shots;
    }
}

TEST_F(SamplingTest, MeasureOnlyQubit0_OutputKeysAreSingleBit) {
    int n = 2;
    std::vector<Gate> gates = { makeH(0), makeH(1) };
    std::vector<bool> measure = {true, false};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    for (const auto& p : counts)
        EXPECT_EQ(p.first.size(), 1u);
    EXPECT_EQ(totalCounts(counts), 1000);
}

TEST_F(SamplingTest, MeasureNoQubits_SingleEmptyKeyWithAllShots) {
    int n = 2;
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {false, false};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 500, zeroStateSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at(""), 500);
}

TEST_F(SamplingTest, StateVectorHasCorrectDimension) {
    int n = 3;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure(n, true);
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 100, zeroStateSparse(n), noNoise, state, counts, cfg);

    EXPECT_EQ(state.rows(), 1L << n);
}

TEST_F(SamplingTest, XGate_StateIsExcitedState) {
    int n = 1;
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 100, zeroStateSparse(n), noNoise, state, counts, cfg);

    // |state[0]|^2 ≈ 0, |state[1]|^2 ≈ 1.
    EXPECT_NEAR(std::norm(state(0, 0)), 0.0, kTol);
    EXPECT_NEAR(std::norm(state(1, 0)), 1.0, kTol);
}

TEST_F(SamplingTest, WithReadoutNoise_TotalCountsStillPreserved) {
    int n = 2;
    auto nm = symmetricReadoutNoise(n, 0.1);
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 1000;

    sampling(gates, measure, n, shots, zeroStateSparse(n), nm, state, counts, cfg);

    EXPECT_EQ(totalCounts(counts), shots);
}

TEST_F(SamplingTest, WithReadoutNoise_DominantOutcomeStillDominates) {
    // X on qubit 0 → should mostly give "10" but with some flips.
    int n = 2;
    auto nm = symmetricReadoutNoise(n, 0.05);
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 5000;

    sampling(gates, measure, n, shots, zeroStateSparse(n), nm, state, counts, cfg);

    EXPECT_GT(fractionOf(counts, "10"), 0.80);
}

TEST_F(SamplingTest, DensityMatrixInitialState_ZeroState_AllCountsAreZero) {
    int n = 2;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 1000, zeroStateDenseSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, CombineSingleQubitGates_SameResultAsWithout) {
    // H then H = identity; with or without combining, result is all-zero.
    int n = 1;
    std::vector<Gate> gates = { makeH(0), makeH(0) };
    std::vector<bool> measure = {true};
    DenseMatrix stateA, stateB;
    std::map<std::string, int> countsA, countsB;

    QiliSimConfig cfgOpt = cfg;
    cfgOpt.set_combine_single_qubit_gates(true);

    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, stateA, countsA, cfg);
    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, stateB, countsB, cfgOpt);

    EXPECT_EQ(countsA, countsB);
}

class SamplingMatrixFreeTest : public ::testing::Test {
protected:
    QiliSimConfig cfg = defaultConfig();
    NoiseModelCpp noNoise = emptyNoise();
};

TEST_F(SamplingMatrixFreeTest, ZeroState_NoGates_AllCountsAreZeroString) {
    int n = 2;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingMatrixFreeTest, XGate_AllCountsAre10) {
    int n = 2;
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("10"), 1000);
}

TEST_F(SamplingMatrixFreeTest, XOnBothQubits_AllCountsAre11) {
    int n = 2;
    std::vector<Gate> gates = { makeX(0), makeX(1) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 1000);
}

TEST_F(SamplingMatrixFreeTest, HadamardOnSingleQubit_ApproxFiftyFifty) {
    int n = 1;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 10000;

    sampling_matrix_free(gates, measure, n, shots, zeroStateSparse(n), noNoise, state, counts, cfg);

    EXPECT_EQ(totalCounts(counts), shots);
    EXPECT_NEAR(fractionOf(counts, "0"), 0.5, kLoose);
    EXPECT_NEAR(fractionOf(counts, "1"), 0.5, kLoose);
}

TEST_F(SamplingMatrixFreeTest, ShotCountAlwaysPreserved) {
    int n = 2;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure = {true, true};
    for (int shots : {1, 10, 100, 1000}) {
        DenseMatrix state;
        std::map<std::string, int> counts;
        sampling_matrix_free(gates, measure, n, shots, zeroStateSparse(n), noNoise, state, counts, cfg);
        EXPECT_EQ(totalCounts(counts), shots) << "shots=" << shots;
    }
}

TEST_F(SamplingMatrixFreeTest, MeasureOnlyQubit0_OutputKeysAreSingleBit) {
    int n = 2;
    std::vector<Gate> gates = { makeH(0), makeH(1) };
    std::vector<bool> measure = {true, false};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg);

    for (const auto& p : counts)
        EXPECT_EQ(p.first.size(), 1u);
    EXPECT_EQ(totalCounts(counts), 1000);
}

TEST_F(SamplingMatrixFreeTest, StateVectorHasCorrectDimension) {
    int n = 3;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure(n, true);
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 100, zeroStateSparse(n), noNoise, state, counts, cfg);

    EXPECT_EQ(state.rows(), 1L << n);
}

TEST_F(SamplingMatrixFreeTest, XGate_StateIsExcitedState) {
    int n = 1;
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 100, zeroStateSparse(n), noNoise, state, counts, cfg);

    EXPECT_NEAR(std::norm(state(0, 0)), 0.0, kTol);
    EXPECT_NEAR(std::norm(state(1, 0)), 1.0, kTol);
}

TEST_F(SamplingMatrixFreeTest, WithReadoutNoise_TotalCountsStillPreserved) {
    int n = 2;
    auto nm = symmetricReadoutNoise(n, 0.1);
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), nm, state, counts, cfg);

    EXPECT_EQ(totalCounts(counts), 1000);
}

TEST_F(SamplingMatrixFreeTest, PureZeroState_MatchesStandardSampling) {
    // Both methods should produce identical counts for a deterministic circuit
    // (no noise, fixed seed).
    int n = 2;
    std::vector<Gate> gates = { makeX(1) };
    std::vector<bool> measure = {true, true};
    DenseMatrix stA, stB;
    std::map<std::string, int> cA, cB;

    sampling(            gates, measure, n, 1000, zeroStateSparse(n), noNoise, stA, cA, cfg);
    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, stB, cB, cfg);

    EXPECT_EQ(cA, cB);
}

TEST_F(SamplingMatrixFreeTest, HadamardCircuit_StatisticsMatchStandardSampling) {
    // For a stochastic circuit, match to within statistical tolerance.
    int n = 2;
    std::vector<Gate> gates = { makeH(0), makeH(1) };
    std::vector<bool> measure = {true, true};
    DenseMatrix stA, stB;
    std::map<std::string, int> cA, cB;
    const int shots = 10000;

    sampling(            gates, measure, n, shots, zeroStateSparse(n), noNoise, stA, cA, cfg);
    sampling_matrix_free(gates, measure, n, shots, zeroStateSparse(n), noNoise, stB, cB, cfg);

    // Both should see each outcome ≈ 25 % ± 5 %.
    for (const std::string& key : {"00", "01", "10", "11"}) {
        EXPECT_NEAR(fractionOf(cA, key), 0.25, kLoose) << "standard key=" << key;
        EXPECT_NEAR(fractionOf(cB, key), 0.25, kLoose) << "matrix_free key=" << key;
    }
}

TEST_F(SamplingMatrixFreeTest, DensityMatrixInitialState_ZeroState_AllCountsAreZero) {
    int n = 2;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 1000, zeroStateDenseSparse(n), noNoise, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

class SamplingMonteCarloTest : public ::testing::Test {
protected:
    QiliSimConfig cfg = defaultConfig();
    NoiseModelCpp noNoise = emptyNoise();
};

TEST_F(SamplingMonteCarloTest, MonteCarloEnabled_ProducesNonDeterministicCounts) {
    int n = 1;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 1000;

    QiliSimConfig cfgMC = cfg;
    cfgMC.set_monte_carlo(true);

    SparseMatrix rho_mixed(2, 2);
    rho_mixed.insert(0, 0) = cx(0.5, 0);
    rho_mixed.insert(1, 1) = cx(0.5, 0);
    rho_mixed.makeCompressed();

    sampling(gates, measure, n, shots, rho_mixed, noNoise, state, counts, cfgMC);

    EXPECT_TRUE(counts.count("0") > 0);
    EXPECT_TRUE(counts.count("1") > 0);
}

TEST_F(SamplingMonteCarloTest, MatrixFreeMonteCarloEnabled_ProducesNonDeterministicCounts) {
    int n = 1;
    std::vector<Gate> gates = { makeH(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 1000;

    QiliSimConfig cfgMC = cfg;
    cfgMC.set_monte_carlo(true);

    SparseMatrix rho_mixed(2, 2);
    rho_mixed.insert(0, 0) = cx(0.5, 0);
    rho_mixed.insert(1, 1) = cx(0.5, 0);
    rho_mixed.makeCompressed();

    sampling_matrix_free(gates, measure, n, shots, rho_mixed, noNoise, state, counts, cfgMC);

    EXPECT_TRUE(counts.count("0") > 0);
    EXPECT_TRUE(counts.count("1") > 0);
}

TEST_F(SamplingTest, BadGate_ThrowsException) {
    int n = 1;
    std::vector<Gate> gates = { Gate("BadGate", SparseMatrix(2, 2), {}, {0}, {}) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    EXPECT_ANY_THROW(sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg));
}

TEST_F(SamplingMatrixFreeTest, BadGate_ThrowsException) {
    int n = 1;
    std::vector<Gate> gates = { Gate("BadGate", SparseMatrix(2, 2), {}, {0}, {}) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    EXPECT_ANY_THROW(sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfg));
}

TEST_F(SamplingTest, PureDensityMatrixInitialState_OutputIsMatrixNotStatevector) {
    int n = 1;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling(gates, measure, n, 1000, zeroStateDenseSparse(n), noNoise, state, counts, cfg);

    EXPECT_EQ(state.rows(), 2);
    EXPECT_EQ(state.cols(), 2);
}

TEST_F(SamplingMatrixFreeTest, PureDensityMatrixInitialState_OutputIsMatrixNotStatevector) {
    int n = 1;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;

    sampling_matrix_free(gates, measure, n, 1000, zeroStateDenseSparse(n), noNoise, state, counts, cfg);

    EXPECT_EQ(state.rows(), 2);
    EXPECT_EQ(state.cols(), 2);
}

// test renormalization (i.e. cfg.set_normalize_after_gate(true)) by applying a non-unitary gate that shrinks the statevector.
TEST_F(SamplingTest, NonUnitaryGate_NormalizationWorks) {
    int n = 1;
    SparseMatrix shrink(2, 2);
    shrink.insert(0, 0) = cx(0.5, 0);
    shrink.insert(1, 1) = cx(0.5, 0);
    shrink.makeCompressed();
    std::vector<Gate> gates = { Gate("Shrink", shrink, {}, {0}, {}) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    QiliSimConfig cfgNorm = cfg;
    cfgNorm.set_normalize_after_gate(true);

    sampling(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfgNorm);

    // The "Shrink" gate halves the amplitude of both |0⟩ and |1⟩, but after normalization we should be back to 0
    EXPECT_NEAR(fractionOf(counts, "0"), 1.0, kLoose);
}

// Same as above but for sampling_matrix_free.
TEST_F(SamplingMatrixFreeTest, NonUnitaryGate_NormalizationWorks) {
    int n = 1;
    SparseMatrix shrink(2, 2);
    shrink.insert(0, 0) = cx(0.5, 0);
    shrink.insert(1, 1) = cx(0.5, 0);
    shrink.makeCompressed();
    std::vector<Gate> gates = { Gate("Shrink", shrink, {}, {0}, {}) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    QiliSimConfig cfgNorm = cfg;
    cfgNorm.set_normalize_after_gate(true);

    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, state, counts, cfgNorm);

    EXPECT_NEAR(fractionOf(counts, "0"), 1.0, kLoose);
}


TEST_F(SamplingTest, KrausNoise_BitflipOnSingleQubit) {
    int n = 1;
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 10000;

    NoiseModelCpp nm;
    SparseMatrix op(2, 2);
    op.insert(0, 1) = cx(1.0, 0);
    op.insert(1, 0) = cx(1.0, 0);
    op.makeCompressed();
    nm.add_kraus_operators_global({ op }); 

    sampling(gates, measure, n, shots, zeroStateSparse(n), nm, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("0"), shots);
}

TEST_F(SamplingMatrixFreeTest, KrausNoise_BitflipOnSingleQubit) {
    int n = 1;
    std::vector<Gate> gates = { makeX(0) };
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::map<std::string, int> counts;
    const int shots = 10000;

    NoiseModelCpp nm;
    SparseMatrix op(2, 2);
    op.insert(0, 1) = cx(1.0, 0);
    op.insert(1, 0) = cx(1.0, 0);
    op.makeCompressed();
    nm.add_kraus_operators_global({ op }); 

    sampling_matrix_free(gates, measure, n, shots, zeroStateSparse(n), nm, state, counts, cfg);

    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("0"), shots);
}

TEST_F(SamplingMatrixFreeTest, CombineSingleQubitGates_SameResultAsWithout) {
    // H then H = identity; with or without combining, result is all-zero.
    int n = 1;
    std::vector<Gate> gates = { makeH(0), makeH(0) };
    std::vector<bool> measure = {true};
    DenseMatrix stateA, stateB;
    std::map<std::string, int> countsA, countsB;

    QiliSimConfig cfgOpt = cfg;
    cfgOpt.set_combine_single_qubit_gates(true);

    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, stateA, countsA, cfg);
    sampling_matrix_free(gates, measure, n, 1000, zeroStateSparse(n), noNoise, stateB, countsB, cfgOpt);

    EXPECT_EQ(countsA, countsB);
}
