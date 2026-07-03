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

#include "../../../src/qilisdk_cpp/backends/qilisim/digital/sampling.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/utils/sample.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <pybind11/embed.h>

namespace {

using cx = std::complex<double>;
constexpr double kTol = 1e-6;
constexpr double kLoose = 0.05;

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
    m.insert(0, 0) = cx(v, 0);
    m.insert(0, 1) = cx(v, 0);
    m.insert(1, 0) = cx(v, 0);
    m.insert(1, 1) = cx(-v, 0);
    m.makeCompressed();
    return m;
}

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

Gate makeM(int qubit) {
    return Gate("M", SparseMatrix(), {}, {qubit}, {});
}

Gate makeH(int qubit) {
    return Gate("H", hadamard2(), {}, {qubit}, {});
}

int totalCounts(const std::map<std::string, int>& counts) {
    int total = 0;
    for (const auto& p : counts)
        total += p.second;
    return total;
}

double fractionOf(const std::map<std::string, int>& counts, const std::string& key) {
    int count = counts.count(key) ? counts.at(key) : 0;
    return static_cast<double>(count) / totalCounts(counts);
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

}  // namespace

class ApplyReadoutErrorTest : public ::testing::Test {};

TEST_F(ApplyReadoutErrorTest, ZeroError_CountsUnchanged) {
    std::map<std::string, int> counts = {{"00", 500}, {"11", 500}};
    auto nm = symmetricReadoutNoise(2, 0.0);
    auto result = apply_readout_error(counts, nm, 2);
    EXPECT_EQ(result.at("00"), 500);
    EXPECT_EQ(result.at("11"), 500);
    EXPECT_EQ(totalCounts(result), 1000);
}

TEST_F(ApplyReadoutErrorTest, TotalShotCountPreserved) {
    std::map<std::string, int> counts = {{"0", 300}, {"1", 700}};
    auto nm = symmetricReadoutNoise(1, 0.1);
    auto result = apply_readout_error(counts, nm, 1);
    EXPECT_EQ(totalCounts(result), 1000);
}

TEST_F(ApplyReadoutErrorTest, PerfectFlip_AllOnesBecomesAllZeros) {
    std::map<std::string, int> counts = {{"11", 1000}};
    NoiseModelCpp nm;
    nm.add_readout_error_per_qubit(0, 0.0, 1.0);
    nm.add_readout_error_per_qubit(1, 0.0, 1.0);
    auto result = apply_readout_error(counts, nm, 2);
    ASSERT_EQ(result.count("00"), 1u);
    EXPECT_EQ(result.at("00"), 1000);
}

TEST_F(ApplyReadoutErrorTest, PerfectFlip_AllZerosBecomesAllOnes) {
    std::map<std::string, int> counts = {{"00", 1000}};
    NoiseModelCpp nm;
    nm.add_readout_error_global(1.0, 0.0);
    auto result = apply_readout_error(counts, nm, 2);
    ASSERT_EQ(result.count("11"), 1u);
    EXPECT_EQ(result.at("11"), 1000);
}

TEST_F(ApplyReadoutErrorTest, PartialError_ReducesDominantOutcome) {
    std::map<std::string, int> counts = {{"0", 10000}};
    auto nm = symmetricReadoutNoise(1, 0.1);
    auto result = apply_readout_error(counts, nm, 1);
    EXPECT_EQ(totalCounts(result), 10000);
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
    std::map<std::string, int> counts = {{"0", 500}, {"1", 500}};
    auto nm = symmetricReadoutNoise(1, 0.05);
    auto result = apply_readout_error(counts, nm, 1);
    for (const auto& p : result) {
        EXPECT_EQ(p.first.size(), 1u);
        EXPECT_TRUE(p.first == "0" || p.first == "1");
    }
}

TEST_F(ApplyReadoutErrorTest, MultiQubit_OutputKeysHaveCorrectLength) {
    std::map<std::string, int> counts = {{"000", 500}, {"111", 500}};
    auto nm = symmetricReadoutNoise(3, 0.05);
    auto result = apply_readout_error(counts, nm, 3);
    for (const auto& p : result) {
        EXPECT_EQ(p.first.size(), 3u);
    }
}

TEST_F(ApplyReadoutErrorTest, Deterministic_SameSeedSameResult) {
    std::map<std::string, int> counts = {{"0", 500}, {"1", 500}};
    auto nm = symmetricReadoutNoise(1, 0.2);
    auto r1 = apply_readout_error(counts, nm, 1);
    auto r2 = apply_readout_error(counts, nm, 1);
    EXPECT_EQ(r1, r2);
}

TEST_F(ApplyReadoutErrorTest, AsymmetricError_DifferentFlipRates) {
    std::map<std::string, int> counts = {{"0", 400}, {"1", 600}};
    NoiseModelCpp nm;
    nm.add_readout_error_per_qubit(0, 0.0, 1.0);
    auto result = apply_readout_error(counts, nm, 1);
    ASSERT_EQ(result.count("0"), 1u);
    EXPECT_EQ(result.at("0"), 1000);
    EXPECT_EQ(result.count("1"), 0u);
}

class FilterCountsTest : public ::testing::Test {};

TEST_F(FilterCountsTest, AllQubitsSelected_CountsUnchanged) {
    std::map<std::string, int> counts = {{"00", 300}, {"11", 700}};
    std::vector<bool> mask = {true, true};
    auto result = filter_counts(counts, mask);
    EXPECT_EQ(result, counts);
}

TEST_F(FilterCountsTest, NoQubitsSelected_SingleEmptyKey) {
    std::map<std::string, int> counts = {{"00", 300}, {"11", 700}};
    std::vector<bool> mask = {false, false};
    auto result = filter_counts(counts, mask);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.at(""), 1000);
}

TEST_F(FilterCountsTest, FirstQubitOnly_ProjectsCorrectly) {
    std::map<std::string, int> counts = {{"00", 200}, {"01", 300}, {"10", 100}, {"11", 400}};
    std::vector<bool> mask = {true, false};
    auto result = filter_counts(counts, mask);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.at("0"), 500);
    EXPECT_EQ(result.at("1"), 500);
}

TEST_F(FilterCountsTest, SecondQubitOnly_ProjectsCorrectly) {
    std::map<std::string, int> counts = {{"00", 200}, {"01", 300}, {"10", 100}, {"11", 400}};
    std::vector<bool> mask = {false, true};
    auto result = filter_counts(counts, mask);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.at("0"), 300);
    EXPECT_EQ(result.at("1"), 700);
}

TEST_F(FilterCountsTest, TotalCountPreserved) {
    std::map<std::string, int> counts = {{"000", 100}, {"010", 200}, {"101", 300}, {"111", 400}};
    std::vector<bool> mask = {true, false, true};
    auto result = filter_counts(counts, mask);
    EXPECT_EQ(totalCounts(result), 1000);
}

TEST_F(FilterCountsTest, MiddleQubitSelected_ThreeQubits) {
    std::map<std::string, int> counts = {{"000", 125}, {"010", 125}, {"100", 125}, {"110", 125}, {"001", 125}, {"011", 125}, {"101", 125}, {"111", 125}};
    std::vector<bool> mask = {false, true, false};
    auto result = filter_counts(counts, mask);
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
    std::map<std::string, int> counts = {{"0", 600}, {"1", 400}};
    std::vector<bool> mask = {true};
    auto result = filter_counts(counts, mask);
    EXPECT_EQ(result.at("0"), 600);
    EXPECT_EQ(result.at("1"), 400);
}

TEST_F(FilterCountsTest, CollapsingDistinctBitstrings_CountsMerge) {
    std::map<std::string, int> counts = {{"00", 300}, {"10", 700}};
    std::vector<bool> mask = {false, true};
    auto result = filter_counts(counts, mask);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.at("0"), 1000);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
class SamplingTest : public ::testing::Test {
   protected:
    QiliSimConfig cfg = defaultConfig();
    NoiseModelCpp noNoise = emptyNoise();
    py::list readout = py::list();
};
#pragma GCC diagnostic pop

TEST_F(SamplingTest, ZeroState_NoGates_AllCountsAreZeroString) {
    int n = 2;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, XGateOnQubit0_AllCountsAre10) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("10"), 1000);
}

TEST_F(SamplingTest, FusionWithSingleQubitGateCombiningEnabled) {
    int n = 2;
    QiliSimConfig cfg_fuse = defaultConfig();
    cfg_fuse.set_fuse_gates(true);
    cfg_fuse.set_combine_single_qubit_gates(true);
    cfg_fuse.set_num_threads(4);
    std::vector<Gate> gates = {makeX(0), makeX(0), makeX(1)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg_fuse, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg_fuse, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("01"), 1000);
}

TEST_F(SamplingTest, DoubleXGateOnQubit0_AllCountsAre00_NoCache) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0), makeX(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    QiliSimConfig cfgNoCache = cfg;
    cfgNoCache.set_max_cache_size(0);
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfgNoCache, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfgNoCache, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, DoubleXGateOnQubit0_AllCountsAre00_SmallCache) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0), makeX(0), makeH(0), makeH(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    QiliSimConfig cfgNoCache = cfg;
    cfgNoCache.set_max_cache_size(1);
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfgNoCache, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfgNoCache, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, XOnBothQubits_AllCountsAre11) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0), makeX(1)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 1000);
}

TEST_F(SamplingTest, UnnormalizedStatevector_IsRenormalized) {
    // A statevector whose probabilities sum to 2 (not 1) must be renormalized before sampling; all
    // shots are still accounted for and the two equally weighted outcomes are both produced.
    int n = 1;
    DenseMatrix state(2, 1);
    state(0, 0) = cx(1, 0);
    state(1, 0) = cx(1, 0);
    std::vector<bool> measure = {true};
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    int total = 0;
    for (const auto& [bitstring, count] : counts) {
        total += count;
    }
    EXPECT_EQ(total, 1000);
}

TEST_F(SamplingTest, HadamardOnSingleQubit_ApproxFiftyFifty) {
    int n = 1;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    const int shots = 10000;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, noNoise, cfg, measure);
    EXPECT_EQ(totalCounts(counts), shots);
    double f0 = fractionOf(counts, "0");
    double f1 = fractionOf(counts, "1");
    EXPECT_NEAR(f0, 0.5, kLoose);
    EXPECT_NEAR(f1, 0.5, kLoose);
}

TEST_F(SamplingTest, HadamardOnBothQubits_FourOutcomesEquallyLikely) {
    int n = 2;
    std::vector<Gate> gates = {makeH(0), makeH(1)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    const int shots = 10000;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, noNoise, cfg, measure);
    EXPECT_EQ(totalCounts(counts), shots);
    for (const std::string& key : {"00", "01", "10", "11"}) {
        EXPECT_NEAR(fractionOf(counts, key), 0.25, kLoose);
    }
}

TEST_F(SamplingTest, ShotCountAlwaysPreserved) {
    int n = 2;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure = {true, true};
    for (int shots : {1, 10, 100, 1000}) {
        DenseMatrix state;
        std::vector<py::object> intermediate_results;
        sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
        std::map<std::string, int> counts = construct_samples(state, n, shots, noNoise, cfg, measure);
        EXPECT_EQ(totalCounts(counts), shots) << "shots=" << shots;
    }
}

TEST_F(SamplingTest, MeasureOnlyQubit0_OutputKeysAreSingleBit) {
    int n = 2;
    std::vector<Gate> gates = {makeH(0), makeH(1)};
    std::vector<bool> measure = {true, false};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    for (const auto& p : counts) {
        EXPECT_EQ(p.first.size(), 1u);
    }
    EXPECT_EQ(totalCounts(counts), 1000);
}

TEST_F(SamplingTest, MeasureNoQubits_SingleEmptyKeyWithAllShots) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {false, false};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 500, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at(""), 500);
}

TEST_F(SamplingTest, StateVectorHasCorrectDimension) {
    int n = 3;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure(n, true);
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_EQ(state.rows(), 1L << n);
}

TEST_F(SamplingTest, XGate_StateIsExcitedState) {
    int n = 1;
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_NEAR(std::norm(state(0, 0)), 0.0, kTol);
    EXPECT_NEAR(std::norm(state(1, 0)), 1.0, kTol);
}

TEST_F(SamplingTest, WithReadoutNoise_TotalCountsStillPreserved) {
    int n = 2;
    auto nm = symmetricReadoutNoise(n, 0.1);
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    const int shots = 1000;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), nm, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, nm, cfg, measure);
    EXPECT_EQ(totalCounts(counts), shots);
}

TEST_F(SamplingTest, WithReadoutNoise_DominantOutcomeStillDominates) {
    int n = 2;
    auto nm = symmetricReadoutNoise(n, 0.05);
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    const int shots = 5000;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), nm, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, nm, cfg, measure);
    EXPECT_GT(fractionOf(counts, "10"), 0.80);
}

TEST_F(SamplingTest, DensityMatrixInitialState_ZeroState_AllCountsAreZero) {
    int n = 2;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateDenseSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingTest, CombineSingleQubitGates_SameResultAsWithout) {
    int n = 1;
    std::vector<Gate> gates = {makeH(0), makeH(0)};
    std::vector<bool> measure = {true};
    DenseMatrix stateA, stateB;
    std::vector<py::object> intermediate_resultsA, intermediate_resultsB;
    QiliSimConfig cfgOpt = cfg;
    cfgOpt.set_combine_single_qubit_gates(true);
    sampling(gates, n, zeroStateSparse(n), noNoise, stateA, intermediate_resultsA, cfg, readout);
    std::map<std::string, int> countsA = construct_samples(stateA, n, 1000, noNoise, cfg, measure);
    sampling(gates, n, zeroStateSparse(n), noNoise, stateB, intermediate_resultsB, cfgOpt, readout);
    std::map<std::string, int> countsB = construct_samples(stateB, n, 1000, noNoise, cfgOpt, measure);
    EXPECT_EQ(countsA, countsB);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
class SamplingMatrixFreeTest : public ::testing::Test {
   protected:
    QiliSimConfig cfg = defaultConfig();
    NoiseModelCpp noNoise = emptyNoise();
    py::list readout = py::list();
};
#pragma GCC diagnostic pop

TEST_F(SamplingMatrixFreeTest, ZeroState_NoGates_AllCountsAreZeroString) {
    int n = 2;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

TEST_F(SamplingMatrixFreeTest, XGate_AllCountsAre10) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("10"), 1000);
}

TEST_F(SamplingMatrixFreeTest, XOnBothQubits_AllCountsAre11) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0), makeX(1)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 1000);
}

TEST_F(SamplingMatrixFreeTest, HadamardOnSingleQubit_ApproxFiftyFifty) {
    int n = 1;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    const int shots = 10000;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, noNoise, cfg, measure);
    EXPECT_EQ(totalCounts(counts), shots);
    EXPECT_NEAR(fractionOf(counts, "0"), 0.5, kLoose);
    EXPECT_NEAR(fractionOf(counts, "1"), 0.5, kLoose);
}

TEST_F(SamplingMatrixFreeTest, ShotCountAlwaysPreserved) {
    int n = 2;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure = {true, true};
    for (int shots : {1, 10, 100, 1000}) {
        DenseMatrix state;
        std::vector<py::object> intermediate_results;
        sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
        std::map<std::string, int> counts = construct_samples(state, n, shots, noNoise, cfg, measure);
        EXPECT_EQ(totalCounts(counts), shots) << "shots=" << shots;
    }
}

TEST_F(SamplingMatrixFreeTest, MeasureOnlyQubit0_OutputKeysAreSingleBit) {
    int n = 2;
    std::vector<Gate> gates = {makeH(0), makeH(1)};
    std::vector<bool> measure = {true, false};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    for (const auto& p : counts) {
        EXPECT_EQ(p.first.size(), 1u);
    }
    EXPECT_EQ(totalCounts(counts), 1000);
}

TEST_F(SamplingMatrixFreeTest, StateVectorHasCorrectDimension) {
    int n = 3;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure(n, true);
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_EQ(state.rows(), 1L << n);
}

TEST_F(SamplingMatrixFreeTest, XGate_StateIsExcitedState) {
    int n = 1;
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_NEAR(std::norm(state(0, 0)), 0.0, kTol);
    EXPECT_NEAR(std::norm(state(1, 0)), 1.0, kTol);
}

TEST_F(SamplingMatrixFreeTest, WithReadoutNoise_TotalCountsStillPreserved) {
    int n = 2;
    auto nm = symmetricReadoutNoise(n, 0.1);
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), nm, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, nm, cfg, measure);
    EXPECT_EQ(totalCounts(counts), 1000);
}

TEST_F(SamplingMatrixFreeTest, PureZeroState_MatchesStandardSampling) {
    int n = 2;
    std::vector<Gate> gates = {makeX(1)};
    std::vector<bool> measure = {true, true};
    DenseMatrix stA, stB;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, stA, intermediate_results, cfg, readout);
    std::map<std::string, int> cA = construct_samples(stA, n, 1000, noNoise, cfg, measure);
    std::vector<py::object> intermediate_results_2;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, stB, intermediate_results_2, cfg, readout);
    std::map<std::string, int> cB = construct_samples(stB, n, 1000, noNoise, cfg, measure);
    EXPECT_EQ(cA, cB);
}

TEST_F(SamplingMatrixFreeTest, HadamardCircuit_StatisticsMatchStandardSampling) {
    int n = 2;
    std::vector<Gate> gates = {makeH(0), makeH(1)};
    std::vector<bool> measure = {true, true};
    DenseMatrix stA, stB;
    const int shots = 10000;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, stA, intermediate_results, cfg, readout);
    std::map<std::string, int> cA = construct_samples(stA, n, shots, noNoise, cfg, measure);
    std::vector<py::object> intermediate_results_mf;
    py::object readout_mf = py::object();
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, stB, intermediate_results_mf, cfg, readout_mf);
    std::map<std::string, int> cB = construct_samples(stB, n, shots, noNoise, cfg, measure);
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
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateDenseSparse(n), noNoise, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 1000);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
class SamplingMonteCarloTest : public ::testing::Test {
   protected:
    QiliSimConfig cfg = defaultConfig();
    NoiseModelCpp noNoise = emptyNoise();
    py::list readout = py::list();
};
#pragma GCC diagnostic pop

TEST_F(SamplingMonteCarloTest, MonteCarloEnabled_ProducesNonDeterministicCounts) {
    int n = 1;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    const int shots = 1000;
    QiliSimConfig cfgMC = cfg;
    cfgMC.set_monte_carlo(true);
    SparseMatrix rho_mixed(2, 2);
    rho_mixed.insert(0, 0) = cx(0.5, 0);
    rho_mixed.insert(1, 1) = cx(0.5, 0);
    rho_mixed.makeCompressed();
    std::vector<py::object> intermediate_results;
    sampling(gates, n, rho_mixed, noNoise, state, intermediate_results, cfgMC, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, noNoise, cfgMC, measure);
    EXPECT_TRUE(counts.count("0") > 0);
    EXPECT_TRUE(counts.count("1") > 0);
}

TEST_F(SamplingMonteCarloTest, MatrixFreeMonteCarloEnabled_ProducesNonDeterministicCounts) {
    int n = 1;
    std::vector<Gate> gates = {makeH(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    const int shots = 1000;
    QiliSimConfig cfgMC = cfg;
    cfgMC.set_monte_carlo(true);
    SparseMatrix rho_mixed(2, 2);
    rho_mixed.insert(0, 0) = cx(0.5, 0);
    rho_mixed.insert(1, 1) = cx(0.5, 0);
    rho_mixed.makeCompressed();
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, rho_mixed, noNoise, state, intermediate_results, cfgMC, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, noNoise, cfgMC, measure);
    EXPECT_TRUE(counts.count("0") > 0);
    EXPECT_TRUE(counts.count("1") > 0);
}

TEST_F(SamplingMatrixFreeTest, BadGate_ThrowsException) {
    // MatrixFreeOperator rejects multi-target gates unless they are SWAP, M, or a
    // dense 2^k x 2^k block (gate fusion). A 2-target gate carrying a 2x2 matrix is
    // none of these (its matrix doesn't match the target count), so it is rejected.
    // sampling() has no gate-name validation, so this test only applies to matrix-free.
    int n = 2;
    std::vector<Gate> gates = {Gate("BadGate", SparseMatrix(2, 2), {}, {0, 1}, {})};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    EXPECT_ANY_THROW(sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout));
}

TEST_F(SamplingMatrixFreeTest, GateFusionEnabled_MatchesUnfusedResult) {
    // With fusion enabled (statevector, no noise, >= 4 threads) the matrix-free
    // path fuses runs of gates into dense blocks. The sampled distribution must
    // match the unfused path. Two H gates on each of two qubits return to |00>.
    int n = 2;
    std::vector<Gate> gates = {makeH(0), makeH(1), makeH(0), makeH(1)};
    std::vector<bool> measure = {true, true};
    QiliSimConfig cfgFuse = cfg;
    cfgFuse.set_fuse_gates(true);
    cfgFuse.set_num_threads(4);
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfgFuse, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfgFuse, measure);
    EXPECT_NEAR(fractionOf(counts, "00"), 1.0, kLoose);
}

TEST_F(SamplingTest, PureDensityMatrixInitialState_OutputIsMatrixNotStatevector) {
    int n = 1;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateDenseSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_EQ(state.rows(), 2);
    EXPECT_EQ(state.cols(), 2);
}

TEST_F(SamplingMatrixFreeTest, PureDensityMatrixInitialState_OutputIsMatrixNotStatevector) {
    int n = 1;
    std::vector<Gate> gates;
    std::vector<bool> measure = {true};
    DenseMatrix state;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateDenseSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_EQ(state.rows(), 2);
    EXPECT_EQ(state.cols(), 2);
}

TEST_F(SamplingTest, NonUnitaryGate_NormalizationWorks) {
    int n = 1;
    SparseMatrix shrink(2, 2);
    shrink.insert(0, 0) = cx(0.5, 0);
    shrink.insert(1, 1) = cx(0.5, 0);
    shrink.makeCompressed();
    std::vector<Gate> gates = {Gate("Shrink", shrink, {}, {0}, {})};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    QiliSimConfig cfgNorm = cfg;
    cfgNorm.set_normalize_after_gate(true);
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfgNorm, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfgNorm, measure);
    EXPECT_NEAR(fractionOf(counts, "0"), 1.0, kLoose);
}

TEST_F(SamplingMatrixFreeTest, NonUnitaryGate_NormalizationWorks) {
    int n = 1;
    SparseMatrix shrink(2, 2);
    shrink.insert(0, 0) = cx(0.5, 0);
    shrink.insert(1, 1) = cx(0.5, 0);
    shrink.makeCompressed();
    std::vector<Gate> gates = {Gate("Shrink", shrink, {}, {0}, {})};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    QiliSimConfig cfgNorm = cfg;
    std::vector<py::object> intermediate_results;
    cfgNorm.set_normalize_after_gate(true);
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfgNorm, readout);
    std::map<std::string, int> counts = construct_samples(state, n, 1000, noNoise, cfgNorm, measure);
    EXPECT_NEAR(fractionOf(counts, "0"), 1.0, kLoose);
}

TEST_F(SamplingTest, KrausNoise_BitflipOnSingleQubit) {
    int n = 1;
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    const int shots = 10000;
    NoiseModelCpp nm;
    SparseMatrix op(2, 2);
    op.insert(0, 1) = cx(1.0, 0);
    op.insert(1, 0) = cx(1.0, 0);
    op.makeCompressed();
    nm.add_kraus_operators_global({op});
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), nm, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, nm, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("0"), shots);
}

TEST_F(SamplingMatrixFreeTest, KrausNoise_BitflipOnSingleQubit) {
    int n = 1;
    std::vector<Gate> gates = {makeX(0)};
    std::vector<bool> measure = {true};
    DenseMatrix state;
    const int shots = 10000;
    NoiseModelCpp nm;
    SparseMatrix op(2, 2);
    op.insert(0, 1) = cx(1.0, 0);
    op.insert(1, 0) = cx(1.0, 0);
    op.makeCompressed();
    nm.add_kraus_operators_global({op});
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), nm, state, intermediate_results, cfg, readout);
    std::map<std::string, int> counts = construct_samples(state, n, shots, nm, cfg, measure);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("0"), shots);
}

TEST_F(SamplingMatrixFreeTest, CombineSingleQubitGates_SameResultAsWithout) {
    int n = 1;
    std::vector<Gate> gates = {makeH(0), makeH(0)};
    std::vector<bool> measure = {true};
    DenseMatrix stateA, stateB;
    QiliSimConfig cfgOpt = cfg;
    cfgOpt.set_combine_single_qubit_gates(true);
    std::vector<py::object> intermediate_resultsA, intermediate_resultsB;
    py::object readoutA = py::object();
    py::object readoutB = py::object();
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, stateA, intermediate_resultsA, cfg, readoutA);
    std::map<std::string, int> countsA = construct_samples(stateA, n, 1000, noNoise, cfg, measure);
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, stateB, intermediate_resultsB, cfgOpt, readoutB);
    std::map<std::string, int> countsB = construct_samples(stateB, n, 1000, noNoise, cfgOpt, measure);
    EXPECT_EQ(countsA, countsB);
}

TEST_F(SamplingMatrixFreeTest, MidCircuitMeasurements) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0), makeM(0), makeX(0), makeM(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    const int shots = 10000;
    std::vector<py::object> intermediate_results;
    sampling_matrix_free(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_EQ(intermediate_results.size(), 1u);
}

TEST_F(SamplingTest, MidCircuitMeasurements) {
    int n = 2;
    std::vector<Gate> gates = {makeX(0), makeM(0), makeX(0), makeM(0)};
    std::vector<bool> measure = {true, true};
    DenseMatrix state;
    const int shots = 10000;
    std::vector<py::object> intermediate_results;
    sampling(gates, n, zeroStateSparse(n), noNoise, state, intermediate_results, cfg, readout);
    EXPECT_EQ(intermediate_results.size(), 1u);
}

// GCOV_EXCL_BR_STOP
