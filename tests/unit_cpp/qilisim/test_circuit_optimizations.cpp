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
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include "../../../src/qilisdk_cpp/backends/qilisim/digital/circuit_optimizations.h"

namespace {

using cx = std::complex<double>;
constexpr double kTol = 1e-9;

Eigen::MatrixXcd toDense(const SparseMatrix& s) {
    return Eigen::MatrixXcd(s);
}

SparseMatrix identity2() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = cx(1, 0);
    m.insert(1, 1) = cx(1, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix pauliX() {
    SparseMatrix m(2, 2);
    m.insert(0, 1) = cx(1, 0);
    m.insert(1, 0) = cx(1, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix pauliY() {
    SparseMatrix m(2, 2);
    m.insert(0, 1) = cx(0, -1);
    m.insert(1, 0) = cx(0, 1);
    m.makeCompressed();
    return m;
}

SparseMatrix pauliZ() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = cx(1, 0);
    m.insert(1, 1) = cx(-1, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix hadamard() {
    double v = 1.0 / std::sqrt(2.0);
    SparseMatrix m(2, 2);
    m.insert(0, 0) = cx(v, 0);
    m.insert(0, 1) = cx(v, 0);
    m.insert(1, 0) = cx(v, 0);
    m.insert(1, 1) = cx(-v, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix rz(double theta) {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = std::exp(cx(0, -theta / 2.0));
    m.insert(1, 1) = std::exp(cx(0, theta / 2.0));
    m.makeCompressed();
    return m;
}

Gate makeGate(const std::string& name, const SparseMatrix& mat, std::vector<int> controls, std::vector<int> targets) {
    return Gate(name, mat, controls, targets, {});
}

Gate X(int q) {
    return makeGate("X", pauliX(), {}, {q});
}
Gate Y(int q) {
    return makeGate("Y", pauliY(), {}, {q});
}
Gate Z(int q) {
    return makeGate("Z", pauliZ(), {}, {q});
}
Gate H(int q) {
    return makeGate("H", hadamard(), {}, {q});
}
Gate makeI(int q) {
    return makeGate("I", identity2(), {}, {q});
}
Gate RZ(int q, double t) {
    return makeGate("RZ", rz(t), {}, {q});
}
Gate CNOT(int c, int t) {
    return makeGate("CNOT", pauliX(), {c}, {t});
}

bool matricesApproxEqual(const SparseMatrix& a, const SparseMatrix& b) {
    return toDense(a).isApprox(toDense(b), kTol);
}

}  // namespace

TEST(CombineSingleQubitGatesTest, EmptyInput_ReturnsEmpty) {
    std::vector<Gate> result = combine_single_qubit_gates({});
    EXPECT_TRUE(result.empty());
}

TEST(CombineSingleQubitGatesTest, SingleSingleQubitGate_PassedThrough) {
    std::vector<Gate> in = {X(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(matricesApproxEqual(out[0].get_base_matrix(), pauliX()));
}

TEST(CombineSingleQubitGatesTest, SingleTwoQubitGate_PassedThrough) {
    std::vector<Gate> in = {CNOT(0, 1)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].get_name(), "X");
}

TEST(CombineSingleQubitGatesTest, AllTwoQubitGates_OutputMatchesInput) {
    std::vector<Gate> in = {CNOT(0, 1), CNOT(1, 2), CNOT(0, 2)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(out[i].get_name(), in[i].get_name());
    }
}

TEST(CombineSingleQubitGatesTest, TwoGatesSameQubit_CombinedIntoOne) {
    std::vector<Gate> in = {X(0), Z(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
}

TEST(CombineSingleQubitGatesTest, TwoGatesSameQubit_MatrixIsProductZX) {
    std::vector<Gate> in = {X(0), Z(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    Eigen::MatrixXcd expected = toDense(pauliZ()) * toDense(pauliX());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, ThreeGatesSameQubit_CombinedIntoOne) {
    std::vector<Gate> in = {H(0), X(0), Z(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
}

TEST(CombineSingleQubitGatesTest, ThreeGatesSameQubit_MatrixIsCorrectProduct) {
    std::vector<Gate> in = {H(0), X(0), Z(0)};
    auto out = combine_single_qubit_gates(in);
    Eigen::MatrixXcd expected = toDense(pauliZ()) * toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, TwoSameGates_MatrixIsSquare) {
    std::vector<Gate> in = {X(0), X(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, HHIsIdentity) {
    std::vector<Gate> in = {H(0), H(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, GatesOnDifferentQubits_NotCombined) {
    std::vector<Gate> in = {X(0), X(1)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
}

TEST(CombineSingleQubitGatesTest, AlternatingQubits_NoMerge) {
    std::vector<Gate> in = {X(0), X(1), X(0), X(1)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
    for (auto& g : out) {
        EXPECT_TRUE(toDense(g.get_base_matrix()).isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
    }
}

TEST(CombineSingleQubitGatesTest, ThreeDifferentQubits_ThreeOutputGates) {
    std::vector<Gate> in = {X(0), X(1), X(2)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
}

TEST(CombineSingleQubitGatesTest, TwoQubitGateBarrier_StopsLookahead) {
    std::vector<Gate> in = {X(0), CNOT(0, 1), X(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
}

TEST(CombineSingleQubitGatesTest, TwoQubitGateBarrier_CorrectGateOrder) {
    std::vector<Gate> in = {H(0), CNOT(0, 1), Z(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(toDense(hadamard()), kTol));
    EXPECT_EQ(out[1].get_name(), "X");
    EXPECT_TRUE(toDense(out[2].get_base_matrix()).isApprox(toDense(pauliZ()), kTol));
}

TEST(CombineSingleQubitGatesTest, TwoQubitGateOnOtherQubit_DoesNotBlock) {
    std::vector<Gate> in = {X(0), CNOT(1, 2), X(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
    bool foundIdentity = false;
    for (auto& g : out) {
        if (g.get_qubits().size() == 1 && g.get_qubits()[0] == 0) {
            EXPECT_TRUE(toDense(g.get_base_matrix()).isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
            foundIdentity = true;
        }
    }
    EXPECT_TRUE(foundIdentity);
}

TEST(CombineSingleQubitGatesTest, MultipleGroupsSeparatedByTwoQubitGates) {
    std::vector<Gate> in = {H(0), X(0), CNOT(0, 1), Z(0), Y(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
    Eigen::MatrixXcd exp1 = toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(exp1, kTol));
    EXPECT_EQ(out[1].get_name(), "X");
    Eigen::MatrixXcd exp2 = toDense(pauliY()) * toDense(pauliZ());
    EXPECT_TRUE(toDense(out[2].get_base_matrix()).isApprox(exp2, kTol));
}

TEST(CombineSingleQubitGatesTest, OrderMatters_XZ_NeqZX) {
    auto outXZ = combine_single_qubit_gates({X(0), Z(0)});
    auto outZX = combine_single_qubit_gates({Z(0), X(0)});
    ASSERT_EQ(outXZ.size(), 1u);
    ASSERT_EQ(outZX.size(), 1u);
    EXPECT_FALSE(toDense(outXZ[0].get_base_matrix()).isApprox(toDense(outZX[0].get_base_matrix()), kTol));
}

TEST(CombineSingleQubitGatesTest, FourGates_CorrectProductOrder) {
    std::vector<Gate> in = {H(0), X(0), Y(0), Z(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    Eigen::MatrixXcd expected = toDense(pauliZ()) * toDense(pauliY()) * toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, CombinedGate_TargetQubitPreserved) {
    std::vector<Gate> in = {X(3), Z(3)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].get_target_qubits().size(), 1u);
    EXPECT_EQ(out[0].get_target_qubits()[0], 3);
}

TEST(CombineSingleQubitGatesTest, CombinedGate_NoControlQubits) {
    std::vector<Gate> in = {X(0), H(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0].get_control_qubits().empty());
}

TEST(CombineSingleQubitGatesTest, CombinedGate_IsStillSingleQubit) {
    std::vector<Gate> in = {H(2), X(2), Z(2)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].get_nqubits(), 1);
    EXPECT_EQ(out[0].get_qubits()[0], 2);
}

TEST(CombineSingleQubitGatesTest, UnchangedGate_RetainsOriginalName) {
    std::vector<Gate> in = {X(0), CNOT(0, 1)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].get_name(), "X");
    EXPECT_EQ(out[1].get_name(), "X");
}

TEST(CombineSingleQubitGatesTest, CombinedGate_NameIsNonEmpty) {
    std::vector<Gate> in = {X(0), Z(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FALSE(out[0].get_name().empty());
}

TEST(CombineSingleQubitGatesTest, CombinedGate_IsNormalized) {
    std::vector<Gate> in = {H(0), X(0), Z(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0].is_normalized());
}

TEST(CombineSingleQubitGatesTest, TwoQubitsEachWithTwoGates_TwoOutputGates) {
    std::vector<Gate> in = {H(0), X(0), Z(1), Y(1)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
}

TEST(CombineSingleQubitGatesTest, TwoQubitsEachWithTwoGates_CorrectMatrices) {
    std::vector<Gate> in = {H(0), X(0), Z(1), Y(1)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
    Eigen::MatrixXcd exp0 = toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(exp0, kTol));
    Eigen::MatrixXcd exp1 = toDense(pauliY()) * toDense(pauliZ());
    EXPECT_TRUE(toDense(out[1].get_base_matrix()).isApprox(exp1, kTol));
}

TEST(CombineSingleQubitGatesTest, TwoRZGates_CombinedMatrixIsProductRZ) {
    double t1 = M_PI / 4.0, t2 = M_PI / 3.0;
    std::vector<Gate> in = {RZ(0, t1), RZ(0, t2)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    Eigen::MatrixXcd expected = toDense(rz(t2)) * toDense(rz(t1));
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, RZFollowedByH_CombinedCorrectly) {
    double t = M_PI / 2.0;
    std::vector<Gate> in = {RZ(0, t), H(0)};
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    Eigen::MatrixXcd expected = toDense(hadamard()) * toDense(rz(t));
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, ManyIdentityGates_CombineToIdentity) {
    const int N = 20;
    std::vector<Gate> in;
    for (int i = 0; i < N; ++i) {
        in.push_back(makeI(0));
    }
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, ManyXGates_EvenCountIsIdentity) {
    const int N = 10;
    std::vector<Gate> in;
    for (int i = 0; i < N; ++i) {
        in.push_back(X(0));
    }
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, ManyXGates_OddCountIsX) {
    const int N = 9;
    std::vector<Gate> in;
    for (int i = 0; i < N; ++i) {
        in.push_back(X(0));
    }
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(toDense(pauliX()), kTol));
}

TEST(CombineSingleQubitGatesTest, InterleavedMultiQubitGates_CountIsCorrect) {
    std::vector<Gate> in;
    const int N = 5;
    for (int i = 0; i < N; ++i) {
        in.push_back(X(0));
        if (i < N - 1) {
            in.push_back(CNOT(0, 1));
        }
    }
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), static_cast<size_t>(2 * N - 1));
}

namespace {

Gate Mgate(int q) {
    return makeGate("M", identity2(), {}, {q});
}
Gate CCX(int c0, int c1, int t) {
    return makeGate("CCX", pauliX(), {c0, c1}, {t});
}

// Overall unitary of a circuit (skipping measurements), as the ordered product
// of each gate's full matrix on n qubits. Used to check that fusion preserves
// the circuit semantics regardless of how gates are grouped.
SparseMatrix circuitUnitary(const std::vector<Gate>& gates, int n) {
    SparseMatrix u(1 << n, 1 << n);
    u.setIdentity();
    for (const auto& g : gates) {
        if (g.get_name() == "M") {
            continue;
        }
        u = g.get_full_matrix(n) * u;
    }
    return u;
}

}  // namespace

TEST(FuseGatesTest, EmptyInput_ReturnsEmpty) {
    EXPECT_TRUE(fuse_gates({}, 4).empty());
}

TEST(FuseGatesTest, SingleGate_PassedThroughUnchanged) {
    auto out = fuse_gates({X(0)}, 4);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].get_name(), "X");
    EXPECT_TRUE(matricesApproxEqual(out[0].get_base_matrix(), pauliX()));
}

TEST(FuseGatesTest, TwoGatesSameQubit_FusedIntoProduct) {
    // X then H on qubit 0 fuses into a single 2x2 gate equal to H * X.
    auto out = fuse_gates({X(0), H(0)}, 4);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].get_name(), "FUSED");
    EXPECT_EQ(out[0].get_target_qubits(), std::vector<int>({0}));
    SparseMatrix expected = hadamard() * pauliX();
    EXPECT_TRUE(matricesApproxEqual(out[0].get_base_matrix(), expected));
}

TEST(FuseGatesTest, MeasurementActsAsBarrier) {
    // Gates cannot fuse across a measurement.
    auto out = fuse_gates({X(0), Mgate(0), H(0)}, 4);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0].get_name(), "X");
    EXPECT_EQ(out[1].get_name(), "M");
    EXPECT_EQ(out[2].get_name(), "H");
}

TEST(FuseGatesTest, GateWiderThanMax_PassedThrough) {
    // A 3-qubit gate cannot fit a max width of 2, so it is passed through and
    // forces the overlapping single-qubit block to close first.
    auto out = fuse_gates({X(0), CCX(0, 1, 2)}, 2);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].get_name(), "X");
    EXPECT_EQ(out[1].get_name(), "X");  // CCX is normalized to "X" with two controls
    EXPECT_EQ(out[1].get_control_qubits().size(), 2u);
}

TEST(FuseGatesTest, PreservesCircuitUnitary_Width3) {
    std::vector<Gate> in = {H(0), CNOT(0, 1), RZ(1, 0.7), H(2), CNOT(1, 2), X(0), Z(2)};
    for (int max_width : {1, 2, 3}) {
        auto out = fuse_gates(in, max_width);
        EXPECT_TRUE(matricesApproxEqual(circuitUnitary(out, 3), circuitUnitary(in, 3))) << "Fusion changed the circuit unitary at max_width=" << max_width;
    }
}

TEST(FuseGatesTest, PreservesCircuitUnitary_DisjointBlocks) {
    // Two independent runs on disjoint qubits, interleaved in time.
    std::vector<Gate> in = {H(0), H(2), X(0), Z(2), CNOT(0, 1), CNOT(2, 3)};
    auto out = fuse_gates(in, 4);
    EXPECT_TRUE(matricesApproxEqual(circuitUnitary(out, 4), circuitUnitary(in, 4)));
}

// GCOV_EXCL_BR_STOP