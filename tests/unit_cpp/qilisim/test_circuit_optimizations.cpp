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
#include "../../../src/qilisdk_cpp/backends/qilisim/digital/circuit_optimizations.h"
#include <complex>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

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
    m.insert(1, 0) = cx(0,  1);
    m.makeCompressed();
    return m;
}

SparseMatrix pauliZ() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = cx( 1, 0);
    m.insert(1, 1) = cx(-1, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix hadamard() {
    double v = 1.0 / std::sqrt(2.0);
    SparseMatrix m(2, 2);
    m.insert(0, 0) = cx( v, 0);
    m.insert(0, 1) = cx( v, 0);
    m.insert(1, 0) = cx( v, 0);
    m.insert(1, 1) = cx(-v, 0);
    m.makeCompressed();
    return m;
}

SparseMatrix rz(double theta) {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = std::exp(cx(0, -theta / 2.0));
    m.insert(1, 1) = std::exp(cx(0,  theta / 2.0));
    m.makeCompressed();
    return m;
}

Gate makeGate(const std::string& name, const SparseMatrix& mat,
              std::vector<int> controls, std::vector<int> targets) {
    return Gate(name, mat, controls, targets, {});
}

Gate X(int q)  { return makeGate("X",    pauliX(),   {}, {q}); }
Gate Y(int q)  { return makeGate("Y",    pauliY(),   {}, {q}); }
Gate Z(int q)  { return makeGate("Z",    pauliZ(),   {}, {q}); }
Gate H(int q)  { return makeGate("H",    hadamard(), {}, {q}); }
Gate makeI(int q)  { return makeGate("I",    identity2(),{}, {q}); }
Gate RZ(int q, double t) { return makeGate("RZ", rz(t),    {}, {q}); }
Gate CNOT(int c, int t)  { return makeGate("CNOT", pauliX(), {c}, {t}); }

bool matricesApproxEqual(const SparseMatrix& a, const SparseMatrix& b) {
    return toDense(a).isApprox(toDense(b), kTol);
}

}

TEST(CombineSingleQubitGatesTest, EmptyInput_ReturnsEmpty) {
    std::vector<Gate> result = combine_single_qubit_gates({});
    EXPECT_TRUE(result.empty());
}

TEST(CombineSingleQubitGatesTest, SingleSingleQubitGate_PassedThrough) {
    std::vector<Gate> in = { X(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(matricesApproxEqual(out[0].get_base_matrix(), pauliX()));
}

TEST(CombineSingleQubitGatesTest, SingleTwoQubitGate_PassedThrough) {
    std::vector<Gate> in = { CNOT(0, 1) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].get_name(), "X");
}

TEST(CombineSingleQubitGatesTest, AllTwoQubitGates_OutputMatchesInput) {
    std::vector<Gate> in = { CNOT(0,1), CNOT(1,2), CNOT(0,2) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(out[i].get_name(), in[i].get_name());
    }
}

TEST(CombineSingleQubitGatesTest, TwoGatesSameQubit_CombinedIntoOne) {
    std::vector<Gate> in = { X(0), Z(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
}

TEST(CombineSingleQubitGatesTest, TwoGatesSameQubit_MatrixIsProductZX) {
    std::vector<Gate> in = { X(0), Z(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    Eigen::MatrixXcd expected = toDense(pauliZ()) * toDense(pauliX());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, ThreeGatesSameQubit_CombinedIntoOne) {
    std::vector<Gate> in = { H(0), X(0), Z(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
}

TEST(CombineSingleQubitGatesTest, ThreeGatesSameQubit_MatrixIsCorrectProduct) {
    std::vector<Gate> in = { H(0), X(0), Z(0) };
    auto out = combine_single_qubit_gates(in);
    Eigen::MatrixXcd expected = toDense(pauliZ()) * toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, TwoSameGates_MatrixIsSquare) {
    // X·X = I
    std::vector<Gate> in = { X(0), X(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix())
                    .isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, HHIsIdentity) {
    std::vector<Gate> in = { H(0), H(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix())
                    .isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}


// ══════════════════════════════════════════════════════════════
// 3. Different qubits — gates must NOT be merged across qubits
// ══════════════════════════════════════════════════════════════

TEST(CombineSingleQubitGatesTest, GatesOnDifferentQubits_NotCombined) {
    std::vector<Gate> in = { X(0), X(1) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
}

TEST(CombineSingleQubitGatesTest, AlternatingQubits_NoMerge) {
    // X(0), X(1), X(0), X(1) — each qubit has two non-consecutive gates;
    // the lookahead skips the interleaved different-qubit gates, so the two
    // X(0) gates CAN still merge (and similarly the two X(1) gates).
    // Output should be 2 combined gates.
    std::vector<Gate> in = { X(0), X(1), X(0), X(1) };
    auto out = combine_single_qubit_gates(in);
    // Two groups: qubit-0 group and qubit-1 group — each combined.
    ASSERT_EQ(out.size(), 2u);
    // Each combined matrix is X·X = I.
    for (auto& g : out)
        EXPECT_TRUE(toDense(g.get_base_matrix())
                        .isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, ThreeDifferentQubits_ThreeOutputGates) {
    std::vector<Gate> in = { X(0), X(1), X(2) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
}


// ══════════════════════════════════════════════════════════════
// 4. Two-qubit gate barrier — lookahead must stop at a multi-qubit
//    gate that acts on the same qubit
// ══════════════════════════════════════════════════════════════

TEST(CombineSingleQubitGatesTest, TwoQubitGateBarrier_StopsLookahead) {
    // X(0), CNOT(0,1), X(0) — the CNOT acts on qubit 0, so the second X(0)
    // must NOT be merged with the first X(0).
    std::vector<Gate> in = { X(0), CNOT(0, 1), X(0) };
    auto out = combine_single_qubit_gates(in);
    // First X(0) alone, then CNOT, then second X(0) alone = 3 gates.
    ASSERT_EQ(out.size(), 3u);
}

TEST(CombineSingleQubitGatesTest, TwoQubitGateBarrier_CorrectGateOrder) {
    std::vector<Gate> in = { H(0), CNOT(0, 1), Z(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);
    // First output gate should be H(0).
    EXPECT_TRUE(toDense(out[0].get_base_matrix())
                    .isApprox(toDense(hadamard()), kTol));
    // Middle gate should be CNOT.
    EXPECT_EQ(out[1].get_name(), "X");
    // Last gate should be Z(0).
    EXPECT_TRUE(toDense(out[2].get_base_matrix())
                    .isApprox(toDense(pauliZ()), kTol));
}

TEST(CombineSingleQubitGatesTest, TwoQubitGateOnOtherQubit_DoesNotBlock) {
    // X(0), CNOT(1,2), X(0) — CNOT doesn't touch qubit 0, so both X(0) merge.
    std::vector<Gate> in = { X(0), CNOT(1, 2), X(0) };
    auto out = combine_single_qubit_gates(in);
    // Qubit-0 gates merge into 1; CNOT passes through = 2 total.
    ASSERT_EQ(out.size(), 2u);
    // Find the qubit-0 combined gate and verify it is X·X = I.
    bool foundIdentity = false;
    for (auto& g : out) {
        if (g.get_qubits().size() == 1 && g.get_qubits()[0] == 0) {
            EXPECT_TRUE(toDense(g.get_base_matrix())
                            .isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
            foundIdentity = true;
        }
    }
    EXPECT_TRUE(foundIdentity);
}

TEST(CombineSingleQubitGatesTest, MultipleGroupsSeparatedByTwoQubitGates) {
    // H(0), X(0), CNOT(0,1), Z(0), Y(0)
    // Group 1: H(0)·X(0) → merged
    // Barrier:  CNOT(0,1)
    // Group 2: Z(0)·Y(0) → merged
    std::vector<Gate> in = { H(0), X(0), CNOT(0, 1), Z(0), Y(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 3u);

    // First combined: X·H
    Eigen::MatrixXcd exp1 = toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(exp1, kTol));

    // Middle: CNOT
    EXPECT_EQ(out[1].get_name(), "X");

    // Second combined: Y·Z
    Eigen::MatrixXcd exp2 = toDense(pauliY()) * toDense(pauliZ());
    EXPECT_TRUE(toDense(out[2].get_base_matrix()).isApprox(exp2, kTol));
}


// ══════════════════════════════════════════════════════════════
// 5. Ordering — verify left-to-right matrix multiplication order
// ══════════════════════════════════════════════════════════════

TEST(CombineSingleQubitGatesTest, OrderMatters_XZ_NeqZX) {
    // X then Z:  combined = Z·X
    auto outXZ = combine_single_qubit_gates({ X(0), Z(0) });
    // Z then X:  combined = X·Z
    auto outZX = combine_single_qubit_gates({ Z(0), X(0) });

    ASSERT_EQ(outXZ.size(), 1u);
    ASSERT_EQ(outZX.size(), 1u);

    // The two results should be different matrices (Z·X ≠ X·Z for Paulis).
    EXPECT_FALSE(toDense(outXZ[0].get_base_matrix())
                     .isApprox(toDense(outZX[0].get_base_matrix()), kTol));
}

TEST(CombineSingleQubitGatesTest, FourGates_CorrectProductOrder) {
    // H, X, Y, Z on qubit 0: combined = Z·Y·X·H
    std::vector<Gate> in = { H(0), X(0), Y(0), Z(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);

    Eigen::MatrixXcd expected =
        toDense(pauliZ()) * toDense(pauliY()) * toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}


// ══════════════════════════════════════════════════════════════
// 6. Output gate properties
// ══════════════════════════════════════════════════════════════

TEST(CombineSingleQubitGatesTest, CombinedGate_TargetQubitPreserved) {
    std::vector<Gate> in = { X(3), Z(3) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].get_target_qubits().size(), 1u);
    EXPECT_EQ(out[0].get_target_qubits()[0], 3);
}

TEST(CombineSingleQubitGatesTest, CombinedGate_NoControlQubits) {
    std::vector<Gate> in = { X(0), H(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0].get_control_qubits().empty());
}

TEST(CombineSingleQubitGatesTest, CombinedGate_IsStillSingleQubit) {
    std::vector<Gate> in = { H(2), X(2), Z(2) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].get_nqubits(), 1);
    EXPECT_EQ(out[0].get_qubits()[0], 2);
}

TEST(CombineSingleQubitGatesTest, UnchangedGate_RetainsOriginalName) {
    // A lone gate that cannot be combined should keep its original gate name.
    std::vector<Gate> in = { X(0), CNOT(0, 1) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].get_name(), "X");
    EXPECT_EQ(out[1].get_name(), "X");
}

TEST(CombineSingleQubitGatesTest, CombinedGate_NameIsNonEmpty) {
    std::vector<Gate> in = { X(0), Z(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FALSE(out[0].get_name().empty());
}

TEST(CombineSingleQubitGatesTest, CombinedGate_IsNormalized) {
    // Product of unitary gates should remain unitary.
    std::vector<Gate> in = { H(0), X(0), Z(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0].is_normalized());
}


// ══════════════════════════════════════════════════════════════
// 7. Parallel independent qubits
// ══════════════════════════════════════════════════════════════

TEST(CombineSingleQubitGatesTest, TwoQubitsEachWithTwoGates_TwoOutputGates) {
    // H(0), X(0) → combined on qubit 0
    // Z(1), Y(1) → combined on qubit 1
    std::vector<Gate> in = { H(0), X(0), Z(1), Y(1) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);
}

TEST(CombineSingleQubitGatesTest, TwoQubitsEachWithTwoGates_CorrectMatrices) {
    std::vector<Gate> in = { H(0), X(0), Z(1), Y(1) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 2u);

    // First combined gate: qubit 0, matrix = X·H
    Eigen::MatrixXcd exp0 = toDense(pauliX()) * toDense(hadamard());
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(exp0, kTol));

    // Second combined gate: qubit 1, matrix = Y·Z
    Eigen::MatrixXcd exp1 = toDense(pauliY()) * toDense(pauliZ());
    EXPECT_TRUE(toDense(out[1].get_base_matrix()).isApprox(exp1, kTol));
}


// ══════════════════════════════════════════════════════════════
// 8. Parameterised gates
// ══════════════════════════════════════════════════════════════

TEST(CombineSingleQubitGatesTest, TwoRZGates_CombinedMatrixIsProductRZ) {
    double t1 = M_PI / 4.0, t2 = M_PI / 3.0;
    std::vector<Gate> in = { RZ(0, t1), RZ(0, t2) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);

    Eigen::MatrixXcd expected = toDense(rz(t2)) * toDense(rz(t1));
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}

TEST(CombineSingleQubitGatesTest, RZFollowedByH_CombinedCorrectly) {
    double t = M_PI / 2.0;
    std::vector<Gate> in = { RZ(0, t), H(0) };
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);

    Eigen::MatrixXcd expected = toDense(hadamard()) * toDense(rz(t));
    EXPECT_TRUE(toDense(out[0].get_base_matrix()).isApprox(expected, kTol));
}


// ══════════════════════════════════════════════════════════════
// 9. Large / stress sequences
// ══════════════════════════════════════════════════════════════

TEST(CombineSingleQubitGatesTest, ManyIdentityGates_CombineToIdentity) {
    const int N = 20;
    std::vector<Gate> in;
    for (int i = 0; i < N; ++i) in.push_back(makeI(0));
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix())
                    .isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, ManyXGates_EvenCountIsIdentity) {
    const int N = 10; // even → X^10 = I
    std::vector<Gate> in;
    for (int i = 0; i < N; ++i) in.push_back(X(0));
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix())
                    .isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(CombineSingleQubitGatesTest, ManyXGates_OddCountIsX) {
    const int N = 9; // odd → X
    std::vector<Gate> in;
    for (int i = 0; i < N; ++i) in.push_back(X(0));
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(toDense(out[0].get_base_matrix())
                    .isApprox(toDense(pauliX()), kTol));
}

TEST(CombineSingleQubitGatesTest, InterleavedMultiQubitGates_CountIsCorrect) {
    // Pattern: single, 2Q-barrier, single, 2Q-barrier, ..., single
    // Each single-qubit gate is isolated by barriers → no combining.
    std::vector<Gate> in;
    const int N = 5;
    for (int i = 0; i < N; ++i) {
        in.push_back(X(0));
        if (i < N - 1) in.push_back(CNOT(0, 1));
    }
    // N single + (N-1) two-qubit = 2N-1 gates, no merging possible.
    auto out = combine_single_qubit_gates(in);
    ASSERT_EQ(out.size(), static_cast<size_t>(2 * N - 1));
}