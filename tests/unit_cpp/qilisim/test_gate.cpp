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

#include "../../../src/qilisdk_cpp/backends/qilisim/digital/gate.h"

#include <cmath>
#include <complex>
#include <sstream>
#include <string>
#include <vector>

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

}

class GateTest : public ::testing::Test {
   protected:
    Gate xGate = Gate("X", pauliX(), {}, {0}, {});
    Gate hGate = Gate("H", hadamard(), {}, {0}, {});
    Gate zGate = Gate("Z", pauliZ(), {}, {0}, {});
    Gate idGate = Gate("I", identity2(), {}, {0}, {});
    Gate rzGate = Gate("RZ", rz(M_PI), {}, {1}, {{"theta", M_PI}});
    Gate cnotGate = Gate("CNOT", pauliX(), {0}, {1}, {});
    Gate cyGate = Gate("CY", pauliY(), {0}, {1}, {});
};

TEST_F(GateTest, GetNQubits_SingleQubitGate) {
    EXPECT_EQ(xGate.get_nqubits(), 1);
    EXPECT_EQ(hGate.get_nqubits(), 1);
}

TEST_F(GateTest, GetNQubits_TwoQubitGate) {
    EXPECT_EQ(cnotGate.get_nqubits(), 2);
}

TEST_F(GateTest, GetNQubits_ControlAndTarget) {
    Gate gate("CZ", pauliZ(), {2}, {5}, {});
    EXPECT_EQ(gate.get_nqubits(), 2);
}

TEST_F(GateTest, GetQubits_SingleTarget) {
    auto q = xGate.get_qubits();
    ASSERT_EQ(q.size(), 1u);
    EXPECT_EQ(q[0], 0);
}

TEST_F(GateTest, GetTargetQubits_ReturnsTargets) {
    auto t = cnotGate.get_target_qubits();
    ASSERT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0], 1);
}

TEST_F(GateTest, GetControlQubits_ReturnsControls) {
    auto c = cnotGate.get_control_qubits();
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0], 0);
}

TEST_F(GateTest, GetControlQubits_EmptyForSingleQubitGate) {
    EXPECT_TRUE(xGate.get_control_qubits().empty());
}

TEST_F(GateTest, GetQubits_ContainsBothControlAndTarget) {
    auto q = cnotGate.get_qubits();
    ASSERT_EQ(q.size(), 2u);
    EXPECT_NE(std::find(q.begin(), q.end(), 0), q.end());
    EXPECT_NE(std::find(q.begin(), q.end(), 1), q.end());
}

TEST_F(GateTest, GetQubits_MultipleTargets) {
    Gate swapGate("SWAP", identity2(), {}, {0, 1}, {});
    auto q = swapGate.get_qubits();
    ASSERT_EQ(q.size(), 2u);
}

TEST_F(GateTest, GetName_ReturnsGateType) {
    EXPECT_EQ(xGate.get_name(), "X");
    EXPECT_EQ(hGate.get_name(), "H");
    EXPECT_EQ(cnotGate.get_name(), "X");
    EXPECT_EQ(rzGate.get_name(), "RZ");
    EXPECT_EQ(cyGate.get_name(), "Y");
}

TEST_F(GateTest, GetName_EmptyStringGate) {
    Gate g("", identity2(), {}, {0}, {});
    EXPECT_EQ(g.get_name(), "");
}

TEST_F(GateTest, GetId_NonEmpty) {
    EXPECT_FALSE(xGate.get_id().empty());
}

TEST_F(GateTest, GetId_DifferentGatesHaveDifferentIds) {
    EXPECT_NE(xGate.get_id(), hGate.get_id());
}

TEST_F(GateTest, GetId_SameConstructionProducesSameId) {
    Gate g1("X", pauliX(), {}, {0}, {});
    Gate g2("X", pauliX(), {}, {0}, {});
    EXPECT_EQ(g1.get_id(), g2.get_id());
}

TEST_F(GateTest, GetId_WithParameters) {
    Gate g1("RZ", rz(M_PI), {}, {0}, {{"theta", M_PI}});
    Gate g2("RZ", rz(M_PI), {}, {0}, {{"theta", M_PI}});
    Gate g3("RZ", rz(M_PI / 2), {}, {0}, {{"theta", M_PI / 2}});
    EXPECT_EQ(g1.get_id(), g2.get_id());
    EXPECT_NE(g1.get_id(), g3.get_id());
}

TEST_F(GateTest, GetParameters_EmptyForNonParametrised) {
    EXPECT_TRUE(xGate.get_parameters().empty());
    EXPECT_TRUE(hGate.get_parameters().empty());
    EXPECT_TRUE(cnotGate.get_parameters().empty());
}

TEST_F(GateTest, GetParameters_ReturnsParametersForRZ) {
    auto params = rzGate.get_parameters();
    ASSERT_EQ(params.size(), 1u);
    EXPECT_EQ(params[0].first, "theta");
    EXPECT_NEAR(params[0].second, M_PI, kTol);
}

TEST_F(GateTest, GetParameters_MultipleParameters) {
    std::vector<std::pair<std::string, double>> ps = {{"alpha", 1.0}, {"beta", 2.0}};
    Gate g("U", identity2(), {}, {0}, ps);
    auto got = g.get_parameters();
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0].first, "alpha");
    EXPECT_NEAR(got[0].second, 1.0, kTol);
    EXPECT_EQ(got[1].first, "beta");
    EXPECT_NEAR(got[1].second, 2.0, kTol);
}

TEST_F(GateTest, GetBaseMatrix_IdentityGate) {
    auto m = toDense(idGate.get_base_matrix());
    EXPECT_NEAR(m(0, 0).real(), 1.0, kTol);
    EXPECT_NEAR(m(1, 1).real(), 1.0, kTol);
    EXPECT_NEAR(m(0, 1).real(), 0.0, kTol);
    EXPECT_NEAR(m(1, 0).real(), 0.0, kTol);
}

TEST_F(GateTest, GetBaseMatrix_PauliX) {
    auto m = toDense(xGate.get_base_matrix());
    EXPECT_NEAR(m(0, 0).real(), 0.0, kTol);
    EXPECT_NEAR(m(0, 1).real(), 1.0, kTol);
    EXPECT_NEAR(m(1, 0).real(), 1.0, kTol);
    EXPECT_NEAR(m(1, 1).real(), 0.0, kTol);
}

TEST_F(GateTest, GetBaseMatrix_Hadamard) {
    double v = 1.0 / std::sqrt(2.0);
    auto m = toDense(hGate.get_base_matrix());
    EXPECT_NEAR(m(0, 0).real(), v, kTol);
    EXPECT_NEAR(m(0, 1).real(), v, kTol);
    EXPECT_NEAR(m(1, 0).real(), v, kTol);
    EXPECT_NEAR(m(1, 1).real(), -v, kTol);
}

TEST_F(GateTest, GetBaseMatrix_Dimensions) {
    EXPECT_EQ(xGate.get_base_matrix().rows(), 2);
    EXPECT_EQ(xGate.get_base_matrix().cols(), 2);
}

TEST_F(GateTest, IsNormalized_StandardGatesAreUnitary) {
    EXPECT_TRUE(xGate.is_normalized());
    EXPECT_TRUE(hGate.is_normalized());
    EXPECT_TRUE(zGate.is_normalized());
    EXPECT_TRUE(idGate.is_normalized());
    EXPECT_TRUE(cnotGate.is_normalized());
}

TEST_F(GateTest, IsNormalized_RZGateIsUnitary) {
    EXPECT_TRUE(rzGate.is_normalized());
}

TEST_F(GateTest, IsNormalized_NonUnitaryMatrix) {
    SparseMatrix nonUnit(2, 2);
    nonUnit.insert(0, 0) = cx(2, 0);
    nonUnit.insert(1, 1) = cx(2, 0);
    nonUnit.makeCompressed();
    Gate g("BAD", nonUnit, {}, {0}, {});
    EXPECT_FALSE(g.is_normalized());
}

TEST_F(GateTest, IsNormalized_ZeroMatrix) {
    SparseMatrix zero(2, 2);
    zero.makeCompressed();
    Gate g("ZERO", zero, {}, {0}, {});
    EXPECT_FALSE(g.is_normalized());
}

TEST_F(GateTest, GetFullMatrix_SingleQubitOn1Qubit) {
    auto full = toDense(xGate.get_full_matrix(1));
    auto base = toDense(pauliX());
    EXPECT_TRUE(full.isApprox(base, kTol));
}

TEST_F(GateTest, GetFullMatrix_XOnQubit0_2Qubits) {
    Gate g("X", pauliX(), {}, {0}, {});
    auto full = toDense(g.get_full_matrix(2));
    ASSERT_EQ(full.rows(), 4);
    ASSERT_EQ(full.cols(), 4);
    Eigen::MatrixXcd expected(4, 4);
    expected << 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0;
    EXPECT_TRUE(full.isApprox(expected, kTol));
}

TEST_F(GateTest, GetFullMatrix_XOnQubit1_2Qubits) {
    Gate g("X", pauliX(), {}, {1}, {});
    auto full = toDense(g.get_full_matrix(2));
    ASSERT_EQ(full.rows(), 4);
    ASSERT_EQ(full.cols(), 4);
    Eigen::MatrixXcd expected(4, 4);
    expected << 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;
    EXPECT_TRUE(full.isApprox(expected, kTol));
}

TEST_F(GateTest, GetFullMatrix_IdentityIsIdentityExpansion) {
    Gate g("I", identity2(), {}, {0}, {});
    auto full = toDense(g.get_full_matrix(3));
    EXPECT_TRUE(full.isApprox(Eigen::MatrixXcd::Identity(8, 8), kTol));
}

TEST_F(GateTest, GetFullMatrix_CNOT_2Qubits) {
    auto full = toDense(cnotGate.get_full_matrix(2));
    ASSERT_EQ(full.rows(), 4);
    Eigen::MatrixXcd expected(4, 4);
    expected << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;
    EXPECT_TRUE(full.isApprox(expected, kTol)) << "Full matrix:\n" << full << "\nExpected:\n" << expected << "\nCNOT base matrix:\n" << toDense(cnotGate.get_base_matrix()) << "\nControl qubits: " << cnotGate.get_control_qubits().size() << " Target qubits: " << cnotGate.get_target_qubits().size();
}

TEST_F(GateTest, GetFullMatrix_CNOT_2Qubits_Swapped) {
    Gate g("CNOT", pauliX(), {1}, {0}, {});
    auto full = toDense(g.get_full_matrix(2));
    ASSERT_EQ(full.rows(), 4);
    Eigen::MatrixXcd expected(4, 4);
    expected << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0;
    EXPECT_TRUE(full.isApprox(expected, kTol)) << "Full matrix:\n" << full << "\nExpected:\n" << expected << "\nCNOT base matrix:\n" << toDense(g.get_base_matrix()) << "\nControl qubits: " << g.get_control_qubits().size() << " Target qubits: " << g.get_target_qubits().size();
}

TEST_F(GateTest, GetFullMatrix_DimensionScalesExponentially) {
    Gate g("X", pauliX(), {}, {0}, {});
    for (int n = 1; n <= 5; ++n) {
        auto full = g.get_full_matrix(n);
        int expected_dim = 1 << n;
        EXPECT_EQ(full.rows(), expected_dim) << "n=" << n;
        EXPECT_EQ(full.cols(), expected_dim) << "n=" << n;
    }
}

TEST_F(GateTest, GetFullMatrix_IsUnitary_XIn3Qubits) {
    Gate g("X", pauliX(), {}, {1}, {});
    auto full = toDense(g.get_full_matrix(3));
    auto product = full * full.adjoint();
    EXPECT_TRUE(product.isApprox(Eigen::MatrixXcd::Identity(8, 8), kTol));
}

TEST_F(GateTest, GetFullMatrix_HadamardSquaredIsIdentity) {
    Gate g("H", hadamard(), {}, {0}, {});
    auto full = toDense(g.get_full_matrix(2));
    auto sq = full * full;
    EXPECT_TRUE(sq.isApprox(Eigen::MatrixXcd::Identity(4, 4), kTol));
}

TEST(GateStreamTest, EmptyVector) {
    std::vector<Gate> gates;
    std::ostringstream oss;
    oss << gates;
    SUCCEED();
}

TEST(GateStreamTest, SingleGate) {
    std::vector<Gate> gates = {Gate("X", pauliX(), {}, {0}, {})};
    std::ostringstream oss;
    oss << gates;
    EXPECT_NE(oss.str().find("X"), std::string::npos);
}

TEST(GateStreamTest, MultipleGates) {
    std::vector<Gate> gates = {Gate("X", pauliX(), {}, {0}, {}), Gate("H", hadamard(), {}, {1}, {}), Gate("CNOT", pauliX(), {0}, {1}, {})};
    std::ostringstream oss;
    EXPECT_NO_THROW(oss << gates);
    EXPECT_FALSE(oss.str().empty());
}

TEST(GateEdgeTest, HighIndexQubit) {
    Gate g("X", pauliX(), {}, {10}, {});
    EXPECT_EQ(g.get_target_qubits()[0], 10);
}

TEST(GateEdgeTest, MultipleControlQubits) {
    SparseMatrix tof(2, 2);
    tof.insert(0, 1) = cx(1, 0);
    tof.insert(1, 0) = cx(1, 0);
    tof.makeCompressed();
    Gate ccx("CCX", tof, {0, 1}, {2}, {});
    EXPECT_EQ(ccx.get_control_qubits().size(), 2u);
    EXPECT_EQ(ccx.get_target_qubits().size(), 1u);
    EXPECT_EQ(ccx.get_nqubits(), 3);
}

TEST(GateEdgeTest, CopyConstructedGateIsEquivalent) {
    Gate original("H", hadamard(), {}, {0}, {});
    Gate copy = original;
    EXPECT_EQ(copy.get_name(), original.get_name());
    EXPECT_EQ(copy.get_id(), original.get_id());
    EXPECT_TRUE(toDense(copy.get_base_matrix()).isApprox(toDense(original.get_base_matrix()), kTol));
}

TEST(GateEdgeTest, RZAtZeroIsIdentity) {
    Gate g("RZ", rz(0.0), {}, {0}, {{"theta", 0.0}});
    auto m = toDense(g.get_base_matrix());
    EXPECT_TRUE(m.isApprox(Eigen::MatrixXcd::Identity(2, 2), kTol));
}

TEST(GateEdgeTest, RZAtTwoPI_IsIdentityUpToGlobalPhase) {
    Gate g("RZ", rz(2 * M_PI), {}, {0}, {{"theta", 2 * M_PI}});
    auto m = toDense(g.get_base_matrix());
    double abs00 = std::abs(m(0, 0));
    double abs11 = std::abs(m(1, 1));
    EXPECT_NEAR(abs00, 1.0, kTol);
    EXPECT_NEAR(abs11, 1.0, kTol);
    EXPECT_NEAR(std::abs(m(0, 1)), 0.0, kTol);
    EXPECT_NEAR(std::abs(m(1, 0)), 0.0, kTol);
}

// GCOV_EXCL_BR_STOP
