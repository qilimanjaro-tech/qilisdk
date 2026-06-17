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
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/matrix_free_operator.h"

namespace {

DenseMatrix ket0() {
    DenseMatrix s(2, 1);
    s(0, 0) = 1.0;
    s(1, 0) = 0.0;
    return s;
}
DenseMatrix ket1() {
    DenseMatrix s(2, 1);
    s(0, 0) = 0.0;
    s(1, 0) = 1.0;
    return s;
}
DenseMatrix ketPlus() {
    DenseMatrix s(2, 1);
    s(0, 0) = s(1, 0) = 1.0 / std::sqrt(2.0);
    return s;
}
DenseMatrix ketMinus() {
    DenseMatrix s(2, 1);
    s(0, 0) = 1.0 / std::sqrt(2.0);
    s(1, 0) = -1.0 / std::sqrt(2.0);
    return s;
}

DenseMatrix ket00() {
    DenseMatrix s(4, 1);
    s.setZero();
    s(0, 0) = 1.0;
    return s;
}
DenseMatrix ket01() {
    DenseMatrix s(4, 1);
    s.setZero();
    s(1, 0) = 1.0;
    return s;
}
DenseMatrix ket10() {
    DenseMatrix s(4, 1);
    s.setZero();
    s(2, 0) = 1.0;
    return s;
}
DenseMatrix ket11() {
    DenseMatrix s(4, 1);
    s.setZero();
    s(3, 0) = 1.0;
    return s;
}

DenseMatrix ket000() {
    DenseMatrix s(8, 1);
    s.setZero();
    s(0, 0) = 1.0;
    return s;
}
DenseMatrix ket101() {
    DenseMatrix s(8, 1);
    s.setZero();
    s(5, 0) = 1.0;
    return s;
}
DenseMatrix ket110() {
    DenseMatrix s(8, 1);
    s.setZero();
    s(6, 0) = 1.0;
    return s;
}
DenseMatrix ket111() {
    DenseMatrix s(8, 1);
    s.setZero();
    s(7, 0) = 1.0;
    return s;
}

DenseMatrix dm0() {
    DenseMatrix d(2, 2);
    d.setZero();
    d(0, 0) = 1.0;
    return d;
}
DenseMatrix dm1() {
    DenseMatrix d(2, 2);
    d.setZero();
    d(1, 1) = 1.0;
    return d;
}
DenseMatrix dmPlus() {
    DenseMatrix d(2, 2);
    d(0, 0) = d(0, 1) = d(1, 0) = d(1, 1) = 0.5;
    return d;
}
DenseMatrix dmMinus() {
    DenseMatrix d(2, 2);
    d(0, 0) = d(1, 1) = 0.5;
    d(0, 1) = d(1, 0) = -0.5;
    return d;
}

DenseMatrix ketbra(const DenseMatrix& ket) {
    return ket * ket.adjoint();
}
DenseMatrix ketbra(const DenseMatrix& ket, const DenseMatrix& bra) {
    return ket * bra.adjoint();
}

const std::complex<double> kImag{0.0, 1.0};
const std::complex<double> kImagConj{0.0, -1.0};
const double kInvSqrt2 = 1.0 / std::sqrt(2.0);
const std::complex<double> kTPhase = std::exp(std::complex<double>(0.0, M_PI / 4.0));
const std::complex<double> kTPhaseConj = std::conj(kTPhase);

// A complex, NON-symmetric single-qubit unitary (a general U3-style rotation).
// The off-diagonal entries differ (U(0,1) != U(1,0)) and are complex, so for this
// gate U* (conjugate) != U† (conjugate transpose). This is exactly the case that
// distinguishes a correct rho -> U rho U† from the buggy rho -> U rho U*.
DenseMatrix asymComplexU() {
    const double theta = 2.0;
    const double phi = 0.7;
    const double lambda = 1.3;
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    DenseMatrix u(2, 2);
    u(0, 0) = c;
    u(0, 1) = -std::exp(std::complex<double>(0.0, lambda)) * s;
    u(1, 0) = std::exp(std::complex<double>(0.0, phi)) * s;
    u(1, 1) = std::exp(std::complex<double>(0.0, phi + lambda)) * c;
    return u;
}

// Embed a single-qubit operator acting on `target` into the full 2^nqubits space.
// Matches the simulator's big-endian convention: qubit q corresponds to bit (nqubits-1-q).
DenseMatrix embedSingleQubit(const DenseMatrix& u, int target, int nqubits) {
    const int dim = 1 << nqubits;
    const int tbit = nqubits - 1 - target;
    DenseMatrix full = DenseMatrix::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            bool rest_matches = true;
            for (int b = 0; b < nqubits; ++b) {
                if (b == tbit) {
                    continue;
                }
                if (((i >> b) & 1) != ((j >> b) & 1)) {
                    rest_matches = false;
                    break;
                }
            }
            if (rest_matches) {
                full(i, j) = u((i >> tbit) & 1, (j >> tbit) & 1);
            }
        }
    }
    return full;
}

// Embed a controlled single-qubit operator (single control) into the full space.
DenseMatrix embedControlledSingleQubit(const DenseMatrix& u, int control, int target, int nqubits) {
    const int dim = 1 << nqubits;
    const int tbit = nqubits - 1 - target;
    const int cbit = nqubits - 1 - control;
    DenseMatrix full = DenseMatrix::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) {
        if (((i >> cbit) & 1) == 0) {
            full(i, i) = 1.0;  // control not set -> identity
            continue;
        }
        for (int j = 0; j < dim; ++j) {
            if (((j >> cbit) & 1) == 0) {
                continue;
            }
            bool rest_matches = true;
            for (int b = 0; b < nqubits; ++b) {
                if (b == tbit) {
                    continue;
                }
                if (((i >> b) & 1) != ((j >> b) & 1)) {
                    rest_matches = false;
                    break;
                }
            }
            if (rest_matches) {
                full(i, j) = u((i >> tbit) & 1, (j >> tbit) & 1);
            }
        }
    }
    return full;
}

// A genuinely mixed two-qubit Hermitian density matrix (trace 1).
DenseMatrix mixedTwoQubitDensityMatrix() {
    DenseMatrix bell = (ket00() + ket11()) * kInvSqrt2;
    DenseMatrix rho = 0.7 * ketbra(bell) + 0.3 * ketbra(ket01());
    return rho;
}

}  // namespace

TEST(MatrixFreeOperator, NameAndTargetQubitAccessors) {
    MatrixFreeOperator op("X", 3);
    EXPECT_EQ(op.get_name(), "X");
    EXPECT_EQ(op.get_target_qubits().size(), 1);
    EXPECT_EQ(op.get_control_qubits().size(), 0);
    EXPECT_EQ(op.get_target_qubits()[0], 3);
}

TEST(MatrixFreeOperator, GetIdSingleQubit) {
    MatrixFreeOperator op("Z", 2);
    EXPECT_EQ(op.get_id(), "Z(2)");
}

TEST(MatrixFreeOperator, InitWithTargetAndControl) {
    MatrixFreeOperator op("CNOT", 1, 0);
    EXPECT_EQ(op.get_name(), "X");
    EXPECT_EQ(op.get_target_qubits().size(), 1);
    EXPECT_EQ(op.get_control_qubits().size(), 1);
    EXPECT_EQ(op.get_target_qubits()[0], 0);
    EXPECT_EQ(op.get_control_qubits()[0], 1);
    EXPECT_EQ(op.get_id(), "X(0)_c1");
}

TEST(MatrixFreeOperator, InitCY) {
    MatrixFreeOperator op("CY", 1, 0);
    EXPECT_EQ(op.get_name(), "Y");
    EXPECT_EQ(op.get_target_qubits().size(), 1);
    EXPECT_EQ(op.get_control_qubits().size(), 1);
    EXPECT_EQ(op.get_target_qubits()[0], 0);
    EXPECT_EQ(op.get_control_qubits()[0], 1);
    EXPECT_EQ(op.get_id(), "Y(0)_c1");
}

TEST(MatrixFreeOperator, InitSWAP) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    EXPECT_EQ(op.get_name(), "SWAP");
    EXPECT_EQ(op.get_target_qubits().size(), 2);
    EXPECT_EQ(op.get_control_qubits().size(), 0);
    EXPECT_EQ(op.get_target_qubits()[0], 0);
    EXPECT_EQ(op.get_target_qubits()[1], 1);
    EXPECT_EQ(op.get_id(), "SWAP(0,1)");
}

TEST(MatrixFreeOperator, InitWithGate) {
    DenseMatrix x_matrix(2, 2);
    x_matrix << 0, 1, 1, 0;
    Gate g("X", x_matrix.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(g);
    EXPECT_EQ(op.get_name(), "X");
    EXPECT_EQ(op.get_target_qubits().size(), 1);
    EXPECT_EQ(op.get_control_qubits().size(), 1);
    EXPECT_EQ(op.get_target_qubits()[0], 1);
    EXPECT_EQ(op.get_control_qubits()[0], 0);
    EXPECT_EQ(op.get_id(), "X(1)_c0");
}

TEST(MatrixFreeOperator, InitWithMultiTargetGate) {
    DenseMatrix x_matrix(2, 2);
    x_matrix << 0, 1, 1, 0;
    Gate g("X", x_matrix.sparseView(), {}, {1, 2}, {});
    EXPECT_ANY_THROW(MatrixFreeOperator op(g));
}

TEST(MatrixFreeOperator, InitWithMultiControlGate) {
    DenseMatrix x_matrix(2, 2);
    x_matrix << 0, 1, 1, 0;
    Gate g("Y", x_matrix.sparseView(), {1, 2}, {1}, {});
    EXPECT_ANY_THROW(MatrixFreeOperator op(g));
}

TEST(MatrixFreeOperator, EqualitySameOperator) {
    MatrixFreeOperator a("X", 0), b("X", 0);
    EXPECT_TRUE(a == b);
}

TEST(MatrixFreeOperator, InequalityDifferentName) {
    EXPECT_FALSE(MatrixFreeOperator("X", 0) == MatrixFreeOperator("Z", 0));
}

TEST(MatrixFreeOperator, InequalityDifferentTarget) {
    EXPECT_FALSE(MatrixFreeOperator("X", 0) == MatrixFreeOperator("X", 1));
}

TEST(MatrixFreeOperator, StreamOutputContainsNameAndTarget) {
    MatrixFreeOperator op("Y", 2, 1);
    std::ostringstream oss;
    oss << op;
    EXPECT_NE(oss.str().find("Y"), std::string::npos);
    EXPECT_NE(oss.str().find("1"), std::string::npos);
    EXPECT_NE(oss.str().find("2"), std::string::npos);
}

TEST(MatrixFreeOperator, UnknownNameThrowsOnApply) {
    MatrixFreeOperator op("UNKNOWN_OP", 0);
    DenseMatrix state = ket0();
    EXPECT_ANY_THROW(op.apply(state, MatrixFreeApplicationType::Left));
}

TEST(MatrixFreeOperator, X_StateVector_Ket0ToKet1) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket1(), 1e-10)) << "X|0> should be approximately |1>, but got:\n" << s;
}

TEST(MatrixFreeOperator, X_StateVector_Ket1ToKet0) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket0(), 1e-10)) << "X|1> should be approximately |0>, but got:\n" << s;
}

TEST(MatrixFreeOperator, X_StateVector_Involutory) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "X² should be approximately the identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, X_StateVector_TwoQubit_QubitZero) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket11(), 1e-10)) << "X_q0 |01> should be approximately |11>, but got:\n" << s;
}

TEST(MatrixFreeOperator, X_StateVector_TwoQubit_QubitOne) {
    MatrixFreeOperator op("X", 1);
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket11(), 1e-10)) << "X_q1 |10> should be approximately |11>, but got:\n" << s;
}

TEST(MatrixFreeOperator, X_Left_DensityMatrix_Dm0ToDm1Rows) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2);
    expected.setZero();
    expected(1, 0) = 1.0;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "X Left |0><0| should be approximately |1><0|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, X_Right_DensityMatrix_Dm0) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2);
    expected.setZero();
    expected(0, 1) = 1.0;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "X Right |0><0| should be approximately |0><1|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, X_LeftAndRight_Dm0GivesDm1) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dm1(), 1e-10)) << "X LAR |0><0| should be approximately |1><1|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, X_LeftAndRight_Dm1GivesDm0) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dm0(), 1e-10)) << "X LAR |1><1| should be approximately |0><0|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Y_StateVector_Ket0) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 0.0;
    expected(1, 0) = kImag;
    ASSERT_TRUE(s.isApprox(expected, 1e-10)) << "Y|0> should be approximately i|1>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Y_StateVector_Ket1) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = kImagConj;
    expected(1, 0) = 0.0;
    ASSERT_TRUE(s.isApprox(expected, 1e-10)) << "Y|1> should be approximately -i|0>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Y_StateVector_Involutory) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "Y² should be approximately the identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, Y_Left_DensityMatrix_Dm0) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2);
    expected.setZero();
    expected(1, 0) = kImag;  // i·<0|0><0|
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Y Left |0><0| should be approximately i|1><0|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Y_Right_DensityMatrix_Dm0) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2);
    expected.setZero();
    expected(0, 1) = kImagConj;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Y Right |0><0| should be approximately -i|0><1|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Y_LeftAndRight_DmPlus_GivesDmMinus) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dmMinus(), 1e-10)) << "Y LAR |+><+| should be approximately |-><-|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Z_StateVector_Ket0Unchanged) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket0(), 1e-10)) << "Z|0> should be approximately |0>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Z_StateVector_Ket1Negated) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ket1();
    expected *= -1.0;
    ASSERT_TRUE(s.isApprox(expected, 1e-10)) << "Z|1> should be approximately -|1>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Z_StateVector_KetPlusGivesKetMinus) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ketPlus();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ketMinus(), 1e-10)) << "Z|+> should be approximately |->, but got:\n" << s;
}

TEST(MatrixFreeOperator, Z_StateVector_Involutory) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "Z² should be approximately the identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, Z_Left_DensityMatrix_Dm1) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected = dm1();
    expected *= -1.0;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Z Left |1><1| should be approximately -|1><1|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Z_Right_DensityMatrix_Dm1) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected = dm1();
    expected *= -1.0;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Z Right |1><1| should be approximately -|1><1|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Z_LeftAndRight_Dm1Unchanged) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dm1(), 1e-10)) << "Z LAR |1><1| should be approximately |1><1|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Z_LeftAndRight_DmPlusGivesDmMinus) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dmMinus(), 1e-10)) << "Z LAR |+><+| should be approximately |-><-|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, H_StateVector_Ket0GivesKetPlus) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ketPlus(), 1e-10)) << "H|0> should be approximately |+>, but got:\n" << s;
}

TEST(MatrixFreeOperator, H_StateVector_Ket1GivesKetMinus) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ketMinus(), 1e-10)) << "H|1> should be approximately |->, but got:\n" << s;
}

TEST(MatrixFreeOperator, H_StateVector_KetPlusGivesKet0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ketPlus();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket0(), 1e-10)) << "H|+> should be approximately |0>, but got:\n" << s;
}

TEST(MatrixFreeOperator, H_StateVector_Involutory) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ket0();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "H² should be approximately the identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, H_Left_DensityMatrix_Dm0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2);
    expected.setZero();
    expected(0, 0) = kInvSqrt2;
    expected(1, 0) = kInvSqrt2;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "H Left |0><0| should be approximately |+><0|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, H_Right_DensityMatrix_Dm0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2);
    expected.setZero();
    expected(0, 0) = kInvSqrt2;
    expected(0, 1) = kInvSqrt2;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "H Right |0><0| should be approximately |0><+|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, H_LeftAndRight_Dm0GivesDmPlus) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dmPlus(), 1e-10)) << "H LAR |0><0| should be approximately |+><+|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, H_LeftAndRight_DmPlusGivesDm0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dm0(), 1e-10)) << "H LAR |+><+| should be approximately |0><0|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, S_StateVector_Ket0Unchanged) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket0(), 1e-10)) << "S|0> should be approximately |0>, but got:\n" << s;
}

TEST(MatrixFreeOperator, S_StateVector_Ket1GivesIKet1) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 0.0;
    expected(1, 0) = kImag;
    ASSERT_TRUE(s.isApprox(expected, 1e-10)) << "S|1> should be approximately i|1>, but got:\n" << s;
}

TEST(MatrixFreeOperator, S_StateVector_FourApplicationsIsIdentity) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    for (int i = 0; i < 4; ++i)
        op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "S⁴ should be approximately the identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, S_Left_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;
    expected(0, 1) = 0.5;
    expected(1, 0) = 0.5 * kImag;
    expected(1, 1) = 0.5 * kImag;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "S Left |+><+| should be approximately |+><+| with i in the bottom row, but got:\n" << rho;
}

TEST(MatrixFreeOperator, S_Right_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;
    expected(0, 1) = 0.5 * kImagConj;
    expected(1, 0) = 0.5;
    expected(1, 1) = 0.5 * kImagConj;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Got:\n" << rho << "\nExpected:\n" << expected;
}

TEST(MatrixFreeOperator, S_LeftAndRight_DmPlus) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;
    expected(0, 1) = 0.5 * kImagConj;
    expected(1, 0) = 0.5 * kImag;
    expected(1, 1) = 0.5;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Got:\n" << rho << "\nExpected:\n" << expected;
}

TEST(MatrixFreeOperator, T_StateVector_Ket0Unchanged) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket0(), 1e-10)) << "T|0> should be approximately |0>, but got:\n" << s;
}

TEST(MatrixFreeOperator, T_StateVector_Ket1GivesTPhaseKet1) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 0.0;
    expected(1, 0) = kTPhase;
    ASSERT_TRUE(s.isApprox(expected, 1e-10)) << "T|1> should be approximately e^{iπ/4}|1>, but got:\n" << s;
}

TEST(MatrixFreeOperator, T_StateVector_EightApplicationsIsIdentity) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    for (int i = 0; i < 8; ++i)
        op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "T⁸ should be approximately the identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, T_Left_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;
    expected(0, 1) = 0.5;
    expected(1, 0) = 0.5 * kTPhase;
    expected(1, 1) = 0.5 * kTPhase;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "T Left |+><+| should be approximately |+><+| with T phase in the bottom row, but got:\n" << rho;
}

TEST(MatrixFreeOperator, T_Right_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;
    expected(0, 1) = 0.5 * kTPhaseConj;
    expected(1, 0) = 0.5;
    expected(1, 1) = 0.5 * kTPhaseConj;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "T Right |+><+| should be approximately |+><+| with T phase in the right column, but got:\n" << rho;
}

TEST(MatrixFreeOperator, T_LeftAndRight_DmPlus) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;
    expected(0, 1) = 0.5 * kTPhaseConj;
    expected(1, 0) = 0.5 * kTPhase;
    expected(1, 1) = 0.5;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "T LAR |+><+| should be approximately |+><+| with T phase in the off-diagonal elements, but got:\n" << rho;
}

TEST(MatrixFreeOperator, T_LeftAndRight_Dm1Unchanged) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(dm1(), 1e-10)) << "T LAR |1><1| should be approximately unchanged, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control0Target1_Ket10FlipsToKet11) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket11(), 1e-10)) << "CNOT(0,1) |10> should be approximately |11>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control0Target1_Ket11FlipsToKet10) {
    MatrixFreeOperator op("CX", 0, 1);
    DenseMatrix s = ket11();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket10(), 1e-10)) << "CNOT(0,1) |11> should be approximately |10>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control0Target1_Ket00Unchanged) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix s = ket00();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket00(), 1e-10)) << "CNOT(0,1) |00> should be approximately |00>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control1Target0_Ket01FlipsToKet11) {
    MatrixFreeOperator op("CNOT", 1, 0);
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket11(), 1e-10)) << "CNOT(1,0) |01> should be approximately |11>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CNOT_StateVector_Involutory) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix s = ket10();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "CNOT² should be approximately identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, CNOT_Left_DensityMatrix_Ket10) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(rho.isApprox(ketbra(ket11(), ket10()), 1e-10)) << "CNOT Left |10><10| should be approximately |11><10|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CNOT_Right_DensityMatrix_Ket10) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::Right);
    ASSERT_TRUE(rho.isApprox(ketbra(ket10(), ket11()), 1e-10)) << "CNOT Right |10><10| should be approximately |10><11|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CNOT_LeftAndRight_Ket10GivesKet11) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(ketbra(ket11()), 1e-10)) << "CNOT LAR |10><10| should be approximately |11><11|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CZ_StateVector_Ket11GetsNegated) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket11();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ket11();
    expected *= -1.0;
    ASSERT_TRUE(s.isApprox(expected, 1e-10)) << "CZ |11> should be approximately -|11>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CZ_StateVector_Ket10Unchanged) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket10(), 1e-10)) << "CZ |10> should be approximately |10>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CZ_StateVector_Ket01Unchanged) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket01(), 1e-10)) << "CZ |01> should be approximately |01>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CZ_StateVector_Involutory) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket11();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "CZ² should be approximately identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, CZ_Left_DensityMatrix_Ket11) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix rho = ketbra(ket11());
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ketbra(ket11());
    expected *= -1.0;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "CZ Left |11><11| should be approximately -|11><11|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CZ_Right_DensityMatrix_Ket11) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix rho = ketbra(ket11());
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected = ketbra(ket11());
    expected *= -1.0;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "CZ Right |11><11| should be approximately -|11><11|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CZ_LeftAndRight_Ket11Unchanged) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix rho = ketbra(ket11());
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(ketbra(ket11()), 1e-10)) << "CZ LAR |11><11| should be approximately unchanged, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CustomBaseMatrix_StateVector_ActsCorrectly) {
    DenseMatrix hmat(2, 2);
    hmat(0, 0) = kInvSqrt2;
    hmat(0, 1) = kInvSqrt2;
    hmat(1, 0) = kInvSqrt2;
    hmat(1, 1) = -kInvSqrt2;
    Gate g("CustomH", hmat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ketPlus(), 1e-10)) << "custom H|0> should be approximately |+>, but got:\n" << s;
}

TEST(MatrixFreeOperator, CustomBaseMatrixLeft_DensityMatrix_ActsCorrectly) {
    DenseMatrix hmat(2, 2);
    hmat(0, 0) = 0;
    hmat(0, 1) = 1;
    hmat(1, 0) = 1;
    hmat(1, 1) = 0;
    Gate g("CustomX", hmat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = ketbra(ket0());
    op.apply(rho, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(rho.isApprox(ketbra(ket1(), ket0()), 1e-10)) << "custom X Left |0><0| should be approximately |1><0|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CustomBaseMatrixRight_DensityMatrix_ActsCorrectly) {
    DenseMatrix hmat(2, 2);
    hmat(0, 0) = 0;
    hmat(0, 1) = 1;
    hmat(1, 0) = 1;
    hmat(1, 1) = 0;
    Gate g("CustomX", hmat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = ketbra(ket0());
    op.apply(rho, MatrixFreeApplicationType::Right);
    ASSERT_TRUE(rho.isApprox(ketbra(ket0(), ket1()), 1e-10)) << "custom X Right |0><0| should be approximately |0><1|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, CustomBaseMatrix_LeftAndRight_UsesConjugateOnRight) {
    DenseMatrix umat(2, 2);
    umat.setZero();
    umat(0, 0) = 1.0;
    umat(1, 1) = kImag;
    Gate g("CustomS", umat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;
    expected(0, 1) = 0.5 * kImagConj;
    expected(1, 0) = 0.5 * kImag;
    expected(1, 1) = 0.5;
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Got:\n" << rho << "\nExpected:\n" << expected;
}

TEST(MatrixFreeOperator, SWAP_StateVector_Ket01FlipsToKet10) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket10(), 1e-10)) << "SWAP |01> should be approximately |10>, but got:\n" << s;
}

TEST(MatrixFreeOperator, SWAP_StateVector_Ket10FlipsToKet01) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket01(), 1e-10)) << "SWAP |10> should be approximately |01>, but got:\n" << s;
}

TEST(MatrixFreeOperator, SWAP_StateVector_Ket00Unchanged) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix s = ket00();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket00(), 1e-10)) << "SWAP |00> should be approximately |00>, but got:\n" << s;
}

TEST(MatrixFreeOperator, SWAP_StateVector_Ket11Unchanged) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix s = ket11();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket11(), 1e-10)) << "SWAP |11> should be approximately |11>, but got:\n" << s;
}

TEST(MatrixFreeOperator, SWAP_StateVector_Involutory) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix s = ket01();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "SWAP² should be approximately identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, SWAP_Left_DensityMatrix_Ket01) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix rho = ketbra(ket01());
    op.apply(rho, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(rho.isApprox(ketbra(ket10(), ket01()), 1e-10)) << "SWAP Left |01><01| should be approximately |10><01|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, SWAP_Right_DensityMatrix_Ket01) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix rho = ketbra(ket01());
    op.apply(rho, MatrixFreeApplicationType::Right);
    ASSERT_TRUE(rho.isApprox(ketbra(ket01(), ket10()), 1e-10)) << "SWAP Right |01><01| should be approximately |01><10|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, SWAP_LeftAndRight_Ket01GivesKet10) {
    MatrixFreeOperator op("SWAP", {}, {0, 1}, DenseMatrix());
    DenseMatrix rho = ketbra(ket01());
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(ketbra(ket10()), 1e-10)) << "SWAP LAR |01><01| should be approximately |10><10|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Toffoli_StateVector_Control0Control1Target2_Ket110FlipsToKet111) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix s = ket110();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket111(), 1e-10)) << "Toffoli(0,1,2) |110> should be approximately |111>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Toffoli_StateVector_Control0Control1Target2_Ket111FlipsToKet110) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix s = ket111();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket110(), 1e-10)) << "Toffoli(0,1,2) |111> should be approximately |110>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Toffoli_StateVector_Control0Control1Target2_Ket000Unchanged) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix s = ket000();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket000(), 1e-10)) << "Toffoli(0,1,2) |000> should be approximately |000>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Toffoli_StateVector_Control0Control1Target2_Ket101Unchanged) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix s = ket101();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket101(), 1e-10)) << "Toffoli(0,1,2) |101> should be approximately |101>, but got:\n" << s;
}

TEST(MatrixFreeOperator, Toffoli_StateVector_Involutory) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix s = ket110();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "Toffoli² should be approximately identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, Toffoli_Left_DensityMatrix_Ket110) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix rho = ketbra(ket110());
    op.apply(rho, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(rho.isApprox(ketbra(ket111(), ket110()), 1e-10)) << "Toffoli Left |110><110| should be approximately |111><110|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Toffoli_Right_DensityMatrix_Ket110) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix rho = ketbra(ket110());
    op.apply(rho, MatrixFreeApplicationType::Right);
    ASSERT_TRUE(rho.isApprox(ketbra(ket110(), ket111()), 1e-10)) << "Toffoli Right |110><110| should be approximately |110><111|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, Toffoli_LeftAndRight_Ket110GivesKet111) {
    MatrixFreeOperator op("Toffoli", {0, 1}, {2}, DenseMatrix());
    DenseMatrix rho = ketbra(ket110());
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(ketbra(ket111()), 1e-10)) << "Toffoli LAR |110><110| should be approximately |111><111|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, ControlledCustomGate_StateVector_Control0Target1_Ket10FlipsToKet11) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket11(), 1e-10)) << "Controlled Custom X |10> should be approximately |11>, but got:\n" << s;
}

TEST(MatrixFreeOperator, ControlledCustomGate_StateVector_Control0Target1_Ket11FlipsToKet10) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix s = ket11();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket10(), 1e-10)) << "Controlled Custom X |11> should be approximately |10>, but got:\n" << s;
}

TEST(MatrixFreeOperator, ControlledCustomGate_StateVector_Control0Target1_Ket00Unchanged) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix s = ket00();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket00(), 1e-10)) << "Controlled Custom X |00> should be approximately |00>, but got:\n" << s;
}

TEST(MatrixFreeOperator, ControlledCustomGate_StateVector_Control0Target1_Ket01Unchanged) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket01(), 1e-10)) << "Controlled Custom X |01> should be approximately |01>, but got:\n" << s;
}

TEST(MatrixFreeOperator, ControlledCustomGate_StateVector_Involutory) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix s = ket10();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(original, 1e-10)) << "Controlled Custom X² should be approximately identity, but got:\n" << s;
}

TEST(MatrixFreeOperator, ControlledCustomGate_Left_DensityMatrix_Control0Target1_Ket10) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(rho.isApprox(ketbra(ket11(), ket10()), 1e-10)) << "Controlled Custom X Left |10><10| should be approximately |11><10|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, ControlledCustomGate_Right_DensityMatrix_Control0Target1_Ket10) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::Right);
    ASSERT_TRUE(rho.isApprox(ketbra(ket10(), ket11()), 1e-10)) << "Controlled Custom X Right |10><10| should be approximately |10><11|, but got:\n" << rho;
}

TEST(MatrixFreeOperator, ControlledCustomGate_LeftAndRight_Control0Target1_Ket10GivesKet11) {
    DenseMatrix xmat(2, 2);
    xmat(0, 0) = 0;
    xmat(0, 1) = 1;
    xmat(1, 0) = 1;
    xmat(1, 1) = 0;
    Gate customX("CustomX", xmat.sparseView(), {0}, {1}, {});
    MatrixFreeOperator op(customX);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(ketbra(ket11()), 1e-10)) << "Controlled Custom X LAR |10><10| should be approximately |11><11|, but got:\n" << rho;
}

// --- Regression tests for the rho -> U rho U† density-matrix path with a COMPLEX,
// --- NON-symmetric gate. The right multiplication must apply U† (conjugate transpose),
// --- not U* (conjugate). For real-symmetric gates (X, Z, H) and diagonal gates (S, T)
// --- these coincide, which is why the bug was only triggered by gates like U2/U3.

TEST(MatrixFreeOperator, AsymComplexGate_Right_DensityMatrix_GivesRhoUdag) {
    DenseMatrix u = asymComplexU();
    Gate g("AsymU", u.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = dmPlus();
    DenseMatrix expected = rho * u.adjoint();
    op.apply(rho, MatrixFreeApplicationType::Right);
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Right must compute rho*U† (conjugate transpose), got:\n" << rho << "\nexpected:\n" << expected;
}

TEST(MatrixFreeOperator, AsymComplexGate_LeftAndRight_GivesUrhoUdagAndStaysHermitian) {
    DenseMatrix u = asymComplexU();
    Gate g("AsymU", u.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = dmPlus();
    DenseMatrix expected = u * rho * u.adjoint();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "LeftAndRight must compute U rho U†, got:\n" << rho << "\nexpected:\n" << expected;
    // The whole point: a Hermitian input must stay Hermitian (the bug produced U rho U*, which is not).
    ASSERT_TRUE(rho.isApprox(rho.adjoint(), 1e-10)) << "U rho U† of a Hermitian rho must stay Hermitian, got:\n" << rho;
}

TEST(MatrixFreeOperator, AsymComplexGate_LeftAndRight_TwoQubitTarget1_MixedState) {
    DenseMatrix u = asymComplexU();
    Gate g("AsymU", u.sparseView(), {}, {1}, {});  // target qubit 1 -> exercises the strided embedding
    MatrixFreeOperator op(g);
    DenseMatrix rho = mixedTwoQubitDensityMatrix();
    DenseMatrix full = embedSingleQubit(u, 1, 2);
    DenseMatrix expected = full * rho * full.adjoint();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "2-qubit U rho U† mismatch, got:\n" << rho << "\nexpected:\n" << expected;
    ASSERT_TRUE(rho.isApprox(rho.adjoint(), 1e-10)) << "Result must stay Hermitian, got:\n" << rho;
}

TEST(MatrixFreeOperator, ControlledAsymComplexGate_LeftAndRight_MixedStateStaysHermitian) {
    DenseMatrix u = asymComplexU();
    Gate g("AsymU", u.sparseView(), {0}, {1}, {});  // control 0, target 1
    MatrixFreeOperator op(g);
    DenseMatrix rho = mixedTwoQubitDensityMatrix();
    DenseMatrix full = embedControlledSingleQubit(u, 0, 1, 2);
    DenseMatrix expected = full * rho * full.adjoint();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    ASSERT_TRUE(rho.isApprox(expected, 1e-10)) << "Controlled U rho U† mismatch, got:\n" << rho << "\nexpected:\n" << expected;
    ASSERT_TRUE(rho.isApprox(rho.adjoint(), 1e-10)) << "Result must stay Hermitian, got:\n" << rho;
}

// --- Dense multi-qubit (fused) gate application ---------------------------

namespace {

DenseMatrix hMat() {
    DenseMatrix h(2, 2);
    Real v = 1.0 / std::sqrt(2.0);
    h(0, 0) = v;
    h(0, 1) = v;
    h(1, 0) = v;
    h(1, 1) = -v;
    return h;
}

DenseMatrix swap4() {
    DenseMatrix m(4, 4);
    m.setZero();
    m(0, 0) = 1.0;
    m(1, 2) = 1.0;
    m(2, 1) = 1.0;
    m(3, 3) = 1.0;
    return m;
}

// Reference: apply a dense base matrix on `targets` of an n-qubit state by
// expanding it to the full register via Gate::get_full_matrix.
DenseMatrix applyViaFullMatrix(const DenseMatrix& base, const std::vector<int>& targets, int n, const DenseMatrix& state) {
    SparseMatrix sparse_base = base.sparseView();
    Gate g("FUSED", sparse_base, {}, targets, {});
    SparseMatrix full = g.get_full_matrix(n);
    return DenseMatrix(full * state);
}

DenseMatrix randomState(int n, unsigned seed) {
    long dim = 1L << n;
    DenseMatrix s(dim, 1);
    // Deterministic pseudo-random fill (no <random> dependency needed here).
    unsigned x = seed * 2654435761u + 1u;
    for (long i = 0; i < dim; ++i) {
        x = x * 1664525u + 1013904223u;
        Real re = static_cast<Real>((x >> 9) & 0xFFFF) / 65535.0 - 0.5;
        x = x * 1664525u + 1013904223u;
        Real im = static_cast<Real>((x >> 9) & 0xFFFF) / 65535.0 - 0.5;
        s(i, 0) = Complex(re, im);
    }
    return s / s.norm();
}

}  // namespace

TEST(MatrixFreeOperator, DenseMultiQubit_ConstructorAllowsTwoTargets) {
    MatrixFreeOperator op("FUSED", {}, {0, 1}, swap4());
    EXPECT_EQ(op.get_target_qubits().size(), 2u);
    EXPECT_EQ(op.get_control_qubits().size(), 0u);
}

TEST(MatrixFreeOperator, DenseMultiQubit_SwapMatrixActsAsSwap) {
    // A dense SWAP matrix applied on targets {0,1} should swap |01> and |10>.
    MatrixFreeOperator op("FUSED", {}, {0, 1}, swap4());
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(s.isApprox(ket10(), 1e-5)) << "Dense SWAP|01> should be |10>, got:\n" << s;
}

TEST(MatrixFreeOperator, DenseMultiQubit_TwoQubitDense_AdjacentTargets) {
    // H⊗H applied on adjacent targets {0,1} of a 3-qubit state.
    DenseMatrix hh = Eigen::kroneckerProduct(hMat(), hMat()).eval();
    DenseMatrix state = randomState(3, 1);
    DenseMatrix expected = applyViaFullMatrix(hh, {0, 1}, 3, state);
    MatrixFreeOperator op("FUSED", {}, {0, 1}, hh);
    op.apply(state, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(state.isApprox(expected, 1e-5)) << "Dense 2-qubit apply mismatch, got:\n" << state << "\nexpected:\n" << expected;
}

TEST(MatrixFreeOperator, DenseMultiQubit_TwoQubitDense_NonAdjacentTargets) {
    // Targets {0,2} on a 4-qubit register exercise the scattered gather/scatter.
    DenseMatrix hh = Eigen::kroneckerProduct(hMat(), hMat()).eval();
    DenseMatrix state = randomState(4, 7);
    DenseMatrix expected = applyViaFullMatrix(hh, {0, 2}, 4, state);
    MatrixFreeOperator op("FUSED", {}, {0, 2}, hh);
    op.apply(state, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(state.isApprox(expected, 1e-5)) << "Dense 2-qubit non-adjacent apply mismatch";
}

TEST(MatrixFreeOperator, DenseMultiQubit_ThreeQubitDense_ScatteredTargets) {
    // A dense 8x8 (H⊗H⊗H) on targets {0,2,4} of a 5-qubit register.
    DenseMatrix hhh = Eigen::kroneckerProduct(hMat(), Eigen::kroneckerProduct(hMat(), hMat()).eval()).eval();
    DenseMatrix state = randomState(5, 13);
    DenseMatrix expected = applyViaFullMatrix(hhh, {0, 2, 4}, 5, state);
    MatrixFreeOperator op("FUSED", {}, {0, 2, 4}, hhh);
    op.apply(state, MatrixFreeApplicationType::Left);
    ASSERT_TRUE(state.isApprox(expected, 1e-5)) << "Dense 3-qubit apply mismatch";
}

TEST(MatrixFreeOperator, DenseMultiQubit_ThrowsOnDensityMatrix) {
    // Fused operators are statevector-only; applying to a density matrix throws.
    MatrixFreeOperator op("FUSED", {}, {0, 1}, swap4());
    DenseMatrix rho(4, 4);
    rho.setZero();
    rho(1, 1) = 1.0;
    EXPECT_ANY_THROW(op.apply(rho, MatrixFreeApplicationType::LeftAndRight));
}

// GCOV_EXCL_BR_STOP