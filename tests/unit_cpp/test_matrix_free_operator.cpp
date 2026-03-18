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
#include "../../src/qilisdk_cpp/backends/qilisim/representations/matrix_free_operator.h"


namespace {

DenseMatrix ket0() {
    DenseMatrix s(2, 1);
    s(0, 0) = 1.0; s(1, 0) = 0.0;
    return s;
}
DenseMatrix ket1() {
    DenseMatrix s(2, 1);
    s(0, 0) = 0.0; s(1, 0) = 1.0;
    return s;
}
DenseMatrix ketPlus() {
    DenseMatrix s(2, 1);
    s(0, 0) = s(1, 0) = 1.0 / std::sqrt(2.0);
    return s;
}
DenseMatrix ketMinus() {
    DenseMatrix s(2, 1);
    s(0, 0) =  1.0 / std::sqrt(2.0);
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
    d(0,0) = d(0,1) = d(1,0) = d(1,1) = 0.5;
    return d;
}
DenseMatrix dmMinus() {
    DenseMatrix d(2, 2);
    d(0,0) = d(1,1) = 0.5; d(0,1) = d(1,0) = -0.5;
    return d;
}

DenseMatrix ketbra(const DenseMatrix& ket) {
    return ket * ket.adjoint();
}
DenseMatrix ketbra(const DenseMatrix& ket, const DenseMatrix& bra) {
    return ket * bra.adjoint();
}

void expectMatrixNear(const DenseMatrix& a, const DenseMatrix& b,
                      double tol = 1e-10,
                      const std::string& label = "") {
    ASSERT_EQ(a.rows(), b.rows()) << label;
    ASSERT_EQ(a.cols(), b.cols()) << label;
    for (int r = 0; r < a.rows(); ++r)
        for (int c = 0; c < a.cols(); ++c)
            EXPECT_NEAR(std::abs(a(r, c)), std::abs(b(r, c)), tol)
                << label << " mismatch at (" << r << "," << c << ")";
}

const std::complex<double> kImag     {0.0,  1.0};
const std::complex<double> kImagConj {0.0, -1.0};
const double kInvSqrt2 = 1.0 / std::sqrt(2.0);
const std::complex<double> kTPhase     = std::exp(std::complex<double>(0.0,  M_PI / 4.0));
const std::complex<double> kTPhaseConj = std::conj(kTPhase);

}

TEST(MatrixFreeOperator, NameAndTargetQubitAccessors) {
    MatrixFreeOperator op("X", 3);
    EXPECT_EQ(op.get_name(), "X");
    EXPECT_EQ(op.get_target_qubit(), 3);
    EXPECT_EQ(op.get_control_qubit(), -1);
}

TEST(MatrixFreeOperator, GetIdSingleQubit) {
    MatrixFreeOperator op("Z", 2);
    EXPECT_EQ(op.get_id(), "Z_t2_c-1");
}

TEST(MatrixFreeOperator, InitWithTargetAndControl) {
    MatrixFreeOperator op("CNOT", 1, 0);
    EXPECT_EQ(op.get_name(), "CNOT");
    EXPECT_EQ(op.get_target_qubit(), 0);
    EXPECT_EQ(op.get_control_qubit(), 1);
    EXPECT_EQ(op.get_id(), "CNOT_t0_c1");
}

TEST(MatrixFreeOperator, InitWithGate) {
    DenseMatrix x_matrix(2, 2);
    x_matrix << 0, 1,
                1, 0;
    Gate g("X", x_matrix.sparseView(), {}, {1}, {});
    MatrixFreeOperator op(g);
    EXPECT_EQ(op.get_name(), "X");
    EXPECT_EQ(op.get_target_qubit(), 1);
    EXPECT_EQ(op.get_control_qubit(), -1);
    EXPECT_EQ(op.get_id(), "X_t1_c-1");
}

TEST(MatrixFreeOperator, InitWithMultiTargetGate) {
    DenseMatrix x_matrix(2, 2);
    x_matrix << 0, 1,
                1, 0;
    Gate g("X", x_matrix.sparseView(), {}, {1,2}, {});
    EXPECT_ANY_THROW(MatrixFreeOperator op(g));
}

TEST(MatrixFreeOperator, InitWithMultiControlGate) {
    DenseMatrix x_matrix(2, 2);
    x_matrix << 0, 1,
                1, 0;
    Gate g("X", x_matrix.sparseView(), {1,2}, {1}, {});
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
    EXPECT_THROW(op.apply(state, MatrixFreeApplicationType::Left),
                 std::runtime_error);
}

TEST(MatrixFreeOperator, X_StateVector_Ket0ToKet1) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket1(), 1e-10, "X|0>=|1>");
}

TEST(MatrixFreeOperator, X_StateVector_Ket1ToKet0) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket0(), 1e-10, "X|1>=|0>");
}

TEST(MatrixFreeOperator, X_StateVector_Involutory) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "X²=I");
}

TEST(MatrixFreeOperator, X_StateVector_TwoQubit_QubitZero) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket11(), 1e-10, "X_q0 |01>=|11>");
}

TEST(MatrixFreeOperator, X_StateVector_TwoQubit_QubitOne) {
    MatrixFreeOperator op("X", 1);
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket11(), 1e-10, "X_q1 |10>=|11>");
}

TEST(MatrixFreeOperator, X_Left_DensityMatrix_Dm0ToDm1Rows) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2); expected.setZero();
    expected(1, 0) = 1.0;
    expectMatrixNear(rho, expected, 1e-10, "X Left |0><0|");
}

TEST(MatrixFreeOperator, X_Right_DensityMatrix_Dm0) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2); expected.setZero();
    expected(0, 1) = 1.0;
    expectMatrixNear(rho, expected, 1e-10, "X Right |0><0|");
}

TEST(MatrixFreeOperator, X_LeftAndRight_Dm0GivesDm1) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm1(), 1e-10, "X LAR |0><0|=|1><1|");
}

TEST(MatrixFreeOperator, X_LeftAndRight_Dm1GivesDm0) {
    MatrixFreeOperator op("X", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm0(), 1e-10, "X LAR |1><1|=|0><0|");
}

TEST(MatrixFreeOperator, Y_StateVector_Ket0) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 0.0;
    expected(1, 0) = kImag;
    expectMatrixNear(s, expected, 1e-10, "Y|0>=i|1>");
}

TEST(MatrixFreeOperator, Y_StateVector_Ket1) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = kImagConj;
    expected(1, 0) = 0.0;
    expectMatrixNear(s, expected, 1e-10, "Y|1>=-i|0>");
}

TEST(MatrixFreeOperator, Y_StateVector_Involutory) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "Y²=I");
}

TEST(MatrixFreeOperator, Y_Left_DensityMatrix_Dm0) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2); expected.setZero();
    expected(1, 0) = kImag;   // i·<0|0><0|
    expectMatrixNear(rho, expected, 1e-10, "Y Left |0><0|");
}

TEST(MatrixFreeOperator, Y_Right_DensityMatrix_Dm0) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2); expected.setZero();
    expected(0, 1) = kImagConj;
    expectMatrixNear(rho, expected, 1e-10, "Y Right |0><0|");
}

TEST(MatrixFreeOperator, Y_LeftAndRight_DmPlus_GivesDmMinus) {
    MatrixFreeOperator op("Y", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dmMinus(), 1e-10, "Y LAR |+><+|=|-><-|");
}

TEST(MatrixFreeOperator, Z_StateVector_Ket0Unchanged) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket0(), 1e-10, "Z|0>=|0>");
}

TEST(MatrixFreeOperator, Z_StateVector_Ket1Negated) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ket1();
    expected *= -1.0;
    expectMatrixNear(s, expected, 1e-10, "Z|1>=-|1>");
}

TEST(MatrixFreeOperator, Z_StateVector_KetPlusGivesKetMinus) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ketPlus();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ketMinus(), 1e-10, "Z|+>=|->");
}

TEST(MatrixFreeOperator, Z_StateVector_Involutory) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "Z²=I");
}

TEST(MatrixFreeOperator, Z_Left_DensityMatrix_Dm1) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected = dm1();
    expected *= -1.0;
    expectMatrixNear(rho, expected, 1e-10, "Z Left |1><1|");
}

TEST(MatrixFreeOperator, Z_Right_DensityMatrix_Dm1) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected = dm1();
    expected *= -1.0;
    expectMatrixNear(rho, expected, 1e-10, "Z Right |1><1|");
}

TEST(MatrixFreeOperator, Z_LeftAndRight_Dm1Unchanged) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm1(), 1e-10, "Z LAR |1><1|=|1><1|");
}

TEST(MatrixFreeOperator, Z_LeftAndRight_DmPlusGivesDmMinus) {
    MatrixFreeOperator op("Z", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dmMinus(), 1e-10, "Z LAR |+><+|=|-><-|");
}

TEST(MatrixFreeOperator, H_StateVector_Ket0GivesKetPlus) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ketPlus(), 1e-10, "H|0>=|+>");
}

TEST(MatrixFreeOperator, H_StateVector_Ket1GivesKetMinus) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ketMinus(), 1e-10, "H|1>=|->");
}

TEST(MatrixFreeOperator, H_StateVector_KetPlusGivesKet0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ketPlus();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket0(), 1e-10, "H|+>=|0>");
}

TEST(MatrixFreeOperator, H_StateVector_Involutory) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix s = ket0();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "H²=I");
}

TEST(MatrixFreeOperator, H_Left_DensityMatrix_Dm0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2); expected.setZero();
    expected(0, 0) = kInvSqrt2;
    expected(1, 0) = kInvSqrt2;
    expectMatrixNear(rho, expected, 1e-10, "H Left |0><0|");
}

TEST(MatrixFreeOperator, H_Right_DensityMatrix_Dm0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2); expected.setZero();
    expected(0, 0) = kInvSqrt2;
    expected(0, 1) = kInvSqrt2;
    expectMatrixNear(rho, expected, 1e-10, "H Right |0><0|");
}

TEST(MatrixFreeOperator, H_LeftAndRight_Dm0GivesDmPlus) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dm0();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dmPlus(), 1e-10, "H LAR |0><0|=|+><+|");
}

TEST(MatrixFreeOperator, H_LeftAndRight_DmPlusGivesDm0) {
    MatrixFreeOperator op("H", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm0(), 1e-10, "H LAR |+><+|=|0><0|");
}

TEST(MatrixFreeOperator, S_StateVector_Ket0Unchanged) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket0(), 1e-10, "S|0>=|0>");
}

TEST(MatrixFreeOperator, S_StateVector_Ket1GivesIKet1) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 0.0;
    expected(1, 0) = kImag;
    expectMatrixNear(s, expected, 1e-10, "S|1>=i|1>");
}

TEST(MatrixFreeOperator, S_StateVector_FourApplicationsIsIdentity) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    for (int i = 0; i < 4; ++i)
        op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "S⁴=I");
}

TEST(MatrixFreeOperator, S_Left_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;    expected(0, 1) = 0.5;
    expected(1, 0) = 0.5 * kImag; expected(1, 1) = 0.5 * kImag;
    expectMatrixNear(rho, expected, 1e-10, "S Left |+><+|");
}

TEST(MatrixFreeOperator, S_Right_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5; expected(0, 1) = 0.5 * kImagConj;
    expected(1, 0) = 0.5; expected(1, 1) = 0.5 * kImagConj;
    expectMatrixNear(rho, expected, 1e-10, "S Right |+><+|");
}

TEST(MatrixFreeOperator, S_LeftAndRight_DmPlus_StaysTheSame) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dmPlus(), 1e-10, "S LAR |+><+| unchanged");
}

TEST(MatrixFreeOperator, S_LeftAndRight_Dm1Unchanged) {
    MatrixFreeOperator op("S", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm1(), 1e-10, "S LAR |1><1| unchanged");
}

TEST(MatrixFreeOperator, T_StateVector_Ket0Unchanged) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket0(), 1e-10, "T|0>=|0>");
}

TEST(MatrixFreeOperator, T_StateVector_Ket1GivesTPhaseKet1) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix s = ket1();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 0.0;
    expected(1, 0) = kTPhase;
    expectMatrixNear(s, expected, 1e-10, "T|1>=e^{iπ/4}|1>");
}

TEST(MatrixFreeOperator, T_StateVector_EightApplicationsIsIdentity) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix s = ketPlus();
    DenseMatrix original = s;
    for (int i = 0; i < 8; ++i)
        op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "T⁸=I");
}

TEST(MatrixFreeOperator, T_Left_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5;             expected(0, 1) = 0.5;
    expected(1, 0) = 0.5 * kTPhase;   expected(1, 1) = 0.5 * kTPhase;
    expectMatrixNear(rho, expected, 1e-10, "T Left |+><+|");
}

TEST(MatrixFreeOperator, T_Right_DensityMatrix_DmPlus) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected(2, 2);
    expected(0, 0) = 0.5; expected(0, 1) = 0.5 * kTPhaseConj;
    expected(1, 0) = 0.5; expected(1, 1) = 0.5 * kTPhaseConj;
    expectMatrixNear(rho, expected, 1e-10, "T Right |+><+|");
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
    expectMatrixNear(rho, expected, 1e-10, "T LAR |+><+|");
}

TEST(MatrixFreeOperator, T_LeftAndRight_Dm1Unchanged) {
    MatrixFreeOperator op("T", 0);
    DenseMatrix rho = dm1();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm1(), 1e-10, "T LAR |1><1| unchanged");
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control0Target1_Ket10FlipsToKet11) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket11(), 1e-10, "CNOT(0,1) |10>=|11>");
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control0Target1_Ket11FlipsToKet10) {
    MatrixFreeOperator op("CX", 0, 1);
    DenseMatrix s = ket11();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket10(), 1e-10, "CNOT(0,1) |11>=|10>");
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control0Target1_Ket00Unchanged) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix s = ket00();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket00(), 1e-10, "CNOT(0,1) |00>=|00>");
}

TEST(MatrixFreeOperator, CNOT_StateVector_Control1Target0_Ket01FlipsToKet11) {
    MatrixFreeOperator op("CNOT", 1, 0);
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket11(), 1e-10, "CNOT(1,0) |01>=|11>");
}

TEST(MatrixFreeOperator, CNOT_StateVector_Involutory) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix s = ket10();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "CNOT²=I");
}

TEST(MatrixFreeOperator, CNOT_Left_DensityMatrix_Ket10) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::Left);
    expectMatrixNear(rho, ketbra(ket11(), ket10()), 1e-10, "CNOT Left |10><10|→|11><10|");
}

TEST(MatrixFreeOperator, CNOT_Right_DensityMatrix_Ket10) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::Right);
    expectMatrixNear(rho, ketbra(ket10(), ket11()), 1e-10, "CNOT Right |10><10|→|10><11|");
}

TEST(MatrixFreeOperator, CNOT_LeftAndRight_Ket10GivesKet11) {
    MatrixFreeOperator op("CNOT", 0, 1);
    DenseMatrix rho = ketbra(ket10());
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, ketbra(ket11()), 1e-10, "CNOT LAR |10><10|=|11><11|");
}

TEST(MatrixFreeOperator, CZ_StateVector_Ket11GetsNegated) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket11();
    op.apply(s, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ket11();
    expected *= -1.0;
    expectMatrixNear(s, expected, 1e-10, "CZ |11>=-|11>");
}

TEST(MatrixFreeOperator, CZ_StateVector_Ket10Unchanged) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket10();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket10(), 1e-10, "CZ |10>=|10>");
}

TEST(MatrixFreeOperator, CZ_StateVector_Ket01Unchanged) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket01();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ket01(), 1e-10, "CZ |01>=|01>");
}

TEST(MatrixFreeOperator, CZ_StateVector_Involutory) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix s = ket11();
    DenseMatrix original = s;
    op.apply(s, MatrixFreeApplicationType::Left);
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, original, 1e-10, "CZ²=I");
}

TEST(MatrixFreeOperator, CZ_Left_DensityMatrix_Ket11) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix rho = ketbra(ket11());
    op.apply(rho, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ketbra(ket11());
    expected *= -1.0;
    expectMatrixNear(rho, expected, 1e-10, "CZ Left |11><11|");
}

TEST(MatrixFreeOperator, CZ_Right_DensityMatrix_Ket11) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix rho = ketbra(ket11());
    op.apply(rho, MatrixFreeApplicationType::Right);
    DenseMatrix expected = ketbra(ket11());
    expected *= -1.0;
    expectMatrixNear(rho, expected, 1e-10, "CZ Right |11><11|");
}

TEST(MatrixFreeOperator, CZ_LeftAndRight_Ket11Unchanged) {
    MatrixFreeOperator op("CZ", 0, 1);
    DenseMatrix rho = ketbra(ket11());
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, ketbra(ket11()), 1e-10, "CZ LAR |11><11| unchanged");
}

TEST(MatrixFreeOperator, CustomBaseMatrix_StateVector_ActsCorrectly) {
    DenseMatrix hmat(2, 2);
    hmat(0,0) =  kInvSqrt2; hmat(0,1) =  kInvSqrt2;
    hmat(1,0) =  kInvSqrt2; hmat(1,1) = -kInvSqrt2;
    Gate g("CustomH", hmat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix s = ket0();
    op.apply(s, MatrixFreeApplicationType::Left);
    expectMatrixNear(s, ketPlus(), 1e-10, "custom H|0>=|+>");
}

TEST(MatrixFreeOperator, CustomBaseMatrixLeft_DensityMatrix_ActsCorrectly) {
    DenseMatrix hmat(2, 2);
    hmat(0,0) =  0; hmat(0,1) = 1;
    hmat(1,0) =  1; hmat(1,1) = 0;
    Gate g("CustomX", hmat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = ketbra(ket0());
    op.apply(rho, MatrixFreeApplicationType::Left);
    expectMatrixNear(rho, ketbra(ket1(), ket0()), 1e-10, "custom X Left |0><0|→|1><0|");
}

TEST(MatrixFreeOperator, CustomBaseMatrixRight_DensityMatrix_ActsCorrectly) {
    DenseMatrix hmat(2, 2);
    hmat(0,0) =  0; hmat(0,1) = 1;
    hmat(1,0) =  1; hmat(1,1) = 0;
    Gate g("CustomX", hmat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = ketbra(ket0());
    op.apply(rho, MatrixFreeApplicationType::Right);
    expectMatrixNear(rho, ketbra(ket0(), ket1()), 1e-10, "custom X Right |0><0|→|0><1|");
}

TEST(MatrixFreeOperator, CustomBaseMatrix_LeftAndRight_UsesConjugateOnRight) {
    DenseMatrix umat(2, 2); 
    umat.setZero();
    umat(0, 0) = 1.0; umat(1, 1) = kImag;
    Gate g("CustomS", umat.sparseView(), {}, {0}, {});
    MatrixFreeOperator op(g);
    DenseMatrix rho = dmPlus();
    op.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dmPlus(), 1e-10, "custom S LAR |+><+|");
}