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

#include "../../../src/qilisdk_cpp/backends/qilisim/utils/matrix_utils.h"

namespace {

constexpr double kTol = 1e-10;

SparseMatrix pauli_x() {
    DenseMatrix X(2, 2);
    X << 0, 1, 1, 0;
    return X.sparseView();
}
SparseMatrix pauli_y() {
    DenseMatrix Y(2, 2);
    Y << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0;
    return Y.sparseView();
}
SparseMatrix pauli_z() {
    DenseMatrix Z(2, 2);
    Z << 1, 0, 0, -1;
    return Z.sparseView();
}

SparseMatrix zero_H_2x2() {
    return SparseMatrix(2, 2);
}

SparseMatrix diag_H() {
    DenseMatrix H(2, 2);
    H << 0, 0, 0, 1;
    return H.sparseView();
}

SparseMatrix mat2x2() {
    DenseMatrix m(2, 2);
    m << std::complex<double>(1, 0), std::complex<double>(2, 0), std::complex<double>(3, 0), std::complex<double>(4, 0);
    return m.sparseView();
}

}

TEST(MatrixUtilsTest, DotProductKet) {
    SparseMatrix v1(3, 1);
    v1.insert(0, 0) = std::complex<double>(1.0, 2.0);
    v1.insert(1, 0) = std::complex<double>(3.0, 4.0);
    v1.insert(2, 0) = std::complex<double>(5.0, 6.0);
    SparseMatrix v2(3, 1);
    v2.insert(0, 0) = std::complex<double>(7.0, 8.0);
    v2.insert(1, 0) = std::complex<double>(9.0, 10.0);
    v2.insert(2, 0) = std::complex<double>(11.0, 12.0);
    v1.makeCompressed();
    v2.makeCompressed();
    std::complex<double> result = dot(v1, v2);
    std::complex<double> expected = 0.0;
    for (int i = 0; i < 3; ++i) {
        expected += std::conj(v1.coeff(i, 0)) * v2.coeff(i, 0);
    }
    EXPECT_NEAR(result.real(), expected.real(), 1e-6);
    EXPECT_NEAR(result.imag(), expected.imag(), 1e-6);
}

TEST(MatrixUtilsTest, DotProductBra) {
    SparseMatrix v1(1, 3);
    v1.insert(0, 0) = std::complex<double>(1.0, 2.0);
    v1.insert(0, 1) = std::complex<double>(3.0, 4.0);
    v1.insert(0, 2) = std::complex<double>(5.0, 6.0);
    SparseMatrix v2(1, 3);
    v2.insert(0, 0) = std::complex<double>(7.0, 8.0);
    v2.insert(0, 1) = std::complex<double>(9.0, 10.0);
    v2.insert(0, 2) = std::complex<double>(11.0, 12.0);
    v1.makeCompressed();
    v2.makeCompressed();
    std::complex<double> result = dot(v1, v2);
    std::complex<double> expected = 0.0;
    for (int i = 0; i < 3; ++i) {
        expected += std::conj(v1.coeff(0, i)) * v2.coeff(0, i);
    }
    EXPECT_NEAR(result.real(), expected.real(), 1e-6);
    EXPECT_NEAR(result.imag(), expected.imag(), 1e-6);
}

TEST(ExpMatActionSparseTest, ZeroHamiltonianIdentityAction) {
    SparseMatrix H = zero_H_2x2();
    DenseMatrix d = DenseMatrix::Identity(2, 2);
    SparseMatrix e1 = d.sparseView();
    std::complex<double> dt(0.0, 0.0);
    SparseMatrix result = exp_mat_action(H, dt, e1);
    DenseMatrix diff = DenseMatrix(result) - d;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatActionSparseTest, RealDtDiagonalHamiltonian) {
    SparseMatrix H = diag_H();
    SparseMatrix e1 = DenseMatrix::Identity(2, 2).sparseView();
    std::complex<double> dt(1.0, 0.0);
    SparseMatrix result = exp_mat_action(H, dt, e1);
    DenseMatrix dense = DenseMatrix(result);
    EXPECT_NEAR(std::abs(dense(0, 0) - 1.0), 0.0, kTol);
    EXPECT_NEAR(std::abs(dense(1, 0)), 0.0, kTol);
    EXPECT_NEAR(std::abs(dense(0, 1)), 0.0, kTol);
    EXPECT_NEAR(std::abs(dense(1, 1) - std::exp(1.0)), 0.0, kTol);
}

TEST(ExpMatActionSparseTest, ImaginaryDtGivesUnitaryAction) {
    SparseMatrix H = pauli_z();
    SparseMatrix e1 = DenseMatrix::Identity(2, 2).sparseView();
    std::complex<double> dt(0.0, -1.0);
    SparseMatrix result = exp_mat_action(H, dt, e1);
    DenseMatrix dense = DenseMatrix(result);
    DenseMatrix product = dense.adjoint() * dense;
    DenseMatrix diff = product - DenseMatrix::Identity(2, 2);
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatActionSparseTest, OutputDimensionsMatchInput) {
    SparseMatrix H = DenseMatrix::Identity(3, 3).sparseView();
    SparseMatrix e1 = DenseMatrix::Identity(3, 3).sparseView();
    std::complex<double> dt(0.5, 0.0);
    SparseMatrix result = exp_mat_action(H, dt, e1);
    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), 3);
}

TEST(ExpMatActionDenseTest, ZeroHamiltonianIdentityAction) {
    SparseMatrix H = SparseMatrix(2, 2);
    DenseMatrix e1 = DenseMatrix::Identity(2, 2);
    std::complex<double> dt(0.0, 0.0);
    DenseMatrix result = exp_mat_action(H, dt, e1);
    DenseMatrix diff = result - e1;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatActionDenseTest, ImaginaryDtGivesUnitaryAction) {
    SparseMatrix H = pauli_x();
    DenseMatrix e1 = DenseMatrix::Identity(2, 2);
    std::complex<double> dt(0.0, -0.5);
    DenseMatrix result = exp_mat_action(H, dt, e1);
    DenseMatrix product = result.adjoint() * result;
    DenseMatrix diff = product - DenseMatrix::Identity(2, 2);
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatActionDenseTest, ConsistencyWithSparseVersion) {
    SparseMatrix H = pauli_z();
    DenseMatrix e1d = DenseMatrix::Identity(2, 2);
    SparseMatrix e1s = e1d.sparseView();
    std::complex<double> dt(0.3, -0.2);
    DenseMatrix dense_result = exp_mat_action(H, dt, e1d);
    SparseMatrix sparse_result = exp_mat_action(H, dt, e1s);
    DenseMatrix diff = dense_result - DenseMatrix(sparse_result);
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatActionDenseTest, OutputDimensionsMatchInput) {
    SparseMatrix H = DenseMatrix::Identity(4, 4).sparseView();
    DenseMatrix e1 = DenseMatrix::Random(4, 2);
    std::complex<double> dt(1.0, 0.0);
    DenseMatrix result = exp_mat_action(H, dt, e1);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 2);
}

TEST(ExpMatTest, ZeroHamiltonianGivesIdentity) {
    SparseMatrix H = SparseMatrix(2, 2);
    std::complex<double> dt(1.0, 0.0);
    SparseMatrix result = exp_mat(H, dt);
    DenseMatrix diff = DenseMatrix(result) - DenseMatrix::Identity(2, 2);
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatTest, DiagonalHamiltonian) {
    DenseMatrix d = DenseMatrix::Zero(2, 2);
    d(0, 0) = std::complex<double>(1.0, 0.0);
    d(1, 1) = std::complex<double>(2.0, 0.0);
    SparseMatrix H = d.sparseView();
    std::complex<double> dt(1.0, 0.0);
    SparseMatrix result = exp_mat(H, dt);
    DenseMatrix dense = DenseMatrix(result);
    EXPECT_NEAR(std::abs(dense(0, 0) - std::exp(1.0)), 0.0, kTol);
    EXPECT_NEAR(std::abs(dense(1, 1) - std::exp(2.0)), 0.0, kTol);
    EXPECT_NEAR(std::abs(dense(0, 1)), 0.0, kTol);
    EXPECT_NEAR(std::abs(dense(1, 0)), 0.0, kTol);
}

TEST(ExpMatTest, ImaginaryDtGivesUnitaryForHermitianH) {
    SparseMatrix H = pauli_x();
    std::complex<double> dt(0.0, -0.7);
    SparseMatrix result = exp_mat(H, dt);
    DenseMatrix dense = DenseMatrix(result);
    DenseMatrix product = dense.adjoint() * dense;
    DenseMatrix diff = product - DenseMatrix::Identity(2, 2);
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatTest, ConsistencyWithExpMatAction) {
    SparseMatrix H = pauli_y();
    SparseMatrix e1 = DenseMatrix::Identity(2, 2).sparseView();
    std::complex<double> dt(0.4, -0.1);
    SparseMatrix exp_H = exp_mat(H, dt);
    DenseMatrix via_mat = DenseMatrix(exp_H) * DenseMatrix(e1);
    DenseMatrix via_act = DenseMatrix(exp_mat_action(H, dt, e1));
    DenseMatrix diff = via_mat - via_act;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpMatTest, OutputDimensionsMatchInput) {
    SparseMatrix H = DenseMatrix::Identity(3, 3).sparseView();
    std::complex<double> dt(1.0, 0.0);
    SparseMatrix result = exp_mat(H, dt);
    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), 3);
}

TEST(DotTest, SparseDotSelfIsNormSquared) {
    DenseMatrix d(3, 1);
    d << std::complex<double>(1, 0), std::complex<double>(0, 1), std::complex<double>(1, 1);
    SparseMatrix v = d.sparseView();
    std::complex<double> result = dot(v, v);
    EXPECT_NEAR(result.imag(), 0.0, kTol);
    EXPECT_NEAR(result.real(), 4.0, kTol);
}

TEST(DotTest, SparseOrthogonalVectorsGiveZero) {
    DenseMatrix a(2, 1), b(2, 1);
    a << 1, 0;
    b << 0, 1;
    SparseMatrix sa = a.sparseView();
    SparseMatrix sb = b.sparseView();
    EXPECT_NEAR(std::abs(dot(sa, sb)), 0.0, kTol);
}

TEST(DotTest, SparseConjugateSymmetry) {
    DenseMatrix a(2, 1), b(2, 1);
    a << std::complex<double>(1, 2), std::complex<double>(3, 4);
    b << std::complex<double>(5, 6), std::complex<double>(7, 8);
    SparseMatrix sa = a.sparseView();
    SparseMatrix sb = b.sparseView();
    std::complex<double> ab = dot(sa, sb);
    std::complex<double> ba = dot(sb, sa);
    EXPECT_NEAR(std::abs(ab - std::conj(ba)), 0.0, kTol);
}

TEST(DotTest, DenseDotSelfIsNormSquared) {
    DenseMatrix v(3, 1);
    v << std::complex<double>(1, 0), std::complex<double>(0, 2), std::complex<double>(3, 0);
    std::complex<double> result = dot(v, v);
    EXPECT_NEAR(result.imag(), 0.0, kTol);
    EXPECT_NEAR(result.real(), 1 + 4 + 9, kTol);
}

TEST(DotTest, DenseOrthogonalVectorsGiveZero) {
    DenseMatrix a(2, 1), b(2, 1);
    a << 1, 0;
    b << 0, 1;
    EXPECT_NEAR(std::abs(dot(a, b)), 0.0, kTol);
}

TEST(DotTest, DenseConjugateSymmetry) {
    DenseMatrix a(2, 1), b(2, 1);
    a << std::complex<double>(1, 2), std::complex<double>(3, 4);
    b << std::complex<double>(5, 6), std::complex<double>(7, 8);
    std::complex<double> ab = dot(a, b);
    std::complex<double> ba = dot(b, a);
    EXPECT_NEAR(std::abs(ab - std::conj(ba)), 0.0, kTol);
}

TEST(DotTest, SparseDenseConsistency) {
    DenseMatrix d(3, 1);
    d << std::complex<double>(1, 2), std::complex<double>(3, 4), std::complex<double>(5, 6);
    SparseMatrix s = d.sparseView();
    std::complex<double> sparse_result = dot(s, s);
    std::complex<double> dense_result = dot(d, d);
    EXPECT_NEAR(std::abs(sparse_result - dense_result), 0.0, kTol);
}

TEST(TraceTest, DenseIdentityTrace) {
    DenseMatrix I = DenseMatrix::Identity(4, 4);
    std::complex<double> t = trace(I);
    EXPECT_NEAR(t.real(), 4.0, kTol);
    EXPECT_NEAR(t.imag(), 0.0, kTol);
}

TEST(TraceTest, TraceOfVectorRaises) {
    DenseMatrix v(3, 1);
    v << std::complex<double>(1, 0), std::complex<double>(2, 0), std::complex<double>(3, 0);
    EXPECT_ANY_THROW(trace(v));
}

TEST(TraceTest, TraceOfSparseVectorRaises) {
    DenseMatrix v(3, 1);
    v << std::complex<double>(1, 0), std::complex<double>(2, 0), std::complex<double>(3, 0);
    SparseMatrix sv = v.sparseView();
    EXPECT_ANY_THROW(trace(sv));
}

TEST(TraceTest, DenseZeroMatrixTrace) {
    DenseMatrix Z = DenseMatrix::Zero(3, 3);
    std::complex<double> t = trace(Z);
    EXPECT_NEAR(std::abs(t), 0.0, kTol);
}

TEST(TraceTest, DenseComplexDiagonal) {
    DenseMatrix M = DenseMatrix::Zero(2, 2);
    M(0, 0) = std::complex<double>(1, 2);
    M(1, 1) = std::complex<double>(3, 4);
    std::complex<double> t = trace(M);
    EXPECT_NEAR(t.real(), 4.0, kTol);
    EXPECT_NEAR(t.imag(), 6.0, kTol);
}

TEST(TraceTest, SparseIdentityTrace) {
    SparseMatrix I = DenseMatrix::Identity(4, 4).sparseView();
    std::complex<double> t = trace(I);
    EXPECT_NEAR(t.real(), 4.0, kTol);
    EXPECT_NEAR(t.imag(), 0.0, kTol);
}

TEST(TraceTest, SparseZeroMatrixTrace) {
    SparseMatrix Z(3, 3);
    std::complex<double> t = trace(Z);
    EXPECT_NEAR(std::abs(t), 0.0, kTol);
}

TEST(TraceTest, SparseDenseConsistency) {
    DenseMatrix d = DenseMatrix::Random(4, 4);
    SparseMatrix s = d.sparseView();
    std::complex<double> dense_t = trace(d);
    std::complex<double> sparse_t = trace(s);
    EXPECT_NEAR(std::abs(dense_t - sparse_t), 0.0, kTol);
}

TEST(VectorizeTest, DenseVectorizeOutputShape) {
    DenseMatrix m = mat2x2();
    DenseMatrix vec = vectorize(m);
    EXPECT_EQ(vec.rows(), 4);
    EXPECT_EQ(vec.cols(), 1);
}

TEST(VectorizeTest, DenseVectorizeColumnMajorOrder) {
    DenseMatrix m = mat2x2();
    DenseMatrix vec = vectorize(m);
    EXPECT_NEAR(std::abs(vec(0, 0) - std::complex<double>(1, 0)), 0.0, kTol);
    EXPECT_NEAR(std::abs(vec(1, 0) - std::complex<double>(3, 0)), 0.0, kTol);
    EXPECT_NEAR(std::abs(vec(2, 0) - std::complex<double>(2, 0)), 0.0, kTol);
    EXPECT_NEAR(std::abs(vec(3, 0) - std::complex<double>(4, 0)), 0.0, kTol);
}

TEST(VectorizeTest, DenseRoundtrip) {
    DenseMatrix m = mat2x2();
    DenseMatrix round = devectorize(vectorize(m));
    DenseMatrix diff = round - m;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(VectorizeTest, SparseVectorizeOutputShape) {
    SparseMatrix s = mat2x2();
    SparseMatrix vec = vectorize(s, 1e-12);
    EXPECT_EQ(vec.rows(), 4);
    EXPECT_EQ(vec.cols(), 1);
}

TEST(VectorizeTest, SparseRoundtrip) {
    SparseMatrix s = mat2x2();
    SparseMatrix vec = vectorize(s, 1e-12);
    SparseMatrix restored = devectorize(vec, 1e-12);
    DenseMatrix diff = DenseMatrix(restored) - mat2x2();
    EXPECT_LT(diff.norm(), kTol);
}

TEST(VectorizeTest, SparseVectorizePreservesNonzeroCount) {
    SparseMatrix s = mat2x2();
    SparseMatrix vec = vectorize(s, 1e-12);
    EXPECT_EQ(vec.nonZeros(), 4);
}

TEST(VectorizeTest, SparseDenseVectorizeConsistency) {
    DenseMatrix d = mat2x2();
    SparseMatrix s = d.sparseView();
    DenseMatrix vd = vectorize(d);
    DenseMatrix vs = DenseMatrix(vectorize(s, 1e-12));
    DenseMatrix diff = vd - vs;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(VectorizeTest, SparseAtolFiltersSmallEntries) {
    DenseMatrix d = mat2x2();
    d(0, 0) = std::complex<double>(1e-15, 0);
    SparseMatrix s = d.sparseView(0.0);
    SparseMatrix vec = vectorize(s, 1e-12);
    DenseMatrix dv = DenseMatrix(vec);
    EXPECT_NEAR(std::abs(dv(0, 0)), 0.0, kTol);
}

TEST(ExpandOperatorTest, SingleQubitOnFirstQubits) {
    SparseMatrix result = expand_operator(0, 1, pauli_x());
    SparseMatrix diff = result - pauli_x();
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpandOperatorTest, SingleQubitExpandedToTwoQubits) {
    SparseMatrix result = expand_operator(0, 2, pauli_x());
    DenseMatrix XI = Eigen::kroneckerProduct(pauli_x(), DenseMatrix::Identity(2, 2)).eval();
    DenseMatrix diff = DenseMatrix(result) - XI;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpandOperatorTest, OperatorHasExactRightNumberOfQubits) {
    SparseMatrix op = Eigen::KroneckerProduct(DenseMatrix(pauli_x()), DenseMatrix(pauli_x())).eval().sparseView();
    SparseMatrix result = expand_operator({0, 1}, 2, op);
    SparseMatrix diff = result - op;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpandOperatorTest, OperatorHasWrongNumberOfQubits) {
    SparseMatrix op = Eigen::KroneckerProduct(pauli_x(), pauli_x()).eval().sparseView();
    EXPECT_ANY_THROW(expand_operator(3, op));
}

TEST(ExpandOperatorTest, SingleQubitOnSecondOfTwoQubits) {
    SparseMatrix result = expand_operator(1, 2, pauli_x());
    DenseMatrix IX = Eigen::kroneckerProduct(DenseMatrix::Identity(2, 2), pauli_x()).eval();
    DenseMatrix diff = DenseMatrix(result) - IX;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpandOperatorTest, OutputDimensionIsTwoToNqubits) {
    int nqubits = 3;
    SparseMatrix result = expand_operator(0, nqubits, pauli_x());
    int dim = 1 << nqubits;
    EXPECT_EQ(result.rows(), dim);
    EXPECT_EQ(result.cols(), dim);
}

TEST(ExpandOperatorTest, GlobalExpandOneQubit) {
    SparseMatrix result = expand_operator(1, pauli_x());
    DenseMatrix diff = DenseMatrix(result) - pauli_x();
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpandOperatorTest, GlobalExpandTwoQubit) {
    SparseMatrix result = expand_operator(2, pauli_x());
    DenseMatrix XX = Eigen::kroneckerProduct(pauli_x(), pauli_x()).eval();
    DenseMatrix diff = DenseMatrix(result) - XX;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(ExpandOperatorTest, MultiTargetOutputDimensionCorrect) {
    SparseMatrix result = expand_operator({0, 1}, 3, pauli_x());
    EXPECT_EQ(result.rows(), 8);
    EXPECT_EQ(result.cols(), 8);
}

TEST(ExpandOperatorTest, ExpandedOperatorIsUnitary) {
    SparseMatrix result = expand_operator(1, 3, pauli_x());
    DenseMatrix dense = DenseMatrix(result);
    DenseMatrix product = dense.adjoint() * dense;
    DenseMatrix diff = product - DenseMatrix::Identity(8, 8);
    EXPECT_LT(diff.norm(), kTol);
}

TEST(NormalizeStateTest, StatevectorNormBecomesOne) {
    DenseMatrix state(4, 1);
    state << std::complex<double>(1, 0), std::complex<double>(1, 0), std::complex<double>(1, 0), std::complex<double>(1, 0);
    normalize_state(state, true, false);
    double norm = state.norm();
    EXPECT_NEAR(norm, 1.0, kTol);
}

TEST(NormalizeStateTest, AlreadyNormalizedStatevectorUnchanged) {
    DenseMatrix state(2, 1);
    state << std::complex<double>(1.0 / std::sqrt(2), 0), std::complex<double>(1.0 / std::sqrt(2), 0);
    DenseMatrix before = state;
    normalize_state(state, true, false);
    DenseMatrix diff = state - before;
    EXPECT_LT(diff.norm(), kTol);
}

TEST(NormalizeStateTest, DensityMatrixTraceBecomesOne) {
    DenseMatrix rho(2, 2);
    rho << std::complex<double>(2, 0), std::complex<double>(0, 0), std::complex<double>(0, 0), std::complex<double>(2, 0);
    normalize_state(rho, false, false);
    std::complex<double> tr = trace(rho);
    EXPECT_NEAR(tr.real(), 1.0, kTol);
    EXPECT_NEAR(tr.imag(), 0.0, kTol);
}

TEST(NormalizeStateTest, MonteCarloFlagDoesNotCorruptNorm) {
    DenseMatrix state(2, 1);
    state << std::complex<double>(3, 0), std::complex<double>(4, 0);
    normalize_state(state, true, true);
    double norm = state.norm();
    EXPECT_TRUE(std::isfinite(norm));
}

TEST(NormalizeStateTest, StatevectorPhasePreserved) {
    DenseMatrix state(2, 1);
    state << std::complex<double>(0, 3), std::complex<double>(0, 4);
    normalize_state(state, true, false);
    EXPECT_NEAR(state.norm(), 1.0, kTol);
}

TEST(NormalizeStateTest, MultiColumnDensityMatrix) {
    DenseMatrix rho = 2.0 * DenseMatrix::Identity(3, 3);
    normalize_state(rho, false, false);
    std::complex<double> tr = trace(rho);
    EXPECT_NEAR(tr.real(), 1.0, kTol);
    EXPECT_NEAR(tr.imag(), 0.0, kTol);
}

// GCOV_EXCL_BR_STOP