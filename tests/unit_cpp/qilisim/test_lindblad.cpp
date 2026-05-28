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
#include "../../../src/qilisdk_cpp/backends/qilisim/analog/lindblad.h"

namespace {

constexpr double kTol = 1e-10;

SparseMatrix to_sparse(const DenseMatrix& M) {
    SparseMatrix S(M.rows(), M.cols());
    S = M.sparseView();
    return S;
}

DenseMatrix pauli_x() {
    DenseMatrix X(2, 2);
    X << 0, 1, 1, 0;
    return X;
}

DenseMatrix pauli_z() {
    DenseMatrix Z(2, 2);
    Z << 1, 0, 0, -1;
    return Z;
}

DenseMatrix maximally_mixed() {
    return DenseMatrix::Identity(2, 2) * 0.5;
}
DenseMatrix pure_zero() {
    DenseMatrix r = DenseMatrix::Zero(2, 2);
    r(0, 0) = 1;
    return r;
}
DenseMatrix pure_plus() {
    DenseMatrix r(2, 2);
    r << 0.5, 0.5, 0.5, 0.5;
    return r;
}

MatrixFreeHamiltonian make_matrix_free_H(std::complex<double> coeff, int qubit, const std::string& pauli) {
    MatrixFreeOperator op(pauli, {}, {qubit}, DenseMatrix());
    return MatrixFreeHamiltonian(1, op, coeff);
}

SparseMatrix amp_damp_jump() {
    DenseMatrix j = DenseMatrix::Zero(2, 2);
    j(0, 1) = 1.0;
    return to_sparse(j);
}

}  // namespace

class CreateSuperoperatorTest : public ::testing::Test {};

TEST_F(CreateSuperoperatorTest, OutputDimension) {
    SparseMatrix H = to_sparse(pauli_z());
    SparseMatrix L = create_superoperator(H, {});
    EXPECT_EQ(L.rows(), 4);
    EXPECT_EQ(L.cols(), 4);
}

TEST_F(CreateSuperoperatorTest, NoJumpsAntiHermitian) {
    SparseMatrix H = to_sparse(pauli_z());
    DenseMatrix Ld = DenseMatrix(create_superoperator(H, {}));
    EXPECT_TRUE((Ld + Ld.adjoint()).isZero(kTol)) << "Superoperator with no jumps should be anti-Hermitian.";
}

TEST_F(CreateSuperoperatorTest, TracePreservingNoJumps) {
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    DenseMatrix L = DenseMatrix(create_superoperator(H, {}));
    Eigen::VectorXcd vecI(4);
    vecI << 1, 0, 0, 1;
    EXPECT_TRUE((L.adjoint() * vecI).isZero(kTol)) << "L†·vec(I) should be zero (trace preservation, no jumps).";
}

TEST_F(CreateSuperoperatorTest, TracePreservingWithJump) {
    SparseMatrix H = to_sparse(pauli_z());
    DenseMatrix L = DenseMatrix(create_superoperator(H, {amp_damp_jump()}));
    Eigen::VectorXcd vecI(4);
    vecI << 1, 0, 0, 1;
    EXPECT_TRUE((L.adjoint() * vecI).isZero(kTol)) << "L†·vec(I) should be zero (trace preservation with jump).";
}

TEST_F(CreateSuperoperatorTest, NoJumpsReproducesCommutator) {
    DenseMatrix H_dense = 0.5 * pauli_z();
    DenseMatrix L = DenseMatrix(create_superoperator(to_sparse(H_dense), {}));
    DenseMatrix rho = pure_plus();
    Eigen::VectorXcd rho_vec = Eigen::Map<Eigen::VectorXcd>(rho.data(), 4);
    DenseMatrix comm = H_dense * rho - rho * H_dense;
    DenseMatrix expected_mat = std::complex<double>(0, -1) * comm;
    Eigen::VectorXcd expected_vec = Eigen::Map<Eigen::VectorXcd>(expected_mat.data(), 4);
    EXPECT_TRUE((L * rho_vec).isApprox(expected_vec, kTol)) << "Superoperator (no jumps) should reproduce -i[H,rho].";
}

TEST_F(CreateSuperoperatorTest, WithJumpReproducesLindbladAction) {
    DenseMatrix H_dense = 0.5 * pauli_z();
    SparseMatrix jump = amp_damp_jump();
    DenseMatrix L = DenseMatrix(create_superoperator(to_sparse(H_dense), {jump}));
    DenseMatrix rho = pure_plus();
    Eigen::VectorXcd rho_vec = Eigen::Map<Eigen::VectorXcd>(rho.data(), 4);
    DenseMatrix J = DenseMatrix(jump);
    DenseMatrix Jdag = J.adjoint();
    DenseMatrix JdagJ = Jdag * J;
    DenseMatrix expected_mat = std::complex<double>(0, -1) * (H_dense * rho - rho * H_dense) + J * rho * Jdag - 0.5 * (JdagJ * rho + rho * JdagJ);
    Eigen::VectorXcd expected_vec = Eigen::Map<Eigen::VectorXcd>(expected_mat.data(), 4);
    EXPECT_TRUE((L * rho_vec).isApprox(expected_vec, kTol)) << "Superoperator (with jump) should reproduce full Lindblad action.";
}

TEST_F(CreateSuperoperatorTest, ZeroHamiltonianNoJumpsGivesZero) {
    SparseMatrix H = to_sparse(DenseMatrix::Zero(2, 2));
    DenseMatrix L = DenseMatrix(create_superoperator(H, {}));
    EXPECT_TRUE(L.isZero(kTol));
}

class LindbladRhsSparseTest : public ::testing::Test {
   protected:
    SparseMatrix H_z = to_sparse(0.5 * pauli_z());
    SparseMatrix jump = amp_damp_jump();
};

TEST_F(LindbladRhsSparseTest, UnitaryBranchStatevector) {
    DenseMatrix psi(2, 1);
    psi << std::complex<double>(1.0 / std::sqrt(2.0), 0), std::complex<double>(0, 1.0 / std::sqrt(2.0));
    DenseMatrix drho(2, 1);
    lindblad_rhs(drho, psi, H_z, {}, true);
    DenseMatrix expected = std::complex<double>(0, -1) * DenseMatrix(H_z) * psi;
    EXPECT_TRUE(drho.isApprox(expected, kTol));
}

TEST_F(LindbladRhsSparseTest, UnitaryBranchLinearity) {
    DenseMatrix psi1(2, 1), psi2(2, 1);
    psi1 << 1, 0;
    psi2 << 0, 1;
    DenseMatrix d1(2, 1), d2(2, 1), dsum(2, 1);
    lindblad_rhs(d1, psi1, H_z, {}, true);
    lindblad_rhs(d2, psi2, H_z, {}, true);
    lindblad_rhs(dsum, psi1 + psi2, H_z, {}, true);
    EXPECT_TRUE(dsum.isApprox(d1 + d2, kTol));
}

TEST_F(LindbladRhsSparseTest, DensityMatrixNoJumpsCommutator) {
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_z, {}, false);
    DenseMatrix H_d = DenseMatrix(H_z);
    DenseMatrix expected = std::complex<double>(0, -1) * (H_d * rho - rho * H_d);
    EXPECT_TRUE(drho.isApprox(expected, kTol));
}

TEST_F(LindbladRhsSparseTest, DensityMatrixOutputIsHermitian) {
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_z, {jump}, false);
    EXPECT_TRUE((drho - drho.adjoint()).isZero(kTol)) << "drho must be Hermitian.";
}

TEST_F(LindbladRhsSparseTest, DensityMatrixTracePreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_z, {jump}, false);
    EXPECT_NEAR(std::abs(drho.trace()), 0.0, kTol);
}

TEST_F(LindbladRhsSparseTest, DensityMatrixWithJumpMatchesManual) {
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_z, {jump}, false);
    DenseMatrix H_d = DenseMatrix(H_z);
    DenseMatrix J = DenseMatrix(jump);
    DenseMatrix Jdag = J.adjoint();
    DenseMatrix JdagJ = Jdag * J;
    DenseMatrix expected = std::complex<double>(0, -1) * (H_d * rho - rho * H_d) + J * rho * Jdag - 0.5 * (JdagJ * rho + rho * JdagJ);
    EXPECT_TRUE(drho.isApprox(expected, kTol));
}

TEST_F(LindbladRhsSparseTest, MaximallyMixedStateOnlyDissipator) {
    DenseMatrix rho = maximally_mixed();
    DenseMatrix drho_no_jump(2, 2), drho_jump(2, 2);
    lindblad_rhs(drho_no_jump, rho, H_z, {}, false);
    lindblad_rhs(drho_jump, rho, H_z, {jump}, false);
    EXPECT_TRUE(drho_no_jump.isZero(kTol));
    DenseMatrix J = DenseMatrix(jump);
    DenseMatrix Jdag = J.adjoint();
    DenseMatrix JdagJ = Jdag * J;
    DenseMatrix expected = J * rho * Jdag - 0.5 * (JdagJ * rho + rho * JdagJ);
    EXPECT_TRUE(drho_jump.isApprox(expected, kTol));
}

TEST_F(LindbladRhsSparseTest, ZeroHamiltonianNoJumpsGivesZero) {
    SparseMatrix H0 = to_sparse(DenseMatrix::Zero(2, 2));
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H0, {}, false);
    EXPECT_TRUE(drho.isZero(kTol));
}

TEST_F(LindbladRhsSparseTest, MultipleJumpsAreAdditive) {
    DenseMatrix rho = pure_plus();
    SparseMatrix jump2 = to_sparse(0.5 * pauli_x());
    DenseMatrix d1(2, 2), d2(2, 2), dboth(2, 2);
    lindblad_rhs(d1, rho, H_z, {jump}, false);
    lindblad_rhs(d2, rho, H_z, {jump2}, false);
    lindblad_rhs(dboth, rho, H_z, {jump, jump2}, false);
    DenseMatrix H_d = DenseMatrix(H_z);
    DenseMatrix unitary = std::complex<double>(0, -1) * (H_d * rho - rho * H_d);
    EXPECT_TRUE(dboth.isApprox(d1 + d2 - unitary, kTol));
}

class LindbladRhsMatrixFreeTest : public ::testing::Test {
   protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5, 0, "Z");
    SparseMatrix H_sparse = to_sparse(0.5 * pauli_z());
    SparseMatrix jump = amp_damp_jump();
};

TEST_F(LindbladRhsMatrixFreeTest, UnitaryBranchMatchesSparse) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix dmf(2, 1), dsp(2, 1);
    lindblad_rhs(dmf, psi, H_mf, {}, true);
    lindblad_rhs(dsp, psi, H_sparse, {}, true);
    EXPECT_TRUE(dmf.isApprox(dsp, kTol));
}

TEST_F(LindbladRhsMatrixFreeTest, DensityMatrixNoJumpsMatchesSparse) {
    DenseMatrix rho = pure_plus();
    DenseMatrix dmf(2, 2), dsp(2, 2);
    lindblad_rhs(dmf, rho, H_mf, {}, false);
    lindblad_rhs(dsp, rho, H_sparse, {}, false);
    EXPECT_TRUE(dmf.isApprox(dsp, kTol));
}

TEST_F(LindbladRhsMatrixFreeTest, OutputIsHermitian) {
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_mf, {jump}, false);
    EXPECT_TRUE((drho - drho.adjoint()).isZero(kTol));
}

TEST_F(LindbladRhsMatrixFreeTest, TraceIsPreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_mf, {jump}, false);
    EXPECT_NEAR(std::abs(drho.trace()), 0.0, kTol);
}

TEST_F(LindbladRhsMatrixFreeTest, WithJumpMatchesManual) {
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_mf, {jump}, false);
    DenseMatrix H_d = DenseMatrix(H_sparse);
    DenseMatrix J = DenseMatrix(jump);
    DenseMatrix Jdag = J.adjoint();
    DenseMatrix JdagJ = Jdag * J;
    DenseMatrix expected = std::complex<double>(0, -1) * (H_d * rho - rho * H_d) + J * rho * Jdag - 0.5 * (JdagJ * rho + rho * JdagJ);
    EXPECT_TRUE(drho.isApprox(expected, kTol));
}

TEST_F(LindbladRhsMatrixFreeTest, ZeroHamiltonianOnlyDissipator) {
    MatrixFreeHamiltonian H_zero = make_matrix_free_H(0.0, 0, "Z");
    DenseMatrix rho = pure_plus();
    DenseMatrix drho(2, 2);
    lindblad_rhs(drho, rho, H_zero, {jump}, false);
    DenseMatrix J = DenseMatrix(jump);
    DenseMatrix Jdag = J.adjoint();
    DenseMatrix JdagJ = Jdag * J;
    DenseMatrix expected = J * rho * Jdag - 0.5 * (JdagJ * rho + rho * JdagJ);
    EXPECT_TRUE(drho.isApprox(expected, kTol));
}

TEST_F(LindbladRhsMatrixFreeTest, SparseAndMatrixFreeOverloadsAgree) {
    DenseMatrix rho = pure_zero();
    DenseMatrix dmf(2, 2), dsp(2, 2);
    lindblad_rhs(dmf, rho, H_mf, {jump}, false);
    lindblad_rhs(dsp, rho, H_sparse, {jump}, false);
    EXPECT_TRUE(dmf.isApprox(dsp, kTol));
}

// GCOV_EXCL_BR_STOP