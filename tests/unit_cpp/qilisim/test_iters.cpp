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
#include "../../../src/qilisdk_cpp/backends/qilisim/analog/iterations.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/utils/matrix_utils.h"

namespace {

constexpr double kTol = 1e-8;
constexpr double kTolLoose = 1e-4;
constexpr double kAtol = 1e-12;


SparseMatrix to_sparse(const DenseMatrix& M) {
    SparseMatrix S(M.rows(), M.cols());
    S = M.sparseView();
    return S;
}

DenseMatrix pure_zero() {
    DenseMatrix r = DenseMatrix::Zero(2, 2);
    r(0, 0) = 1.0;
    return r;
}

DenseMatrix pure_plus() {
    DenseMatrix r(2, 2);
    r << 0.5, 0.5, 0.5, 0.5;
    return r;
}

DenseMatrix maximally_mixed() {
    return DenseMatrix::Identity(2, 2) * 0.5;
}

SparseMatrix amp_damp_jump() {
    DenseMatrix j = DenseMatrix::Zero(2, 2);
    j(0, 1) = 1.0;
    return to_sparse(j);
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

MatrixFreeHamiltonian make_matrix_free_H(const DenseMatrix& base_matrix) {
    MatrixFreeOperator op("custom", {}, {0}, base_matrix);
    return MatrixFreeHamiltonian(op);
}

}

class IterDirectValidationTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
};

TEST_F(IterDirectValidationTest, NonSquareHamiltonianThrows) {
    SparseMatrix H_rect(2, 3);
    DenseMatrix rho = pure_zero();
    EXPECT_ANY_THROW(iter_direct(rho, 0.1, H_rect, {}, false));
}

TEST_F(IterDirectValidationTest, NonSquareDensityMatrixThrows) {
    DenseMatrix rho_rect(2, 3);
    rho_rect.setZero();
    EXPECT_ANY_THROW(iter_direct(rho_rect, 0.1, H, {}, false));
}

TEST_F(IterDirectValidationTest, MismatchedHamiltonianAndRhoDimensionsThrows) {
    DenseMatrix rho_4 = DenseMatrix::Identity(4, 4) * 0.25;
    EXPECT_ANY_THROW(iter_direct(rho_4, 0.1, H, {}, false));
}

TEST_F(IterDirectValidationTest, MismatchedJumpOperatorDimensionThrows) {
    DenseMatrix rho = pure_zero();
    SparseMatrix bad_jump(4, 4);
    EXPECT_ANY_THROW(iter_direct(rho, 0.1, H, {bad_jump}, false));
}

TEST_F(IterDirectValidationTest, NonSquareJumpOperatorThrows) {
    DenseMatrix rho = pure_zero();
    SparseMatrix bad_jump(2, 3);
    EXPECT_ANY_THROW(iter_direct(rho, 0.1, H, {bad_jump}, false));
}

class IterDirectUnitaryStatevectorTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
};

TEST_F(IterDirectUnitaryStatevectorTest, OutputDimensionMatchesInput) {
    DenseMatrix psi(2, 1);
    psi << 1, 0;
    DenseMatrix result = iter_direct(psi, dt, H, {}, true);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 1);
}

TEST_F(IterDirectUnitaryStatevectorTest, NormPreserved) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix result = iter_direct(psi, dt, H, {}, true);
    EXPECT_NEAR(result.norm(), 1.0, kTol);
}

TEST_F(IterDirectUnitaryStatevectorTest, EigenstateLeavesPopulationUnchanged) {
    DenseMatrix psi(2, 1);
    psi << 1, 0;
    DenseMatrix result = iter_direct(psi, dt, H, {}, true);
    EXPECT_NEAR(std::abs(result(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::abs(result(1, 0)), 0.0, kTol);
}

TEST_F(IterDirectUnitaryStatevectorTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix result = iter_direct(psi, 0.0, H, {}, true);
    EXPECT_TRUE(result.isApprox(psi, kTol));
}

TEST_F(IterDirectUnitaryStatevectorTest, PhaseEvolutionMatchesAnalytic) {
    DenseMatrix psi(2, 1);
    psi << 1, 0;
    DenseMatrix result = iter_direct(psi, dt, H, {}, true);
    std::complex<double> expected_phase = std::exp(std::complex<double>(0, -0.5 * dt));
    EXPECT_NEAR(std::abs(result(0, 0) - expected_phase), 0.0, kTol);
}

TEST_F(IterDirectUnitaryStatevectorTest, TwoHalfStepsEqualsOneFullStep) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix half1 = iter_direct(psi, dt / 2.0, H, {}, true);
    DenseMatrix half2 = iter_direct(half1, dt / 2.0, H, {}, true);
    DenseMatrix full = iter_direct(psi, dt, H, {}, true);
    EXPECT_TRUE(half2.isApprox(full, kTol));
}

class IterDirectUnitaryDensityMatrixTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
};

TEST_F(IterDirectUnitaryDensityMatrixTest, OutputDimensionMatchesInput) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {}, false);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
}

TEST_F(IterDirectUnitaryDensityMatrixTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {}, false);
    EXPECT_NEAR(std::real(result.trace()), 1.0, kTol);
}

TEST_F(IterDirectUnitaryDensityMatrixTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {}, false);
    EXPECT_TRUE((result - result.adjoint()).isZero(kTol));
}

TEST_F(IterDirectUnitaryDensityMatrixTest, PurityPreservedForPureState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {}, false);
    double purity = std::real((result * result).trace());
    EXPECT_NEAR(purity, 1.0, kTol);
}

TEST_F(IterDirectUnitaryDensityMatrixTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, 0.0, H, {}, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterDirectUnitaryDensityMatrixTest, EigenstatePopulationsUnchanged) {
    DenseMatrix rho = pure_zero();
    DenseMatrix result = iter_direct(rho, dt, H, {}, false);
    EXPECT_NEAR(std::real(result(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::real(result(1, 1)), 0.0, kTol);
}

TEST_F(IterDirectUnitaryDensityMatrixTest, CoherenceEvolutionMatchesAnalytic) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {}, false);
    std::complex<double> expected_offdiag = 0.5 * std::exp(std::complex<double>(0, -dt));
    EXPECT_NEAR(std::abs(result(0, 1) - expected_offdiag), 0.0, kTol);
}

TEST_F(IterDirectUnitaryDensityMatrixTest, TwoHalfStepsEqualsOneFullStep) {
    DenseMatrix rho = pure_plus();
    DenseMatrix half1 = iter_direct(rho, dt / 2.0, H, {}, false);
    DenseMatrix half2 = iter_direct(half1, dt / 2.0, H, {}, false);
    DenseMatrix full = iter_direct(rho, dt, H, {}, false);
    EXPECT_TRUE(half2.isApprox(full, kTol));
}

TEST_F(IterDirectUnitaryDensityMatrixTest, MaximallyMixedStateInvariant) {
    DenseMatrix rho = maximally_mixed();
    DenseMatrix result = iter_direct(rho, dt, H, {}, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

class IterDirectLindbladTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    SparseMatrix jump = amp_damp_jump();
    double dt = 0.1;
};

TEST_F(IterDirectLindbladTest, OutputDimensionMatchesInput) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {jump}, false);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
}

TEST_F(IterDirectLindbladTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {jump}, false);
    EXPECT_NEAR(std::real(result.trace()), 1.0, kTol);
}

TEST_F(IterDirectLindbladTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {jump}, false);
    EXPECT_TRUE((result - result.adjoint()).isZero(kTol));
}

TEST_F(IterDirectLindbladTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, 0.0, H, {jump}, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterDirectLindbladTest, GroundStateIsFixedPointOfAmpDamping) {
    DenseMatrix rho = pure_zero();
    DenseMatrix result = iter_direct(rho, dt, H, {jump}, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterDirectLindbladTest, ExcitedStateDecaysTowardGroundState) {
    DenseMatrix rho = DenseMatrix::Zero(2, 2);
    rho(1, 1) = 1.0;
    DenseMatrix result = iter_direct(rho, dt, H, {jump}, false);
    EXPECT_GT(std::real(result(0, 0)), std::real(rho(0, 0)));
    EXPECT_LT(std::real(result(1, 1)), std::real(rho(1, 1)));
}

TEST_F(IterDirectLindbladTest, PurityDecreasesForMixedEvolution) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {jump}, false);
    double initial_purity = std::real((rho * rho).trace());
    double final_purity = std::real((result * result).trace());
    EXPECT_LT(final_purity, initial_purity);
}

TEST_F(IterDirectLindbladTest, TwoHalfStepsEqualsOneFullStep) {
    DenseMatrix rho = pure_plus();
    DenseMatrix half1 = iter_direct(rho, dt / 2.0, H, {jump}, false);
    DenseMatrix half2 = iter_direct(half1, dt / 2.0, H, {jump}, false);
    DenseMatrix full = iter_direct(rho, dt, H, {jump}, false);
    EXPECT_TRUE(half2.isApprox(full, kTol));
}

TEST_F(IterDirectLindbladTest, LongTimeEvolutionConvergesToGroundState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, 20.0, H, {jump}, false);
    EXPECT_NEAR(std::real(result(0, 0)), 1.0, 1e-4);
    EXPECT_NEAR(std::real(result(1, 1)), 0.0, 1e-4);
}

TEST_F(IterDirectLindbladTest, MultipleJumpsPreserveTrace) {
    SparseMatrix jump2 = to_sparse(0.5 * pauli_x());
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_direct(rho, dt, H, {jump, jump2}, false);
    EXPECT_NEAR(std::real(result.trace()), 1.0, kTol);
}

class ArnoldiTest : public ::testing::Test {
protected:
    SparseMatrix L = to_sparse(pauli_z().cast<std::complex<double>>());
    DenseMatrix v0 = DenseMatrix::Zero(2, 1);
    int m = 2;

    void SetUp() override {
        v0(0, 0) = 1.0;
    }
};

TEST_F(ArnoldiTest, OutputVectorCountIsAtMostMPlusOne) {
    std::vector<DenseMatrix> V;
    SparseMatrix H;
    arnoldi(L, v0, m, V, H, kAtol);
    EXPECT_LE(int(V.size()), m + 1);
}

TEST_F(ArnoldiTest, HessenbergMatrixDimension) {
    std::vector<DenseMatrix> V;
    SparseMatrix H;
    arnoldi(L, v0, m, V, H, kAtol);
    EXPECT_EQ(H.rows(), m + 1);
    EXPECT_EQ(H.cols(), m);
}

TEST_F(ArnoldiTest, BasisVectorsAreOrthonormal) {
    std::vector<DenseMatrix> V;
    SparseMatrix H;
    arnoldi(L, v0, m, V, H, kAtol);
    for (int i = 0; i < int(V.size()); ++i) {
        for (int j = 0; j < int(V.size()); ++j) {
            std::complex<double> inner = dot(V[i], V[j]);
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(std::abs(inner - expected), 0.0, kTol)
                << "Orthonormality failed for i=" << i << " j=" << j;
        }
    }
}

TEST_F(ArnoldiTest, FirstBasisVectorIsNormalisedInput) {
    std::vector<DenseMatrix> V;
    SparseMatrix H;
    arnoldi(L, v0, m, V, H, kAtol);
    EXPECT_NEAR(V[0].norm(), 1.0, kTol);
    DenseMatrix v0_norm = v0 / v0.norm();
    EXPECT_TRUE(V[0].isApprox(v0_norm, kTol));
}

TEST_F(ArnoldiTest, SubdiagonalEntriesAreNonNegativeReal) {
    std::vector<DenseMatrix> V;
    SparseMatrix H;
    arnoldi(L, v0, m, V, H, kAtol);
    for (int j = 0; j < m; ++j) {
        double val = std::real(H.coeff(j + 1, j));
        EXPECT_GE(val, 0.0);
        EXPECT_NEAR(std::imag(H.coeff(j + 1, j)), 0.0, kTol);
    }
}

TEST_F(ArnoldiTest, ZeroInitialVectorProducesBasis) {
    DenseMatrix zero_v0 = DenseMatrix::Zero(2, 1);
    std::vector<DenseMatrix> V;
    SparseMatrix H;
    arnoldi(L, zero_v0, m, V, H, kAtol);
    EXPECT_EQ(int(V.size()), 0);
}

TEST_F(ArnoldiTest, LargerSubspaceDimension) {
    DenseMatrix rho_vec = DenseMatrix::Zero(4, 1);
    rho_vec(0, 0) = 0.5;
    rho_vec(3, 0) = 0.5;
    SparseMatrix L4 = to_sparse(DenseMatrix::Identity(4, 4));
    std::vector<DenseMatrix> V;
    SparseMatrix H;
    arnoldi(L4, rho_vec, 3, V, H, kAtol);
    EXPECT_LE(int(V.size()), 4);
    for (int i = 0; i < int(V.size()); ++i) {
        EXPECT_NEAR(V[i].norm(), 1.0, kTol);
    }
}

class ArnoldiMatTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    DenseMatrix rho0 = pure_plus();
    int m = 2;
};

TEST_F(ArnoldiMatTest, OutputVectorCountIsAtMostMPlusOne) {
    std::vector<DenseMatrix> V;
    SparseMatrix Hk;
    arnoldi_mat(H, rho0, m, V, Hk, kAtol);
    EXPECT_LE(int(V.size()), m + 1);
}

TEST_F(ArnoldiMatTest, HessenbergMatrixDimension) {
    std::vector<DenseMatrix> V;
    SparseMatrix Hk;
    arnoldi_mat(H, rho0, m, V, Hk, kAtol);
    EXPECT_EQ(Hk.rows(), m + 1);
    EXPECT_EQ(Hk.cols(), m);
}

TEST_F(ArnoldiMatTest, BasisVectorsAreOrthonormal) {
    std::vector<DenseMatrix> V;
    SparseMatrix Hk;
    arnoldi_mat(H, rho0, m, V, Hk, kAtol);
    for (int i = 0; i < int(V.size()); ++i) {
        for (int j = 0; j < int(V.size()); ++j) {
            std::complex<double> inner = dot(V[i], V[j]);
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(std::abs(inner - expected), 0.0, kTol)
                << "Orthonormality failed for i=" << i << " j=" << j;
        }
    }
}

TEST_F(ArnoldiMatTest, FirstBasisVectorIsNormalisedInput) {
    std::vector<DenseMatrix> V;
    SparseMatrix Hk;
    arnoldi_mat(H, rho0, m, V, Hk, kAtol);
    EXPECT_NEAR(V[0].norm(), 1.0, kTol);
    DenseMatrix rho0_norm = rho0 / rho0.norm();
    EXPECT_TRUE(V[0].isApprox(rho0_norm, kTol));
}

TEST_F(ArnoldiMatTest, SubdiagonalEntriesAreNonNegativeReal) {
    std::vector<DenseMatrix> V;
    SparseMatrix Hk;
    arnoldi_mat(H, rho0, m, V, Hk, kAtol);
    for (int j = 0; j < m; ++j) {
        double val = std::real(Hk.coeff(j + 1, j));
        EXPECT_GE(val, 0.0);
        EXPECT_NEAR(std::imag(Hk.coeff(j + 1, j)), 0.0, kTol);
    }
}

TEST_F(ArnoldiMatTest, ZeroInitialMatrixProducesEmptyBasis) {
    DenseMatrix zero_rho = DenseMatrix::Zero(2, 2);
    std::vector<DenseMatrix> V;
    SparseMatrix Hk;
    arnoldi_mat(H, zero_rho, m, V, Hk, kAtol);
    EXPECT_EQ(int(V.size()), 0);
}

class IterArnoldiValidationTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    DenseMatrix rho = pure_zero();
    double dt = 0.1;
    int arnoldi_dim = 4;
    int num_substeps = 1;
};

TEST_F(IterArnoldiValidationTest, NonPositiveArnoldiDimThrows) {
    EXPECT_ANY_THROW(iter_arnoldi(rho, dt, H, {}, 0, num_substeps, false, kAtol));
}

TEST_F(IterArnoldiValidationTest, NegativeArnoldiDimThrows) {
    EXPECT_ANY_THROW(iter_arnoldi(rho, dt, H, {}, -1, num_substeps, false, kAtol));
}

TEST_F(IterArnoldiValidationTest, NonPositiveNumSubstepsThrows) {
    EXPECT_ANY_THROW(iter_arnoldi(rho, dt, H, {}, arnoldi_dim, 0, false, kAtol));
}

TEST_F(IterArnoldiValidationTest, NonSquareHamiltonianThrows) {
    SparseMatrix H_rect(2, 3);
    EXPECT_ANY_THROW(iter_arnoldi(rho, dt, H_rect, {}, arnoldi_dim, num_substeps, false, kAtol));
}

TEST_F(IterArnoldiValidationTest, NonSquareDensityMatrixThrows) {
    DenseMatrix rho_rect(2, 3);
    rho_rect.setZero();
    EXPECT_ANY_THROW(iter_arnoldi(rho_rect, dt, H, {}, arnoldi_dim, num_substeps, false, kAtol));
}

TEST_F(IterArnoldiValidationTest, MismatchedHamiltonianAndRhoDimensionsThrows) {
    DenseMatrix rho_4 = DenseMatrix::Identity(4, 4) * 0.25;
    EXPECT_ANY_THROW(iter_arnoldi(rho_4, dt, H, {}, arnoldi_dim, num_substeps, false, kAtol));
}

TEST_F(IterArnoldiValidationTest, MismatchedJumpOperatorDimensionThrows) {
    SparseMatrix bad_jump(4, 4);
    EXPECT_ANY_THROW(iter_arnoldi(rho, dt, H, {bad_jump}, arnoldi_dim, num_substeps, false, kAtol));
}

class IterArnoldiUnitaryStatevectorTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
    int arnoldi_dim = 4;
    int num_substeps = 1;
};

TEST_F(IterArnoldiUnitaryStatevectorTest, OutputDimensionMatchesInput) {
    DenseMatrix psi(2, 1);
    psi << 1, 0;
    DenseMatrix result = iter_arnoldi(psi, dt, H, {}, arnoldi_dim, num_substeps, true, kAtol);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 1);
}

TEST_F(IterArnoldiUnitaryStatevectorTest, NormPreserved) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix result = iter_arnoldi(psi, dt, H, {}, arnoldi_dim, num_substeps, true, kAtol);
    EXPECT_NEAR(result.norm(), 1.0, kTol);
}

TEST_F(IterArnoldiUnitaryStatevectorTest, EigenstateLeavesPopulationUnchanged) {
    DenseMatrix psi(2, 1);
    psi << 1, 0;
    DenseMatrix result = iter_arnoldi(psi, dt, H, {}, arnoldi_dim, num_substeps, true, kAtol);
    EXPECT_NEAR(std::abs(result(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::abs(result(1, 0)), 0.0, kTol);
}

TEST_F(IterArnoldiUnitaryStatevectorTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix result = iter_arnoldi(psi, 0.0, H, {}, arnoldi_dim, num_substeps, true, kAtol);
    EXPECT_TRUE(result.isApprox(psi, kTol));
}

class IterArnoldiUnitaryDensityMatrixTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
    int arnoldi_dim = 4;
    int num_substeps = 1;
};

TEST_F(IterArnoldiUnitaryDensityMatrixTest, OutputDimensionMatchesInput) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
}

TEST_F(IterArnoldiUnitaryDensityMatrixTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_NEAR(std::real(result.trace()), 1.0, kTol);
}

TEST_F(IterArnoldiUnitaryDensityMatrixTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_TRUE((result - result.adjoint()).isZero(kTol));
}

TEST_F(IterArnoldiUnitaryDensityMatrixTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, 0.0, H, {}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterArnoldiUnitaryDensityMatrixTest, MaximallyMixedStateInvariant) {
    DenseMatrix rho = maximally_mixed();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterArnoldiUnitaryDensityMatrixTest, MatchesIterDirectForShortTime) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result_arnoldi = iter_arnoldi(rho, dt, H, {}, arnoldi_dim, num_substeps, false, kAtol);
    DenseMatrix result_direct = iter_direct(rho, dt, H, {}, false);
    EXPECT_TRUE(result_arnoldi.isApprox(result_direct, kTolLoose));
}

TEST_F(IterArnoldiUnitaryDensityMatrixTest, MultipleSubstepsConsistent) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result_1 = iter_arnoldi(rho, dt, H, {}, arnoldi_dim, 1, false, kAtol);
    DenseMatrix result_4 = iter_arnoldi(rho, dt, H, {}, arnoldi_dim, 4, false, kAtol);
    EXPECT_TRUE(result_1.isApprox(result_4, kTolLoose));
}

class IterArnoldiLindbladTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    SparseMatrix jump = amp_damp_jump();
    double dt = 0.1;
    int arnoldi_dim = 4;
    int num_substeps = 1;
};

TEST_F(IterArnoldiLindbladTest, OutputDimensionMatchesInput) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {jump}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
}

TEST_F(IterArnoldiLindbladTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {jump}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_NEAR(std::real(result.trace()), 1.0, kTol);
}

TEST_F(IterArnoldiLindbladTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {jump}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_TRUE((result - result.adjoint()).isZero(kTol));
}

TEST_F(IterArnoldiLindbladTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, 0.0, H, {jump}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterArnoldiLindbladTest, GroundStateIsFixedPoint) {
    DenseMatrix rho = pure_zero();
    DenseMatrix result = iter_arnoldi(rho, dt, H, {jump}, arnoldi_dim, num_substeps, false, kAtol);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterArnoldiLindbladTest, LongTimeConvergesToGroundState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_arnoldi(rho, 20.0, H, {jump}, arnoldi_dim, 20, false, kAtol);
    EXPECT_NEAR(std::real(result(0, 0)), 1.0, kTolLoose);
    EXPECT_NEAR(std::real(result(1, 1)), 0.0, kTolLoose);
}

class IterIntegrateSparseValidationTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    DenseMatrix rho = pure_zero();
    double dt = 0.1;
};

TEST_F(IterIntegrateSparseValidationTest, NonSquareHamiltonianThrows) {
    SparseMatrix H_rect(2, 3);
    EXPECT_ANY_THROW(iter_integrate(rho, dt, H_rect, {}, 1, false));
}

TEST_F(IterIntegrateSparseValidationTest, NonSquareDensityMatrixThrows) {
    DenseMatrix rho_rect(2, 3);
    rho_rect.setZero();
    EXPECT_ANY_THROW(iter_integrate(rho_rect, dt, H, {}, 1, false));
}

TEST_F(IterIntegrateSparseValidationTest, MismatchedHamiltonianAndRhoDimensionsThrows) {
    DenseMatrix rho_4 = DenseMatrix::Identity(4, 4) * 0.25;
    EXPECT_ANY_THROW(iter_integrate(rho_4, dt, H, {}, 1, false));
}

TEST_F(IterIntegrateSparseValidationTest, MismatchedJumpOperatorDimensionThrows) {
    SparseMatrix bad_jump(4, 4);
    EXPECT_ANY_THROW(iter_integrate(rho, dt, H, {bad_jump}, 1, false));
}

class IterIntegrateSparseUnitaryStatevectorTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
};

TEST_F(IterIntegrateSparseUnitaryStatevectorTest, NormPreserved) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix result = iter_integrate(psi, dt, H, {}, 1, true);
    EXPECT_NEAR(result.norm(), 1.0, kTol);
}

TEST_F(IterIntegrateSparseUnitaryStatevectorTest, EigenstateLeavesPopulationUnchanged) {
    DenseMatrix psi(2, 1);
    psi << 1, 0;
    DenseMatrix result = iter_integrate(psi, dt, H, {}, 1, true);
    EXPECT_NEAR(std::abs(result(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::abs(result(1, 0)), 0.0, kTol);
}

TEST_F(IterIntegrateSparseUnitaryStatevectorTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix result = iter_integrate(psi, 0.0, H, {}, 1, true);
    EXPECT_TRUE(result.isApprox(psi, kTol));
}

TEST_F(IterIntegrateSparseUnitaryStatevectorTest, MatchesIterDirectForShortTime) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix result_rk4 = iter_integrate(psi, dt, H, {}, 10, true);
    DenseMatrix result_direct = iter_direct(psi, dt, H, {}, true);
    EXPECT_TRUE(result_rk4.isApprox(result_direct, kTolLoose));
}

TEST_F(IterIntegrateSparseUnitaryStatevectorTest, MoreSubstepsImproveAccuracy) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix ref = iter_direct(psi, dt, H, {}, true);
    DenseMatrix result_1 = iter_integrate(psi, dt, H, {}, 1, true);
    DenseMatrix result_10 = iter_integrate(psi, dt, H, {}, 10, true);
    double err_1 = (result_1 - ref).norm();
    double err_10 = (result_10 - ref).norm();
    EXPECT_LT(err_10, err_1);
}

class IterIntegrateSparseUnitaryDensityMatrixTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
};

TEST_F(IterIntegrateSparseUnitaryDensityMatrixTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, dt, H, {}, 1, false);
    EXPECT_NEAR(std::real(result.trace()), 1.0, kTol);
}

TEST_F(IterIntegrateSparseUnitaryDensityMatrixTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, dt, H, {}, 1, false);
    EXPECT_TRUE((result - result.adjoint()).isZero(kTol));
}

TEST_F(IterIntegrateSparseUnitaryDensityMatrixTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, 0.0, H, {}, 1, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterIntegrateSparseUnitaryDensityMatrixTest, MaximallyMixedStateInvariant) {
    DenseMatrix rho = maximally_mixed();
    DenseMatrix result = iter_integrate(rho, dt, H, {}, 1, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterIntegrateSparseUnitaryDensityMatrixTest, MatchesIterDirectForShortTime) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result_rk4 = iter_integrate(rho, dt, H, {}, 10, false);
    DenseMatrix result_direct = iter_direct(rho, dt, H, {}, false);
    EXPECT_TRUE(result_rk4.isApprox(result_direct, kTolLoose));
}

TEST_F(IterIntegrateSparseUnitaryDensityMatrixTest, PurityPreservedForPureState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, dt, H, {}, 10, false);
    double purity = std::real((result * result).trace());
    EXPECT_NEAR(purity, 1.0, kTolLoose);
}

class IterIntegrateSparseLindbladTest : public ::testing::Test {
protected:
    SparseMatrix H = to_sparse(0.5 * pauli_z());
    SparseMatrix jump = amp_damp_jump();
    double dt = 0.1;
};

TEST_F(IterIntegrateSparseLindbladTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, dt, H, {jump}, 1, false);
    EXPECT_NEAR(std::real(result.trace()), 1.0, kTol);
}

TEST_F(IterIntegrateSparseLindbladTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, dt, H, {jump}, 1, false);
    EXPECT_TRUE((result - result.adjoint()).isZero(kTol));
}

TEST_F(IterIntegrateSparseLindbladTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, 0.0, H, {jump}, 1, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterIntegrateSparseLindbladTest, GroundStateIsFixedPoint) {
    DenseMatrix rho = pure_zero();
    DenseMatrix result = iter_integrate(rho, dt, H, {jump}, 1, false);
    EXPECT_TRUE(result.isApprox(rho, kTol));
}

TEST_F(IterIntegrateSparseLindbladTest, ExcitedStateDecaysTowardGroundState) {
    DenseMatrix rho = DenseMatrix::Zero(2, 2);
    rho(1, 1) = 1.0;
    DenseMatrix result = iter_integrate(rho, dt, H, {jump}, 1, false);
    EXPECT_GT(std::real(result(0, 0)), std::real(rho(0, 0)));
    EXPECT_LT(std::real(result(1, 1)), std::real(rho(1, 1)));
}

TEST_F(IterIntegrateSparseLindbladTest, MatchesIterDirectForShortTime) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result_rk4 = iter_integrate(rho, dt, H, {jump}, 10, false);
    DenseMatrix result_direct = iter_direct(rho, dt, H, {jump}, false);
    EXPECT_TRUE(result_rk4.isApprox(result_direct, kTolLoose));
}

TEST_F(IterIntegrateSparseLindbladTest, LongTimeConvergesToGroundState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix result = iter_integrate(rho, 20.0, H, {jump}, 100, false);
    EXPECT_NEAR(std::real(result(0, 0)), 1.0, kTolLoose);
    EXPECT_NEAR(std::real(result(1, 1)), 0.0, kTolLoose);
}

class IterIntegrateMatrixFreeValidationTest : public ::testing::Test {
protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5 * pauli_z());
};

TEST_F(IterIntegrateMatrixFreeValidationTest, NonSquareDensityMatrixThrows) {
    DenseMatrix rho_rect(2, 3);
    rho_rect.setZero();
    EXPECT_ANY_THROW(iter_integrate(rho_rect, 0.1, H_mf, {}, 1, false));
}

TEST_F(IterIntegrateMatrixFreeValidationTest, MismatchedJumpOperatorDimensionThrows) {
    DenseMatrix rho = pure_zero();
    SparseMatrix bad_jump(4, 4);
    EXPECT_ANY_THROW(iter_integrate(rho, 0.1, H_mf, {bad_jump}, 1, false));
}

class IterIntegrateMatrixFreeUnitaryStatevectorTest : public ::testing::Test {
protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5 * pauli_z());
    SparseMatrix H_sparse = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
};

TEST_F(IterIntegrateMatrixFreeUnitaryStatevectorTest, NormPreserved) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    iter_integrate(psi, dt, H_mf, {}, 1, true);
    EXPECT_NEAR(psi.norm(), 1.0, kTol);
}

TEST_F(IterIntegrateMatrixFreeUnitaryStatevectorTest, EigenstateLeavesPopulationUnchanged) {
    DenseMatrix psi(2, 1);
    psi << 1, 0;
    iter_integrate(psi, dt, H_mf, {}, 1, true);
    EXPECT_NEAR(std::abs(psi(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::abs(psi(1, 0)), 0.0, kTol);
}

TEST_F(IterIntegrateMatrixFreeUnitaryStatevectorTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix psi(2, 1);
    psi << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    DenseMatrix psi_orig = psi;
    iter_integrate(psi, 0.0, H_mf, {}, 1, true);
    EXPECT_TRUE(psi.isApprox(psi_orig, kTol));
}

TEST_F(IterIntegrateMatrixFreeUnitaryStatevectorTest, MatchesSparseOverloadForShortTime) {
    DenseMatrix psi_mf(2, 1), psi_sp(2, 1);
    psi_mf << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    psi_sp = psi_mf;
    iter_integrate(psi_mf, dt, H_mf, {}, 10, true);
    DenseMatrix result_sp = iter_integrate(psi_sp, dt, H_sparse, {}, 10, true);
    EXPECT_TRUE(psi_mf.isApprox(result_sp, kTolLoose));
}

class IterIntegrateMatrixFreeUnitaryDensityMatrixTest : public ::testing::Test {
protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5 * pauli_z());
    SparseMatrix H_sparse = to_sparse(0.5 * pauli_z());
    double dt = 0.1;
};

TEST_F(IterIntegrateMatrixFreeUnitaryDensityMatrixTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    iter_integrate(rho, dt, H_mf, {}, 1, false);
    EXPECT_NEAR(std::real(rho.trace()), 1.0, kTol);
}

TEST_F(IterIntegrateMatrixFreeUnitaryDensityMatrixTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    iter_integrate(rho, dt, H_mf, {}, 1, false);
    EXPECT_TRUE((rho - rho.adjoint()).isZero(kTol));
}

TEST_F(IterIntegrateMatrixFreeUnitaryDensityMatrixTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix rho_orig = rho;
    iter_integrate(rho, 0.0, H_mf, {}, 1, false);
    EXPECT_TRUE(rho.isApprox(rho_orig, kTol));
}

TEST_F(IterIntegrateMatrixFreeUnitaryDensityMatrixTest, MaximallyMixedStateInvariant) {
    DenseMatrix rho = maximally_mixed();
    iter_integrate(rho, dt, H_mf, {}, 1, false);
    EXPECT_TRUE(rho.isApprox(maximally_mixed(), kTol));
}

TEST_F(IterIntegrateMatrixFreeUnitaryDensityMatrixTest, MatchesSparseOverloadForShortTime) {
    DenseMatrix rho_mf = pure_plus();
    DenseMatrix rho_sp = pure_plus();
    iter_integrate(rho_mf, dt, H_mf, {}, 10, false);
    DenseMatrix result_sp = iter_integrate(rho_sp, dt, H_sparse, {}, 10, false);
    EXPECT_TRUE(rho_mf.isApprox(result_sp, kTolLoose));
}

class IterIntegrateMatrixFreeLindbladTest : public ::testing::Test {
protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5 * pauli_z());
    SparseMatrix H_sparse = to_sparse(0.5 * pauli_z());
    SparseMatrix jump = amp_damp_jump();
    double dt = 0.1;
};

TEST_F(IterIntegrateMatrixFreeLindbladTest, TracePreserved) {
    DenseMatrix rho = pure_plus();
    iter_integrate(rho, dt, H_mf, {jump}, 1, false);
    EXPECT_NEAR(std::real(rho.trace()), 1.0, kTol);
}

TEST_F(IterIntegrateMatrixFreeLindbladTest, HermiticityPreserved) {
    DenseMatrix rho = pure_plus();
    iter_integrate(rho, dt, H_mf, {jump}, 1, false);
    EXPECT_TRUE((rho - rho.adjoint()).isZero(kTol));
}

TEST_F(IterIntegrateMatrixFreeLindbladTest, ZeroTimeStepReturnsInitialState) {
    DenseMatrix rho = pure_plus();
    DenseMatrix rho_orig = rho;
    iter_integrate(rho, 0.0, H_mf, {jump}, 1, false);
    EXPECT_TRUE(rho.isApprox(rho_orig, kTol));
}

TEST_F(IterIntegrateMatrixFreeLindbladTest, GroundStateIsFixedPoint) {
    DenseMatrix rho = pure_zero();
    DenseMatrix rho_orig = rho;
    iter_integrate(rho, dt, H_mf, {jump}, 1, false);
    EXPECT_TRUE(rho.isApprox(rho_orig, kTol));
}

TEST_F(IterIntegrateMatrixFreeLindbladTest, ExcitedStateDecaysTowardGroundState) {
    DenseMatrix rho = DenseMatrix::Zero(2, 2);
    rho(1, 1) = 1.0;
    double initial_excited = std::real(rho(1, 1));
    iter_integrate(rho, dt, H_mf, {jump}, 1, false);
    EXPECT_GT(std::real(rho(0, 0)), 0.0);
    EXPECT_LT(std::real(rho(1, 1)), initial_excited);
}

TEST_F(IterIntegrateMatrixFreeLindbladTest, MatchesSparseOverloadForShortTime) {
    DenseMatrix rho_mf = pure_plus();
    DenseMatrix rho_sp = pure_plus();
    iter_integrate(rho_mf, dt, H_mf, {jump}, 10, false);
    DenseMatrix result_sp = iter_integrate(rho_sp, dt, H_sparse, {jump}, 10, false);
    EXPECT_TRUE(rho_mf.isApprox(result_sp, kTolLoose));
}

TEST_F(IterIntegrateMatrixFreeLindbladTest, LongTimeConvergesToGroundState) {
    DenseMatrix rho = pure_plus();
    iter_integrate(rho, 20.0, H_mf, {jump}, 100, false);
    EXPECT_NEAR(std::real(rho(0, 0)), 1.0, kTolLoose);
    EXPECT_NEAR(std::real(rho(1, 1)), 0.0, kTolLoose);
}