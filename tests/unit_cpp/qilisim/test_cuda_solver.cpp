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
#include <array>
#include <cmath>
#include "../../../src/qilisdk_cpp/libs/cuda_solver.h"

namespace {

constexpr double kTol = 1e-9;

// Deterministic dense matrix, values spread across positive/negative so that
// transposes and products are non-trivial. Reused as the building block for the
// GPU vs Eigen comparisons below.
Eigen::MatrixXd make_det(int rows, int cols, double scale = 1.0) {
    Eigen::MatrixXd A(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A(i, j) = scale * (0.5 + std::sin(0.7 * i + 1.3 * j) + 0.25 * (((i + j) % 3) - 1));
        }
    }
    return A;
}

// cuda_available() lazily probes once and caches; repeated calls must agree and
// must never throw, on GPU and CPU-only machines alike.
TEST(CudaSolver, AvailabilityIsIdempotent) {
    const bool first = qilisdk::gpu::cuda_available();
    EXPECT_EQ(first, qilisdk::gpu::cuda_available());
}

// Dimension guards return false before any device work, so these hold whether or
// not a GPU is present (CPU-only short-circuits on !cuda_available(); with a GPU
// the dimension check itself triggers the early return).
TEST(CudaSolver, GemmRejectsIncompatibleDims) {
    const Eigen::MatrixXd A = make_det(4, 3);
    const Eigen::MatrixXd B = make_det(5, 2);  // A.cols()=3 != B.rows()=5
    Eigen::MatrixXd C;
    EXPECT_FALSE(qilisdk::gpu::gemm(A, B, C));
}

TEST(CudaSolver, CholeskySolveRejectsNonSquareA) {
    const Eigen::MatrixXd A = make_det(4, 3);  // not square
    const Eigen::MatrixXd B = make_det(4, 1);
    Eigen::MatrixXd X;
    EXPECT_FALSE(qilisdk::gpu::cholesky_solve(A, B, X));
}

TEST(CudaSolver, CholeskySolveRejectsRhsRowMismatch) {
    const Eigen::MatrixXd A = make_det(4, 4);
    const Eigen::MatrixXd B = make_det(3, 1);  // rows != A.rows()
    Eigen::MatrixXd X;
    EXPECT_FALSE(qilisdk::gpu::cholesky_solve(A, B, X));
}

// ---- Numeric correctness: GPU path vs Eigen reference (skipped CPU-only) ----

TEST(CudaSolver, GemmMatchesEigen) {
    if (!qilisdk::gpu::cuda_available()) {
        GTEST_SKIP() << "No CUDA device/libraries available at runtime.";
    }
    for (const auto& dims : {std::array<int, 3>{16, 24, 8}, std::array<int, 3>{64, 64, 64}}) {
        const Eigen::MatrixXd A = make_det(dims[0], dims[1]);
        const Eigen::MatrixXd B = make_det(dims[1], dims[2], 0.5);
        Eigen::MatrixXd C;
        ASSERT_TRUE(qilisdk::gpu::gemm(A, B, C));
        ASSERT_EQ(C.rows(), dims[0]);
        ASSERT_EQ(C.cols(), dims[2]);
        EXPECT_LT((C - A * B).cwiseAbs().maxCoeff(), kTol);
    }
}

TEST(CudaSolver, GramAtaMatchesEigen) {
    if (!qilisdk::gpu::cuda_available()) {
        GTEST_SKIP() << "No CUDA device/libraries available at runtime.";
    }
    const Eigen::MatrixXd A = make_det(50, 12);
    Eigen::MatrixXd G;
    ASSERT_TRUE(qilisdk::gpu::gram_ata(A, G));
    ASSERT_EQ(G.rows(), 12);
    ASSERT_EQ(G.cols(), 12);
    // Lower triangle is what cuBLAS fills; compare it against Aᵀ A.
    const Eigen::MatrixXd ref = A.transpose() * A;
    EXPECT_LT((G.triangularView<Eigen::Lower>().toDenseMatrix() -
               ref.triangularView<Eigen::Lower>().toDenseMatrix())
                  .cwiseAbs()
                  .maxCoeff(),
              kTol);
}

TEST(CudaSolver, CholeskySolveMatchesEigen) {
    if (!qilisdk::gpu::cuda_available()) {
        GTEST_SKIP() << "No CUDA device/libraries available at runtime.";
    }
    for (int n : {5, 32}) {
        // SPD matrix A = RᵀR + nI guarantees positive definiteness for potrf.
        const Eigen::MatrixXd R = make_det(n, n);
        const Eigen::MatrixXd A = R.transpose() * R + n * Eigen::MatrixXd::Identity(n, n);
        const Eigen::MatrixXd B = make_det(n, 3, 0.3);  // multiple right-hand sides
        Eigen::MatrixXd X;
        ASSERT_TRUE(qilisdk::gpu::cholesky_solve(A, B, X));
        ASSERT_EQ(X.rows(), n);
        ASSERT_EQ(X.cols(), 3);
        // Residual A X - B must vanish; also cross-check against Eigen's LLT.
        EXPECT_LT((A * X - B).cwiseAbs().maxCoeff(), kTol);
        const Eigen::MatrixXd ref = Eigen::LLT<Eigen::MatrixXd>(A).solve(B);
        EXPECT_LT((X - ref).cwiseAbs().maxCoeff(), kTol);
    }
}

// A non-SPD matrix makes cusolverDnDpotrf report dev_info != 0, which must surface
// as false so callers fall back to Eigen rather than using garbage.
TEST(CudaSolver, CholeskySolveRejectsNonSpd) {
    if (!qilisdk::gpu::cuda_available()) {
        GTEST_SKIP() << "No CUDA device/libraries available at runtime.";
    }
    Eigen::MatrixXd A(2, 2);
    A << 1.0, 2.0, 2.0, 1.0;  // symmetric, eigenvalues 3 and -1 -> indefinite
    const Eigen::MatrixXd B = Eigen::MatrixXd::Ones(2, 1);
    Eigen::MatrixXd X;
    EXPECT_FALSE(qilisdk::gpu::cholesky_solve(A, B, X));
}

}  // namespace

// GCOV_EXCL_BR_STOP
