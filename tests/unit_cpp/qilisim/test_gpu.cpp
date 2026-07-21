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

// GCOVR_EXCL_START

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "../../../src/qilisdk_cpp/libs/gpu.h"

namespace {

using cx = std::complex<double>;
constexpr double kTol = 1e-9;

// cuda_available() lazily probes once and caches; repeated calls must agree and
// must never throw, on GPU and CPU-only machines alike.
TEST(CudaSolver, AvailabilityIsIdempotent) {
    const bool first = qilisdk::gpu::cuda_available();
    EXPECT_EQ(first, qilisdk::gpu::cuda_available());
}

// Reference stochastic-reconfiguration solve on the CPU, mirroring sr_adot_cpu
// in lindblad.cpp: adot = M^{-1} V with M = OᵀO/N_s - ōōᵀ + εI and
// V = -(Oᵀ El/N_s - ō Ēl). The GPU sr_solve must reproduce this.
Eigen::VectorXcd sr_reference(const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double eps) {
    const int N_s = static_cast<int>(O.rows());
    const int p = static_cast<int>(O.cols());
    const Eigen::VectorXd om = O.colwise().mean();
    const cx Elm = El.mean();
    Eigen::MatrixXd M = O.transpose() * O / static_cast<double>(N_s) - om * om.transpose();
    M += eps * Eigen::MatrixXd::Identity(p, p);
    const Eigen::VectorXcd V = -((O.transpose().cast<cx>() * El) / static_cast<double>(N_s) - om.cast<cx>() * Elm);
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    Eigen::VectorXcd adot(p);
    adot.real() = llt.solve(V.real());
    adot.imag() = llt.solve(V.imag());
    return adot;
}

// Build a deterministic ±1 sample-operator matrix (stand-in for samples.O_mat).
Eigen::MatrixXd make_pm1(int N_s, int p) {
    Eigen::MatrixXd O(N_s, p);
    for (int i = 0; i < N_s; ++i) {
        for (int j = 0; j < p; ++j) {
            O(i, j) = (((i * 31 + j * 17) % 2) == 0) ? 1.0 : -1.0;
        }
    }
    return O;
}

// The GPU resident SR solve must match the CPU reference to ~machine precision.
// Skips when no CUDA device/libraries are reachable at runtime (e.g. CPU-only CI).
TEST(GpuSrSolve, MatchesCpuReference) {
    if (!qilisdk::gpu::cuda_available()) {
        GTEST_SKIP() << "No CUDA device/libraries available at runtime.";
    }

    for (const auto& dims : {std::pair<int, int>{200, 37}, std::pair<int, int>{1000, 64}}) {
        const int N_s = dims.first;
        const int p = dims.second;
        const Eigen::MatrixXd O = make_pm1(N_s, p);
        Eigen::VectorXcd El(N_s);
        for (int i = 0; i < N_s; ++i) {
            El(i) = cx(0.3 * i - 5.0, 0.1 * i + 1.0);
        }
        const double eps = 0.1 / std::sqrt(static_cast<double>(N_s));

        const Eigen::VectorXcd ref = sr_reference(O, El, eps);
        Eigen::VectorXcd adot;
        ASSERT_TRUE(qilisdk::gpu::sr_solve(O, El, eps, adot)) << "sr_solve failed for N_s=" << N_s << " p=" << p;
        ASSERT_EQ(adot.size(), p);
        EXPECT_LT((adot - ref).cwiseAbs().maxCoeff(), kTol) << "mismatch for N_s=" << N_s << " p=" << p;
    }
}

}  // namespace

// GCOVR_EXCL_STOP
