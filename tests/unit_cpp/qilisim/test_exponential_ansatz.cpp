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
#include <sstream>
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/exponential_ansatz.h"

namespace {
constexpr double kTolMC = 0.2;       // Monte Carlo tolerance (stochastic)
constexpr double kTolExact = 1e-10;  // Exact tolerance for deterministic results
}  // namespace

// --- Construction ---

TEST(ExponentialAnsatz, Order0HasNoTerms) {
    ExponentialAnsatz ea(2, 0, 100, 10);
    EXPECT_EQ(ea.get_terms().size(), 0u);
}

TEST(ExponentialAnsatz, Order1TermCount) {
    // N single-body Z terms
    ExponentialAnsatz ea(3, 1, 100, 10);
    EXPECT_EQ(ea.get_terms().size(), 3u);
}

TEST(ExponentialAnsatz, Order2TermCount) {
    // N + C(N,2) terms
    ExponentialAnsatz ea(3, 2, 100, 10);
    EXPECT_EQ(ea.get_terms().size(), 6u);
}

TEST(ExponentialAnsatz, Order3TermCount) {
    // N + C(N,2) + C(N,3) terms for N=3: 3+3+1=7
    ExponentialAnsatz ea(3, 3, 100, 10);
    EXPECT_EQ(ea.get_terms().size(), 7u);
}

TEST(ExponentialAnsatz, Order4TermCount) {
    // C(4,1)+C(4,2)+C(4,3)+C(4,4) = 4+6+4+1 = 15
    ExponentialAnsatz ea(4, 4, 100, 10);
    EXPECT_EQ(ea.get_terms().size(), 15u);
}

TEST(ExponentialAnsatz, GettersAfterConstruction) {
    ExponentialAnsatz ea(2, 1, 50, 5);
    EXPECT_EQ(ea.get_order(), 1);
    EXPECT_EQ(ea.get_shots(), 50);
    EXPECT_EQ(ea.get_warmups(), 5);
}

TEST(ExponentialAnsatz, SettersModifyFields) {
    ExponentialAnsatz ea(2, 1, 50, 5);
    ea.set_shots(200);
    ea.set_warmups(20);
    ea.set_order(2);
    EXPECT_EQ(ea.get_shots(), 200);
    EXPECT_EQ(ea.get_warmups(), 20);
    EXPECT_EQ(ea.get_order(), 2);
}

// --- zeroed() ---

TEST(ExponentialAnsatz, ZeroedHasSameTermCount) {
    ExponentialAnsatz ea(2, 2, 100, 10);
    ExponentialAnsatz z = ea.zeroed();
    EXPECT_EQ(z.get_terms().size(), ea.get_terms().size());
}

TEST(ExponentialAnsatz, ZeroedAllCoefficientsAreZero) {
    ExponentialAnsatz ea(2, 1, 100, 10);
    ExponentialAnsatz delta = ea.zeroed();
    for (auto& [ps, coeff] : delta.get_terms().get_operators()) {
        coeff = 1.0;
    }
    ea += delta;
    ExponentialAnsatz z = ea.zeroed();
    for (const auto& [ps, coeff] : z.get_terms().get_operators()) {
        EXPECT_NEAR(std::abs(coeff), 0.0, 1e-14);
    }
}

// --- Arithmetic operators ---

TEST(ExponentialAnsatz, ScalarMultiplyInPlaceDoubles) {
    ExponentialAnsatz ea(2, 1, 100, 10);
    ExponentialAnsatz ones = ea.zeroed();
    for (auto& [ps, coeff] : ones.get_terms().get_operators()) {
        coeff = 1.0;
    }
    ea += ones;
    ea *= 2.0;
    for (const auto& [ps, coeff] : ea.get_terms().get_operators()) {
        EXPECT_NEAR(coeff.real(), 2.0, 1e-12);
    }
}

TEST(ExponentialAnsatz, ScalarMultiplyReturnsNewInstance) {
    ExponentialAnsatz ea(2, 1, 100, 10);
    ExponentialAnsatz ones = ea.zeroed();
    for (auto& [ps, coeff] : ones.get_terms().get_operators()) {
        coeff = 1.0;
    }
    ea += ones;
    ExponentialAnsatz scaled = ea * 3.0;
    for (const auto& [ps, coeff] : scaled.get_terms().get_operators()) {
        EXPECT_NEAR(coeff.real(), 3.0, 1e-12);
    }
    for (const auto& [ps, coeff] : ea.get_terms().get_operators()) {
        EXPECT_NEAR(coeff.real(), 1.0, 1e-12);
    }
}

TEST(ExponentialAnsatz, AdditionInPlaceAccumulates) {
    ExponentialAnsatz ea1(2, 1, 100, 10);
    ExponentialAnsatz delta = ea1.zeroed();
    for (auto& [ps, coeff] : delta.get_terms().get_operators()) {
        coeff = 0.5;
    }
    ea1 += delta;
    ExponentialAnsatz ea2 = ea1.zeroed();
    for (auto& [ps, coeff] : ea2.get_terms().get_operators()) {
        coeff = 0.3;
    }
    ea1 += ea2;
    for (const auto& [ps, coeff] : ea1.get_terms().get_operators()) {
        EXPECT_NEAR(coeff.real(), 0.8, 1e-12);
    }
}

TEST(ExponentialAnsatz, AdditionReturnsSum) {
    ExponentialAnsatz ea1(2, 1, 100, 10);
    ExponentialAnsatz d1 = ea1.zeroed();
    for (auto& [ps, coeff] : d1.get_terms().get_operators()) {
        coeff = 0.4;
    }
    ea1 += d1;
    ExponentialAnsatz ea2 = ea1.zeroed();
    for (auto& [ps, coeff] : ea2.get_terms().get_operators()) {
        coeff = 0.6;
    }
    ExponentialAnsatz sum = ea1 + ea2;
    for (const auto& [ps, coeff] : sum.get_terms().get_operators()) {
        EXPECT_NEAR(coeff.real(), 1.0, 1e-12);
    }
}

// --- Stream operator ---

TEST(ExponentialAnsatz, StreamOutputContainsExpAndKetPlus) {
    ExponentialAnsatz ea(1, 1, 100, 10);
    std::ostringstream oss;
    oss << ea;
    EXPECT_NE(oss.str().find("exp("), std::string::npos);
    EXPECT_NE(oss.str().find("|+>"), std::string::npos);
}

// --- draw_samples() ---

TEST(ExponentialAnsatz, DrawSamplesDefaultHasCorrectShape) {
    ExponentialAnsatz ea(2, 1, 50, 5);
    SampleSet s = ea.draw_samples();
    EXPECT_EQ(static_cast<int>(s.configs.size()), 50);
    EXPECT_EQ(s.O_mat.rows(), 50);
    EXPECT_EQ(s.O_mat.cols(), static_cast<int>(ea.get_terms().size()));
}

TEST(ExponentialAnsatz, DrawSamplesCustomNsAndWarmup) {
    ExponentialAnsatz ea(2, 1, 100, 0);
    SampleSet s = ea.draw_samples(30, 0);
    EXPECT_EQ(static_cast<int>(s.configs.size()), 30);
}

// --- local_energy() ---

TEST(ExponentialAnsatz, LocalEnergyHasCorrectSize) {
    ExponentialAnsatz ea(2, 1, 50, 0);
    MatrixFreeHamiltonian H_Z(2);
    H_Z.add(std::complex<double>(1.0, 0.0), MatrixFreeOperator("Z", 0));
    SampleSet s = ea.draw_samples();
    Eigen::VectorXcd El = ea.local_energy(s, H_Z);
    EXPECT_EQ(El.size(), 50);
}

namespace {
DenseMatrix dense_pauli_string(const PauliString& ps, int n_qubits) {
    // Same qubit ordering as ExponentialAnsatz::to_dense, so that basis index == sample bitset value
    DenseMatrix pauli_x(2, 2), pauli_y(2, 2), pauli_z(2, 2);
    pauli_x << Complex(0), Complex(1), Complex(1), Complex(0);
    pauli_y << Complex(0), Complex(0, -1), Complex(0, 1), Complex(0);
    pauli_z << Complex(1), Complex(0), Complex(0), Complex(-1);
    DenseMatrix op = DenseMatrix::Identity(1, 1);
    for (int i = 0; i < n_qubits; ++i) {
        DenseMatrix single = DenseMatrix::Identity(2, 2);
        if (ps.x_mask[i] && ps.z_mask[i])
            single = pauli_y;
        else if (ps.x_mask[i])
            single = pauli_x;
        else if (ps.z_mask[i])
            single = pauli_z;
        op = Eigen::kroneckerProduct(op, single).eval();
    }
    return op;
}
}  // namespace

TEST(ExponentialAnsatz, LocalEnergyMatchesExactAmplitudeRatios) {
    // E_loc(x) must equal (H psi)(x) / psi(x) for the dense state, including X, Y and mixed terms
    const int n = 3;
    ExponentialAnsatz ea(n, 2, 40, 2, 11);
    double c = 0.05;
    for (auto& [ps, coeff] : ea.get_terms().get_operators()) {
        coeff = Complex(c, -0.5 * c);
        c += 0.07;
    }
    MatrixFreeHamiltonian H(n);
    H.add(Complex(0.7, 0.0), MatrixFreeOperator("X", 0));
    H.add(Complex(-0.4, 0.0), MatrixFreeOperator("Y", 1));
    H.add(Complex(0.3, 0.0), MatrixFreeOperator("Z", 2));
    H.add(Complex(0.9, 0.0), std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("X", 1)});
    H.add(Complex(-0.6, 0.0), std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)});
    H.add(Complex(0.2, 0.0), std::vector<MatrixFreeOperator>{MatrixFreeOperator("Y", 0), MatrixFreeOperator("Z", 2)});

    DenseMatrix H_dense = DenseMatrix::Zero(1 << n, 1 << n);
    for (const auto& [ps, coeff] : H.get_operators()) {
        H_dense += coeff * dense_pauli_string(ps, n);
    }
    DenseMatrix psi = ea.to_dense();
    DenseMatrix H_psi = H_dense * psi;

    SampleSet samples = ea.draw_samples();
    DenseVector El = ea.local_energy(samples, H);
    for (size_t s = 0; s < samples.configs.size(); ++s) {
        const auto index = static_cast<Eigen::Index>(samples.configs[s].to_ulong());
        Complex expected = H_psi(index, 0) / psi(index, 0);
        EXPECT_NEAR(El(s).real(), expected.real(), 1e-9);
        EXPECT_NEAR(El(s).imag(), expected.imag(), 1e-9);
    }
}

TEST(ExponentialAnsatz, LocalEnergyRejectsSamplesFromDifferentAnsatz) {
    ExponentialAnsatz order_one(3, 1, 20, 0);
    ExponentialAnsatz order_two(3, 2, 20, 0);
    SampleSet samples = order_one.draw_samples();
    MatrixFreeHamiltonian H(3);
    H.add(Complex(1.0, 0.0), MatrixFreeOperator("X", 0));
    EXPECT_THROW(order_two.local_energy(samples, H), std::invalid_argument);
}

// --- seeding ---

TEST(ExponentialAnsatz, SameSeedDrawsSameSamples) {
    ExponentialAnsatz first(5, 2, 60, 3, 1234);
    ExponentialAnsatz second(5, 2, 60, 3, 1234);
    SampleSet a = first.draw_samples();
    SampleSet b = second.draw_samples();
    EXPECT_EQ(a.configs, b.configs);
    EXPECT_EQ(a.O_mat, b.O_mat);
}

TEST(ExponentialAnsatz, SetSeedRestartsTheStream) {
    ExponentialAnsatz ea(5, 2, 60, 3, 99);
    SampleSet first = ea.draw_samples();
    ea.set_seed(99);
    SampleSet again = ea.draw_samples();
    EXPECT_EQ(first.configs, again.configs);
}

TEST(ExponentialAnsatz, CopiesShareTheRandomStream) {
    // RK4 stage copies must not replay the samples of the state they were copied from
    ExponentialAnsatz original(5, 2, 60, 3, 7);
    ExponentialAnsatz copy = original.zeroed();
    ExponentialAnsatz fresh(5, 2, 60, 3, 7);
    SampleSet from_fresh = fresh.draw_samples();
    original.draw_samples();
    SampleSet from_copy = copy.draw_samples();
    EXPECT_NE(from_copy.configs, from_fresh.configs);
}

// --- to_dense() ---

TEST(ExponentialAnsatz, ToDenseCorrectShape) {
    ExponentialAnsatz ea(2, 1, 100, 0);
    DenseMatrix d = ea.to_dense();
    EXPECT_EQ(d.rows(), 4);
    EXPECT_EQ(d.cols(), 1);
}

TEST(ExponentialAnsatz, ToDenseIsNormalized) {
    ExponentialAnsatz ea(2, 1, 100, 0);
    DenseMatrix d = ea.to_dense();
    EXPECT_NEAR(d.norm(), 1.0, kTolExact);
}

TEST(ExponentialAnsatz, ToDenseZeroCoeffIsUniformSuperposition) {
    // exp(0)|+^2> = |+^2>: all amplitudes have equal magnitude 1/2
    ExponentialAnsatz ea(2, 1, 100, 0);
    DenseMatrix d = ea.to_dense();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(std::abs(d(i, 0)), 0.5, kTolExact);
    }
}

// --- expectation_value() ---

TEST(ExponentialAnsatz, LocalEnergyWithYHamiltonian_IsNonZero) {
    ExponentialAnsatz ea(1, 1, 100, 0);
    SampleSet samples = ea.draw_samples();
    MatrixFreeHamiltonian H_Y(1);
    H_Y.add(std::complex<double>(1.0, 0.0), MatrixFreeOperator("Y", 0));
    Eigen::VectorXcd el = ea.local_energy(samples, H_Y);
    double total_abs = 0.0;
    for (int i = 0; i < el.size(); ++i)
        total_abs += std::abs(el(i));
    EXPECT_GT(total_abs, 0.0);
}

TEST(ExponentialAnsatz, ToDenseWithXTerm_IsNormalized) {
    ExponentialAnsatz ea(1, 1, 100, 0);
    PauliString ps_x(1);
    ps_x.x_mask.set(0);
    ea.get_terms().add(std::complex<double>(0.1, 0.0), ps_x);
    DenseMatrix d = ea.to_dense();
    EXPECT_EQ(d.rows(), 2);
    EXPECT_NEAR(d.norm(), 1.0, 1e-10);
}

TEST(ExponentialAnsatz, ToDenseWithYTerm_IsNormalized) {
    ExponentialAnsatz ea(1, 1, 100, 0);
    PauliString ps_y(1);
    ps_y.x_mask.set(0);
    ps_y.z_mask.set(0);
    ea.get_terms().add(std::complex<double>(0.1, 0.0), ps_y);
    DenseMatrix d = ea.to_dense();
    EXPECT_EQ(d.rows(), 2);
    EXPECT_NEAR(d.norm(), 1.0, 1e-10);
}

TEST(ExponentialAnsatz, ExpectationValueZIsZeroForPlusState) {
    // |+> state has <Z> = 0
    ExponentialAnsatz ea(1, 1, 500, 20);
    MatrixFreeHamiltonian H_Z(1);
    H_Z.add(std::complex<double>(1.0, 0.0), MatrixFreeOperator("Z", 0));
    double ev = ea.expectation_value(H_Z);
    EXPECT_NEAR(ev, 0.0, kTolMC);
}

TEST(ExponentialAnsatz, ExpectationValueXIsOneForPlusState) {
    // |+> state has <X> = 1 (X|+> = |+>, so local energy is 1 everywhere → no variance)
    ExponentialAnsatz ea(1, 1, 200, 10);
    MatrixFreeHamiltonian H_X(1);
    H_X.add(std::complex<double>(1.0, 0.0), MatrixFreeOperator("X", 0));
    double ev = ea.expectation_value(H_X);
    EXPECT_NEAR(ev, 1.0, kTolMC);
}

// GCOV_EXCL_BR_STOP
