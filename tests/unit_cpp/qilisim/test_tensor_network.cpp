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

#include <cmath>
#include <complex>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "../../../src/qilisdk_cpp/backends/qilisim/digital/gate.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/matrix_free_hamiltonian.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/tensor_network.h"

namespace {

SparseMatrix to_sparse(const DenseMatrix& dense) {
    SparseMatrix out(dense.rows(), dense.cols());
    for (int row = 0; row < dense.rows(); ++row) {
        for (int col = 0; col < dense.cols(); ++col) {
            if (dense(row, col) != Complex(0.0, 0.0)) {
                out.coeffRef(row, col) = dense(row, col);
            }
        }
    }
    out.makeCompressed();
    return out;
}

DenseMatrix hadamard() {
    DenseMatrix h(2, 2);
    h << 1.0, 1.0, 1.0, -1.0;
    return h / std::sqrt(2.0);
}

DenseMatrix pauli_x_matrix() {
    DenseMatrix x = DenseMatrix::Zero(2, 2);
    x(0, 1) = 1.0;
    x(1, 0) = 1.0;
    return x;
}

DenseMatrix pauli_z_matrix() {
    DenseMatrix z = DenseMatrix::Zero(2, 2);
    z(0, 0) = 1.0;
    z(1, 1) = -1.0;
    return z;
}

// A deterministic pseudo-random unitary, so the tests exercise generic matrices
// rather than only the sparse structured ones.
DenseMatrix random_unitary(int dim, uint64_t seed) {
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    DenseMatrix m(dim, dim);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            m(row, col) = Complex(uniform(engine), uniform(engine));
        }
    }
    Eigen::HouseholderQR<DenseMatrix> qr(m);
    return DenseMatrix(qr.householderQ() * DenseMatrix::Identity(dim, dim));
}

Gate make_gate(const std::string& name, const DenseMatrix& base, const std::vector<int>& controls, const std::vector<int>& targets) {
    return Gate(name, to_sparse(base), controls, targets, {});
}

// The statevector of a circuit, built by brute force so the MPS can be checked against it
DenseMatrix statevector(const std::vector<Gate>& gates, int nqubits) {
    DenseMatrix state = DenseMatrix::Zero(1 << nqubits, 1);
    state(0, 0) = 1.0;
    for (const Gate& gate : gates) {
        DenseMatrix next = DenseMatrix(gate.get_full_matrix(nqubits)) * state;
        state = next;
    }
    return state;
}

std::string bitstring(int index, int nqubits) {
    std::string b(nqubits, '0');
    for (int q = 0; q < nqubits; ++q) {
        b[q] = ((index >> (nqubits - 1 - q)) & 1) ? '1' : '0';
    }
    return b;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

TEST(TensorTest, ConstructsZeroFilledAndFromValues) {
    Tensor zeros({2, 3, 4});
    EXPECT_EQ(zeros.rank(), 3);
    EXPECT_EQ(zeros.size(), 24);
    EXPECT_EQ(zeros.extent(1), 3);
    EXPECT_EQ(zeros.get_shape(), std::vector<int>({2, 3, 4}));
    EXPECT_EQ(zeros(std::vector<int>{1, 2, 3}), Complex(0.0, 0.0));

    Tensor filled({2, 2}, {1.0, 2.0, 3.0, 4.0});
    // Column-major: leg 0 varies fastest
    EXPECT_EQ(filled(std::vector<int>{0, 0}), Complex(1.0, 0.0));
    EXPECT_EQ(filled(std::vector<int>{1, 0}), Complex(2.0, 0.0));
    EXPECT_EQ(filled(std::vector<int>{0, 1}), Complex(3.0, 0.0));
    EXPECT_EQ(filled(std::vector<int>{1, 1}), Complex(4.0, 0.0));
    EXPECT_EQ(filled.raw().size(), 4u);

    // A rank-0 tensor is a scalar, which contraction of every leg produces
    Tensor scalar(std::vector<int>{});
    EXPECT_EQ(scalar.rank(), 0);
    EXPECT_EQ(scalar.size(), 1);
}

TEST(TensorTest, MutatesThroughIndexAndRawBuffer) {
    Tensor t({2, 2});
    t(std::vector<int>{1, 0}) = Complex(5.0, 1.0);
    EXPECT_EQ(t(std::vector<int>{1, 0}), Complex(5.0, 1.0));
    t.raw()[0] = Complex(9.0, 0.0);
    EXPECT_EQ(t(std::vector<int>{0, 0}), Complex(9.0, 0.0));
}

TEST(TensorTest, RejectsInvalidShapesAndIndices) {
    EXPECT_THROW(Tensor({2, 0}), std::invalid_argument);
    EXPECT_THROW(Tensor({2, -1}), std::invalid_argument);
    EXPECT_THROW(Tensor({2, 2}, {1.0, 2.0}), std::invalid_argument);

    Tensor t({2, 2});
    EXPECT_THROW(t(std::vector<int>{0}), std::invalid_argument);
    EXPECT_THROW(t(std::vector<int>{0, 2}), std::out_of_range);
    EXPECT_THROW(t(std::vector<int>{-1, 0}), std::out_of_range);
}

TEST(TensorTest, PermuteMovesLegsAndRejectsNonPermutations) {
    Tensor t({2, 3});
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            t(std::vector<int>{i, j}) = Complex(i * 10.0 + j, 0.0);
        }
    }
    Tensor transposed = t.permute({1, 0});
    EXPECT_EQ(transposed.get_shape(), std::vector<int>({3, 2}));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(transposed(std::vector<int>{j, i}), t(std::vector<int>{i, j}));
        }
    }

    // A three-leg permutation, where the odometer has to carry more than once
    Tensor cube({2, 3, 4});
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                cube(std::vector<int>{i, j, k}) = Complex(i + 10.0 * j + 100.0 * k, 0.0);
            }
        }
    }
    Tensor rolled = cube.permute({2, 0, 1});
    EXPECT_EQ(rolled.get_shape(), std::vector<int>({4, 2, 3}));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                EXPECT_EQ(rolled(std::vector<int>{k, i, j}), cube(std::vector<int>{i, j, k}));
            }
        }
    }

    EXPECT_THROW(cube.permute({0, 1}), std::invalid_argument);
    EXPECT_THROW(cube.permute({0, 1, 3}), std::invalid_argument);
    EXPECT_THROW(cube.permute({0, 1, 1}), std::invalid_argument);
}

TEST(TensorTest, ReshapeIsMetadataOnly) {
    Tensor t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    t.reshape({6});
    EXPECT_EQ(t.rank(), 1);
    EXPECT_EQ(t(std::vector<int>{4}), Complex(5.0, 0.0));
    EXPECT_THROW(t.reshape({4}), std::invalid_argument);
}

TEST(TensorTest, FusesGroupsOfLegs) {
    Tensor t({2, 3, 4});
    t(std::vector<int>{1, 2, 3}) = Complex(7.0, 0.0);
    Tensor fused = t.fuse({{0, 1}, {2}});
    EXPECT_EQ(fused.get_shape(), std::vector<int>({6, 4}));
    EXPECT_EQ(fused(std::vector<int>{1 + 2 * 2, 3}), Complex(7.0, 0.0));

    // An empty group adds a dummy leg of extent one
    Tensor padded = t.fuse({{0, 1, 2}, {}});
    EXPECT_EQ(padded.get_shape(), std::vector<int>({24, 1}));

    EXPECT_THROW(t.fuse({{0, 1}}), std::invalid_argument);
    EXPECT_THROW(t.fuse({{0, 5}, {1, 2}}), std::out_of_range);
}

TEST(TensorTest, MatrixViewsAreZeroCopy) {
    Tensor t({2, 2, 2});
    t(std::vector<int>{1, 0, 1}) = Complex(3.0, 0.0);
    EXPECT_EQ(t.matrix_view(1).rows(), 2);
    EXPECT_EQ(t.matrix_view(1).cols(), 4);
    EXPECT_EQ(t.matrix_view(2).rows(), 4);
    EXPECT_EQ(t.matrix_view(2).cols(), 2);
    EXPECT_EQ(t.matrix_view(2)(1, 1), Complex(3.0, 0.0));
    EXPECT_EQ(t.matrix_view(0).rows(), 1);
    EXPECT_EQ(t.matrix_view(3).cols(), 1);

    // The non-const view writes straight into the buffer
    t.matrix_view(2)(0, 0) = Complex(4.0, 0.0);
    EXPECT_EQ(t(std::vector<int>{0, 0, 0}), Complex(4.0, 0.0));

    EXPECT_THROW(t.matrix_view(4), std::out_of_range);
    EXPECT_THROW(t.matrix_view(-1), std::out_of_range);
    const Tensor& const_t = t;
    EXPECT_THROW(const_t.matrix_view(4), std::out_of_range);
    EXPECT_THROW(const_t.matrix_view(-1), std::out_of_range);
}

TEST(TensorTest, AsMatrixAndFromMatrixRoundTrip) {
    Tensor t({2, 3});
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            t(std::vector<int>{i, j}) = Complex(i + 3.0 * j, 0.0);
        }
    }
    DenseMatrix leading = t.as_matrix({0});
    EXPECT_EQ(leading.rows(), 2);
    EXPECT_EQ(leading.cols(), 3);
    EXPECT_EQ(leading(1, 2), t(std::vector<int>{1, 2}));

    // Row legs that are not leading force a permutation first
    DenseMatrix trailing = t.as_matrix({1});
    EXPECT_EQ(trailing.rows(), 3);
    EXPECT_EQ(trailing(2, 1), t(std::vector<int>{1, 2}));

    Tensor rebuilt = Tensor::from_matrix(leading, {2, 3});
    EXPECT_EQ(rebuilt.get_shape(), t.get_shape());
    EXPECT_EQ(rebuilt(std::vector<int>{1, 2}), t(std::vector<int>{1, 2}));
    EXPECT_THROW(Tensor::from_matrix(leading, {2, 2}), std::invalid_argument);
}

TEST(TensorTest, ContractsAgainstAnExplicitReference) {
    Tensor a({2, 3, 2});
    Tensor b({3, 4});
    std::mt19937_64 engine(5);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (Complex& value : a.raw()) {
        value = Complex(uniform(engine), uniform(engine));
    }
    for (Complex& value : b.raw()) {
        value = Complex(uniform(engine), uniform(engine));
    }

    Tensor c = a.contract(b, {1}, {0});
    ASSERT_EQ(c.get_shape(), std::vector<int>({2, 2, 4}));
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 2; ++k) {
            for (int l = 0; l < 4; ++l) {
                Complex expected = 0.0;
                for (int j = 0; j < 3; ++j) {
                    expected += a(std::vector<int>{i, j, k}) * b(std::vector<int>{j, l});
                }
                EXPECT_NEAR(std::abs(c(std::vector<int>{i, k, l}) - expected), 0.0, 1e-12);
            }
        }
    }

    // Contracting every leg leaves a scalar
    Tensor full = a.contract(a, {0, 1, 2}, {0, 1, 2});
    EXPECT_EQ(full.rank(), 0);
    EXPECT_NEAR(std::abs(full(std::vector<int>{}) - a.trace_all_with(a)), 0.0, 1e-12);

    EXPECT_THROW(a.contract(b, {1}, {0, 1}), std::invalid_argument);
    EXPECT_THROW(a.contract(b, {0}, {0}), std::invalid_argument);
    EXPECT_THROW(a.contract(b, {5}, {0}), std::out_of_range);
}

TEST(TensorTest, SplitFactorisesAndReportsDiscardedWeight) {
    Tensor t({2, 3});
    std::mt19937_64 engine(9);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (Complex& value : t.raw()) {
        value = Complex(uniform(engine), uniform(engine));
    }

    Tensor left;
    Tensor right;
    RealVector singular_values;
    Real error = -1.0;
    t.split({0}, 0, 0.0, left, singular_values, right, &error);
    EXPECT_EQ(singular_values.size(), 2);
    EXPECT_NEAR(error, 0.0, 1e-12);
    ASSERT_EQ(left.get_shape(), std::vector<int>({2, 2}));
    ASSERT_EQ(right.get_shape(), std::vector<int>({2, 3}));

    // Reassembling left * diag(s) * right must return the original
    DenseMatrix reassembled = left.matrix_view(1) * singular_values.cast<Complex>().asDiagonal() * right.matrix_view(1);
    EXPECT_NEAR((reassembled - t.matrix_view(1)).norm(), 0.0, 1e-12);

    // Capping the bond discards the smaller singular value
    Tensor capped_left;
    Tensor capped_right;
    RealVector capped_values;
    Real capped_error = -1.0;
    t.split({0}, 1, 0.0, capped_left, capped_values, capped_right, &capped_error);
    EXPECT_EQ(capped_values.size(), 1);
    Real expected_error = singular_values(1) * singular_values(1) / singular_values.squaredNorm();
    EXPECT_NEAR(capped_error, expected_error, 1e-12);

    // The truncation error is optional
    t.split({0}, 1, 0.0, capped_left, capped_values, capped_right);
    EXPECT_EQ(capped_values.size(), 1);

    // A cutoff above every singular value still keeps one, so a bond never vanishes
    t.split({0}, 0, 2.0, capped_left, capped_values, capped_right, &capped_error);
    EXPECT_EQ(capped_values.size(), 1);

    // A zero tensor has no weight at all to discard
    Tensor zeros({2, 2});
    zeros.split({0}, 0, 1e-10, capped_left, capped_values, capped_right, &capped_error);
    EXPECT_NEAR(capped_error, 0.0, 1e-15);
}

TEST(TensorTest, ElementwiseHelpers) {
    Tensor t({2, 2}, {Complex(1.0, 1.0), Complex(2.0, 0.0), Complex(0.0, -1.0), Complex(1.0, 0.0)});
    Tensor conjugated = t.conjugate();
    EXPECT_EQ(conjugated(std::vector<int>{0, 0}), Complex(1.0, -1.0));
    EXPECT_EQ(conjugated(std::vector<int>{0, 1}), Complex(0.0, 1.0));

    EXPECT_NEAR(t.norm(), std::sqrt(2.0 + 4.0 + 1.0 + 1.0), 1e-12);

    Tensor other({2, 2}, {1.0, 1.0, 1.0, 1.0});
    EXPECT_NEAR(std::abs(t.trace_all_with(other) - Complex(4.0, 0.0)), 0.0, 1e-12);
    EXPECT_THROW(t.trace_all_with(Tensor({4})), std::invalid_argument);

    t.scale(Complex(0.0, 2.0));
    EXPECT_EQ(t(std::vector<int>{0, 0}), Complex(-2.0, 2.0));

    EXPECT_FALSE(t.has_nan());
    t.raw()[0] = Complex(std::nan(""), 0.0);
    EXPECT_TRUE(t.has_nan());
    t.raw()[0] = Complex(0.0, std::nan(""));
    EXPECT_TRUE(t.has_nan());

    t.set_zero();
    EXPECT_NEAR(t.norm(), 0.0, 1e-15);
    EXPECT_FALSE(t.has_nan());
}

// ---------------------------------------------------------------------------
// MPSTensor
// ---------------------------------------------------------------------------

TEST(MPSTensorTest, ConstructsAndExposesViews) {
    MPSTensor site(2, 3);
    EXPECT_EQ(site.rank(), 3);
    EXPECT_EQ(site.left(), 2);
    EXPECT_EQ(site.right(), 3);
    EXPECT_EQ(site.get_shape(), std::vector<int>({2, MPSTensor::PHYSICAL_DIMENSION, 3}));

    site(1, 1, 2) = Complex(4.0, 0.0);
    EXPECT_EQ(site(1, 1, 2), Complex(4.0, 0.0));
    const MPSTensor& const_site = site;
    EXPECT_EQ(const_site(1, 1, 2), Complex(4.0, 0.0));

    // The three groupings all view the same buffer
    EXPECT_EQ(site.right_fused().rows(), 2);
    EXPECT_EQ(site.right_fused().cols(), 6);
    EXPECT_EQ(site.left_fused().rows(), 4);
    EXPECT_EQ(site.left_fused().cols(), 3);
    EXPECT_EQ(site.physical_slice(1).rows(), 2);
    EXPECT_EQ(site.physical_slice(1).cols(), 3);
    EXPECT_EQ(site.physical_slice(1)(1, 2), Complex(4.0, 0.0));
    EXPECT_EQ(site.physical_slice(0)(1, 2), Complex(0.0, 0.0));

    // A default site tensor is a one-dimensional bond on each side
    MPSTensor smallest;
    EXPECT_EQ(smallest.left(), 1);
    EXPECT_EQ(smallest.right(), 1);

    MPSTensor from_values(1, 1, {Complex(1.0, 0.0), Complex(0.0, 0.0)});
    EXPECT_EQ(from_values(0, 0, 0), Complex(1.0, 0.0));

    EXPECT_THROW(site.physical_slice(2), std::out_of_range);
    EXPECT_THROW(site.physical_slice(-1), std::out_of_range);
}

TEST(MPSTensorTest, AdoptsOnlyRankThreeTensorsWithAPhysicalLeg) {
    Tensor good({2, 2, 2});
    good(std::vector<int>{1, 1, 1}) = Complex(6.0, 0.0);
    MPSTensor adopted(good);
    EXPECT_EQ(adopted(1, 1, 1), Complex(6.0, 0.0));

    EXPECT_THROW(MPSTensor(Tensor({2, 2})), std::invalid_argument);
    EXPECT_THROW(MPSTensor(Tensor({2, 3, 2})), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// MPSState
// ---------------------------------------------------------------------------

TEST(MPSStateTest, ConstructsProductStates) {
    MPSState zero(3);
    EXPECT_EQ(zero.get_nqubits(), 3);
    EXPECT_EQ(zero.get_bond_dimension(0), 1);
    EXPECT_EQ(zero.get_max_bond_dimension_used(), 1);
    EXPECT_NEAR(zero.get_truncation_error(), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(zero.amplitude("000") - Complex(1.0, 0.0)), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(zero.amplitude("001")), 0.0, 1e-15);
    EXPECT_NEAR(zero.norm(), 1.0, 1e-15);

    MPSState mixed(3, "101");
    EXPECT_NEAR(std::abs(mixed.amplitude("101") - Complex(1.0, 0.0)), 0.0, 1e-15);
    EXPECT_EQ(mixed.get_site_tensor(0)(0, 1, 0), Complex(1.0, 0.0));

    // A single qubit state has no bonds at all
    MPSState lonely(1);
    EXPECT_EQ(lonely.get_max_bond_dimension_used(), 1);
    EXPECT_THROW(lonely.get_bond_dimension(0), std::out_of_range);

    EXPECT_THROW(MPSState(0), std::invalid_argument);
    EXPECT_THROW(MPSState(-2), std::invalid_argument);
    EXPECT_THROW(MPSState(3, "01"), std::invalid_argument);
    EXPECT_THROW(MPSState(3, "0x1"), std::invalid_argument);
    EXPECT_THROW(zero.get_bond_dimension(2), std::out_of_range);
    EXPECT_THROW(zero.get_bond_dimension(-1), std::out_of_range);
    EXPECT_THROW(zero.amplitude("00"), std::invalid_argument);
    EXPECT_THROW(zero.amplitude("0x0"), std::invalid_argument);
}

TEST(MPSStateTest, MoveCentreLeavesTheStateUnchanged) {
    MPSState state(4);
    state.set_truncation_cutoff(0.0);
    state.apply_one_site(hadamard(), 0);
    state.apply_two_site(DenseMatrix(random_unitary(4, 1)), 0);
    state.apply_two_site(DenseMatrix(random_unitary(4, 2)), 2);
    state.apply_two_site(DenseMatrix(random_unitary(4, 3)), 1);
    DenseMatrix before = state.as_dense();

    for (int q : {3, 0, 2, 1, 0}) {
        state.move_centre(q);
        EXPECT_NEAR((state.as_dense() - before).norm(), 0.0, 1e-12);
    }
    EXPECT_THROW(state.move_centre(4), std::out_of_range);
    EXPECT_THROW(state.move_centre(-1), std::out_of_range);
}

TEST(MPSStateTest, ApplyOneSiteMatchesTheStatevector) {
    MPSState state(2);
    state.apply_one_site(hadamard(), 0);
    EXPECT_NEAR(std::abs(state.amplitude("00") - Complex(1.0 / std::sqrt(2.0), 0.0)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(state.amplitude("10") - Complex(1.0 / std::sqrt(2.0), 0.0)), 0.0, 1e-12);
    EXPECT_NEAR(state.apply_one_site(pauli_x_matrix(), 1), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(state.amplitude("01") - Complex(1.0 / std::sqrt(2.0), 0.0)), 0.0, 1e-12);

    EXPECT_THROW(state.apply_one_site(DenseMatrix::Identity(4, 4), 0), std::invalid_argument);
    EXPECT_THROW(state.apply_one_site(hadamard(), 2), std::out_of_range);
    EXPECT_THROW(state.apply_one_site(hadamard(), -1), std::out_of_range);
}

TEST(MPSStateTest, ApplyTwoSiteEntanglesAndValidates) {
    MPSState state(2);
    state.apply_one_site(hadamard(), 0);

    // CNOT with qubit 0 the control, i.e. the most significant index
    DenseMatrix cnot = DenseMatrix::Zero(4, 4);
    cnot(0, 0) = 1.0;
    cnot(1, 1) = 1.0;
    cnot(2, 3) = 1.0;
    cnot(3, 2) = 1.0;
    EXPECT_NEAR(state.apply_two_site(cnot, 0), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(state.amplitude("00") - Complex(1.0 / std::sqrt(2.0), 0.0)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(state.amplitude("11") - Complex(1.0 / std::sqrt(2.0), 0.0)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(state.amplitude("01")), 0.0, 1e-12);
    EXPECT_EQ(state.get_bond_dimension(0), 2);

    EXPECT_THROW(state.apply_two_site(hadamard(), 0), std::invalid_argument);
    EXPECT_THROW(state.apply_two_site(cnot, 1), std::out_of_range);
    EXPECT_THROW(state.apply_two_site(cnot, -1), std::out_of_range);
}

TEST(MPSStateTest, ReproducesTheStatevectorForARandomCircuit) {
    const int nqubits = 5;
    std::vector<Gate> gates;
    gates.push_back(make_gate("U1", random_unitary(2, 11), {}, {0}));
    gates.push_back(make_gate("U2", random_unitary(4, 12), {}, {1, 2}));
    gates.push_back(make_gate("X", pauli_x_matrix(), {0}, {1}));
    gates.push_back(make_gate("U1", random_unitary(2, 13), {}, {3}));
    gates.push_back(make_gate("U2", random_unitary(4, 14), {}, {2, 3}));
    // Descending target order, so the gate matrix has to be reordered
    gates.push_back(make_gate("U2", random_unitary(4, 15), {}, {3, 2}));
    // Long-range pairs, routed with swaps
    gates.push_back(make_gate("X", pauli_x_matrix(), {0}, {4}));
    gates.push_back(make_gate("U2", random_unitary(4, 16), {}, {4, 1}));

    MPSState state(nqubits);
    state.set_max_bond_dimension(64);
    state.set_truncation_cutoff(0.0);
    Real error = 0.0;
    for (const Gate& gate : gates) {
        error += state.apply_gate(gate);
    }
    EXPECT_NEAR(error, 0.0, 1e-12);
    EXPECT_NEAR(state.get_truncation_error(), 0.0, 1e-12);

    DenseMatrix exact = statevector(gates, nqubits);
    EXPECT_NEAR((state.as_dense() - exact).norm(), 0.0, 1e-10);
    for (int index = 0; index < (1 << nqubits); ++index) {
        EXPECT_NEAR(std::abs(state.amplitude(bitstring(index, nqubits)) - exact(index, 0)), 0.0, 1e-10);
    }
    EXPECT_NEAR(state.norm(), 1.0, 1e-10);
}

TEST(MPSStateTest, ApplyGateRejectsWhatItCannotDo) {
    MPSState state(3);
    Gate three_qubit("X", to_sparse(pauli_x_matrix()), {0, 1}, {2}, {});
    EXPECT_THROW(state.apply_gate(three_qubit), std::invalid_argument);
    Gate out_of_range("X", to_sparse(pauli_x_matrix()), {}, {5}, {});
    EXPECT_THROW(state.apply_gate(out_of_range), std::out_of_range);
    Gate repeated_qubit("U2", to_sparse(random_unitary(4, 17)), {}, {1, 1}, {});
    EXPECT_THROW(state.apply_gate(repeated_qubit), std::invalid_argument);
}

TEST(MPSStateTest, TruncationIsReportedAndBounded) {
    const int nqubits = 4;
    std::vector<Gate> gates;
    for (int q = 0; q < nqubits; ++q) {
        gates.push_back(make_gate("U1", random_unitary(2, 20 + q), {}, {q}));
    }
    for (int q = 0; q + 1 < nqubits; ++q) {
        gates.push_back(make_gate("U2", random_unitary(4, 30 + q), {}, {q, q + 1}));
    }
    gates.push_back(make_gate("U2", random_unitary(4, 40), {}, {1, 2}));

    MPSState exact_state(nqubits);
    exact_state.set_max_bond_dimension(16);
    exact_state.set_truncation_cutoff(0.0);
    MPSState capped(nqubits);
    capped.set_max_bond_dimension(1);
    Real capped_error = 0.0;
    for (const Gate& gate : gates) {
        exact_state.apply_gate(gate);
        capped_error += capped.apply_gate(gate);
    }

    // At bond dimension one the state is forced to stay a product state, and the
    // discarded weight is what says so
    EXPECT_EQ(capped.get_max_bond_dimension_used(), 1);
    EXPECT_GT(capped_error, 0.0);
    EXPECT_NEAR(capped.get_truncation_error(), capped_error, 1e-12);
    EXPECT_GT(exact_state.get_max_bond_dimension_used(), 1);
    EXPECT_LT(std::abs(capped.overlap(exact_state)) / capped.norm(), 1.0);
}

TEST(MPSStateTest, NormalizeAndOverlap) {
    MPSState state(3);
    state.apply_one_site(hadamard(), 0);
    state.apply_one_site(DenseMatrix(3.0 * DenseMatrix::Identity(2, 2)), 1);
    EXPECT_NEAR(state.norm(), 3.0, 1e-12);
    state.normalize();
    EXPECT_NEAR(state.norm(), 1.0, 1e-12);

    MPSState other(3);
    other.apply_one_site(hadamard(), 0);
    EXPECT_NEAR(std::abs(state.overlap(other) - Complex(1.0, 0.0)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(state.overlap(MPSState(3, "111"))), 0.0, 1e-12);
    EXPECT_THROW(state.overlap(MPSState(2)), std::invalid_argument);

    // A state annihilated to zero has no norm to divide by
    MPSState annihilated(2);
    annihilated.apply_one_site(DenseMatrix::Zero(2, 2), 0);
    EXPECT_NEAR(annihilated.norm(), 0.0, 1e-15);
    EXPECT_THROW(annihilated.normalize(), std::runtime_error);
    EXPECT_THROW(annihilated.sample(), std::runtime_error);
    EXPECT_THROW(annihilated.expectation_value(pauli_z_matrix(), {0}), std::runtime_error);
    MatrixFreeHamiltonian h(2, PauliString(2, 'Z', 0), 1.0);
    EXPECT_THROW(annihilated.expectation_value(h), std::runtime_error);
}

TEST(MPSStateTest, ExpectationValuesMatchTheStatevector) {
    const int nqubits = 4;
    std::vector<Gate> gates;
    gates.push_back(make_gate("U1", random_unitary(2, 51), {}, {0}));
    gates.push_back(make_gate("U2", random_unitary(4, 52), {}, {0, 1}));
    gates.push_back(make_gate("U2", random_unitary(4, 53), {}, {2, 3}));
    gates.push_back(make_gate("U2", random_unitary(4, 54), {}, {1, 2}));

    MPSState state(nqubits);
    state.set_truncation_cutoff(0.0);
    for (const Gate& gate : gates) {
        state.apply_gate(gate);
    }
    DenseMatrix exact = statevector(gates, nqubits);

    // One-site observables on every qubit, and a two-site observable on every pair
    for (int q = 0; q < nqubits; ++q) {
        DenseMatrix full = DenseMatrix::Identity(1, 1);
        for (int i = 0; i < nqubits; ++i) {
            full = DenseMatrix(Eigen::kroneckerProduct(full, i == q ? pauli_z_matrix() : DenseMatrix(DenseMatrix::Identity(2, 2))).eval());
        }
        Complex expected = (exact.adjoint() * full * exact)(0, 0);
        EXPECT_NEAR(std::abs(state.expectation_value(pauli_z_matrix(), {q}) - expected), 0.0, 1e-10);
    }

    DenseMatrix observable = random_unitary(4, 55);
    observable = DenseMatrix(observable + observable.adjoint());
    for (int q = 0; q + 1 < nqubits; ++q) {
        DenseMatrix full = DenseMatrix::Identity(1, 1);
        for (int i = 0; i < nqubits;) {
            if (i == q) {
                full = DenseMatrix(Eigen::kroneckerProduct(full, observable).eval());
                i += 2;
            } else {
                full = DenseMatrix(Eigen::kroneckerProduct(full, DenseMatrix(DenseMatrix::Identity(2, 2))).eval());
                i += 1;
            }
        }
        Complex expected = (exact.adjoint() * full * exact)(0, 0);
        EXPECT_NEAR(std::abs(state.expectation_value(observable, {q, q + 1}) - expected), 0.0, 1e-10);
    }

    // The whole register at once, so the block spans the entire chain
    DenseMatrix everything = DenseMatrix(random_unitary(1 << nqubits, 56));
    everything = DenseMatrix(everything + everything.adjoint());
    Complex expected_all = (exact.adjoint() * everything * exact)(0, 0);
    EXPECT_NEAR(std::abs(state.expectation_value(everything, {0, 1, 2, 3}) - expected_all), 0.0, 1e-10);

    EXPECT_THROW(state.expectation_value(pauli_z_matrix(), {}), std::invalid_argument);
    EXPECT_THROW(state.expectation_value(observable, {0, 2}), std::invalid_argument);
    EXPECT_THROW(state.expectation_value(observable, {1, 0}), std::invalid_argument);
    EXPECT_THROW(state.expectation_value(observable, {0}), std::invalid_argument);
    EXPECT_THROW(state.expectation_value(pauli_z_matrix(), {4}), std::out_of_range);
    EXPECT_THROW(state.expectation_value(pauli_z_matrix(), {-1}), std::out_of_range);
}

TEST(MPSStateTest, HamiltonianExpectationMatchesTheStatevector) {
    const int nqubits = 3;
    std::vector<Gate> gates;
    gates.push_back(make_gate("U1", random_unitary(2, 61), {}, {0}));
    gates.push_back(make_gate("U2", random_unitary(4, 62), {}, {0, 1}));
    gates.push_back(make_gate("U2", random_unitary(4, 63), {}, {1, 2}));

    MPSState state(nqubits);
    state.set_truncation_cutoff(0.0);
    for (const Gate& gate : gates) {
        state.apply_gate(gate);
    }
    DenseMatrix exact = statevector(gates, nqubits);

    // Every single-qubit Pauli, plus a two-body and a three-body term, so that the
    // I / X / Y / Z branches of the local operator all get used
    MatrixFreeHamiltonian h(nqubits);
    h.add(0.5, PauliString(nqubits, 'X', 0));
    h.add(-1.25, PauliString(nqubits, 'Y', 1));
    h.add(2.0, PauliString(nqubits, 'Z', 2));
    PauliString zz(nqubits);
    zz.z_mask.set(0);
    zz.z_mask.set(1);
    h.add(0.75, zz);
    PauliString xyz(nqubits);
    xyz.x_mask.set(0);
    xyz.x_mask.set(1);
    xyz.z_mask.set(1);
    xyz.z_mask.set(2);
    h.add(-0.5, xyz);

    // The same Hamiltonian as a dense matrix, built from the Pauli factors directly
    DenseMatrix identity = DenseMatrix::Identity(2, 2);
    DenseMatrix pauli_y_matrix = DenseMatrix::Zero(2, 2);
    pauli_y_matrix(0, 1) = Complex(0.0, -1.0);
    pauli_y_matrix(1, 0) = Complex(0.0, 1.0);
    DenseMatrix dense_h = DenseMatrix::Zero(1 << nqubits, 1 << nqubits);
    for (const auto& term : h.get_operators()) {
        DenseMatrix full = DenseMatrix::Identity(1, 1);
        for (int q = 0; q < nqubits; ++q) {
            bool x = term.first.x_mask[q];
            bool z = term.first.z_mask[q];
            const DenseMatrix& factor = (x && z) ? pauli_y_matrix : (x ? pauli_x_matrix() : (z ? pauli_z_matrix() : identity));
            full = DenseMatrix(Eigen::kroneckerProduct(full, factor).eval());
        }
        dense_h += term.second * full;
    }
    Complex expected = (exact.adjoint() * dense_h * exact)(0, 0);
    EXPECT_NEAR(state.expectation_value(h), expected.real(), 1e-10);

    EXPECT_THROW(state.expectation_value(MatrixFreeHamiltonian(2)), std::invalid_argument);
}

TEST(MPSStateTest, SamplingFollowsTheExactDistribution) {
    const int nqubits = 3;
    MPSState state(nqubits);
    state.set_truncation_cutoff(0.0);
    state.apply_gate(make_gate("U1", random_unitary(2, 71), {}, {0}));
    state.apply_gate(make_gate("U2", random_unitary(4, 72), {}, {0, 1}));
    state.apply_gate(make_gate("U2", random_unitary(4, 73), {}, {1, 2}));
    state.normalize();

    const int nshots = 40000;
    state.set_seed(1234);
    std::map<std::string, int> counts = state.sample(nshots);
    int total = 0;
    for (const auto& pair : counts) {
        total += pair.second;
    }
    EXPECT_EQ(total, nshots);
    for (int index = 0; index < (1 << nqubits); ++index) {
        std::string b = bitstring(index, nqubits);
        Real expected = std::norm(state.amplitude(b));
        Real observed = counts.count(b) ? Real(counts[b]) / nshots : Real(0);
        EXPECT_NEAR(observed, expected, 0.02) << "outcome " << b;
    }

    // The same seed gives the same shots, and an explicit engine is honoured
    state.set_seed(1234);
    EXPECT_EQ(state.sample(nshots), counts);
    std::mt19937_64 engine(99);
    EXPECT_EQ(state.sample(engine).size(), size_t(nqubits));

    // A product state is deterministic where its amplitudes are
    MPSState product(2, "10");
    EXPECT_EQ(product.sample(), "10");

    EXPECT_THROW(state.sample(0), std::invalid_argument);
    EXPECT_THROW(state.sample(-1), std::invalid_argument);
}

TEST(MPSStateTest, AsDenseRefusesHugeRegisters) {
    MPSState small(2);
    EXPECT_EQ(small.as_dense().rows(), 4);
    EXPECT_EQ(small.as_dense().cols(), 1);
    // A product state of 25 qubits is cheap to hold as an MPS and impossible as a vector
    MPSState huge(25);
    EXPECT_THROW(huge.as_dense(), std::runtime_error);
}

TEST(MPSStateTest, StreamsASummary) {
    MPSState state(2);
    state.apply_one_site(hadamard(), 0);
    std::ostringstream os;
    os << state;
    EXPECT_NE(os.str().find("nqubits=2"), std::string::npos);
    EXPECT_NE(os.str().find("bond_dimensions="), std::string::npos);
    EXPECT_NE(os.str().find("truncation_error="), std::string::npos);
}

// GCOV_EXCL_BR_STOP
