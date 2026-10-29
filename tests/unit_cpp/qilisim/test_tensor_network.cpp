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

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "../../../src/qilisdk_cpp/backends/qilisim/digital/gate.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/bp.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/matrix_free_hamiltonian.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/mps.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/tensor.h"

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

// CNOT with the first qubit of the pair the control, i.e. the most significant index
DenseMatrix cnot_matrix() {
    DenseMatrix cnot = DenseMatrix::Zero(4, 4);
    cnot(0, 0) = 1.0;
    cnot(1, 1) = 1.0;
    cnot(2, 3) = 1.0;
    cnot(3, 2) = 1.0;
    return cnot;
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
    EXPECT_EQ(filled.raw().size(), 4);

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

TEST(TensorTest, PermuteMatchesTheIndexDefinitionForEveryPermutation) {
    // Every permutation of four legs covers merged runs, a contiguous fastest leg and a strided one
    const std::vector<int> shape = {2, 3, 4, 5};
    Tensor t(shape);
    for (int64_t flat = 0; flat < t.size(); ++flat) {
        t.raw()(flat) = Complex(static_cast<double>(flat), -0.5 * static_cast<double>(flat));
    }
    std::vector<int> perm = {0, 1, 2, 3};
    do {
        Tensor permuted = t.permute(perm);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 4; ++k) {
                    for (int l = 0; l < 5; ++l) {
                        std::vector<int> index = {i, j, k, l};
                        std::vector<int> moved(4);
                        for (int leg = 0; leg < 4; ++leg) {
                            moved[leg] = index[perm[leg]];
                        }
                        ASSERT_EQ(permuted(moved), t(index));
                    }
                }
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
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

    // A block of a taller matrix has an outer stride longer than its columns, which has to be honored
    DenseMatrix tall = DenseMatrix::Constant(4, 3, Complex(-1.0, 0.0));
    tall.topRows(2) = leading;
    Tensor from_block = Tensor::from_matrix(tall.topRows(2), {2, 3});
    EXPECT_NEAR((from_block.matrix_view(1) - leading).norm(), 0.0, 0.0);
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
    EXPECT_NEAR(std::abs(full(std::vector<int>{}) - a.contract_all(a)), 0.0, 1e-12);

    // Contracting against a leg that is not the other tensor's first has to reorder it
    Tensor d({4, 3});
    for (Complex& value : d.raw()) {
        value = Complex(uniform(engine), uniform(engine));
    }
    Tensor e = a.contract(d, {1}, {1});
    ASSERT_EQ(e.get_shape(), std::vector<int>({2, 2, 4}));
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 2; ++k) {
            for (int l = 0; l < 4; ++l) {
                Complex expected = 0.0;
                for (int j = 0; j < 3; ++j) {
                    expected += a(std::vector<int>{i, j, k}) * d(std::vector<int>{l, j});
                }
                EXPECT_NEAR(std::abs(e(std::vector<int>{i, k, l}) - expected), 0.0, 1e-12);
            }
        }
    }

    EXPECT_THROW(a.contract(b, {1}, {0, 1}), std::invalid_argument);
    EXPECT_THROW(a.contract(b, {0}, {0}), std::invalid_argument);
    EXPECT_THROW(a.contract(b, {5}, {0}), std::out_of_range);
}

TEST(TensorTest, SplitFactorizesAndReportsDiscardedWeight) {
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

    // Splitting on a leg that is not the leading one has to reorder before factorizing
    Tensor trailing_left;
    Tensor trailing_right;
    RealVector trailing_values;
    t.split({1}, 0, 0.0, trailing_left, trailing_values, trailing_right);
    EXPECT_EQ(trailing_values.size(), 2);
    ASSERT_EQ(trailing_left.get_shape(), std::vector<int>({3, 2}));
    ASSERT_EQ(trailing_right.get_shape(), std::vector<int>({2, 2}));
    DenseMatrix trailing = trailing_left.matrix_view(1) * trailing_values.cast<Complex>().asDiagonal() * trailing_right.matrix_view(1);
    EXPECT_NEAR((trailing - t.as_matrix({1})).norm(), 0.0, 1e-12);

    // A zero tensor has no weight at all to discard
    Tensor zeros({2, 2});
    zeros.split({0}, 0, 1e-10, capped_left, capped_values, capped_right, &capped_error);
    EXPECT_NEAR(capped_error, 0.0, 1e-15);
}

TEST(TensorTest, SplitCutoffIsExactBelowTheGramResolution) {
    // Singular values 1 and 1e-9, hidden behind a random unitary on each side
    DenseMatrix diagonal = DenseMatrix::Zero(2, 2);
    diagonal(0, 0) = 1.0;
    diagonal(1, 1) = 1e-9;
    DenseMatrix m = random_unitary(2, 61) * diagonal * random_unitary(2, 62);
    Tensor t = Tensor::from_matrix(m, {2, 2});
    Tensor left;
    Tensor right;
    RealVector values;

    // A cutoff below the small singular value keeps it, resolved by the full SVD
    t.split({0}, 0, 1e-10, left, values, right);
    ASSERT_EQ(values.size(), 2);
    EXPECT_NEAR(values(1), 1e-9, 1e-15);
    DenseMatrix reassembled = left.matrix_view(1) * values.cast<Complex>().asDiagonal() * right.matrix_view(1);
    EXPECT_NEAR((reassembled - m).norm(), 0.0, 1e-14);

    // A cutoff above it discards it, whichever factorization is used
    t.split({0}, 0, 1e-8, left, values, right);
    EXPECT_EQ(values.size(), 1);
    t.split({0}, 0, 1e-7, left, values, right);
    EXPECT_EQ(values.size(), 1);
}

TEST(TensorTest, ElementwiseHelpers) {
    Tensor t({2, 2}, {Complex(1.0, 1.0), Complex(2.0, 0.0), Complex(0.0, -1.0), Complex(1.0, 0.0)});
    Tensor conjugated = t.conjugate();
    EXPECT_EQ(conjugated(std::vector<int>{0, 0}), Complex(1.0, -1.0));
    EXPECT_EQ(conjugated(std::vector<int>{0, 1}), Complex(0.0, 1.0));

    EXPECT_NEAR(t.norm(), std::sqrt(2.0 + 4.0 + 1.0 + 1.0), 1e-12);

    Tensor other({2, 2}, {1.0, 1.0, 1.0, 1.0});
    EXPECT_NEAR(std::abs(t.contract_all(other) - Complex(4.0, 0.0)), 0.0, 1e-12);
    EXPECT_THROW(t.contract_all(Tensor({4})), std::invalid_argument);

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

TEST(MPSStateTest, MoveCenterLeavesTheStateUnchanged) {
    MPSState state(4);
    state.set_truncation_cutoff(0.0);
    state.apply_one_site(hadamard(), 0);
    state.apply_two_site(DenseMatrix(random_unitary(4, 1)), 0);
    state.apply_two_site(DenseMatrix(random_unitary(4, 2)), 2);
    state.apply_two_site(DenseMatrix(random_unitary(4, 3)), 1);
    DenseMatrix before = state.as_dense();

    for (int q : {3, 0, 2, 1, 0}) {
        state.move_center(q);
        EXPECT_NEAR((state.as_dense() - before).norm(), 0.0, 1e-12);
    }
    EXPECT_THROW(state.move_center(4), std::out_of_range);
    EXPECT_THROW(state.move_center(-1), std::out_of_range);
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

TEST(MPSStateTest, ApplyTwoSiteKeepsTheCenterOnEitherSite) {
    // Build the same Bell state twice, leaving the center on a different site each time
    Real amplitude = 1.0 / std::sqrt(2.0);
    MPSState right_centered(2);
    right_centered.apply_one_site(hadamard(), 0);
    EXPECT_NEAR(right_centered.apply_two_site(cnot_matrix(), 0), 0.0, 1e-15);
    MPSState left_centered(2);
    left_centered.apply_one_site(hadamard(), 0);
    EXPECT_NEAR(left_centered.apply_two_site(cnot_matrix(), 0, true), 0.0, 1e-15);
    EXPECT_EQ(right_centered.get_bond_dimension(0), 2);
    EXPECT_EQ(left_centered.get_bond_dimension(0), 2);

    // Both hold the same physical state
    for (const std::string& b : {"00", "01", "10", "11"}) {
        Complex expected = (b == "00" || b == "11") ? Complex(amplitude, 0.0) : Complex(0.0, 0.0);
        EXPECT_NEAR(std::abs(right_centered.amplitude(b) - expected), 0.0, 1e-12) << "bitstring " << b;
        EXPECT_NEAR(std::abs(left_centered.amplitude(b) - expected), 0.0, 1e-12) << "bitstring " << b;
    }
    EXPECT_NEAR(right_centered.norm(), 1.0, 1e-12);
    EXPECT_NEAR(left_centered.norm(), 1.0, 1e-12);

    // Center on the right means site 0 is left canonical, sum_p A^p dagger A^p = I
    const MPSTensor& left_site = right_centered.get_site_tensor(0);
    DenseMatrix left_gram = DenseMatrix::Zero(left_site.right(), left_site.right());
    for (int p = 0; p < MPSTensor::PHYSICAL_DIMENSION; ++p) {
        left_gram += left_site.physical_slice(p).adjoint() * left_site.physical_slice(p);
    }
    EXPECT_NEAR((left_gram - DenseMatrix::Identity(left_gram.rows(), left_gram.cols())).norm(), 0.0, 1e-12);

    // Center on the left means site 1 is right canonical instead, sum_p A^p A^p dagger = I
    const MPSTensor& right_site = left_centered.get_site_tensor(1);
    DenseMatrix right_gram = DenseMatrix::Zero(right_site.left(), right_site.left());
    for (int p = 0; p < MPSTensor::PHYSICAL_DIMENSION; ++p) {
        right_gram += right_site.physical_slice(p) * right_site.physical_slice(p).adjoint();
    }
    EXPECT_NEAR((right_gram - DenseMatrix::Identity(right_gram.rows(), right_gram.cols())).norm(), 0.0, 1e-12);
}

TEST(MPSStateTest, SettersRejectInvalidTruncation) {
    MPSState state(2);
    EXPECT_THROW(state.set_max_bond_dimension(0), std::invalid_argument);
    EXPECT_THROW(state.set_max_bond_dimension(-3), std::invalid_argument);
    EXPECT_THROW(state.set_truncation_cutoff(-1e-3), std::invalid_argument);
    EXPECT_THROW(state.set_truncation_cutoff(1.0), std::invalid_argument);
    EXPECT_THROW(state.set_truncation_cutoff(std::nan("")), std::invalid_argument);
    EXPECT_NO_THROW(state.set_max_bond_dimension(1));
    EXPECT_NO_THROW(state.set_truncation_cutoff(0.0));
}

TEST(MPSStateTest, TruncationErrorIsTheDiscardedSchmidtWeight) {
    // RY(theta) then CNOT gives cos(theta / 2) |00> + sin(theta / 2) |11>, whose Schmidt weights are known
    const Real theta = 0.6;
    DenseMatrix ry(2, 2);
    ry << std::cos(theta / 2), -std::sin(theta / 2), std::sin(theta / 2), std::cos(theta / 2);
    MPSState state(2);
    state.set_max_bond_dimension(1);
    state.apply_one_site(ry, 0);
    Real error = state.apply_two_site(cnot_matrix(), 0);
    Real expected = std::sin(theta / 2) * std::sin(theta / 2);
    EXPECT_NEAR(error, expected, 1e-12);
    EXPECT_NEAR(state.get_truncation_error(), expected, 1e-12);

    // Only the larger Schmidt vector survives, with its amplitude unchanged
    EXPECT_NEAR(std::abs(state.amplitude("00") - Complex(std::cos(theta / 2), 0.0)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(state.amplitude("11")), 0.0, 1e-12);
}

TEST(MPSStateTest, BondCapAboveOneHoldsOnEveryBond) {
    const int nqubits = 6;
    const int cap = 3;
    MPSState state(nqubits);
    state.set_max_bond_dimension(cap);
    for (int layer = 0; layer < 3; ++layer) {
        for (int q = 0; q + 1 < nqubits; ++q) {
            state.apply_two_site(random_unitary(4, 70 + 10 * layer + q), q);
        }
    }
    EXPECT_EQ(state.get_max_bond_dimension_used(), cap);
    for (int bond = 0; bond + 1 < nqubits; ++bond) {
        EXPECT_LE(state.get_bond_dimension(bond), cap) << "bond " << bond;
    }
    EXPECT_GT(state.get_truncation_error(), 0.0);
}

TEST(MPSStateTest, NonUnitaryGatesOnEntangledAndDistantQubitsMatchTheStatevector) {
    const int nqubits = 4;
    std::vector<Gate> gates;
    gates.push_back(make_gate("H", hadamard(), {}, {0}));
    gates.push_back(make_gate("X", pauli_x_matrix(), {0}, {1}));
    gates.push_back(make_gate("U2", random_unitary(4, 81), {}, {1, 2}));

    // A damping on a qubit that is entangled with its neighbours
    DenseMatrix damping = DenseMatrix::Zero(2, 2);
    damping(0, 0) = 1.0;
    damping(1, 1) = 0.3;
    gates.push_back(make_gate("D", damping, {}, {1}));

    // A generic non-unitary two-qubit matrix on a distant pair, which has to be routed with swaps
    DenseMatrix generic = random_unitary(4, 82) * 0.5 + random_unitary(4, 83) * 0.25;
    gates.push_back(make_gate("G", generic, {}, {3, 0}));

    MPSState state(nqubits);
    state.set_truncation_cutoff(0.0);
    for (const Gate& gate : gates) {
        state.apply_gate(gate);
    }
    EXPECT_NEAR((state.as_dense() - statevector(gates, nqubits)).norm(), 0.0, 1e-10);
}

TEST(MPSStateTest, NaNPropagatesInsteadOfLookingLikeZero) {
    MPSState state(2);
    DenseMatrix poisoned = hadamard();
    poisoned(0, 1) = Complex(std::nan(""), 0.0);
    state.apply_one_site(poisoned, 0);
    EXPECT_TRUE(std::isnan(state.norm()));
    EXPECT_THROW(state.normalize(), std::runtime_error);
    EXPECT_THROW(state.sample(), std::runtime_error);
}

TEST(MPSStateTest, StreamsASummary) {
    MPSState state(2);
    state.apply_one_site(hadamard(), 0);
    std::ostringstream os;
    os << state;
    EXPECT_NE(os.str().find("nqubits=2"), std::string::npos);
    EXPECT_NE(os.str().find("bond_dimensions="), std::string::npos);
    EXPECT_NE(os.str().find("truncation_error="), std::string::npos);

    // A single bond needs no separator, so three qubits to exercise the comma between bonds
    MPSState chain(3);
    std::ostringstream chain_os;
    chain_os << chain;
    EXPECT_NE(chain_os.str().find("nqubits=3"), std::string::npos);
    EXPECT_NE(chain_os.str().find("bond_dimensions=[1, 1]"), std::string::npos);

    // Entangling the first pair grows only that bond, so the two entries differ
    chain.apply_one_site(hadamard(), 0);
    chain.apply_two_site(cnot_matrix(), 0);
    std::ostringstream entangled_os;
    entangled_os << chain;
    EXPECT_NE(entangled_os.str().find("bond_dimensions=[2, 1]"), std::string::npos);
}

// ---------------------------------------------------------------------------
// BPState
// ---------------------------------------------------------------------------

namespace {

// Apply a circuit to a fresh tensor network with no truncation
BPState bp_state(const std::vector<Gate>& gates, int nqubits, int max_bond_dimension = 64) {
    BPState state(nqubits);
    state.set_max_bond_dimension(max_bond_dimension);
    state.set_truncation_cutoff(0.0);
    for (const Gate& gate : gates) {
        state.apply_gate(gate);
    }
    state.run_belief_propagation();
    return state;
}

// A layer of random one-qubit gates followed by a random two-qubit gate on every edge, repeated
std::vector<Gate> random_graph_circuit(int nqubits, const std::vector<std::pair<int, int>>& edges, int layers, uint64_t seed) {
    std::vector<Gate> gates;
    for (int layer = 0; layer < layers; ++layer) {
        for (int q = 0; q < nqubits; ++q) {
            gates.push_back(make_gate("U1", random_unitary(2, seed++), {}, {q}));
        }
        for (const auto& edge : edges) {
            gates.push_back(make_gate("U2", random_unitary(4, seed++), {}, {edge.first, edge.second}));
        }
    }
    return gates;
}

// The dense matrix of a Pauli Hamiltonian, with qubit 0 the most significant
DenseMatrix dense_hamiltonian(const MatrixFreeHamiltonian& h, int nqubits) {
    DenseMatrix identity = DenseMatrix::Identity(2, 2);
    DenseMatrix pauli_y = DenseMatrix::Zero(2, 2);
    pauli_y(0, 1) = Complex(0.0, -1.0);
    pauli_y(1, 0) = Complex(0.0, 1.0);
    DenseMatrix dense = DenseMatrix::Zero(1 << nqubits, 1 << nqubits);
    for (const auto& term : h.get_operators()) {
        DenseMatrix full = DenseMatrix::Identity(1, 1);
        for (int q = 0; q < nqubits; ++q) {
            bool x = term.first.x_mask[q];
            bool z = term.first.z_mask[q];
            const DenseMatrix& factor = (x && z) ? pauli_y : (x ? pauli_x_matrix() : (z ? pauli_z_matrix() : identity));
            full = DenseMatrix(Eigen::kroneckerProduct(full, factor).eval());
        }
        dense += term.second * full;
    }
    return dense;
}

// A Pauli string from a map of qubit to 'X', 'Y' or 'Z'
PauliString pauli_string(int nqubits, const std::map<int, char>& factors) {
    PauliString pauli(nqubits);
    for (const auto& factor : factors) {
        if (factor.second != 'Z') {
            pauli.x_mask.set(factor.first);
        }
        if (factor.second != 'X') {
            pauli.z_mask.set(factor.first);
        }
    }
    return pauli;
}

Real exact_expectation(const DenseMatrix& state, const MatrixFreeHamiltonian& h, int nqubits) {
    DenseMatrix normalized = state / state.norm();
    return (normalized.adjoint() * dense_hamiltonian(h, nqubits) * normalized)(0, 0).real();
}

}  // namespace

TEST(BPStateTest, ConstructsProductStates) {
    BPState zero(3);
    EXPECT_EQ(zero.get_nqubits(), 3);
    EXPECT_EQ(zero.get_num_edges(), 0);
    EXPECT_EQ(zero.get_max_bond_dimension_used(), 1);
    DenseMatrix expected_zero = DenseMatrix::Zero(8, 1);
    expected_zero(0, 0) = 1.0;
    EXPECT_NEAR((zero.as_dense() - expected_zero).norm(), 0.0, 1e-14);

    BPState bits(3, "101");
    DenseMatrix expected_bits = DenseMatrix::Zero(8, 1);
    expected_bits(5, 0) = 1.0;
    EXPECT_NEAR((bits.as_dense() - expected_bits).norm(), 0.0, 1e-14);
    EXPECT_EQ(bits.get_site_tensor(0).get_shape(), std::vector<int>({2}));

    EXPECT_THROW(BPState(0), std::invalid_argument);
    EXPECT_THROW(BPState(-2), std::invalid_argument);
    EXPECT_THROW(BPState(2, "1"), std::invalid_argument);
    EXPECT_THROW(BPState(2, "12"), std::invalid_argument);
}

TEST(BPStateTest, SettersValidateTheirArguments) {
    BPState state(2);
    EXPECT_NO_THROW(state.set_max_bond_dimension(3));
    EXPECT_NO_THROW(state.set_truncation_cutoff(0.0));
    EXPECT_NO_THROW(state.set_max_iterations(1));
    EXPECT_NO_THROW(state.set_tolerance(1e-3));
    EXPECT_THROW(state.set_max_bond_dimension(0), std::invalid_argument);
    for (Real cutoff : {-1e-3, 1.0, std::nan("")}) {
        EXPECT_THROW(state.set_truncation_cutoff(cutoff), std::invalid_argument) << "cutoff " << cutoff;
    }
    EXPECT_THROW(state.set_max_iterations(0), std::invalid_argument);
    for (Real tolerance : {0.0, -1.0, std::nan("")}) {
        EXPECT_THROW(state.set_tolerance(tolerance), std::invalid_argument) << "tolerance " << tolerance;
    }
}

TEST(BPStateTest, ReproducesTheStatevectorOnATree) {
    // A star with a branch, so qubit 0 has degree three and the gauge has several legs to handle
    const int nqubits = 6;
    std::vector<std::pair<int, int>> tree = {{0, 1}, {0, 2}, {0, 3}, {3, 4}, {4, 5}};
    std::vector<Gate> gates = random_graph_circuit(nqubits, tree, 2, 100);
    BPState state = bp_state(gates, nqubits);
    EXPECT_EQ(state.get_num_edges(), 5);
    EXPECT_NEAR(state.get_truncation_error(), 0.0, 1e-12);
    EXPECT_GT(state.get_max_bond_dimension_used(), 1);
    DenseMatrix exact = statevector(gates, nqubits);
    EXPECT_NEAR(std::abs((state.as_dense().adjoint() * exact)(0, 0)), 1.0, 1e-10);
}

TEST(BPStateTest, ReproducesTheStatevectorOnLoopyGraphsWithoutTruncation) {
    // Gauging and ungauging are exact, so even on loops an untruncated network holds the exact state
    const int nqubits = 5;
    std::vector<std::pair<int, int>> ring = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}};
    std::vector<std::pair<int, int>> complete = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
    for (const auto& edges : {ring, complete}) {
        std::vector<Gate> gates = random_graph_circuit(nqubits, edges, 2, 200);
        BPState state = bp_state(gates, nqubits);
        DenseMatrix exact = statevector(gates, nqubits);
        EXPECT_NEAR(std::abs((state.as_dense().adjoint() * exact)(0, 0)), 1.0, 1e-9);
    }
}

TEST(BPStateTest, GateQubitOrderIsRespected) {
    // A controlled gate whose control has the higher index, which the local matrix orders ascending
    const int nqubits = 3;
    std::vector<Gate> gates;
    gates.push_back(make_gate("H", hadamard(), {}, {2}));
    gates.push_back(make_gate("X", pauli_x_matrix(), {2}, {0}));
    gates.push_back(make_gate("U2", random_unitary(4, 301), {}, {2, 1}));
    gates.push_back(make_gate("U2", random_unitary(4, 302), {}, {1, 0}));
    BPState state = bp_state(gates, nqubits);
    DenseMatrix exact = statevector(gates, nqubits);
    EXPECT_NEAR(std::abs((state.as_dense().adjoint() * exact)(0, 0)), 1.0, 1e-10);
}

TEST(BPStateTest, ExpectationValuesAreExactOnATree) {
    const int nqubits = 6;
    std::vector<std::pair<int, int>> tree = {{0, 1}, {1, 2}, {1, 3}, {3, 4}, {4, 5}};
    std::vector<Gate> gates = random_graph_circuit(nqubits, tree, 2, 400);
    BPState state = bp_state(gates, nqubits);
    DenseMatrix exact = statevector(gates, nqubits);

    // Single-site terms of every Pauli, a term on an edge, terms joined only through a path, and a three-body term
    MatrixFreeHamiltonian h(nqubits);
    h.add(0.5, pauli_string(nqubits, {{0, 'X'}}));
    h.add(-1.25, pauli_string(nqubits, {{2, 'Y'}}));
    h.add(2.0, pauli_string(nqubits, {{5, 'Z'}}));
    h.add(0.75, pauli_string(nqubits, {{3, 'Z'}, {4, 'Z'}}));
    h.add(-0.4, pauli_string(nqubits, {{0, 'X'}, {5, 'Y'}}));
    h.add(0.3, pauli_string(nqubits, {{2, 'Z'}, {4, 'X'}, {5, 'Z'}}));
    h.add(1.5, PauliString(nqubits));
    EXPECT_NEAR(state.expectation_value(h), exact_expectation(exact, h, nqubits), 1e-10);

    // Each term on its own too, so a single failing cluster is visible
    for (const auto& term : h.get_operators()) {
        MatrixFreeHamiltonian single(nqubits);
        single.add(1.0, term.first);
        EXPECT_NEAR(state.expectation_value(term.first).real(), exact_expectation(exact, single, nqubits), 1e-10) << term.first;
    }
    EXPECT_NEAR(std::abs(state.expectation_value(PauliString(nqubits)) - Complex(1.0, 0.0)), 0.0, 1e-14);
    EXPECT_THROW(state.expectation_value(MatrixFreeHamiltonian(nqubits + 1)), std::invalid_argument);
}

TEST(BPStateTest, ClusterPathsCanRunThroughLaterSupportQubits) {
    // In the chain 0 - 2 - 1 the path joining qubit 1 to qubit 0 already covers qubit 2
    const int nqubits = 3;
    std::vector<Gate> gates = random_graph_circuit(nqubits, {{0, 2}, {2, 1}}, 2, 450);
    BPState state = bp_state(gates, nqubits);
    DenseMatrix exact = statevector(gates, nqubits);
    MatrixFreeHamiltonian h(nqubits);
    h.add(1.0, pauli_string(nqubits, {{0, 'Z'}, {1, 'X'}, {2, 'Y'}}));
    EXPECT_NEAR(state.expectation_value(h), exact_expectation(exact, h, nqubits), 1e-10);
}

TEST(BPStateTest, ExpectationValuesFactorizeAcrossDisconnectedPieces) {
    // Two Bell pairs that never interact, and a lone qubit, so clusters cannot be joined by a path
    const int nqubits = 5;
    std::vector<Gate> gates;
    gates.push_back(make_gate("U2", random_unitary(4, 501), {}, {0, 1}));
    gates.push_back(make_gate("U2", random_unitary(4, 502), {}, {2, 3}));
    gates.push_back(make_gate("U1", random_unitary(2, 503), {}, {4}));
    BPState state = bp_state(gates, nqubits);
    DenseMatrix exact = statevector(gates, nqubits);
    MatrixFreeHamiltonian h(nqubits);
    h.add(1.0, pauli_string(nqubits, {{0, 'Z'}, {2, 'X'}, {4, 'Y'}}));
    h.add(0.5, pauli_string(nqubits, {{1, 'Y'}, {3, 'Z'}}));
    EXPECT_NEAR(state.expectation_value(h), exact_expectation(exact, h, nqubits), 1e-10);
}

TEST(BPStateTest, ExpectationValuesAreApproximateButCloseOnAWeaklyEntangledLoop) {
    // Small rotations keep loop correlations weak, so belief propagation is close but not exact
    const int nqubits = 6;
    std::vector<Gate> gates;
    DenseMatrix rx = DenseMatrix::Identity(2, 2) * std::cos(0.4) - Complex(0.0, std::sin(0.4)) * pauli_x_matrix();
    DenseMatrix zz = DenseMatrix::Zero(4, 4);
    for (int index = 0; index < 4; ++index) {
        Real sign = (index == 0 || index == 3) ? 1.0 : -1.0;
        zz(index, index) = std::exp(Complex(0.0, -0.3 * sign));
    }
    for (int step = 0; step < 3; ++step) {
        for (int q = 0; q < nqubits; ++q) {
            gates.push_back(make_gate("U1", rx, {}, {q}));
        }
        for (int q = 0; q < nqubits; ++q) {
            gates.push_back(make_gate("U2", zz, {}, {q, (q + 1) % nqubits}));
        }
    }
    BPState state = bp_state(gates, nqubits);
    DenseMatrix exact = statevector(gates, nqubits);
    MatrixFreeHamiltonian h(nqubits);
    for (int q = 0; q < nqubits; ++q) {
        h.add(1.0, pauli_string(nqubits, {{q, 'Z'}, {(q + 1) % nqubits, 'Z'}}));
        h.add(1.0, pauli_string(nqubits, {{q, 'X'}}));
    }
    EXPECT_NEAR(state.expectation_value(h), exact_expectation(exact, h, nqubits), 5e-2);
    EXPECT_TRUE(state.get_bp_converged());
}

TEST(BPStateTest, BeliefPropagationRunsOncePerLayerAndReportsConvergence) {
    BPState state(4);
    EXPECT_EQ(state.run_belief_propagation(), 0);

    // Two gates on disjoint pairs form one layer, so the second needs no refresh
    state.apply_two_site(random_unitary(4, 601), 0, 1);
    state.apply_two_site(random_unitary(4, 602), 2, 3);
    EXPECT_EQ(state.get_bp_runs(), 0);

    // A gate on a qubit already used in the layer starts a new one
    state.apply_two_site(random_unitary(4, 603), 1, 2);
    EXPECT_EQ(state.get_bp_runs(), 1);
    EXPECT_GE(state.get_bp_iterations(), 1);

    // Unitary one-qubit gates leave the messages alone, so nothing needs refreshing
    state.run_belief_propagation();
    int runs = state.get_bp_runs();
    int iterations = state.get_bp_iterations();
    state.apply_one_site(random_unitary(2, 604), 0);
    state.run_belief_propagation();
    EXPECT_EQ(state.get_bp_runs(), runs + 1);
    EXPECT_EQ(state.get_bp_iterations(), iterations + 1);
    EXPECT_TRUE(state.get_bp_converged());

    // On a loop a single sweep is not enough, which is remembered
    BPState loop(3);
    loop.set_max_iterations(1);
    loop.set_tolerance(1e-15);
    for (const Gate& gate : random_graph_circuit(3, {{0, 1}, {1, 2}, {2, 0}}, 2, 610)) {
        loop.apply_gate(gate);
    }
    loop.run_belief_propagation();
    EXPECT_FALSE(loop.get_bp_converged());
}

TEST(BPStateTest, TruncationIsReportedAndBounded) {
    // A bond of one cannot hold a Bell pair, so half the weight is discarded
    BPState state(2);
    state.set_max_bond_dimension(1);
    state.apply_one_site(hadamard(), 0);
    Real error = state.apply_two_site(cnot_matrix(), 0, 1);
    EXPECT_NEAR(error, 0.5, 1e-12);
    EXPECT_NEAR(state.get_truncation_error(), 0.5, 1e-12);
    EXPECT_EQ(state.get_max_bond_dimension_used(), 1);
    EXPECT_NEAR(state.as_dense().norm(), 1.0, 1e-12);

    // A tighter cap can only lose more, and a large enough one is exact
    const int nqubits = 6;
    std::vector<std::pair<int, int>> chain = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
    std::vector<Gate> gates = random_graph_circuit(nqubits, chain, 3, 700);
    DenseMatrix exact = statevector(gates, nqubits);
    Real previous = 0.0;
    for (int chi : {1, 2, 4, 8}) {
        BPState capped = bp_state(gates, nqubits, chi);
        Real fidelity = std::norm((capped.as_dense().adjoint() * exact)(0, 0));
        EXPECT_GE(fidelity, previous - 1e-10) << "chi " << chi;
        EXPECT_LE(capped.get_max_bond_dimension_used(), chi);
        previous = fidelity;
    }
    EXPECT_NEAR(previous, 1.0, 1e-10);
}

TEST(BPStateTest, ApplyGateRejectsWhatItCannotDo) {
    BPState state(3);
    Gate three_qubit("X", to_sparse(pauli_x_matrix()), {0, 1}, {2}, {});
    EXPECT_THROW(state.apply_gate(three_qubit), std::invalid_argument);
    EXPECT_THROW(state.apply_one_site(cnot_matrix(), 0), std::invalid_argument);
    EXPECT_THROW(state.apply_one_site(hadamard(), 3), std::out_of_range);
    EXPECT_THROW(state.apply_one_site(hadamard(), -1), std::out_of_range);
    EXPECT_THROW(state.apply_two_site(hadamard(), 0, 1), std::invalid_argument);
    EXPECT_THROW(state.apply_two_site(cnot_matrix(), 0, 3), std::out_of_range);
    EXPECT_THROW(state.apply_two_site(cnot_matrix(), -1, 0), std::out_of_range);
    EXPECT_THROW(state.apply_two_site(cnot_matrix(), 1, 1), std::invalid_argument);
    EXPECT_EQ(state.get_num_edges(), 0);
}

TEST(BPStateTest, NonUnitaryGatesMatchTheNormalizedStatevector) {
    const int nqubits = 4;
    DenseMatrix damping = DenseMatrix::Zero(2, 2);
    damping(0, 0) = 1.0;
    damping(1, 1) = 0.3;
    std::vector<Gate> gates = random_graph_circuit(nqubits, {{0, 1}, {1, 2}, {2, 3}}, 1, 800);
    gates.push_back(make_gate("D", damping, {}, {1}));
    gates.push_back(make_gate("U2", random_unitary(4, 810), {}, {1, 2}));
    gates.push_back(make_gate("D", damping, {}, {2}));
    gates.push_back(make_gate("U2", random_unitary(4, 811), {}, {0, 1}));
    BPState state = bp_state(gates, nqubits);
    DenseMatrix exact = statevector(gates, nqubits);
    EXPECT_NEAR(std::abs((state.as_dense().adjoint() * exact)(0, 0)) / exact.norm(), 1.0, 1e-10);
    MatrixFreeHamiltonian h(nqubits);
    h.add(1.0, pauli_string(nqubits, {{1, 'Z'}}));
    h.add(1.0, pauli_string(nqubits, {{0, 'X'}, {3, 'X'}}));
    EXPECT_NEAR(state.expectation_value(h), exact_expectation(exact, h, nqubits), 1e-10);
}

TEST(BPStateTest, ZeroStateHasNoExpectationValue) {
    BPState state(2);
    state.apply_one_site(DenseMatrix::Zero(2, 2), 1);
    state.run_belief_propagation();
    EXPECT_THROW(state.expectation_value(pauli_string(2, {{1, 'Z'}})), std::runtime_error);

    // A gate that zeroes the whole bond leaves uniform messages rather than NaN ones
    BPState bond(2);
    bond.apply_two_site(DenseMatrix::Zero(4, 4), 0, 1);
    bond.run_belief_propagation();
    EXPECT_FALSE(bond.get_site_tensor(0).has_nan());
    EXPECT_THROW(bond.expectation_value(pauli_string(2, {{0, 'Z'}})), std::runtime_error);
}

TEST(BPStateTest, ExpectationValuesNeedFreshMessages) {
    // After a two-qubit gate the messages around it are stale until belief propagation runs again
    BPState state(3);
    state.apply_two_site(cnot_matrix(), 0, 1);
    EXPECT_THROW(state.expectation_value(pauli_string(3, {{0, 'Z'}})), std::logic_error);
    state.run_belief_propagation();
    EXPECT_NEAR(state.expectation_value(pauli_string(3, {{0, 'Z'}})).real(), 1.0, 1e-12);

    // A unitary one-qubit gate leaves the messages valid, a non-unitary one does not
    state.apply_one_site(hadamard(), 2);
    EXPECT_NO_THROW(state.expectation_value(pauli_string(3, {{2, 'X'}})));
    state.apply_one_site(DenseMatrix::Identity(2, 2) * 2.0, 2);
    EXPECT_THROW(state.expectation_value(pauli_string(3, {{2, 'X'}})), std::logic_error);
}

TEST(BPStateTest, AsDenseRefusesHugeRegisters) {
    BPState state(BPState::MAX_DENSE_QUBITS + 1);
    EXPECT_THROW(state.as_dense(), std::runtime_error);
}

TEST(BPStateTest, NaNReachesTheSiteTensor) {
    BPState state(2);
    DenseMatrix poisoned = hadamard();
    poisoned(0, 1) = Complex(std::nan(""), 0.0);
    state.apply_one_site(poisoned, 0);
    EXPECT_TRUE(state.get_site_tensor(0).has_nan());
    EXPECT_FALSE(state.get_site_tensor(1).has_nan());
}

TEST(BPStateTest, StreamsASummary) {
    BPState state(3);
    state.apply_one_site(hadamard(), 0);
    state.apply_two_site(cnot_matrix(), 0, 2);
    std::ostringstream os;
    os << state;
    EXPECT_NE(os.str().find("nqubits=3"), std::string::npos);
    EXPECT_NE(os.str().find("edges=1"), std::string::npos);
    EXPECT_NE(os.str().find("max_bond_dimension_used=2"), std::string::npos);
    EXPECT_NE(os.str().find("truncation_error="), std::string::npos);
    EXPECT_NE(os.str().find("bp_runs=0"), std::string::npos);
}

// GCOV_EXCL_BR_STOP
