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
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/matrix_free_hamiltonian.h"

namespace {

DenseMatrix ket0() {
    DenseMatrix s = DenseMatrix::Zero(2, 1);
    s(0, 0) = 1.0;
    return s;
}
DenseMatrix ket1() {
    DenseMatrix s = DenseMatrix::Zero(2, 1);
    s(1, 0) = 1.0;
    return s;
}
DenseMatrix ket00() {
    DenseMatrix s = DenseMatrix::Zero(4, 1);
    s(0, 0) = 1.0;
    return s;
}
DenseMatrix ket10() {
    DenseMatrix s = DenseMatrix::Zero(4, 1);
    s(2, 0) = 1.0;
    return s;
}
DenseMatrix ket11() {
    DenseMatrix s = DenseMatrix::Zero(4, 1);
    s(3, 0) = 1.0;
    return s;
}
DenseMatrix ketPlus() {
    DenseMatrix s = DenseMatrix::Zero(2, 1);
    s(0, 0) = 1.0 / std::sqrt(2.0);
    s(1, 0) = 1.0 / std::sqrt(2.0);
    return s;
}

DenseMatrix dm0() {
    DenseMatrix d = DenseMatrix::Zero(2, 2);
    d(0, 0) = 1.0;
    return d;
}
DenseMatrix dm1() {
    DenseMatrix d = DenseMatrix::Zero(2, 2);
    d(1, 1) = 1.0;
    return d;
}

void expectMatrixNear(const DenseMatrix& a, const DenseMatrix& b, double tol = 1e-10) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (int r = 0; r < a.rows(); ++r) {
        for (int c = 0; c < a.cols(); ++c) {
            EXPECT_NEAR(std::abs(a(r, c)), std::abs(b(r, c)), tol) << "Expected \n" << a << "\nto be near \n" << b << ",\nbut mismatch at (" << r << ", " << c << ") with values " << a(r, c) << " and " << b(r, c);
        }
    }
}

}  // namespace

TEST(MatrixFreeHamiltonian, DefaultConstructorIsEmpty) {
    MatrixFreeHamiltonian h(1);
    MatrixFreeHamiltonian h2(1);
    EXPECT_TRUE(h == h2);
}

TEST(MatrixFreeHamiltonian, ConstructFromSingleOperator) {
    MatrixFreeOperator z("Z", 0);
    MatrixFreeHamiltonian h(1, z);
    MatrixFreeHamiltonian h2(1);
    h2.add(std::complex<double>(1.0, 0.0), z);
    EXPECT_TRUE(h == h2);
}

TEST(MatrixFreeHamiltonian, ConstructFromVectorOfTerms) {
    MatrixFreeOperator z("Z", 0);
    MatrixFreeOperator x("X", 0);
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, z);
    h.add({2.0, 0.0}, x);
    MatrixFreeHamiltonian ref(1);
    ref.add({1.0, 0.0}, z);
    ref.add({2.0, 0.0}, x);
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, AddSingleOperator) {
    MatrixFreeHamiltonian h(1);
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian ref(1);
    ref.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, AddVectorOfOperators) {
    MatrixFreeHamiltonian h(2);
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    MatrixFreeHamiltonian ref(2);
    ref.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, AddMultipleTermsAccumulates) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    h.add({3.0, 0.0}, MatrixFreeOperator("I", 0));
    MatrixFreeHamiltonian ref(1);
    ref.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    ref.add({3.0, 0.0}, MatrixFreeOperator("I", 0));
    ref.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, EqualityEmptyHamiltonians) {
    EXPECT_TRUE(MatrixFreeHamiltonian(1) == MatrixFreeHamiltonian(1));
}

TEST(MatrixFreeHamiltonian, EqualityDifferentOrder) {
    MatrixFreeHamiltonian a(2), b(2);
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({1.0, 0.0}, {MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, EqualitySameTermsSameOrder) {
    MatrixFreeHamiltonian a(1), b(1);
    a.add({2.0, 1.0}, MatrixFreeOperator("Z", 0));
    b.add({2.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(a == b);
}

TEST(MatrixFreeHamiltonian, InequalityDifferentCoefficients) {
    MatrixFreeHamiltonian a(1), b(1);
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, InequalityDifferentNumberOfTerms) {
    MatrixFreeHamiltonian a(1), b(1);
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    a.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    b.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, InequalityDifferentOperators) {
    MatrixFreeHamiltonian a(1), b(1);
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, ScaleInPlaceByComplexScalar) {
    MatrixFreeHamiltonian h(1);
    h.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({1.0, 1.0}, MatrixFreeOperator("X", 0));
    std::complex<double> scalar{3.0, 0.0};
    h *= scalar;
    MatrixFreeHamiltonian ref(1);
    ref.add({6.0, 0.0}, MatrixFreeOperator("Z", 0));
    ref.add({3.0, 3.0}, MatrixFreeOperator("X", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, ScaleReturnsNewHamiltonian) {
    MatrixFreeHamiltonian h(1);
    h.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian scaled = h * std::complex<double>(0.5, 0.0);
    MatrixFreeHamiltonian ref(1);
    ref.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(scaled == ref);
    MatrixFreeHamiltonian original(1);
    original.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == original);
}

TEST(MatrixFreeHamiltonian, ScaleByImaginaryUnit) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h *= std::complex<double>(0.0, 1.0);
    MatrixFreeHamiltonian ref(1);
    ref.add({0.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, ScaleByZeroGivesZeroCoefficients) {
    MatrixFreeHamiltonian h(1);
    h.add({5.0, 3.0}, MatrixFreeOperator("Z", 0));
    h *= std::complex<double>(0.0, 0.0);
    MatrixFreeHamiltonian ref(1);
    ref.add({0.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, ScaleByRealDouble) {
    MatrixFreeHamiltonian h(1);
    h.add({4.0, 2.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian scaled = h * 0.5;
    MatrixFreeHamiltonian ref(1);
    ref.add({2.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(scaled == ref);
}

TEST(MatrixFreeHamiltonian, ScaleByRealDoesNotMutateOriginal) {
    MatrixFreeHamiltonian h(1);
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    auto _ = h * 10.0;
    MatrixFreeHamiltonian ref(1);
    ref.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, PlusEqualsAddsNewTerms) {
    MatrixFreeHamiltonian a(1), b(1);
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    a += b;
    MatrixFreeHamiltonian ref(1);
    ref.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    ref.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_TRUE(a == ref);
}

TEST(MatrixFreeHamiltonian, PlusEqualsMergesDuplicateTerms) {
    MatrixFreeHamiltonian a(1), b(1);
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    a += b;
    MatrixFreeHamiltonian ref(1);
    ref.add({4.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(a == ref);
}

TEST(MatrixFreeHamiltonian, PlusEqualsWithEmptyOtherIsNoop) {
    MatrixFreeHamiltonian a(1), empty(1);
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian before = a;
    a += empty;
    EXPECT_TRUE(a == before);
}

TEST(MatrixFreeHamiltonian, PlusEqualsAddingToEmptyHamiltonian) {
    MatrixFreeHamiltonian a(1), b(1);
    b.add({5.0, 0.0}, MatrixFreeOperator("X", 0));
    a += b;
    EXPECT_TRUE(a == b);
}

TEST(MatrixFreeHamiltonian, PlusEqualsMergesProductTermsByFullId) {
    MatrixFreeHamiltonian a(2), b(2);
    a.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    b.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("X", 1)});
    a += b;
    MatrixFreeHamiltonian ref(2);
    ref.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    ref.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("X", 1)});
    EXPECT_TRUE(a == ref);
}

TEST(MatrixFreeHamiltonian, ApplyZtoKet0GivesKet0) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix state = ket0();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket0());
}

TEST(MatrixFreeHamiltonian, ApplyZtoKet1GivesMinusKet1) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix state = ket1();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    DenseMatrix expected = ket1();
    expected *= -1.0;
    expectMatrixNear(output_state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyXtoKet0GivesKet1) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix state = ket0();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket1());
}

TEST(MatrixFreeHamiltonian, ApplyXtoKet1GivesKet0) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix state = ket1();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket0());
}

TEST(MatrixFreeHamiltonian, ApplyWithCoefficientScalesOutput) {
    MatrixFreeHamiltonian h(1);
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix state = ket0();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    DenseMatrix expected = ket0();
    expected *= 3.0;
    expectMatrixNear(output_state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyTwoTermsSummed) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix state = ket0();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 1.0;
    expected(1, 0) = 1.0;
    expectMatrixNear(output_state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyProductOfTwoOperators) {
    MatrixFreeHamiltonian h(2);
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("X", 1)});
    DenseMatrix state = ket00();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket11());
}

TEST(MatrixFreeHamiltonian, ApplyProductXZtoKet00) {
    MatrixFreeHamiltonian h(2);
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    DenseMatrix state = ket00();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket10());
}

TEST(MatrixFreeHamiltonian, ApplyProductZXtoKet10) {
    MatrixFreeHamiltonian h(2);
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("X", 1)});
    DenseMatrix state = ket10();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    DenseMatrix expected = ket11();
    expected *= -1.0;
    expectMatrixNear(output_state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyLeftToDensityMatrix) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix rho = dm0();
    DenseMatrix output_state;
    h.apply(rho, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, dm0());
}

TEST(MatrixFreeHamiltonian, ApplyRightToDensityMatrix) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix rho = dm0();
    DenseMatrix output_state;
    h.apply(rho, MatrixFreeApplicationType::Right, output_state);
    expectMatrixNear(output_state, dm0());
}

TEST(MatrixFreeHamiltonian, ApplyLeftAndRightToDensityMatrix) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix rho = dm0();
    DenseMatrix output_state;
    h.apply(rho, MatrixFreeApplicationType::LeftAndRight, output_state);
    expectMatrixNear(output_state, dm0());
}

TEST(MatrixFreeHamiltonian, ApplyLeftAndRightXonDm0) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix rho = dm0();
    DenseMatrix output_state;
    h.apply(rho, MatrixFreeApplicationType::LeftAndRight, output_state);
    expectMatrixNear(output_state, dm1());
}

TEST(MatrixFreeHamiltonian, ApplyTwoArgOverloadWritesToOutput) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix input = ket0();
    DenseMatrix output(2, 1);
    output.setZero();
    h.apply(input, MatrixFreeApplicationType::Left, output);
    expectMatrixNear(output, ket1());
}

TEST(MatrixFreeHamiltonian, ApplyTwoArgOverloadDoesNotMutateInput) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix input = ket0();
    DenseMatrix output(2, 1);
    output.setZero();
    h.apply(input, MatrixFreeApplicationType::Left, output);
    expectMatrixNear(input, ket0());
}

TEST(MatrixFreeHamiltonian, ApplyTwoArgOverloadResetsOutput) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix input = ket0();
    DenseMatrix output = ket0();
    h.apply(input, MatrixFreeApplicationType::Left, output);
    DenseMatrix expected = ket0();
    expectMatrixNear(output, expected);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZonKet0IsPlus1) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZonKet1IsMinues1) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket1()), -1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueXonKet0IsZero) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 0.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueXonKetPlusIs1) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_NEAR(h.expectation_value(ketPlus()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueWithRealCoefficientScales) {
    MatrixFreeHamiltonian h(1);
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 3.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueImaginaryCoefficientContributesZero) {
    MatrixFreeHamiltonian h(1);
    h.add({0.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 0.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueSumOfTerms) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZZonKetIsPlus1) {
    MatrixFreeHamiltonian h(2);
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_NEAR(h.expectation_value(ket00()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZZonSingleQubitRaisesError) {
    MatrixFreeHamiltonian h(2);
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_ANY_THROW(h.expectation_value(ket0()));
}

TEST(MatrixFreeHamiltonian, StreamOutputSingleTerm) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    std::ostringstream oss;
    oss << h;
    EXPECT_NE(oss.str().find("1"), std::string::npos);
    EXPECT_NE(oss.str().find("Z"), std::string::npos);
}

TEST(MatrixFreeHamiltonian, StreamOutputMultipleTermsContainsPlusSeparator) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({2.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("X", 1)});
    std::ostringstream oss;
    oss << h;
    EXPECT_NE(oss.str().find("+"), std::string::npos);
}

TEST(MatrixFreeHamiltonian, StreamOutputSingleTermHasNoPlusSeparator) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    std::ostringstream oss;
    oss << h;
    EXPECT_EQ(oss.str().find("+"), std::string::npos);
}

TEST(MatrixFreeHamiltonian, StreamOutputEmptyHamiltonian) {
    MatrixFreeHamiltonian h(1);
    std::ostringstream oss;
    EXPECT_NO_THROW(oss << h);
}

TEST(MatrixFreeHamiltonian, YActingOnKet0GivesKet1TimesI) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Y", 0));
    DenseMatrix state = ket0();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket1() * std::complex<double>(0.0, 1.0));
}

TEST(MatrixFreeHamiltonian, YActingOnKet1GivesKet0TimesMinusI) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("Y", 0));
    DenseMatrix state = ket1();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket0() * std::complex<double>(0.0, -1.0));
}

TEST(MatrixFreeHamiltonian, IActingOnKet0GivesKet0) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("I", 0));
    DenseMatrix state = ket0();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket0());
}

TEST(MatrixFreeHamiltonian, IActingOnKet1GivesKet1) {
    MatrixFreeHamiltonian h(1);
    h.add({1.0, 0.0}, MatrixFreeOperator("I", 0));
    DenseMatrix state = ket1();
    DenseMatrix output_state;
    h.apply(state, MatrixFreeApplicationType::Left, output_state);
    expectMatrixNear(output_state, ket1());
}

TEST(MatrixFreeHamiltonian, UnsupportedPauliThrowsError) {
    MatrixFreeHamiltonian h(1);
    EXPECT_ANY_THROW(h.add({1.0, 0.0}, MatrixFreeOperator("Q", 0)));
}

// --- PauliString construction and stream ---

TEST(PauliString, YConstructorSetsBothMasks) {
    PauliString ps(1, 'Y', 0);
    EXPECT_TRUE(ps.x_mask[0]);
    EXPECT_TRUE(ps.z_mask[0]);
}

TEST(PauliString, VectorConstructorSetsCorrectMasks) {
    std::vector<MatrixFreeOperator> ops = {MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)};
    PauliString ps(2, ops);
    EXPECT_TRUE(ps.x_mask[0]);
    EXPECT_FALSE(ps.z_mask[0]);
    EXPECT_FALSE(ps.x_mask[1]);
    EXPECT_TRUE(ps.z_mask[1]);
}

TEST(PauliString, StreamOutputContainsY) {
    PauliString ps(1, 'Y', 0);
    std::ostringstream oss;
    oss << ps;
    EXPECT_NE(oss.str().find("Y(0)"), std::string::npos);
}

TEST(PauliString, StreamOutputIdentityPrintsI) {
    PauliString ps(2);  // all zeros = identity
    std::ostringstream oss;
    oss << ps;
    EXPECT_EQ(oss.str(), "I");
}

// --- MatrixFreeHamiltonian missing methods ---

TEST(MatrixFreeHamiltonian, AddWithPauliString) {
    MatrixFreeHamiltonian h(1);
    PauliString ps(1, 'Z', 0);
    h.add(std::complex<double>(2.0, 0.0), ps);
    MatrixFreeHamiltonian ref(1);
    ref.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, SizeReturnsTermCount) {
    MatrixFreeHamiltonian h(1);
    EXPECT_EQ(h.size(), 0u);
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_EQ(h.size(), 1u);
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_EQ(h.size(), 2u);
}

TEST(MatrixFreeHamiltonian, Addition) {
    MatrixFreeHamiltonian h1(1), h2(1);
    h1.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h2.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    MatrixFreeHamiltonian sum = h1 + h2;
    EXPECT_EQ(sum.size(), 2u);
    MatrixFreeHamiltonian ref(1);
    ref.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    ref.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_TRUE(sum == ref);
}

TEST(MatrixFreeHamiltonian, Subtraction) {
    MatrixFreeHamiltonian h1(1), h2(1);
    h1.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    h2.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian diff = h1 - h2;
    MatrixFreeHamiltonian ref(1);
    ref.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(diff == ref);
}

TEST(MatrixFreeHamiltonian, HamiltonianMultiplicationXX_GivesIdentity) {
    // X * X = I (with coefficient 1)
    MatrixFreeHamiltonian H_X(1);
    H_X.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    MatrixFreeHamiltonian result = H_X * H_X;
    MatrixFreeHamiltonian expected(1);
    PauliString iden(1);  // identity: all masks zero
    expected.add({1.0, 0.0}, iden);
    EXPECT_TRUE(result == expected) << "Expected " << expected << " but got " << result;
}

TEST(MatrixFreeHamiltonian, HamiltonianMultiplicationXZ_GivesMinusIY) {
    // X * Z = -iY
    MatrixFreeHamiltonian H_X(1), H_Z(1);
    H_X.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    H_Z.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian result = H_X * H_Z;
    // X*Z = -iY: coefficient = -i, PauliString = Y(0)
    MatrixFreeHamiltonian expected(1);
    expected.add({0.0, -1.0}, MatrixFreeOperator("Y", 0));
    EXPECT_TRUE(result == expected) << "Expected " << expected << " but got " << result;
}

TEST(MatrixFreeHamiltonian, LeftScalarMultiplication) {
    MatrixFreeHamiltonian H(1);
    H.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    std::complex<double> scalar{3.0, 0.0};
    MatrixFreeHamiltonian result = scalar * H;
    MatrixFreeHamiltonian expected(1);
    expected.add({6.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(result == expected);
}

TEST(MatrixFreeHamiltonian, PruneByThresholdRemovesSmallTerms) {
    MatrixFreeHamiltonian H(1);
    H.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    H.add({0.001, 0.0}, MatrixFreeOperator("X", 0));
    H.prune(0.1, 10);
    EXPECT_EQ(H.size(), 1u);
}

TEST(MatrixFreeHamiltonian, PruneByMaxTermsKeepsLargest) {
    MatrixFreeHamiltonian H(1);
    H.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    H.add({0.5, 0.0}, MatrixFreeOperator("X", 0));
    H.prune(0.0, 1);
    EXPECT_EQ(H.size(), 1u);
}

TEST(MatrixFreeHamiltonian, ConjugateNegatesImaginaryPart) {
    MatrixFreeHamiltonian H(1);
    H.add({1.0, 2.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian conj = H.conjugate();
    for (const auto& [ps, coeff] : conj.get_operators()) {
        EXPECT_NEAR(coeff.real(), 1.0, 1e-12);
        EXPECT_NEAR(coeff.imag(), -2.0, 1e-12);
    }
}

TEST(MatrixFreeHamiltonian, ExpectationValueXOnPlusState) {
    // state = I|+> = |+>; <+|X|+> = 1
    MatrixFreeHamiltonian state(1);
    state.add({1.0, 0.0}, MatrixFreeOperator("I", 0));
    MatrixFreeHamiltonian H_X(1);
    H_X.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_NEAR(state.expectation_value(H_X), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZOnPlusState) {
    // state = I|+> = |+>; <+|Z|+> = 0
    MatrixFreeHamiltonian state(1);
    state.add({1.0, 0.0}, MatrixFreeOperator("I", 0));
    MatrixFreeHamiltonian H_Z(1);
    H_Z.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(state.expectation_value(H_Z), 0.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueIdentityOnPlusState) {
    // <+|I|+> = 1
    MatrixFreeHamiltonian state(1);
    state.add({1.0, 0.0}, MatrixFreeOperator("I", 0));
    MatrixFreeHamiltonian H_I(1);
    H_I.add({1.0, 0.0}, MatrixFreeOperator("I", 0));
    EXPECT_NEAR(state.expectation_value(H_I), 1.0, 1e-10);
}

// GCOV_EXCL_BR_STOP
