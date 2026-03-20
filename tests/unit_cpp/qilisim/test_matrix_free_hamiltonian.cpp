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
DenseMatrix dmPlus() {
    DenseMatrix d = DenseMatrix::Zero(2, 2);
    d(0, 0) = 0.5;
    d(0, 1) = 0.5;
    d(1, 0) = 0.5;
    d(1, 1) = 0.5;
    return d;
}

void expectMatrixNear(const DenseMatrix& a, const DenseMatrix& b, double tol = 1e-10) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (int r = 0; r < a.rows(); ++r)
        for (int c = 0; c < a.cols(); ++c)
            EXPECT_NEAR(std::abs(a(r, c)), std::abs(b(r, c)), tol)
                << "Mismatch at (" << r << ", " << c << ")";
}

}

TEST(MatrixFreeHamiltonian, DefaultConstructorIsEmpty) {
    MatrixFreeHamiltonian h;
    MatrixFreeHamiltonian h2;
    EXPECT_TRUE(h == h2);
}

TEST(MatrixFreeHamiltonian, ConstructFromSingleOperator) {
    MatrixFreeOperator z("Z", 0);
    MatrixFreeHamiltonian h(z);
    MatrixFreeHamiltonian h2;
    h2.add(std::complex<double>(1.0, 0.0), z);
    EXPECT_TRUE(h == h2);
}

TEST(MatrixFreeHamiltonian, ConstructFromVectorOfTerms) {
    MatrixFreeOperator z("Z", 0);
    MatrixFreeOperator x("X", 0);
    std::vector<std::pair<std::complex<double>, std::vector<MatrixFreeOperator>>> terms = {
        { {1.0, 0.0}, {z} },
        { {2.0, 0.0}, {x} }
    };
    MatrixFreeHamiltonian h(terms);
    MatrixFreeHamiltonian ref;
    ref.add({1.0, 0.0}, z);
    ref.add({2.0, 0.0}, x);
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, AddSingleOperator) {
    MatrixFreeHamiltonian h;
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian ref;
    ref.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, AddVectorOfOperators) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    MatrixFreeHamiltonian ref;
    ref.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, AddMultipleTermsAccumulates) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    h.add({3.0, 0.0}, MatrixFreeOperator("I", 0));
    MatrixFreeHamiltonian ref;
    ref.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    ref.add({3.0, 0.0}, MatrixFreeOperator("I", 0));
    ref.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, EqualityEmptyHamiltonians) {
    EXPECT_TRUE(MatrixFreeHamiltonian() == MatrixFreeHamiltonian());
}

TEST(MatrixFreeHamiltonian, EqualityDifferentOrder) {
    MatrixFreeHamiltonian a, b;
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({1.0, 0.0}, {MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, EqualitySameTermsSameOrder) {
    MatrixFreeHamiltonian a, b;
    a.add({2.0, 1.0}, MatrixFreeOperator("Z", 0));
    b.add({2.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(a == b);
}

TEST(MatrixFreeHamiltonian, InequalityDifferentCoefficients) {
    MatrixFreeHamiltonian a, b;
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, InequalityDifferentNumberOfTerms) {
    MatrixFreeHamiltonian a, b;
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    a.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    b.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, InequalityDifferentOperators) {
    MatrixFreeHamiltonian a, b;
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_FALSE(a == b);
}

TEST(MatrixFreeHamiltonian, ScaleInPlaceByComplexScalar) {
    MatrixFreeHamiltonian h;
    h.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({1.0, 1.0}, MatrixFreeOperator("X", 0));
    std::complex<double> scalar{3.0, 0.0};
    h *= scalar;
    MatrixFreeHamiltonian ref;
    ref.add({6.0, 0.0}, MatrixFreeOperator("Z", 0));
    ref.add({3.0, 3.0}, MatrixFreeOperator("X", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, ScaleReturnsNewHamiltonian) {
    MatrixFreeHamiltonian h;
    h.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian scaled = h * std::complex<double>(0.5, 0.0);
    MatrixFreeHamiltonian ref;
    ref.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(scaled == ref);
    MatrixFreeHamiltonian original;
    original.add({2.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == original);
}

TEST(MatrixFreeHamiltonian, ScaleByImaginaryUnit) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h *= std::complex<double>(0.0, 1.0);
    MatrixFreeHamiltonian ref;
    ref.add({0.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, ScaleByZeroGivesZeroCoefficients) {
    MatrixFreeHamiltonian h;
    h.add({5.0, 3.0}, MatrixFreeOperator("Z", 0));
    h *= std::complex<double>(0.0, 0.0);
    MatrixFreeHamiltonian ref;
    ref.add({0.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, ScaleByRealDouble) {
    MatrixFreeHamiltonian h;
    h.add({4.0, 2.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian scaled = h * 0.5;
    MatrixFreeHamiltonian ref;
    ref.add({2.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(scaled == ref);
}

TEST(MatrixFreeHamiltonian, ScaleByRealDoesNotMutateOriginal) {
    MatrixFreeHamiltonian h;
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    auto _ = h * 10.0;
    MatrixFreeHamiltonian ref;
    ref.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_TRUE(h == ref);
}

TEST(MatrixFreeHamiltonian, PlusEqualsAddsNewTerms) {
    MatrixFreeHamiltonian a, b;
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    a += b;
    MatrixFreeHamiltonian ref;
    ref.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    ref.add({2.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_TRUE(a == ref);
}

TEST(MatrixFreeHamiltonian, PlusEqualsMergesDuplicateTerms) {
    MatrixFreeHamiltonian a, b;
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    b.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    a += b;
    MatrixFreeHamiltonian ref;
    ref.add({4.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(a.get_operators()[0].first.real(), 4.0, 1e-10);
    EXPECT_NEAR(a.get_operators()[0].first.imag(), 0.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, PlusEqualsWithEmptyOtherIsNoop) {
    MatrixFreeHamiltonian a, empty;
    a.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    MatrixFreeHamiltonian before = a;
    a += empty;
    EXPECT_TRUE(a == before);
}

TEST(MatrixFreeHamiltonian, PlusEqualsAddingToEmptyHamiltonian) {
    MatrixFreeHamiltonian a, b;
    b.add({5.0, 0.0}, MatrixFreeOperator("X", 0));
    a += b;
    EXPECT_TRUE(a == b);
}

TEST(MatrixFreeHamiltonian, PlusEqualsMergesProductTermsByFullId) {
    MatrixFreeHamiltonian a, b;
    a.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    b.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("X", 1)});
    a += b;
    MatrixFreeHamiltonian ref;
    ref.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    ref.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("X", 1)});
    EXPECT_TRUE(a == ref);
}

TEST(MatrixFreeHamiltonian, ApplyZtoKet0GivesKet0) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix state = ket0();
    h.apply(state, MatrixFreeApplicationType::Left);
    expectMatrixNear(state, ket0());
}

TEST(MatrixFreeHamiltonian, ApplyZtoKet1GivesMinusKet1) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix state = ket1();
    h.apply(state, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ket1();
    expected *= -1.0;
    expectMatrixNear(state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyXtoKet0GivesKet1) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix state = ket0();
    h.apply(state, MatrixFreeApplicationType::Left);
    expectMatrixNear(state, ket1());
}

TEST(MatrixFreeHamiltonian, ApplyXtoKet1GivesKet0) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix state = ket1();
    h.apply(state, MatrixFreeApplicationType::Left);
    expectMatrixNear(state, ket0());
}

TEST(MatrixFreeHamiltonian, ApplyWithCoefficientScalesOutput) {
    MatrixFreeHamiltonian h;
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix state = ket0();
    h.apply(state, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ket0();
    expected *= 3.0;
    expectMatrixNear(state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyTwoTermsSummed) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix state = ket0();
    h.apply(state, MatrixFreeApplicationType::Left);
    DenseMatrix expected(2, 1);
    expected(0, 0) = 1.0;
    expected(1, 0) = 1.0;
    expectMatrixNear(state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyProductOfTwoOperators) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("X", 1)});
    DenseMatrix state = ket00();
    h.apply(state, MatrixFreeApplicationType::Left);
    expectMatrixNear(state, ket11());
}

TEST(MatrixFreeHamiltonian, ApplyProductXZtoKet00) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    DenseMatrix state = ket00();
    h.apply(state, MatrixFreeApplicationType::Left);
    expectMatrixNear(state, ket10());
}

TEST(MatrixFreeHamiltonian, ApplyProductZXtoKet10) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("X", 1)});
    DenseMatrix state = ket10();
    h.apply(state, MatrixFreeApplicationType::Left);
    DenseMatrix expected = ket11();
    expected *= -1.0;
    expectMatrixNear(state, expected);
}

TEST(MatrixFreeHamiltonian, ApplyModifiesStateInPlace) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix state = ket0();
    const DenseMatrix* ptr_before = &state;
    h.apply(state, MatrixFreeApplicationType::Left);
    EXPECT_EQ(&state, ptr_before);
    expectMatrixNear(state, ket1());
}

TEST(MatrixFreeHamiltonian, ApplyLeftToDensityMatrix) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix rho = dm0();
    h.apply(rho, MatrixFreeApplicationType::Left);
    expectMatrixNear(rho, dm0());
}

TEST(MatrixFreeHamiltonian, ApplyRightToDensityMatrix) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix rho = dm0();
    h.apply(rho, MatrixFreeApplicationType::Right);
    expectMatrixNear(rho, dm0());
}

TEST(MatrixFreeHamiltonian, ApplyLeftAndRightToDensityMatrix) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix rho = dm0();
    h.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm0());
}

TEST(MatrixFreeHamiltonian, ApplyLeftAndRightXonDm0) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix rho = dm0();
    h.apply(rho, MatrixFreeApplicationType::LeftAndRight);
    expectMatrixNear(rho, dm1());
}

TEST(MatrixFreeHamiltonian, ApplyTwoArgOverloadWritesToOutput) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix input = ket0();
    DenseMatrix output(2, 1);
    output.setZero();
    h.apply(input, MatrixFreeApplicationType::Left, output);
    expectMatrixNear(output, ket1());
}

TEST(MatrixFreeHamiltonian, ApplyTwoArgOverloadDoesNotMutateInput) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    DenseMatrix input = ket0();
    DenseMatrix output(2, 1);
    output.setZero();
    h.apply(input, MatrixFreeApplicationType::Left, output);
    expectMatrixNear(input, ket0());
}

TEST(MatrixFreeHamiltonian, ApplyTwoArgOverloadAccumulatesIntoOutput) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    DenseMatrix input = ket0();
    DenseMatrix output = ket0();
    h.apply(input, MatrixFreeApplicationType::Left, output);
    DenseMatrix expected = ket0();
    expected *= 2.0;
    expectMatrixNear(output, expected);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZonKet0IsPlus1) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZonKet1IsMinues1) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket1()), -1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueXonKet0IsZero) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 0.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueXonKetPlusIs1) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_NEAR(h.expectation_value(ketPlus()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueCNOTOnKet00IsPlus1) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0, 1)});
    EXPECT_NEAR(h.expectation_value(ket00()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueWithRealCoefficientScales) {
    MatrixFreeHamiltonian h;
    h.add({3.0, 0.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 3.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueImaginaryCoefficientContributesZero) {
    MatrixFreeHamiltonian h;
    h.add({0.0, 1.0}, MatrixFreeOperator("Z", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 0.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueSumOfTerms) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({1.0, 0.0}, MatrixFreeOperator("X", 0));
    EXPECT_NEAR(h.expectation_value(ket0()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZZonKetIsPlus1) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_NEAR(h.expectation_value(ket00()), 1.0, 1e-10);
}

TEST(MatrixFreeHamiltonian, ExpectationValueZZonSingleQubitRaisesError) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_ANY_THROW(h.expectation_value(ket0()));
}


TEST(MatrixFreeHamiltonian, StreamOutputSingleTerm) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    std::ostringstream oss;
    oss << h;
    EXPECT_NE(oss.str().find("1"), std::string::npos);
    EXPECT_NE(oss.str().find("Z"), std::string::npos);
}

TEST(MatrixFreeHamiltonian, StreamOutputMultipleTermsContainsPlusSeparator) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    h.add({2.0, 0.0}, std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("X", 1)});
    std::ostringstream oss;
    oss << h;
    EXPECT_NE(oss.str().find("+"), std::string::npos);
}

TEST(MatrixFreeHamiltonian, StreamOutputSingleTermHasNoPlusSeparator) {
    MatrixFreeHamiltonian h;
    h.add({1.0, 0.0}, MatrixFreeOperator("Z", 0));
    std::ostringstream oss;
    oss << h;
    EXPECT_EQ(oss.str().find("+"), std::string::npos);
}

TEST(MatrixFreeHamiltonian, StreamOutputEmptyHamiltonian) {
    MatrixFreeHamiltonian h;
    std::ostringstream oss;
    EXPECT_NO_THROW(oss << h);
}