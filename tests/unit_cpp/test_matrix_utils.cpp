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

#include "../../src/qilisdk_cpp/backends/qilisim/utils/matrix_utils.h"

TEST(MatrixUtilsTest, DotProductKet) {
    SparseMatrix v1(3, 1);
    v1.insert(0, 0) = std::complex<double>(1.0, 2.0);
    v1.insert(1, 0) = std::complex<double>(3.0, 4.0);
    v1.insert(2, 0) = std::complex<double>(5.0, 6.0);
    SparseMatrix v2(3, 1);
    v2.insert(0, 0) = std::complex<double>(7.0, 8.0);
    v2.insert(1, 0) = std::complex<double>(9.0, 10.0);
    v2.insert(2, 0) = std::complex<double>(11.0, 12.0);
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
    std::complex<double> result = dot(v1, v2);
    std::complex<double> expected = 0.0;
    for (int i = 0; i < 3; ++i) {
        expected += std::conj(v1.coeff(0, i)) * v2.coeff(0, i);
    }
    EXPECT_NEAR(result.real(), expected.real(), 1e-6);
    EXPECT_NEAR(result.imag(), expected.imag(), 1e-6);
}
