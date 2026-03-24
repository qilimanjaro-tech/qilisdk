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
#include <pybind11/embed.h>
#include "../../../src/qilisdk_cpp/libs/numpy.h"

namespace py = pybind11;

TEST(MatrixConversion, FromNumpyBasic2x2Identity) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        identity_2x2 = np.eye(2, dtype=complex)
    )");
    py::object np_matrix = py::globals()["identity_2x2"];
    SparseMatrix result = from_numpy(np_matrix, 1e-10);
    ASSERT_EQ(result.rows(), 2);
    ASSERT_EQ(result.cols(), 2);
    EXPECT_NEAR(std::abs(result.coeff(0, 0) - std::complex<double>(1.0, 0.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 1) - std::complex<double>(1.0, 0.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(0, 1)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 0)), 0.0, 1e-10);
}

TEST(MatrixConversion, FromNumpyComplexValues) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        complex_matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=complex)
    )");
    py::object np_matrix = py::globals()["complex_matrix"];
    SparseMatrix result = from_numpy(np_matrix, 1e-10);
    ASSERT_EQ(result.rows(), 2);
    ASSERT_EQ(result.cols(), 2);
    EXPECT_NEAR(std::abs(result.coeff(0, 0) - std::complex<double>(1.0, 2.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(0, 1) - std::complex<double>(3.0, 4.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 0) - std::complex<double>(5.0, 6.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 1) - std::complex<double>(7.0, 8.0)), 0.0, 1e-10);
}

TEST(MatrixConversion, FromNumpyAtolFiltersSmallValues) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        nearly_diagonal = np.array([[1.0+0j, 0.01+0j], [0.01+0j, 1.0+0j]], dtype=complex)
    )");
    py::object np_matrix = py::globals()["nearly_diagonal"];
    SparseMatrix result = from_numpy(np_matrix, 0.1);
    EXPECT_NEAR(std::abs(result.coeff(0, 1)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(0, 0) - 1.0), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 1) - 1.0), 0.0, 1e-10);
}

TEST(MatrixConversion, FromNumpyAtolKeepsValuesAboveThreshold) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        nearly_diagonal = np.array([[1.0+0j, 0.01+0j], [0.01+0j, 1.0+0j]], dtype=complex)
    )");
    py::object np_matrix = py::globals()["nearly_diagonal"];
    SparseMatrix result = from_numpy(np_matrix, 0.001);
    EXPECT_NEAR(std::abs(result.coeff(0, 1) - 0.01), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 0) - 0.01), 0.0, 1e-10);
}

TEST(MatrixConversion, FromNumpyThrowsOn1DInput) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        vector_1d = np.array([1.0, 2.0, 3.0], dtype=complex)
    )");
    py::object np_vector = py::globals()["vector_1d"];
    EXPECT_THROW(from_numpy(np_vector, 1e-10), py::value_error);
}

TEST(MatrixConversion, FromNumpyAllZerosMatrix) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        zero_matrix = np.zeros((3, 3), dtype=complex)
    )");
    py::object np_matrix = py::globals()["zero_matrix"];
    SparseMatrix result = from_numpy(np_matrix, 1e-10);
    ASSERT_EQ(result.rows(), 3);
    ASSERT_EQ(result.cols(), 3);
    EXPECT_EQ(result.nonZeros(), 0);
}

TEST(MatrixConversion, FromNumpyNonSquareMatrix) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        rect_matrix = np.array([[1+0j, 2+0j, 3+0j], [4+0j, 5+0j, 6+0j]], dtype=complex)
    )");
    py::object np_matrix = py::globals()["rect_matrix"];
    SparseMatrix result = from_numpy(np_matrix, 1e-10);
    ASSERT_EQ(result.rows(), 2);
    ASSERT_EQ(result.cols(), 3);
    EXPECT_NEAR(std::abs(result.coeff(0, 2) - 3.0), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 1) - 5.0), 0.0, 1e-10);
}

TEST(MatrixConversion, SparseToNumpyShape) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        mat = np.eye(4, dtype=complex)
    )");
    py::object np_matrix = py::globals()["mat"];
    SparseMatrix sparse = from_numpy(np_matrix, 1e-10);
    py::array_t<std::complex<double>> result = to_numpy(sparse);
    py::buffer_info buf = result.request();
    ASSERT_EQ(buf.ndim, 2);
    EXPECT_EQ(buf.shape[0], 4);
    EXPECT_EQ(buf.shape[1], 4);
}

TEST(MatrixConversion, DenseToNumpyValues) {
    DenseMatrix dense(2, 2);
    dense(0, 0) = std::complex<double>(1.0, 0.0);
    dense(0, 1) = std::complex<double>(0.0, 1.0);
    dense(1, 0) = std::complex<double>(-1.0, 0.0);
    dense(1, 1) = std::complex<double>(0.0, -1.0);
    py::gil_scoped_acquire gil;
    py::array_t<std::complex<double>> result = to_numpy(dense);
    py::buffer_info buf = result.request();
    auto ptr = static_cast<std::complex<double>*>(buf.ptr);
    EXPECT_NEAR(std::abs(ptr[0] - std::complex<double>(1.0, 0.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(ptr[1] - std::complex<double>(0.0, 1.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(ptr[2] - std::complex<double>(-1.0, 0.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(ptr[3] - std::complex<double>(0.0, -1.0)), 0.0, 1e-10);
}

TEST(MatrixConversion, DenseToNumpyShape) {
    DenseMatrix dense(3, 5);
    dense.setZero();
    py::gil_scoped_acquire gil;
    py::array_t<std::complex<double>> result = to_numpy(dense);
    py::buffer_info buf = result.request();
    ASSERT_EQ(buf.ndim, 2);
    EXPECT_EQ(buf.shape[0], 3);
    EXPECT_EQ(buf.shape[1], 5);
}

TEST(MatrixConversion, VectorToNumpyValues) {
    std::vector<double> vec = {1.0, 2.5, -3.0, 0.0, 42.0};
    py::gil_scoped_acquire gil;
    py::array_t<double> result = to_numpy(vec);
    py::buffer_info buf = result.request();
    auto ptr = static_cast<double*>(buf.ptr);
    ASSERT_EQ(buf.shape[0], 5);
    EXPECT_NEAR(ptr[0], 1.0, 1e-10);
    EXPECT_NEAR(ptr[1], 2.5, 1e-10);
    EXPECT_NEAR(ptr[2], -3.0, 1e-10);
    EXPECT_NEAR(ptr[3], 0.0, 1e-10);
    EXPECT_NEAR(ptr[4], 42.0, 1e-10);
}

TEST(MatrixConversion, VectorToNumpyEmpty) {
    std::vector<double> vec;
    py::gil_scoped_acquire gil;
    py::array_t<double> result = to_numpy(vec);
    py::buffer_info buf = result.request();
    EXPECT_EQ(buf.shape[0], 0);
}

TEST(MatrixConversion, VectorOfVectorsToNumpyValues) {
    std::vector<std::vector<double>> vecs = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    py::gil_scoped_acquire gil;
    py::array_t<double> result = to_numpy(vecs);
    py::buffer_info buf = result.request();
    auto ptr = static_cast<double*>(buf.ptr);
    ASSERT_EQ(buf.ndim, 2);
    EXPECT_EQ(buf.shape[0], 2);
    EXPECT_EQ(buf.shape[1], 3);
    EXPECT_NEAR(ptr[0], 1.0, 1e-10);
    EXPECT_NEAR(ptr[4], 5.0, 1e-10);
    EXPECT_NEAR(ptr[5], 6.0, 1e-10);
}

TEST(MatrixConversion, VectorOfVectorsToNumpyShape) {
    std::vector<std::vector<double>> vecs(4, std::vector<double>(7, 0.0));
    py::gil_scoped_acquire gil;
    py::array_t<double> result = to_numpy(vecs);
    py::buffer_info buf = result.request();
    EXPECT_EQ(buf.shape[0], 4);
    EXPECT_EQ(buf.shape[1], 7);
}

TEST(MatrixConversion, FromSpmatrixIdentity) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp
        import numpy as np
        sp_identity = sp.eye(3, dtype=complex, format='csr')
    )");
    py::object sp_matrix = py::globals()["sp_identity"];
    SparseMatrix result = from_spmatrix(sp_matrix, 1e-10);
    ASSERT_EQ(result.rows(), 3);
    ASSERT_EQ(result.cols(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(std::abs(result.coeff(i, i) - 1.0), 0.0, 1e-10);
    }
    EXPECT_NEAR(std::abs(result.coeff(0, 1)), 0.0, 1e-10);
}

TEST(MatrixConversion, FromSpmatrixAtolFiltersValues) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp
        import numpy as np
        data   = np.array([1.0+0j, 0.005+0j, 0.005+0j, 1.0+0j], dtype=complex)
        row    = np.array([0, 0, 1, 1])
        col    = np.array([0, 1, 0, 1])
        sp_small_offdiag = sp.coo_matrix((data, (row, col)), shape=(2, 2))
    )");
    py::object sp_matrix = py::globals()["sp_small_offdiag"];
    SparseMatrix result = from_spmatrix(sp_matrix, 0.01);
    EXPECT_NEAR(std::abs(result.coeff(0, 1)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(0, 0) - 1.0), 0.0, 1e-10);
}

TEST(MatrixConversion, FromSpmatrixComplexValues) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp
        import numpy as np
        data = np.array([1+2j, 3+4j], dtype=complex)
        row  = np.array([0, 1])
        col  = np.array([1, 0])
        sp_complex = sp.coo_matrix((data, (row, col)), shape=(2, 2))
    )");
    py::object sp_matrix = py::globals()["sp_complex"];
    SparseMatrix result = from_spmatrix(sp_matrix, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(0, 1) - std::complex<double>(1.0, 2.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(1, 0) - std::complex<double>(3.0, 4.0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.coeff(0, 0)), 0.0, 1e-10);
}

TEST(MatrixConversion, ToSpmatrixNonZeroCount) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        # 3x3 with exactly 4 non-zeros
        nnz_input = np.array([
            [1+0j, 0+0j, 2+0j],
            [0+0j, 0+0j, 0+0j],
            [3+0j, 0+0j, 4+0j]
        ], dtype=complex)
    )");
    py::object np_matrix = py::globals()["nnz_input"];
    SparseMatrix sparse = from_numpy(np_matrix, 1e-10);
    py::object sp_result = to_spmatrix(sparse);
    int nnz = sp_result.attr("nnz").cast<int>();
    EXPECT_EQ(nnz, 4);
}

TEST(MatrixConversion, ToSpmatrixShape) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        shape_input = np.eye(5, dtype=complex)
    )");
    py::object np_matrix = py::globals()["shape_input"];
    SparseMatrix sparse = from_numpy(np_matrix, 1e-10);
    py::object sp_result = to_spmatrix(sparse);
    py::object shape = sp_result.attr("shape");
    int rows = shape.attr("__getitem__")(0).cast<int>();
    int cols = shape.attr("__getitem__")(1).cast<int>();
    EXPECT_EQ(rows, 5);
    EXPECT_EQ(cols, 5);
}

// GCOV_EXCL_BR_STOP