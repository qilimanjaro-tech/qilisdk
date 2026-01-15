// Copyright 2025 Qilimanjaro Quantum Tech
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "qilisim.h"
namespace py = pybind11;

SparseMatrix QiliSimCpp::from_numpy(const py::buffer& matrix_buffer) const {
    /*
    Convert a numpy array buffer to a SparseMatrix.

    Args:
        matrix_buffer (py::buffer): The numpy array buffer.

    Returns:
        SparseMatrix: The converted sparse matrix.
    */
    py::buffer_info buf = matrix_buffer.request();
    if (buf.ndim != 2) {
        throw py::value_error("Input array must be 2D.");
    }
    int rows = int(buf.shape[0]);
    int cols = int(buf.shape[1]);
    auto ptr = static_cast<std::complex<double>*>(buf.ptr);
    Triplets entries;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::complex<double> val = ptr[r * cols + c];
            if (std::abs(val) > atol_) {
                entries.emplace_back(Triplet(r, c, val));
            }
        }
    }
    SparseMatrix mat(rows, cols);
    mat.setFromTriplets(entries.begin(), entries.end());
    return mat;
}

py::array_t<std::complex<double>> QiliSimCpp::to_numpy(const SparseMatrix& matrix) const {
    /*
    Convert a SparseMatrix to a NumPy array.

    Args:
        matrix (SparseMatrix): The input sparse matrix.

    Returns:
        py::array_t<std::complex<double>>: The corresponding NumPy array.
    */
    int rows = int(matrix.rows());
    int cols = int(matrix.cols());
    py::array_t<std::complex<double>> np_array({rows, cols});
    py::buffer_info buf = np_array.request();
    auto ptr = static_cast<std::complex<double>*>(buf.ptr);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ptr[r * cols + c] = matrix.coeff(r, c);
        }
    }
    return np_array;
}

py::array_t<double> QiliSimCpp::to_numpy(const std::vector<double>& vec) const {
    /*
    Convert a vector of complex numbers to a NumPy array.

    Args:
        vec (std::vector<double>): The input vector.

    Returns:
        py::array_t<double>: The corresponding NumPy array.
    */
    int size = vec.size();
    py::array_t<double> np_array({size});
    py::buffer_info buf = np_array.request();
    auto ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < size; ++i) {
        ptr[i] = vec[i];
    }
    return np_array;
}

py::array_t<double> QiliSimCpp::to_numpy(const std::vector<std::vector<double>>& vecs) const {
    /*
    Convert a vector of vectors of complex numbers to a 2D NumPy array.

    Args:
        vecs (std::vector<std::vector<double>>): The input vector of vectors.

    Returns:
        py::array_t<double>: The corresponding 2D NumPy array.
    */
    int rows = vecs.size();
    int cols = vecs[0].size();
    py::array_t<double> np_array({rows, cols});
    py::buffer_info buf = np_array.request();
    auto ptr = static_cast<double*>(buf.ptr);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ptr[r * cols + c] = vecs[r][c];
        }
    }
    return np_array;
}
