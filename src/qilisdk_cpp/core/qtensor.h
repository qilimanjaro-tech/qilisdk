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
#pragma once

#include "../libs/pybind.h"
#include "../libs/eigen.h"

// The main QiliSim C++ class
class QTensorCpp {
    private:
        SparseMatrix _data;
    public:
        QTensorCpp() {}
        QTensorCpp(const SparseMatrix& data) : _data(data) {}
        QTensorCpp(const py::object& data);
        QTensorCpp(int rows, int cols) : _data(rows, cols) {}
        const SparseMatrix& get_data() const { return _data; }
        py::object get_data_as_scipy() const;
        py::object get_data_as_numpy() const;
        int get_nqubits() const;
        std::pair<int, int> get_shape() const;
        bool is_ket() const;
        bool is_bra() const;
        bool is_operator() const;
        bool is_scalar() const;
        bool is_density_matrix(double atol) const;
        bool is_self_adjoint(double atol) const;
        bool is_positive_semidefinite(double atol) const;
        QTensorCpp conjugate() const;
        QTensorCpp transpose() const;
        QTensorCpp adjoint() const;
        QTensorCpp exponential() const;
        std::complex<double> trace() const;
        QTensorCpp partial_trace(const std::vector<int>& keep) const;
        double norm(const std::string& norm_type) const;
        QTensorCpp normalized(const std::string& norm_type) const;
        QTensorCpp ket(const std::string& bitstring) const;
        QTensorCpp bra(const std::string& bitstring) const;
        QTensorCpp tensor_product_python(const py::list& others) const;
        QTensorCpp tensor_product(const std::vector<QTensorCpp>& others) const;
        QTensorCpp add_python(const py::object& other) const;
        QTensorCpp add(const QTensorCpp& other) const;
        QTensorCpp sub_python(const py::object& other) const;
        QTensorCpp sub(const QTensorCpp& other) const;
        QTensorCpp mul_python(const py::object& other) const;
        QTensorCpp mul(const QTensorCpp& other) const;
        QTensorCpp matmul_python(const py::object& other) const;
        QTensorCpp matmul(const QTensorCpp& other) const;
        QTensorCpp to_density_matrix() const;
        QTensorCpp identity(int dim) const;
        bool equals_python(const py::object& other) const;
        bool equals(const QTensorCpp& other) const;
        std::string as_string() const;

};
