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

#include "../libs/pybind.h"
#include "qtensor.h"

PYBIND11_MODULE(qtensor_module, m) {

    // Make the QTensor class available in Python as well as the various methods
    py::class_<QTensorCpp>(m, "QTensorCpp")
        .def("get_data_as_scipy", &QTensorCpp::get_data_as_scipy)
        .def("get_data_as_numpy", &QTensorCpp::get_data_as_numpy)
        .def("get_nqubits", &QTensorCpp::get_nqubits)
        .def("get_shape", &QTensorCpp::get_shape)
        .def("is_ket", &QTensorCpp::is_ket)
        .def("is_bra", &QTensorCpp::is_bra)
        .def("is_operator", &QTensorCpp::is_operator)
        .def("is_scalar", &QTensorCpp::is_scalar)
        .def("is_density_matrix", &QTensorCpp::is_density_matrix)
        .def("to_density_matrix", &QTensorCpp::to_density_matrix)
        .def("adjoint", &QTensorCpp::adjoint)
        .def("is_self_adjoint", &QTensorCpp::is_self_adjoint)
        .def("is_positive_semidefinite", &QTensorCpp::is_positive_semidefinite)
        .def("transpose", &QTensorCpp::transpose)
        .def("conjugate", &QTensorCpp::conjugate)
        .def("identity", &QTensorCpp::identity)
        .def("trace", &QTensorCpp::trace)
        .def("partial_trace", &QTensorCpp::partial_trace)
        .def("norm", &QTensorCpp::norm)
        .def("normalized", &QTensorCpp::normalized)
        .def("ket", &QTensorCpp::ket)
        .def("bra", &QTensorCpp::bra)
        .def("tensor_product_python", &QTensorCpp::tensor_product_python)
        .def("tensor_product", &QTensorCpp::tensor_product)
        .def("add_python", &QTensorCpp::add_python)
        .def("add", &QTensorCpp::add)
        .def("sub_python", &QTensorCpp::sub_python)
        .def("sub", &QTensorCpp::sub)
        .def("mul_python", &QTensorCpp::mul_python)
        .def("mul", &QTensorCpp::mul)
        .def("matmul_python", &QTensorCpp::matmul_python)
        .def("matmul", &QTensorCpp::matmul)
        .def("equals_python", &QTensorCpp::equals_python)
        .def("equals", &QTensorCpp::equals)
        .def("as_string", &QTensorCpp::as_string)
        .def(py::init<>())
        .def(py::init<const py::object&>());

}
