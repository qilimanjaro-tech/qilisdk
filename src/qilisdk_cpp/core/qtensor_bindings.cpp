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
        .def("as_scipy", &QTensorCpp::as_scipy)
        .def("as_numpy", &QTensorCpp::as_numpy)
        .def("get_nqubits", &QTensorCpp::get_nqubits)
        .def("get_shape", &QTensorCpp::get_shape)
        .def("is_ket", &QTensorCpp::is_ket)
        .def("is_bra", &QTensorCpp::is_bra)
        .def("is_operator", &QTensorCpp::is_operator)
        .def("is_scalar", &QTensorCpp::is_scalar)
        .def("is_density_matrix", &QTensorCpp::is_density_matrix)
        .def("as_density_matrix", &QTensorCpp::as_density_matrix)
        .def("adjoint", &QTensorCpp::adjoint)
        .def("is_self_adjoint", &QTensorCpp::is_self_adjoint)
        .def("is_positive_semidefinite", &QTensorCpp::is_positive_semidefinite)
        .def("transpose", &QTensorCpp::transpose)
        .def("conjugate", &QTensorCpp::conjugate)
        .def("identity", &QTensorCpp::identity)
        .def("trace", &QTensorCpp::trace)
        .def("partial_trace_python", &QTensorCpp::partial_trace_python)
        .def("norm", &QTensorCpp::norm)
        .def("normalized", &QTensorCpp::normalized)
        .def("zero", &QTensorCpp::zero)
        .def("ket_python", &QTensorCpp::ket_python)
        .def("bra_python", &QTensorCpp::bra_python)
        .def("expectation_value_python", &QTensorCpp::expectation_value_python)
        .def("tensor_product_python", &QTensorCpp::tensor_product_python)
        .def("add_python", &QTensorCpp::add_python)
        .def("sub_python", &QTensorCpp::sub_python)
        .def("mul_python", &QTensorCpp::mul_python)
        .def("coeff", &QTensorCpp::coeff)
        .def("matmul_python", &QTensorCpp::matmul_python)
        .def("equals_python", &QTensorCpp::equals_python)
        .def("as_string", &QTensorCpp::as_string)
        .def("as_dense", &QTensorCpp::as_dense)
        .def("pow", &QTensorCpp::pow)
        .def("sqrt", &QTensorCpp::sqrt)
        .def("log", &QTensorCpp::log)
        .def("exp", &QTensorCpp::exp)
        .def("rank", &QTensorCpp::rank)
        .def("get_eigenvalues_python", &QTensorCpp::get_eigenvalues_python)
        .def("get_eigenvectors_python", &QTensorCpp::get_eigenvectors_python)
        .def("dot_python", &QTensorCpp::dot_python)
        .def("entropy_von_neumann", &QTensorCpp::entropy_von_neumann)
        .def("entropy_renyi", &QTensorCpp::entropy_renyi)
        .def("fidelity_python", &QTensorCpp::fidelity_python)
        .def("purity", &QTensorCpp::purity)
        .def("div", &QTensorCpp::div)
        .def("inverse", &QTensorCpp::inverse)
        .def("ghz", &QTensorCpp::ghz)
        .def("commutator_python", &QTensorCpp::commutator_python)
        .def("anticommutator_python", &QTensorCpp::anticommutator_python)
        .def("is_hermitian", &QTensorCpp::is_hermitian)
        .def("is_unitary", &QTensorCpp::is_unitary)
        .def("is_pure", &QTensorCpp::is_pure)
        .def("clear_cache", &QTensorCpp::clear_cache)
        .def("probabilities_python", &QTensorCpp::probabilities_python)
        .def("compute_eigendecomposition", &QTensorCpp::compute_eigendecomposition)
        .def("reset_qubits_python", &QTensorCpp::reset_qubits_python)
        .def(py::init<>())
        .def(py::init<const py::object&>());
}
