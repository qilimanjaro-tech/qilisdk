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
#include "../../src/qilisdk_cpp/backends/qilisim/utils/parsers.h"
#include <pybind11/embed.h>

namespace py = pybind11;

// One interpreter for the whole test binary
// class PybindEnvironment : public ::testing::Environment {
// public:
//     void SetUp() override { py::initialize_interpreter(); }
//     void TearDown() override { py::finalize_interpreter(); }
// };

// parse_hamiltonians_matrix_free: empty list throws
TEST(ParseHamiltonians, MatrixFreeEmptyThrows) {
    py::list empty;
    EXPECT_ANY_THROW(parse_hamiltonians_matrix_free(empty));
}

// parse_hamiltonians_matrix_free: single Hamiltonian with one term parsed correctly
// TEST(ParseHamiltonians, MatrixFreeSingleTerm) {
//     py::gil_scoped_acquire gil;

//     // Build a minimal fake Hamiltonian object in Python
//     py::exec(R"(
//         class FakePauli:
//             def __init__(self, name, qubit):
//                 self.name = name
//                 self.qubit = qubit

//         class FakeHamiltonian:
//             def __init__(self):
//                 self.elements = {(FakePauli('X', 0),): (1+0j)}
//             # elements.items() must return iterable of (pauli_list, coeff)
//             # but our code does hamiltonian.attr("elements").attr("items")()
//             # so elements needs to be a dict-like with .items()

//         fake_H = FakeHamiltonian()
//     )");

//     py::object fake_H = py::globals()["fake_H"];
//     py::list Hs;
//     Hs.append(fake_H);

//     auto result = parse_hamiltonians_matrix_free(Hs);
//     EXPECT_EQ(result.size(), 1);
// }

// parse_hamiltonians: empty list throws
// TEST(ParseHamiltonians, SparseEmptyThrows) {
//     py::list empty;
//     EXPECT_ANY_THROW(parse_hamiltonians(empty, 1e-10));
// }

// parse_hamiltonians: single Hamiltonian parsed correctly
// TEST(ParseHamiltonians, SparseSingleHamiltonian) {
//     py::gil_scoped_acquire gil;

//     py::exec(R"(
//         import scipy.sparse as sp
//         import numpy as np

//         class FakeSparseHamiltonian:
//             def to_matrix(self):
//                 # 2x2 identity as a scipy sparse matrix
//                 return sp.eye(2, format='csr', dtype=complex)

//         fake_sparse_H = FakeSparseHamiltonian()
//     )");

//     py::object fake_sparse_H = py::globals()["fake_sparse_H"];
//     py::list Hs;
//     Hs.append(fake_sparse_H);

//     auto result = parse_hamiltonians(Hs, 1e-10);
//     EXPECT_EQ(result.size(), 1);
//     EXPECT_EQ(result[0].rows(), 2);
//     EXPECT_EQ(result[0].cols(), 2);
// }