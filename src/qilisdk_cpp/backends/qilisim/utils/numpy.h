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
#pragma once

#include <complex>
#include <vector>
#include "../libs/eigen.h"
#include "../libs/pybind.h"

SparseMatrix from_numpy(const py::buffer& matrix_buffer, double atol);
SparseMatrix from_spmatrix(const py::object& matrix, double atol);
py::array_t<double> to_numpy(const std::vector<double>& vec);
py::array_t<double> to_numpy(const std::vector<std::vector<double>>& vecs);
py::array_t<std::complex<double>> to_numpy(const SparseMatrix& matrix);