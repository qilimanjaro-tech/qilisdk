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

#include "qilisim.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Make the QiliSimCpp class available in Python, as well as the two main methods
PYBIND11_MODULE(qilisim_module, m) {
    py::class_<QiliSimCpp>(m, "QiliSimCpp").def(py::init<>()).def("execute_sampling", &QiliSimCpp::execute_sampling).def("execute_time_evolution", &QiliSimCpp::execute_time_evolution);
}