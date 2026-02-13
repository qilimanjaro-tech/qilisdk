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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Shorthand
namespace py = pybind11;

// Needed for _a literals
using namespace py::literals;

// Get the Python functional classes
const py::object Sampling = py::module_::import("qilisdk.functionals.sampling").attr("Sampling");
const py::object TimeEvolution = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolution");
const py::object SamplingResult = py::module_::import("qilisdk.functionals.sampling").attr("SamplingResult");
const py::object TimeEvolutionResult = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolutionResult");
const py::object numpy_array = py::module_::import("numpy").attr("array");
const py::object csrmatrix = py::module_::import("scipy.sparse").attr("csr_matrix");
const py::object QTensor = py::module_::import("qilisdk.core.qtensor").attr("QTensor");
const py::object Hamiltonian = py::module_::import("qilisdk.analog.hamiltonian").attr("Hamiltonian");
const py::object PauliOperator = py::module_::import("qilisdk.analog.hamiltonian").attr("PauliOperator");
const py::object NoiseModel = py::module_::import("qilisdk.noise.noise_model").attr("NoiseModel");
const py::object SupportsStaticKraus = py::module_::import("qilisdk.noise.protocols").attr("SupportsStaticKraus");
const py::object SupportsStaticLindblad = py::module_::import("qilisdk.noise.protocols").attr("SupportsStaticLindblad");
const py::object SupportsTimeDerivedKraus = py::module_::import("qilisdk.noise.protocols").attr("SupportsTimeDerivedKraus");
const py::object SupportsTimeDerivedLindblad = py::module_::import("qilisdk.noise.protocols").attr("SupportsTimeDerivedLindblad");
const py::object ReadoutAssignment = py::module_::import("qilisdk.noise.readout_assignment").attr("ReadoutAssignment");
const py::object NoiseConfig = py::module_::import("qilisdk.noise.noise_config").attr("NoiseConfig");
const py::dtype dtype = py::dtype("complex128");
