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

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// GCOV_EXCL_BR_START

// Shorthand
namespace py = pybind11;

// Needed for _a literals
using namespace py::literals;

#pragma GCC visibility push(default)

// External types
extern py::object numpy_array;
extern py::object numpy_array_type;
extern py::object csrmatrix;
extern py::object cscmatrix;
extern py::object coomatrix;
extern py::object sparray;
extern py::dtype dtype;
extern py::object py_complex;

// Internal types
extern py::object Circuit;
extern py::object Schedule;
extern py::object DigitalPropagation;
extern py::object AnalogEvolution;
extern py::object QuantumReservoir;
extern py::object FunctionalResult;
extern py::object QTensor;
extern py::object InitialState;
extern py::object Hamiltonian;
extern py::object PauliOperator;
extern py::object NoiseModel;
extern py::object SupportsStaticKraus;
extern py::object SupportsStaticLindblad;
extern py::object SupportsTimeDerivedKraus;
extern py::object SupportsTimeDerivedLindblad;
extern py::object ReadoutAssignment;
extern py::object NoiseConfig;
extern py::object ExpectationReadout;
extern py::object ReadoutMethod;
extern py::object SamplingReadout;
extern py::object StateTomographyReadout;
extern py::object ExpectationReadoutResult;
extern py::object ReadoutResult;
extern py::object SamplingReadoutResult;
extern py::object StateTomographyReadoutResult;
extern py::object ReadoutCompositeResults;

void initialize_all_pybind_types();
void initialize_external_pybind_types();
void finalize_all_pybind_types();

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP