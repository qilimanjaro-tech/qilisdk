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
#include "pybind.h"

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

py::object numpy_array;
py::object numpy_array_type;
py::object csrmatrix;
py::object cscmatrix;
py::object coomatrix;
py::object sparray;
py::dtype dtype;
py::object py_complex;
py::object QTensor;
py::object Hamiltonian;
py::object PauliOperator;
py::object NoiseModel;
py::object SupportsStaticKraus;
py::object SupportsStaticLindblad;
py::object SupportsTimeDerivedKraus;
py::object SupportsTimeDerivedLindblad;
py::object ReadoutAssignment;
py::object NoiseConfig;
py::object Circuit;
py::object Schedule;
py::object DigitalPropagation;
py::object AnalogEvolution;
py::object QuantumReservoir;
py::object FunctionalResult;
py::object ExpectationReadout;
py::object ReadoutMethod;
py::object SamplingReadout;
py::object StateTomographyReadout;
py::object SamplingReadoutResult;
py::object ExpectationReadoutResult;
py::object StateTomographyReadoutResult;
py::object ReadoutResult;
py::object ReadoutCompositeResults;

void initialize_all_pybind_types() {
    Circuit = py::module_::import("qilisdk.digital.circuit").attr("Circuit");
    Schedule = py::module_::import("qilisdk.analog.schedule").attr("Schedule");
    DigitalPropagation = py::module_::import("qilisdk.functionals.digital_propagation").attr("DigitalPropagation");
    AnalogEvolution = py::module_::import("qilisdk.functionals.analog_evolution").attr("AnalogEvolution");
    QuantumReservoir = py::module_::import("qilisdk.functionals.quantum_reservoirs").attr("QuantumReservoir");
    FunctionalResult = py::module_::import("qilisdk.functionals.functional_result").attr("FunctionalResult");
    ExpectationReadout = py::module_::import("qilisdk.readout.readout").attr("ExpectationReadout");
    ReadoutMethod = py::module_::import("qilisdk.readout.readout").attr("ReadoutMethod");
    SamplingReadout = py::module_::import("qilisdk.readout.readout").attr("SamplingReadout");
    StateTomographyReadout = py::module_::import("qilisdk.readout.readout").attr("StateTomographyReadout");
    ExpectationReadoutResult = py::module_::import("qilisdk.readout.readout_result").attr("ExpectationReadoutResult");
    ReadoutResult = py::module_::import("qilisdk.readout.readout_result").attr("ReadoutResult");
    SamplingReadoutResult = py::module_::import("qilisdk.readout.readout_result").attr("SamplingReadoutResult");
    StateTomographyReadoutResult = py::module_::import("qilisdk.readout.readout_result").attr("StateTomographyReadoutResult");
    ReadoutCompositeResults = py::module_::import("qilisdk.readout.readout_result").attr("ReadoutCompositeResults");
    QTensor = py::module_::import("qilisdk.core.qtensor").attr("QTensor");
    Hamiltonian = py::module_::import("qilisdk.analog.hamiltonian").attr("Hamiltonian");
    PauliOperator = py::module_::import("qilisdk.analog.hamiltonian").attr("PauliOperator");
    NoiseModel = py::module_::import("qilisdk.noise.noise_model").attr("NoiseModel");
    SupportsStaticKraus = py::module_::import("qilisdk.noise.protocols").attr("SupportsStaticKraus");
    SupportsStaticLindblad = py::module_::import("qilisdk.noise.protocols").attr("SupportsStaticLindblad");
    SupportsTimeDerivedKraus = py::module_::import("qilisdk.noise.protocols").attr("SupportsTimeDerivedKraus");
    SupportsTimeDerivedLindblad = py::module_::import("qilisdk.noise.protocols").attr("SupportsTimeDerivedLindblad");
    ReadoutAssignment = py::module_::import("qilisdk.noise.readout_assignment").attr("ReadoutAssignment");
    NoiseConfig = py::module_::import("qilisdk.noise.noise_config").attr("NoiseConfig");
    initialize_external_pybind_types();
}

void initialize_external_pybind_types() {
    numpy_array = py::module_::import("numpy").attr("array");
    numpy_array_type = py::module_::import("numpy").attr("ndarray");
    csrmatrix = py::module_::import("scipy.sparse").attr("csr_matrix");
    cscmatrix = py::module_::import("scipy.sparse").attr("csc_matrix");
    coomatrix = py::module_::import("scipy.sparse").attr("coo_matrix");
    sparray = py::module_::import("scipy.sparse").attr("sparray");
    dtype = py::dtype("complex128");
    py_complex = py::module_::import("builtins").attr("complex");
}

// GCOVR_EXCL_START
void finalize_all_pybind_types() {
    Circuit = py::object();
    Schedule = py::object();
    DigitalPropagation = py::object();
    AnalogEvolution = py::object();
    QuantumReservoir = py::object();
    FunctionalResult = py::object();
    ExpectationReadout = py::object();
    ReadoutMethod = py::object();
    SamplingReadout = py::object();
    StateTomographyReadout = py::object();
    ExpectationReadoutResult = py::object();
    ReadoutResult = py::object();
    SamplingReadoutResult = py::object();
    StateTomographyReadoutResult = py::object();
    ReadoutCompositeResults = py::object();
    QTensor = py::object();
    Hamiltonian = py::object();
    PauliOperator = py::object();
    NoiseModel = py::object();
    SupportsStaticKraus = py::object();
    SupportsStaticLindblad = py::object();
    SupportsTimeDerivedKraus = py::object();
    SupportsTimeDerivedLindblad = py::object();
    ReadoutAssignment = py::object();
    NoiseConfig = py::object();
    numpy_array = py::object();
    numpy_array_type = py::object();
    csrmatrix = py::object();
    cscmatrix = py::object();
    coomatrix = py::object();
    sparray = py::object();
    dtype = py::object();
    py_complex = py::object();
}
// GCOVR_EXCL_STOP

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
