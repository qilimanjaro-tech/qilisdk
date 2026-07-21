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

#include "../../libs/pybind.h"
#include "qilisim.h"

// GCOV_EXCL_BR_START
// GCOVR_EXCL_START

PYBIND11_MODULE(qilisim_module, m) {
    initialize_all_pybind_types();
    // Release the module-global py::object handles (see pybind.cpp) before the
    // interpreter finalizes. The capsule destructor runs at module teardown with
    // the GIL held and the interpreter still alive; without it the globals'
    // static destructors run Py_DECREF after Py_Finalize and abort the process.
    m.add_object("_qilisdk_cleanup", py::capsule(&finalize_all_pybind_types));
    py::class_<QiliSimCpp>(m, "QiliSimCpp").def(py::init<>()).def("execute_analog_evolution", &QiliSimCpp::execute_analog_evolution).def("execute_digital_propagation", &QiliSimCpp::execute_digital_propagation).def("execute_quantum_reservoir", &QiliSimCpp::execute_quantum_reservoir);
}

// GCOVR_EXCL_STOP
// GCOV_EXCL_BR_STOP