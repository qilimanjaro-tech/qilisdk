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

#include "expression.h"

// GCOVR_EXCL_START
PYBIND11_MODULE(expression_module, m) {
    namespace expr = qilisdk::expr;
    namespace bound = qilisdk::expr::python;
    m.def("initialize", &expr::initialize);
    m.def("constant_node", &bound::constant_node);
    m.def("node", &bound::node);
    m.def("function", &bound::function);
    m.def("variable_hash", &bound::variable_hash);
    m.def("add", &bound::add);
    m.def("sub", &bound::sub);
    m.def("rsub", &bound::rsub);
    m.def("mul", &bound::mul);
    m.def("neg", &bound::neg);
    m.def("truediv", &bound::truediv);
    m.def("rtruediv", &bound::rtruediv);
    m.def("pow", &bound::pow);
    m.def("rpow", &bound::rpow);
    m.def("build_add", &bound::build_add);
    m.def("build_mul", &bound::build_mul);
    m.def("build_pow", &bound::build_pow);
    m.def("evaluate", &bound::evaluate);
    m.def("free_symbols", &bound::free_symbols);
    m.def("variables", &bound::variables);
    m.def("is_parameterized", &bound::is_parameterized);
    m.def("degree", &bound::degree);
    m.def("derivative", &bound::derivative);
    m.def("expand", &bound::expand);
    m.def("substitute", &bound::substitute);
    m.def("to_binary", &bound::to_binary);
    m.def("get_constant", &bound::get_constant);
    m.def("as_coefficients_dict", &bound::as_coefficients_dict);
    m.def("monomial_factors", &bound::monomial_factors);
    m.def("to_list", &bound::to_list);
    m.def("args", &bound::args);
    m.def("value", &bound::value);
    m.def("coefficient", &bound::coefficient);
    m.def("monomial", &bound::monomial);
    m.def("hash", &bound::hash);
    m.def("equal", &bound::equal);
    m.def("repr", &bound::repr);
}
// GCOVR_EXCL_STOP
