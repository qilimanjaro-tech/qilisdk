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

// GCOV_EXCL_BR_START

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <cmath>
#include <limits>
#include <string>
#include "../../../src/qilisdk_cpp/core/expression/expression.h"

namespace py = pybind11;
using namespace qilisdk::expr;

namespace {

// Point the engine at the real Python classes and define the variables the tests share
class ExpressionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        py::exec(R"(
            import numpy as np
            from qilisdk.core.exceptions import NonPolynomialError, NotSupportedOperation
            from qilisdk.core.expression import Add, Constant, Cos, Expression, Function, Log, Mul, Pow, Sin, Abs
            from qilisdk.core.variables import BinaryVariable, Domain, Parameter, SpinVariable, Variable
            from qilisdk.settings import get_settings
            x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
            s = SpinVariable("s")
            p, q = Parameter("p", 2.0), Parameter("q", 3.0)
        )");
        initialize(g("Constant"), g("Add"), g("Mul"), g("Pow"), g("Expression"), g("Log"), g("NonPolynomialError"), g("get_settings")().attr("atol").cast<double>());
    }

    static py::object g(const char* name) { return py::globals()[name]; }
    static py::object ev(const std::string& code) { return py::eval(code, py::globals()); }
    static std::string r(const std::string& code) { return python::repr(ev(code)); }
    static bool same(const std::string& a, const std::string& b) { return python::equal(ev(a), ev(b)); }
};

bool is_int(const Num& n, int64_t value) {
    return n.tag == Num::Tag::Int && n.i == value;
}

bool is_float(const Num& n, double value) {
    return n.tag == Num::Tag::Float && n.re == value;
}

}  // namespace

// ---------------------------------------------------------------- numbers

TEST_F(ExpressionTest, IntegerArithmeticStaysExactUntilItOverflows) {
    const int64_t big = std::numeric_limits<int64_t>::max();
    EXPECT_TRUE(is_int(num_add(Num::integer(2), Num::integer(3)), 5));
    EXPECT_EQ(num_add(Num::integer(big), Num::integer(1)).tag, Num::Tag::Float);
    EXPECT_TRUE(is_int(num_mul(Num::integer(4), Num::integer(5)), 20));
    EXPECT_EQ(num_mul(Num::integer(big), Num::integer(2)).tag, Num::Tag::Float);
    EXPECT_TRUE(is_float(num_add(Num::integer(1), Num::real(0.5)), 1.5));
    EXPECT_TRUE(is_float(num_mul(Num::integer(2), Num::real(0.5)), 1.0));
    const Num sum = num_add(Num::integer(1), Num::complex(0.0, 2.0));
    EXPECT_EQ(sum.tag, Num::Tag::Complex);
    EXPECT_EQ(sum.re, 1.0);
    const Num product = num_mul(Num::complex(0.0, 1.0), Num::complex(0.0, 1.0));
    EXPECT_EQ(product.re, -1.0);
}

TEST_F(ExpressionTest, PowersFollowPythonSemantics) {
    EXPECT_TRUE(is_int(num_pow(Num::integer(2), Num::integer(10)), 1024));
    EXPECT_EQ(num_pow(Num::integer(3), Num::integer(100)).tag, Num::Tag::Float);
    EXPECT_TRUE(is_float(num_pow(Num::integer(2), Num::integer(-1)), 0.5));
    EXPECT_TRUE(is_float(num_pow(Num::real(4.0), Num::real(0.5)), 2.0));
    // A negative base under a fractional power is complex, as in Python
    const Num cube_root = num_pow(Num::real(-8.0), Num::real(1.0 / 3.0));
    EXPECT_EQ(cube_root.tag, Num::Tag::Complex);
    EXPECT_NEAR(cube_root.re, 1.0, 1e-12);
    EXPECT_NEAR(cube_root.im, std::sqrt(3.0), 1e-12);
    // Integer powers of a complex number multiply out, negative ones divide through
    const Num i_squared = num_pow(Num::complex(0.0, 1.0), Num::integer(2));
    EXPECT_EQ(i_squared.re, -1.0);
    const Num wide = num_pow(Num::complex(2.0, 1.0), Num::integer(-1));
    EXPECT_NEAR(wide.re, 0.4, 1e-12);
    EXPECT_NEAR(wide.im, -0.2, 1e-12);
    const Num tall = num_pow(Num::complex(1.0, 2.0), Num::integer(-1));
    EXPECT_NEAR(tall.re, 0.2, 1e-12);
    EXPECT_NEAR(tall.im, -0.4, 1e-12);
    // Other powers go through the polar form
    const Num root = num_pow(Num::complex(0.0, 1.0), Num::real(0.5));
    EXPECT_NEAR(root.re, std::sqrt(0.5), 1e-12);
    const Num complex_exponent = num_pow(Num::real(std::exp(1.0)), Num::complex(0.0, std::acos(-1.0)));
    EXPECT_NEAR(complex_exponent.re, -1.0, 1e-12);
    const Num zero = num_pow(Num::complex(0.0, 0.0), Num::real(0.5));
    EXPECT_EQ(zero.re, 0.0);
    EXPECT_THROW(num_pow(Num::complex(0.0, 0.0), Num::real(-0.5)), py::value_error);
    EXPECT_THROW(num_pow(Num::complex(0.0, 0.0), Num::integer(-1)), py::value_error);
}

TEST_F(ExpressionTest, NumbersCompareByValueAcrossTypes) {
    EXPECT_TRUE(num_eq(Num::integer(2), Num::integer(2)));
    EXPECT_TRUE(num_eq(Num::integer(2), Num::real(2.0)));
    EXPECT_TRUE(num_eq(Num::real(2.0), Num::integer(2)));
    EXPECT_FALSE(num_eq(Num::integer(2), Num::real(2.5)));
    EXPECT_FALSE(num_eq(Num::real(2.0), Num::complex(2.0, 1.0)));
    EXPECT_TRUE(num_eq(Num::real(0.5), Num::real(0.5)));
    EXPECT_EQ(num_abs(Num::complex(3.0, 4.0)), 5.0);
    EXPECT_EQ(num_abs(Num::integer(-2)), 2.0);
    EXPECT_EQ(float_if_real(Num::complex(2.0, 1e-15)).tag, Num::Tag::Float);
    EXPECT_EQ(float_if_real(Num::complex(2.0, 1.0)).tag, Num::Tag::Complex);
    EXPECT_TRUE(is_float(finalize(Num::integer(3)), 3.0));
    EXPECT_EQ(finalize(Num::complex(1.0, 1.0)).tag, Num::Tag::Complex);
}

TEST_F(ExpressionTest, NumbersConvertToAndFromPython) {
    Num out;
    EXPECT_TRUE(num_from_py(py::bool_(true), out));
    EXPECT_TRUE(is_int(out, 1));
    EXPECT_TRUE(num_from_py(py::int_(7), out));
    EXPECT_TRUE(is_int(out, 7));
    EXPECT_TRUE(num_from_py(ev("2**70"), out));
    EXPECT_TRUE(is_float(out, std::ldexp(1.0, 70)));
    EXPECT_THROW(num_from_py(ev("10**400"), out), py::error_already_set);
    EXPECT_TRUE(num_from_py(py::float_(1.5), out));
    EXPECT_TRUE(is_float(out, 1.5));
    EXPECT_TRUE(num_from_py(ev("2+3j"), out));
    EXPECT_EQ(out.im, 3.0);
    EXPECT_TRUE(num_from_py(ev("np.int64(4)"), out));
    EXPECT_TRUE(is_int(out, 4));
    EXPECT_FALSE(num_from_py(py::str("four"), out));
    EXPECT_TRUE(num_to_py(Num::integer(5)).equal(py::int_(5)));
    EXPECT_TRUE(num_to_py(Num::real(0.5)).equal(py::float_(0.5)));
    EXPECT_TRUE(num_to_py(Num::complex(1.0, 2.0)).equal(ev("1+2j")));
}

// ---------------------------------------------------------------- construction

TEST_F(ExpressionTest, ConstantsAndTrustingConstructors) {
    py::object three = g("Constant").attr("__new__")(g("Constant"));
    three.attr("_node") = python::constant_node(py::int_(3));
    EXPECT_EQ(python::repr(three), "3");
    EXPECT_THROW(python::constant_node(py::str("three")), py::type_error);
    py::object sum = g("Add").attr("__new__")(g("Add"));
    sum.attr("_node") = python::node(static_cast<int>(Kind::Add), py::make_tuple(g("x"), g("y")));
    EXPECT_TRUE(python::equal(sum, ev("x + y")));
    EXPECT_THROW(python::node(static_cast<int>(Kind::Add), py::make_tuple(g("x"))), py::value_error);
    EXPECT_THROW(python::node(static_cast<int>(Kind::Mul), py::tuple()), py::value_error);
    EXPECT_THROW(python::node(static_cast<int>(Kind::Pow), py::make_tuple(g("x"), g("y"), g("z"))), py::value_error);
    EXPECT_THROW(python::node(static_cast<int>(Kind::Variable), py::make_tuple(g("x"), g("y"))), py::value_error);
    EXPECT_EQ(python::variable_hash("x"), python::hash(g("x")));
    EXPECT_TRUE(python::value(ev("Constant(2.5)")).equal(py::float_(2.5)));
}

TEST_F(ExpressionTest, ArithmeticCanonicalizes) {
    py::object x = g("x");
    py::object y = g("y");
    EXPECT_TRUE(python::equal(python::add(x, y), python::add(y, x)));
    EXPECT_TRUE(python::equal(python::add(x, x), python::mul(x, py::int_(2))));
    EXPECT_TRUE(python::equal(python::sub(x, x), ev("Constant(0)")));
    EXPECT_TRUE(python::equal(python::rsub(x, py::int_(1)), ev("1 - x")));
    EXPECT_TRUE(python::equal(python::neg(x), ev("-1 * x")));
    EXPECT_TRUE(python::equal(python::mul(x, x), x));
    EXPECT_TRUE(python::equal(python::truediv(x, py::int_(2)), ev("0.5 * x")));
    EXPECT_THROW(python::truediv(x, py::int_(0)), py::value_error);
    EXPECT_TRUE(python::equal(python::truediv(x, y), ev("x * y**-1")));
    EXPECT_TRUE(python::equal(python::rtruediv(x, py::int_(1)), ev("x**-1")));
    EXPECT_TRUE(python::equal(python::pow(g("p"), py::int_(2)), ev("p * p")));
    EXPECT_TRUE(python::equal(python::rpow(g("p"), py::int_(2)), ev("2**p")));
    // Anything that is not a number or an Expression is left to Python
    py::str other("other");
    for (auto op : {&python::add, &python::sub, &python::rsub, &python::mul, &python::truediv, &python::rtruediv, &python::pow, &python::rpow}) {
        EXPECT_TRUE(op(x, other).is(py::handle(Py_NotImplemented)));
    }
}

TEST_F(ExpressionTest, BuildersCollectTermsAndPowers) {
    py::object x = g("x");
    py::object p = g("p");
    EXPECT_TRUE(python::equal(python::build_add(py::make_tuple(x, py::int_(1), py::int_(-1))), x));
    EXPECT_TRUE(python::equal(python::build_add(py::tuple()), ev("Constant(0)")));
    EXPECT_THROW(python::build_add(py::make_tuple(x, py::str("bad"))), py::type_error);
    EXPECT_TRUE(python::equal(python::build_mul(py::make_tuple(p, p, p)), ev("p**3")));
    EXPECT_TRUE(python::equal(python::build_mul(py::make_tuple(ev("p**2"), p)), ev("p**3")));
    EXPECT_TRUE(python::equal(python::build_mul(py::make_tuple(p, py::int_(0))), ev("Constant(0)")));
    EXPECT_TRUE(python::equal(python::build_mul(py::make_tuple(p, ev("p**-1"))), ev("Constant(1)")));
    EXPECT_TRUE(python::equal(python::build_mul(py::make_tuple(ev("x * y"), py::int_(2))), ev("2 * x * y")));
    EXPECT_TRUE(python::equal(python::build_mul(py::make_tuple(x, ev("x**3"))), x));
    EXPECT_THROW(python::build_pow(py::str("bad"), x), py::type_error);
    EXPECT_TRUE(python::equal(python::build_pow(p, py::int_(1)), p));
    EXPECT_TRUE(python::equal(python::build_pow(p, py::int_(0)), ev("Constant(1)")));
    EXPECT_TRUE(python::equal(python::build_pow(py::int_(2), py::int_(3)), ev("Constant(8)")));
    EXPECT_TRUE(python::equal(python::build_pow(ev("p**2"), py::int_(3)), ev("p**6")));
    EXPECT_EQ(python::repr(python::build_pow(ev("p**2"), py::float_(0.5))), "(p**2)**0.5");
    EXPECT_TRUE(python::equal(python::build_pow(x, py::int_(3)), x));
    EXPECT_TRUE(python::equal(python::build_pow(py::int_(1), g("q")), ev("Constant(1)")));
    EXPECT_THROW(python::build_pow(py::int_(0), py::int_(-1)), py::value_error);
}

TEST_F(ExpressionTest, AddingOneTermToASumMergesIt) {
    // A constant, with and without an existing one, in either order
    EXPECT_EQ(r("(x + 1) + 2"), "3 + x");
    EXPECT_EQ(r("2 + (x + 1)"), "3 + x");
    EXPECT_EQ(r("(x + y) + 2"), "2 + x + y");
    EXPECT_EQ(r("(x + 1) + (-1)"), "x");
    // A monomial that is new, that cancels, that merges to a coefficient of one, in either order
    EXPECT_EQ(r("(x + y) + z"), "x + y + z");
    EXPECT_EQ(r("(x + y) - x"), "y");
    EXPECT_EQ(r("(2 * x + y) - x"), "x + y");
    EXPECT_EQ(r("x + (x + y)"), "y + 2 * x");
    // Every way a summand can fail to share the new term's base
    EXPECT_EQ(r("(2 * x * y + 1) + z"), "1 + z + 2 * x * y");
    EXPECT_EQ(r("(2 * x * y + 1) + x * y * z"), "1 + 2 * x * y + x * y * z");
    EXPECT_EQ(r("(2 * x * y + 1) + x * z"), "1 + 2 * x * y + x * z");
    EXPECT_EQ(r("(2 * x * y + 1) + 3 * x * y"), "1 + 5 * x * y");
}

TEST_F(ExpressionTest, RunningSumsMatchTheirFreshlyBuiltForm) {
    py::exec(R"(
        terms = [2 * x * y, -x, 3, y * z, x, -2 * x * y, 0.5 * z, -3, s, x * y * z, 1j * s]
        running = 0
        for term in terms:
            running = running + term
        fresh = Add.build(terms)
    )");
    const NodePtr running = node_of(g("running"));
    const NodePtr fresh = node_of(g("fresh"));
    EXPECT_TRUE(equal(running, fresh));
    EXPECT_EQ(running->hash, fresh->hash);
    EXPECT_EQ(running->monomial_hash, fresh->monomial_hash);
    // Terms that cancelled may leave their bits behind, but every variable still present has its bit
    EXPECT_EQ(running->symbol_mask & fresh->symbol_mask, fresh->symbol_mask);
}

TEST_F(ExpressionTest, ProductsKnowTheHashOfTheirMonomial) {
    EXPECT_EQ(node_of(ev("3 * x * y"))->monomial_hash, node_of(ev("x * y"))->hash);
    EXPECT_EQ(node_of(ev("3 * x"))->monomial_hash, node_of(g("x"))->hash);
    EXPECT_EQ(node_of(ev("x * y"))->monomial_hash, node_of(ev("x * y"))->hash);
    EXPECT_EQ(node_of(ev("x + y"))->monomial_hash, node_of(ev("x + y"))->hash);
}

TEST_F(ExpressionTest, SymbolMasksCoverExactlyTheVariablesBuiltIn) {
    const uint64_t x_bit = node_of(g("x"))->symbol_mask;
    const uint64_t p_bit = node_of(g("p"))->symbol_mask;
    EXPECT_NE(x_bit, 0U);
    EXPECT_EQ(node_of(ev("Constant(2)"))->symbol_mask, 0U);
    EXPECT_EQ(node_of(ev("2 * x * p + Sin(p)"))->symbol_mask, x_bit | p_bit);
    EXPECT_EQ(node_of(ev("Cos(p)"))->symbol_mask, p_bit);
}

TEST_F(ExpressionTest, FunctionsFoldConstantsAndCheckTheirOperand) {
    EXPECT_TRUE(python::equal(python::function(g("Sin"), py::int_(0)), ev("Constant(0.0)")));
    EXPECT_EQ(python::repr(python::function(g("Sin"), g("p"))), "Sin(p)");
    EXPECT_THROW(python::function(g("Sin"), py::str("bad")), py::type_error);
    EXPECT_THROW(python::function(g("Sin"), ev("1+1j")), py::value_error);
}

// ---------------------------------------------------------------- structure

TEST_F(ExpressionTest, OperandsSortCanonically) {
    // Constants by value, variables by label, functions by name then operand, compounds lexicographically
    EXPECT_EQ(r("3 * x + 2 * y"), "2 * y + 3 * x");
    EXPECT_EQ(r("2j * x + 1j * y"), "1j * y + 2j * x");
    EXPECT_EQ(r("2 * y + 2 * x"), "2 * x + 2 * y");
    EXPECT_EQ(r("Sin(y) + Cos(x) + Sin(x)"), "Cos(x) + Sin(x) + Sin(y)");
    EXPECT_EQ(r("x * y * z + x * y"), "x * y + x * y * z");
    EXPECT_EQ(r("p**3 + p**2"), "p**2 + p**3");
    EXPECT_EQ(r("(x + y)**3 + (y + x)**2"), "(x + y)**2 + (x + y)**3");
}

TEST_F(ExpressionTest, EqualityIsStructural) {
    EXPECT_TRUE(same("Constant(2)", "Constant(2.0)"));
    EXPECT_TRUE(same("Sin(x)", "Sin(x)"));
    EXPECT_FALSE(same("Sin(x)", "Cos(x)"));
    EXPECT_TRUE(same("x * y + 1", "1 + y * x"));
    EXPECT_FALSE(same("x", "y"));
    EXPECT_FALSE(python::equal(g("x"), py::int_(1)));
    EXPECT_TRUE(NodeEqual()(node_of(g("x")), node_of(g("x"))));
}

TEST_F(ExpressionTest, ExponentsAndCoefficientsAreReadOff) {
    EXPECT_EQ(int_exponent(node_of(ev("Constant(2.0)"))).value(), 2);
    EXPECT_FALSE(int_exponent(node_of(ev("Constant(2.5)"))).has_value());
    EXPECT_FALSE(int_exponent(node_of(g("x"))).has_value());
    EXPECT_TRUE(python::coefficient(ev("3 * x * y")).equal(py::int_(3)));
    EXPECT_TRUE(python::equal(python::monomial(ev("3 * x * y")), ev("x * y")));
    EXPECT_TRUE(python::equal(python::monomial(ev("3 * x")), g("x")));
    EXPECT_TRUE(python::coefficient(g("x")).equal(py::int_(1)));
}

// ---------------------------------------------------------------- symbolic operations

TEST_F(ExpressionTest, ExpandDistributesProducts) {
    EXPECT_TRUE(same("((p + q) * (p - q)).expand()", "p**2 - q**2"));
    EXPECT_TRUE(python::equal(python::expand(ev("(p + q)**2")), ev("p**2 + 2 * p * q + q**2")));
    EXPECT_TRUE(python::equal(python::expand(ev("p * q + 1")), ev("p * q + 1")));
    EXPECT_TRUE(python::equal(python::expand(ev("(p + q)**0.5")), ev("(p + q)**0.5")));
    EXPECT_TRUE(python::equal(python::expand(ev("Sin(p * (q + 1))")), ev("Sin(p * q + p)")));
    EXPECT_TRUE(python::equal(python::expand(g("p")), g("p")));
}

TEST_F(ExpressionTest, DerivativesApplyTheUsualRules) {
    py::object p = g("p");
    py::object q = g("q");
    EXPECT_TRUE(python::equal(python::derivative(ev("Constant(3)"), p), ev("Constant(0)")));
    EXPECT_TRUE(python::equal(python::derivative(p, p), ev("Constant(1)")));
    EXPECT_TRUE(python::equal(python::derivative(q, p), ev("Constant(0)")));
    EXPECT_TRUE(python::equal(python::derivative(ev("p * q + p"), p), ev("q + 1")));
    // Summands and factors without the variable are skipped rather than differentiated to zero
    EXPECT_TRUE(python::equal(python::derivative(ev("2 * x * y + q * x + 3 * p * q * x"), p), ev("3 * q * x")));
    EXPECT_TRUE(python::equal(python::derivative(ev("2 * x * y"), p), ev("Constant(0)")));
    EXPECT_TRUE(python::equal(python::derivative(ev("p**3"), p), ev("3 * p**2")));
    EXPECT_NEAR(python::evaluate(python::derivative(ev("p**q"), q), py::none()).cast<double>(), 8.0 * std::log(2.0), 1e-12);
    EXPECT_TRUE(python::equal(python::derivative(ev("Sin(p)"), p), ev("Cos(p)")));
    EXPECT_TRUE(python::equal(python::derivative(ev("Sin(q)"), p), ev("Constant(0)")));
    EXPECT_THROW(python::derivative(ev("Abs(p)"), p), py::error_already_set);
}

TEST_F(ExpressionTest, SubstituteReplacesWholeSubtrees) {
    EXPECT_TRUE(python::equal(python::substitute(ev("p + q"), ev("{p: 2 * q}")), ev("3 * q")));
    EXPECT_TRUE(python::equal(python::substitute(ev("p * q"), ev("{p * q: 1}")), ev("Constant(1)")));
    EXPECT_TRUE(python::equal(python::substitute(ev("p**q"), ev("{q: 2}")), ev("p**2")));
    EXPECT_TRUE(python::equal(python::substitute(ev("Sin(p)"), ev("{p: q}")), ev("Sin(q)")));
    EXPECT_TRUE(python::equal(python::substitute(ev("p * q"), ev("{q: p}")), ev("p**2")));
    EXPECT_TRUE(python::equal(python::substitute(ev("p + 1"), ev("{p: 'not a number', 'not an expression': 1}")), ev("p + 1")));
    // Any mapping works, not only a dict
    EXPECT_TRUE(python::equal(python::substitute(g("p"), ev("__import__('types').MappingProxyType({p: q})")), g("q")));
}

TEST_F(ExpressionTest, ToBinaryEncodesEachVariableOnce) {
    py::exec(R"(
        class NotBinary(SpinVariable):
            def to_binary(self):
                return "not an expression"
        v = Variable("v", domain=Domain.POSITIVE_INTEGER, bounds=(0, 3))
    )");
    EXPECT_TRUE(python::equal(python::to_binary(ev("s + s * x")), ev("(s + s * x).to_binary()")));
    EXPECT_TRUE(python::equal(python::to_binary(ev("Sin(v) + v**2 + 1")), ev("Sin(v.to_binary()) + v.to_binary()**2 + 1")));
    EXPECT_TRUE(python::equal(python::to_binary(ev("x * y")), ev("x * y")));
    EXPECT_TRUE(python::equal(python::to_binary(ev("Constant(2)")), ev("Constant(2)")));
    EXPECT_THROW(python::to_binary(ev("NotBinary('n') + 1")), py::type_error);
}

TEST_F(ExpressionTest, EvaluateAsksEachVariableOnce) {
    py::exec(R"(
        class NotANumber(BinaryVariable):
            def evaluate(self, env=None):
                return "not a number"
        class Broken(Function):
            NAME = "broken"
            @staticmethod
            def _numeric(value):
                return "not a number"
            def _outer_derivative(self, operand):
                return "not an expression"
    )");
    EXPECT_TRUE(python::evaluate(ev("x * y + x"), ev("{x: 1, y: 1}")).equal(py::float_(2.0)));
    EXPECT_TRUE(python::evaluate(ev("p**2 + p * q"), py::none()).equal(py::float_(10.0)));
    EXPECT_NEAR(python::evaluate(ev("Sin(p)"), py::none()).cast<double>(), std::sin(2.0), 1e-15);
    EXPECT_TRUE(python::evaluate(ev("Constant(1j)"), py::none()).equal(ev("1j")));
    EXPECT_THROW(python::evaluate(ev("NotANumber('n') + 1"), py::none()), py::type_error);
    EXPECT_THROW(python::evaluate(ev("Broken(p)"), py::none()), py::type_error);
    EXPECT_THROW(python::derivative(ev("Broken(p)"), g("p")), py::type_error);
    EXPECT_THROW(python::function(g("Broken"), py::int_(1)), py::type_error);
    EXPECT_THROW(python::evaluate(ev("Sin(1j * p)"), py::none()), py::value_error);
}

TEST_F(ExpressionTest, PolynomialAccessors) {
    py::object poly = ev("2 * x * y + 3 * x + 5");
    EXPECT_EQ(python::degree(poly), 2);
    EXPECT_EQ(python::degree(ev("p**3 + Constant(1)")), 3);
    EXPECT_THROW(python::degree(ev("p**0.5")), py::error_already_set);
    EXPECT_THROW(python::degree(ev("Sin(p)")), py::error_already_set);
    // A function of a sum that was never canonicalized has no symbols, so degree zero
    py::object constants = g("Add").attr("__new__")(g("Add"));
    constants.attr("_node") = python::node(static_cast<int>(Kind::Add), py::make_tuple(ev("Constant(1)"), ev("Constant(2)")));
    EXPECT_EQ(python::degree(python::function(g("Sin"), constants)), 0);
    EXPECT_TRUE(python::get_constant(poly).equal(py::int_(5)));
    EXPECT_TRUE(python::get_constant(ev("Constant(4)")).equal(py::int_(4)));
    EXPECT_TRUE(python::get_constant(g("x")).equal(py::int_(0)));
    EXPECT_EQ(py::len(python::as_coefficients_dict(poly)), 2u);
    EXPECT_EQ(py::len(python::as_coefficients_dict(ev("Constant(4)"))), 0u);
    EXPECT_EQ(py::len(python::as_coefficients_dict(g("x"))), 1u);
    EXPECT_EQ(py::len(python::monomial_factors(ev("2 * x * p**3"))), 2u);
    EXPECT_EQ(py::len(python::monomial_factors(ev("Constant(4)"))), 0u);
    EXPECT_THROW(python::monomial_factors(ev("p**-1")), py::error_already_set);
    EXPECT_EQ(py::len(python::to_list(poly)), 3u);
    EXPECT_EQ(py::len(python::to_list(g("x"))), 1u);
    EXPECT_EQ(py::len(python::args(ev("p**2"))), 2u);
}

TEST_F(ExpressionTest, SymbolsAreCollectedOncePerLabel) {
    EXPECT_EQ(py::len(python::free_symbols(ev("x * y + x + p"))), 3u);
    py::list ordered = python::variables(ev("y + x * p"));
    EXPECT_EQ(py::str(ordered[0]).cast<std::string>(), "p");
    EXPECT_EQ(py::str(ordered[2]).cast<std::string>(), "y");
    EXPECT_TRUE(python::is_parameterized(ev("p * q + 1")));
    EXPECT_FALSE(python::is_parameterized(ev("p * x")));
}

TEST_F(ExpressionTest, PrintingMatchesPython) {
    EXPECT_EQ(r("-2 - 2 * x"), "-2 - 2 * x");
    EXPECT_EQ(r("p - q"), "p - q");
    EXPECT_EQ(r("-p"), "-p");
    EXPECT_EQ(r("-(p + q)"), "-(p + q)");
    EXPECT_EQ(r("p - (q + x)"), "p - (q + x)");
    EXPECT_EQ(r("p + 1j * q"), "p + 1j * q");
    EXPECT_EQ(r("(p * q)**x"), "(p * q)**x");
    EXPECT_EQ(r("p**(q + 1)"), "p**(1 + q)");
    EXPECT_EQ(r("p**(2 * q)"), "p**(2 * q)");
    EXPECT_EQ(r("Sin(p)**2 + Cos(p)"), "Sin(p)**2 + Cos(p)");
}

TEST_F(ExpressionTest, WrappingRejectsANodeThatIsNotACapsule) {
    py::exec(R"(
        class NoCapsule(Expression):
            _node = 5
    )");
    EXPECT_THROW(node_of(ev("NoCapsule()")), py::error_already_set);
}

TEST_F(ExpressionTest, NanAndSignedZeroConstantsAreStructurallyEqual) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const NodePtr first = make_constant(Num::real(nan));
    const NodePtr second = make_constant(Num::real(-nan));
    EXPECT_TRUE(equal(first, second));
    EXPECT_EQ(first->hash, second->hash);
    EXPECT_TRUE(equal(make_constant(Num::real(-0.0)), make_constant(Num::integer(0))));
    // Integers past 2**53 share a double, which must not make them equal
    EXPECT_FALSE(equal(make_constant(Num::integer(int64_t{1} << 53)), make_constant(Num::integer((int64_t{1} << 53) + 1))));
    EXPECT_FALSE(equal(first, make_constant(Num::real(1.0))));
    EXPECT_EQ(make_constant(Num::real(-0.0))->hash, make_constant(Num::real(0.0))->hash);
    // NaN sorts after every number, consistently in both directions
    const NodePtr one = make_constant(Num::real(1.0));
    EXPECT_EQ(compare(first, second), 0);
    EXPECT_EQ(compare(first, one), 1);
    EXPECT_EQ(compare(one, first), -1);
    EXPECT_EQ(compare(make_constant(Num::real(2.0)), one), 1);
    EXPECT_EQ(compare(one, make_constant(Num::real(2.0))), -1);
    EXPECT_EQ(compare(make_constant(Num::complex(1.0, 2.0)), make_constant(Num::complex(1.0, 1.0))), 1);
    EXPECT_EQ(r("float('nan') * x + 2 * y"), "2 * y + nan * x");
}

TEST_F(ExpressionTest, LongProductsAndSumsSpillAndSort) {
    // More than 8 factors spill the inline buffers, and more than 16 out of order take the stable sort
    py::exec(R"(
        many = [BinaryVariable(f"v{i:02d}") for i in range(20)]
    )");
    py::list reversed = ev("many[::-1]");
    py::object product = python::build_mul(reversed);
    EXPECT_EQ(py::len(python::args(product)), 20u);
    EXPECT_EQ(python::repr(python::args(product)[0]), "v00");
    py::object sum = python::build_add(ev("[(i + 1) * v for i, v in enumerate(many[::-1])]"));
    EXPECT_EQ(python::repr(python::args(sum)[0]), "v19");
}

TEST_F(ExpressionTest, MergedMonomialsKeepOrRebuildTheirNode) {
    py::object xy = python::mul(g("x"), g("y"));
    py::object twice = python::mul(xy, py::int_(2));
    py::object minus = python::mul(xy, py::int_(-2));
    // A scaled monomial merged back to coefficient 1 needs a node of its own
    EXPECT_TRUE(python::equal(python::build_add(py::make_tuple(python::mul(xy, py::int_(3)), minus)), xy));
    // A plain one keeps its node
    EXPECT_TRUE(python::equal(python::build_add(py::make_tuple(xy, twice, minus)), xy));
}

TEST_F(ExpressionTest, ComparingANodeWithItselfOrAnEqualCopy) {
    const NodePtr sum = node_of(python::add(g("x"), g("y")));
    EXPECT_EQ(compare(sum, sum), 0);
    const NodePtr copy = node_of(python::add(g("y"), g("x")));
    EXPECT_EQ(compare(sum, copy), 0);
    EXPECT_EQ(python::repr(python::build_add(py::make_tuple(python::pow(python::add(g("x"), g("y")), py::int_(3)), python::pow(python::add(g("y"), g("x")), py::int_(2))))), "(x + y)**2 + (x + y)**3");
}

TEST_F(ExpressionTest, LabelsCompareByTheirFirstEightBytesThenInFull) {
    py::exec(R"(
        short, longer = BinaryVariable("abc"), BinaryVariable("abc\x00")
        long_a, long_b = BinaryVariable("prefix__a"), BinaryVariable("prefix__b")
    )");
    EXPECT_EQ(compare(node_of(g("short")), node_of(g("longer"))), -1);
    EXPECT_EQ(compare(node_of(g("longer")), node_of(g("short"))), 1);
    EXPECT_EQ(compare(node_of(g("long_a")), node_of(g("long_b"))), -1);
    EXPECT_EQ(compare(node_of(g("long_b")), node_of(g("long_a"))), 1);
    EXPECT_EQ(compare(node_of(g("long_a")), node_of(ev("BinaryVariable('prefix__a')"))), 0);
    EXPECT_EQ(compare(node_of(g("x")), node_of(g("y"))), -1);
}

TEST_F(ExpressionTest, ArgsHandBackTheLiveWrapperOfEachOperand) {
    py::exec("shared = Sin(x + y)\nparent = shared * z + shared");
    EXPECT_TRUE(python::args(ev("parent"))[0].is(python::args(ev("parent"))[0]));
    EXPECT_TRUE(wrap_shared(node_of(g("x"))).is(g("x")));
    // Once the wrapper dies, the next call makes a new one
    const NodePtr node = node_of(ev("Cos(x)"));
    py::object wrapper = wrap_shared(node);
    EXPECT_TRUE(wrap_shared(node).is(wrapper));
    wrapper = py::none();
    EXPECT_TRUE(python::equal(wrap_shared(node), ev("Cos(x)")));
}
