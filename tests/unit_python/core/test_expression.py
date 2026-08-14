# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the :mod:`qilisdk.core.expression` expression-tree AST."""

import math

import numpy as np
import pytest

from qilisdk.core.exceptions import NonPolynomialError, NotSupportedOperation
from qilisdk.core.expression import Abs, Add, Constant, Cos, Exp, Expression, Inv, Log, Mul, Pow, Sin, Sqrt, Tan
from qilisdk.core.variables import BinaryVariable, Domain, Parameter, Variable
from qilisdk.utils.serialization import deserialize, serialize


@pytest.fixture
def params():
    return Parameter("x", 1.0), Parameter("y", 2.0), Parameter("z", 3.0)


# --------------------------------------------------------------------------- canonicalization & equality
def test_addition_is_order_independent(params):
    x, y, _ = params
    assert (x + y) == (y + x)
    assert hash(x + y) == hash(y + x)


def test_multiplication_is_order_independent(params):
    x, y, _ = params
    assert (x * y) == (y * x)
    assert hash(x * y) == hash(y * x)


def test_like_terms_combine(params):
    x, _, _ = params
    assert (x + x) == (2 * x)
    assert (x + 3 * x) == (4 * x)
    assert (x - x) == Constant(0)


def test_like_powers_collect(params):
    x, _, _ = params
    assert (x * x) == (x**2)
    assert (x * x * x) == (x**3)


def test_constants_fold():
    assert (Constant(1) + Constant(1)) == Constant(2)
    assert (Constant(2) * Constant(3)) == Constant(6)
    assert (Constant(2) ** Constant(3)) == Constant(8)
    assert Constant(2) == Constant(2.0) == Constant(2 + 0j)


def test_identity_and_zero_elimination(params):
    x, _, _ = params
    assert (x + 0) == x
    assert (x * 1) == x
    assert (x * 0) == Constant(0)
    assert (x**1) == x
    assert (x**0) == Constant(1)


def test_binary_variable_is_idempotent_under_mul():
    b = BinaryVariable("b")
    assert (b * b) == b
    assert (b**3) == b


def test_safe_power_merge_only_for_integers(params):
    x, _, _ = params
    assert ((x**2) ** 3) == (x**6)
    # unsafe over reals: (x**2)**0.5 != x  -- left as an inert Pow node
    assert ((x**2) ** 0.5) != x


def test_mul_does_not_distribute(params):
    # Intentional behaviour change vs the old flattened Term model.
    x, y, z = params
    assert (x * (y + z)) != (x * y + x * z)
    assert (x * (y + z)).expand() == (x * y + x * z)


# --------------------------------------------------------------------------- evaluation
def test_evaluate_uses_parameter_values(params):
    x, y, _ = params  # x=1, y=2
    assert (x + 2 * y).evaluate() == 5.0


def test_evaluate_with_environment(params):
    x, y, _ = params
    assert (x * y).evaluate({x: 3, y: 4}) == 12.0


def test_evaluate_function():
    x = Parameter("x", 0.0)
    assert Sin(x).evaluate() == 0.0
    assert Cos(x).evaluate() == 1.0


def test_constant_folding_of_functions():
    assert Sin(0) == Constant(0.0)
    assert isinstance(Cos(0), Constant)


# --------------------------------------------------------------------------- degree
def test_degree(params):
    x, y, _ = params
    assert Constant(5).degree == 0
    assert x.degree == 1
    assert (x * y).degree == 2
    assert (x**3).degree == 3
    assert (x * y + x).degree == 2


def test_degree_of_non_polynomial_raises(params):
    x, _, _ = params
    with pytest.raises(NonPolynomialError):
        _ = (x**0.5).degree
    with pytest.raises(NonPolynomialError):
        _ = Sin(x).degree


# --------------------------------------------------------------------------- symbolic powers (new capability)
def test_non_integer_and_symbolic_powers(params):
    x, y, _ = params
    assert isinstance(x**0.5, Pow)
    assert isinstance(x**y, Pow)
    assert isinstance(x**-1, Pow)  # previously raised
    assert (x**0.5).evaluate({x: 4.0}) == pytest.approx(2.0)


# --------------------------------------------------------------------------- differentiation
def test_diff_rules(params):
    x, y, _ = params
    assert x.diff(x) == Constant(1)
    assert x.diff(y) == Constant(0)
    assert (x * y).diff(x) == y
    assert (x**2).diff(x) == (2 * x)
    assert (x + y).diff(x) == Constant(1)


def test_diff_chain_rule(params):
    x, _, _ = params
    assert Sin(x).diff(x) == Cos(x)
    assert Cos(x).diff(x) == -Sin(x)
    assert Exp(x).diff(x) == Exp(x)
    # numerical check of the chain rule on sin(x**2): 2x cos(x**2)
    d = Sin(x**2).diff(x)
    assert d.evaluate({x: 1.3}) == pytest.approx(2 * 1.3 * math.cos(1.3**2))


# --------------------------------------------------------------------------- expand / accessors
def test_expand_binomial(params):
    x, y, _ = params
    assert ((x + y) ** 2).expand() == (x**2 + 2 * x * y + y**2)


def test_polynomial_accessors(params):
    x, y, _ = params
    expr = 2 * x * y + 3 * x + 5
    assert expr.get_constant() == 5
    coeffs = expr.as_coefficients_dict()
    assert coeffs[x] == 3
    assert coeffs[x * y] == 2


def test_free_symbols_and_parameters(params):
    x, y, _ = params
    b = BinaryVariable("b")
    expr = x + 2 * b
    assert expr.free_symbols() == {x, b}
    assert expr.free_parameters() == {x}
    assert expr.variables() == [b, x]  # sorted by label
    assert (x + y).is_parameterized()
    assert not expr.is_parameterized()


def test_substitute(params):
    x, y, _ = params
    assert (x + y).substitute({x: 2 * y}) == (3 * y)


def test_to_binary_encodes_variables():
    v = Variable("v", domain=Domain.POSITIVE_INTEGER, bounds=(0, 3))
    binary = v.to_binary()
    assert all(isinstance(s, BinaryVariable) for s in binary.free_symbols())


# --------------------------------------------------------------------------- serialization
@pytest.mark.parametrize(
    "expr_factory",
    [
        lambda x, y: 2 * x + 3 * y + 1,
        lambda x, y: x * y - x**2,
        lambda x, y: Sin(x) + Cos(y),
        lambda x, y: Exp(x * y) + Log(x + 2),
        lambda x, y: (x + y) ** 3,
    ],
)
def test_yaml_round_trip(params, expr_factory):
    x, y, _ = params
    expr = expr_factory(x, y)
    restored = deserialize(serialize(expr))
    assert restored == expr
    assert isinstance(restored, Expression)


def test_simplify_does_not_affect_equality(params):
    x, _, _ = params
    expr = x * (x + 1)
    # simplify/expand may differ structurally from the unexpanded form
    assert expr.expand() != expr
    # but evaluating both agrees
    assert expr.evaluate({x: 4.0}) == expr.expand().evaluate({x: 4.0})


# --------------------------------------------------------------------------- derivatives
@pytest.mark.parametrize(
    "factory",
    [
        Sin,
        Cos,
        Exp,
        Log,
        Tan,
        Sqrt,
        Inv,
        lambda x: x**3,
        lambda x: x**-2,
        lambda x: x**0.5,
        lambda x: Sin(x**2),
        lambda x: Sin(x) * Cos(x),
        lambda x: x**x,
        lambda x: 2**x,
        lambda x: Log(Sqrt(x)) + Tan(x),
    ],
)
def test_diff_matches_finite_differences(params, factory):
    x, _, _ = params
    expr = factory(x)
    at, h = 0.7, 1e-6
    numeric = (expr.evaluate({x: at + h}) - expr.evaluate({x: at - h})) / (2 * h)
    assert expr.diff(x).evaluate({x: at}) == pytest.approx(numeric, rel=1e-4)


def test_diff_with_symbolic_exponent(params):
    x, y, _ = params
    # d/dx x**y = x**y * (y' ln x + y/x); with y independent of x this is y * x**(y-1).
    assert (x**y).diff(x).evaluate({x: 2.0, y: 3.0}) == pytest.approx(3 * 2.0**2)
    # d/dy x**y = x**y * ln x
    assert (x**y).diff(y).evaluate({x: 2.0, y: 3.0}) == pytest.approx(2.0**3 * math.log(2.0))


def test_diff_of_abs_is_not_supported(params):
    x, _, _ = params
    expr = Abs(x)
    with pytest.raises(NotSupportedOperation, match=r"The derivative of abs is not supported."):
        expr.diff(x)


def test_diff_of_a_constant_and_an_unrelated_symbol(params):
    x, y, _ = params
    assert Constant(7).diff(x) == Constant(0)
    assert Sin(y).diff(x) == Constant(0)


# --------------------------------------------------------------------------- simplify
def test_simplify_recurses_into_every_node_type(params):
    x, y, _ = params
    assert (Sin(x * 1) + 0).simplify() == Sin(x)
    assert ((x + 0) * (y * 1)).simplify() == x * y
    assert ((x + 0) ** (y * 1)).simplify() == x**y
    assert (x - x).simplify() == Constant(0)


def test_simplify_leaves_semantics_untouched(params):
    x, y, _ = params
    expr = Sin(x) ** 2 + Cos(x) ** 2 + x * y
    assert expr.simplify().evaluate({x: 0.3, y: 1.5}) == pytest.approx(expr.evaluate({x: 0.3, y: 1.5}))


# --------------------------------------------------------------------------- substitute
def test_substitute_into_every_node_type(params):
    x, y, _ = params
    assert (x + y).substitute({x: 5}) == 5 + y
    assert (x * y).substitute({x: y}) == y**2
    assert (x**y).substitute({y: 2}) == x**2
    assert Sin(x).substitute({x: y}) == Sin(y)


def test_substitute_replaces_a_whole_subtree(params):
    x, y, _ = params
    assert Sin(x).substitute({Sin(x): y}) == y
    assert (x + y).substitute({x + y: 1}) == Constant(1)
    assert (x * y).substitute({x * y: 1}) == Constant(1)
    assert (x**2).substitute({x**2: 9}) == Constant(9)


def test_substitute_ignores_unmappable_replacements(params):
    x, y, _ = params
    # A replacement that is neither an Expression nor a number leaves the node alone.
    assert x.substitute({x: "not a number"}) == x
    assert (x + y).substitute({}) == x + y


# --------------------------------------------------------------------------- expand / monomials
def test_expand_is_a_noop_for_non_integer_powers(params):
    x, y, _ = params
    assert ((x + y) ** -1).expand() == (x + y) ** -1
    assert ((x + y) ** 0.5).expand() == (x + y) ** 0.5
    assert ((x + y) ** x).expand() == (x + y) ** x


def test_expand_recurses_into_functions(params):
    x, y, _ = params
    assert Sin(x * (y + 1)).expand() == Sin(x + x * y)


def test_monomial_factors_of_a_power(params):
    x, _, _ = params
    assert (x**3).monomial_factors() == [(x, 3)]
    for expr in (x**-1, x**0.5):
        with pytest.raises(NonPolynomialError, match=r"is not a monomial with an integer power"):
            expr.monomial_factors()


def test_a_function_of_a_constant_folds_to_a_constant():
    folded = Sin(Constant(2))
    assert isinstance(folded, Constant)
    assert folded.degree == 0
    assert Constant(3).degree == 0


# --------------------------------------------------------------------------- to_binary
def test_to_binary_recurses_through_functions_and_powers():
    v = Variable("v", domain=Domain.POSITIVE_INTEGER, bounds=(0, 3))
    for expr in (Sin(v), v**2, v + 1):
        assert all(isinstance(symbol, BinaryVariable) for symbol in expr.to_binary().free_symbols())


# --------------------------------------------------------------------------- guards and coercion
def test_zero_base_with_a_negative_exponent_is_an_error(params):
    x, _, _ = params
    with pytest.raises(ValueError, match=r"Division by zero is not allowed"):
        (x**-1).evaluate({x: 0})
    with pytest.raises(ValueError, match=r"Division by zero is not allowed"):
        _ = Constant(0) ** -1
    with pytest.raises(ValueError, match=r"Division by zero is not allowed"):
        _ = x / 0


def test_tan_rejects_the_poles(params):
    x, _, _ = params
    expr = Tan(x)
    with pytest.raises(ValueError, match=r"Tangent is not defined"):
        expr.evaluate({x: math.pi / 2})


def test_numeric_coercion(params):
    x, _, _ = params
    assert x + True == x + 1
    assert x + np.float64(1.5) == x + 1.5
    assert Constant(np.int64(3)).value == 3
    assert Constant(True).value == 1
    # a complex value with a negligible imaginary part collapses to a real one
    assert Constant(2 + 0j).value == 2.0
    assert (x * Constant(1j)).evaluate({x: 2.0}) == 2j


def test_evaluation_drops_a_negligible_imaginary_part(params):
    x, _, _ = params
    # Squaring an imaginary coefficient gives a real result that arrives as a complex number.
    expr = (Constant(1j) * x) ** 2
    result = expr.evaluate({x: 2.0})
    assert result == -4.0
    assert isinstance(result, float)


def test_one_to_a_symbolic_power_is_one(params):
    _, y, _ = params
    assert 1**y == Constant(1)
    assert Pow.build(Constant(1), y) == Constant(1)


def test_unsupported_operands_return_not_implemented(params):
    x, _, _ = params
    for op in ("__add__", "__sub__", "__mul__", "__pow__"):
        assert getattr(x, op)("not a number") is NotImplemented
    with pytest.raises(NotSupportedOperation):
        _ = 3 // x


def test_empty_products_and_sums_are_identities():
    assert Add.build(()) == Constant(0)
    assert Mul.build(()) == Constant(1)


def test_a_product_of_the_same_function_is_a_square(params):
    # The old flattened Term model turned this into 2 * sin(x) and evaluated it wrongly.
    x, _, _ = params
    assert Sin(x) * Sin(x) == Sin(x) ** 2
    assert (Sin(x) * Sin(x)).evaluate({x: 0.5}) == pytest.approx(math.sin(0.5) ** 2)


# --------------------------------------------------------------------------- immutability
@pytest.mark.parametrize(
    ("factory", "attribute"),
    [
        (lambda x: Constant(2), "value"),
        (lambda x: x + 1, "args"),
        (lambda x: 2 * x, "args"),
        (lambda x: x**2, "base"),
        (lambda x: x**2, "exp"),
        (Sin, "arg"),
    ],
)
def test_node_payloads_are_read_only(params, factory, attribute):
    x, _, _ = params
    node = factory(x)
    # The hash is cached on first use, so a writable payload would let it go stale.
    before = hash(node)
    with pytest.raises(AttributeError):
        setattr(node, attribute, None)
    assert hash(node) == before
    assert getattr(node, attribute) is not None


# --------------------------------------------------------------------------- to_list
def test_to_list_returns_the_operands(params):
    x, y, _ = params
    assert (x + 2 * y + 1).to_list() == [Constant(1), x, 2 * y]
    assert (2 * x * y).to_list() == [Constant(2), x, y]
    # A leaf has no operands, so it yields itself.
    assert x.to_list() == [x]
    assert Constant(3).to_list() == [Constant(3)]
    assert Sin(x).to_list() == [Sin(x)]


# --------------------------------------------------------------------------- printing
def test_sums_print_negative_terms_with_a_minus(params):
    x, y, _ = params
    b = BinaryVariable("b")
    assert repr(-2 - 2 * b) == "-2 - 2 * b"
    assert repr(x - y) == "x - y"
    assert repr(x + y) == "x + y"
    assert repr(-x - y) == "-x - y"
    assert repr(x - Sin(y)) == "x - sin(y)"
    # A coefficient of exactly -1 reads as a leading minus, other coefficients keep the product.
    assert repr(-x) == "-x"
    assert repr(-2 * x) == "-2 * x"
    assert repr(-(x + y)) == "-(x + y)"
