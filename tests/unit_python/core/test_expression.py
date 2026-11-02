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

import copy
import itertools
import math
import pickle
import random

import numpy as np
import pytest

from qilisdk.core.exceptions import EvaluationError, NonPolynomialError, NotSupportedOperation
from qilisdk.core.expression import (
    Abs,
    Add,
    Constant,
    Cos,
    Exp,
    Expression,
    Function,
    Inv,
    Log,
    Mul,
    Pow,
    Sin,
    Sqrt,
    Tan,
)
from qilisdk.core.variables import BinaryVariable, Domain, Parameter, SpinVariable, Variable
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
    assert x.derivative(x) == Constant(1)
    assert x.derivative(y) == Constant(0)
    assert (x * y).derivative(x) == y
    assert (x**2).derivative(x) == (2 * x)
    assert (x + y).derivative(x) == Constant(1)


def test_derivative_of_a_function_without_the_symbol_is_zero(params):
    x, y, _ = params
    # Abs has no derivative, but nothing has to be differentiated when x does not occur in it.
    assert (Abs(y) + 1).derivative(x) == Constant(0)
    abs_y = Abs(y)
    assert (abs_y * x).derivative(x) == abs_y
    with pytest.raises(NotSupportedOperation, match="derivative of Abs"):
        abs_y.derivative(y)


def test_functions_survive_copying(params):
    x, _, _ = params
    expr = Sin(x) ** 2 + Cos(x)
    assert copy.deepcopy(expr) == expr
    assert pickle.loads(pickle.dumps(expr)) == expr


def test_diff_chain_rule(params):
    x, _, _ = params
    assert Sin(x).derivative(x) == Cos(x)
    assert Cos(x).derivative(x) == -Sin(x)
    assert Exp(x).derivative(x) == Exp(x)
    # numerical check of the chain rule on sin(x**2): 2x cos(x**2)
    d = Sin(x**2).derivative(x)
    assert d.evaluate({x: 1.3}) == pytest.approx(2 * 1.3 * math.cos(1.3**2))


# --------------------------------------------------------------------------- expand / accessors
def test_expand_binomial(params):
    x, y, _ = params
    assert ((x + y) ** 2).expand() == (x**2 + 2 * x * y + y**2)


def test_expand_expands_a_float_valued_integer_power(params):
    x, _, _ = params
    assert (x**2.0).expand() == (x * x)


@pytest.mark.parametrize("exponent", [0.5, -1.5])
def test_expand_leaves_a_non_integer_power_inert(params, exponent):
    x, _, _ = params
    expr = x**exponent
    assert expr.expand() == expr
    assert isinstance(expr.expand(), Pow)


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


def test_expand_does_not_affect_equality(params):
    x, _, _ = params
    expr = x * (x + 1)
    # expand may differ structurally from the unexpanded form
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
    assert expr.derivative(x).evaluate({x: at}) == pytest.approx(numeric, rel=1e-4)


def test_diff_with_symbolic_exponent(params):
    x, y, _ = params
    # d/dx x**y = x**y * (y' ln x + y/x); with y independent of x this is y * x**(y-1).
    assert (x**y).derivative(x).evaluate({x: 2.0, y: 3.0}) == pytest.approx(3 * 2.0**2)
    # d/dy x**y = x**y * ln x
    assert (x**y).derivative(y).evaluate({x: 2.0, y: 3.0}) == pytest.approx(2.0**3 * math.log(2.0))


def test_diff_of_abs_is_not_supported(params):
    x, _, _ = params
    expr = Abs(x)
    with pytest.raises(NotSupportedOperation, match=r"The derivative of Abs is not supported."):
        expr.derivative(x)


def test_diff_of_a_constant_and_an_unrelated_symbol(params):
    x, y, _ = params
    assert Constant(7).derivative(x) == Constant(0)
    assert Sin(y).derivative(x) == Constant(0)


# --------------------------------------------------------------------------- canonicalization on construction
def test_construction_already_normalizes_every_node_type(params):
    x, y, _ = params
    # There is no separate simplify() pass: building the node is what normalizes it.
    assert (Sin(x * 1) + 0) == Sin(x)
    assert ((x + 0) * (y * 1)) == x * y
    assert ((x + 0) ** (y * 1)) == x**y
    assert (x - x) == Constant(0)
    assert (2 * x + 3 * x) == 5 * x
    assert (x / x) == Constant(1)


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
    assert repr(x - Sin(y)) == "x - Sin(y)"
    # A coefficient of exactly -1 reads as a leading minus, other coefficients keep the product.
    assert repr(-x) == "-x"
    assert repr(-2 * x) == "-2 * x"
    assert repr(-(x + y)) == "-(x + y)"


def test_a_subtracted_sum_keeps_its_parentheses(params):
    x, y, z = params
    # "a - (b + c)" is not "a - b + c", so the parentheses are load bearing.
    assert repr(x - (y + z)) == "x - (y + z)"
    assert repr(x - (y + 1)) == "x - (1 + y)"
    # and the printed form still agrees with what the expression evaluates to
    expr = x - (y + z)
    assert expr.evaluate({x: 5.0, y: 2.0, z: 1.0}) == 5.0 - (2.0 + 1.0)


def test_a_power_of_a_power_keeps_its_parentheses(params):
    x, _, _ = params
    # ``**`` is right-associative, so "x**2**0.5" would read back as x**(2**0.5).
    assert repr((x**2) ** 0.5) == "(x**2)**0.5"
    assert repr(x ** (2**0.5)) == "x**1.4142135623730951"
    assert ((x**2) ** 0.5).evaluate({x: 9.0}) == pytest.approx(9.0)


def test_functions_print_the_name_you_construct_them_with(params):
    x, _, _ = params
    assert repr(Sin(x)) == "Sin(x)"
    assert repr(Sqrt(x)) == "Sqrt(x)"
    assert repr(Abs(x)) == "Abs(x)"
    assert repr(Sin(x) ** 2 + Cos(x)) == "Sin(x)**2 + Cos(x)"


def test_derivative_of_a_power(params):
    x, _, _ = params
    assert (x**3).derivative(x) == 3 * x**2


# --------------------------------------------------------------------------- trusting constructors
def test_trusting_constructors_keep_their_operands(params):
    x, y, _ = params
    assert Add((x, y)) == x + y
    assert Mul((Constant(2), x)) == 2 * x
    assert Pow(x, Constant(3)) == x**3


def test_a_product_splits_into_coefficient_and_monomial(params):
    x, y, _ = params
    product = 3 * x * y
    assert product.coefficient() == 3
    assert product.monomial() == x * y
    assert (x * y).coefficient() == 1
    assert (x * y).monomial() == x * y


@pytest.fixture
def binaries():
    return BinaryVariable("a"), BinaryVariable("b"), BinaryVariable("c")


# --------------------------------------------------------------------------- numbers
def test_integer_coefficients_stay_integers(params):
    x, _, _ = params
    expr = 2 * x * 3
    assert isinstance(expr.coefficient(), int)
    assert repr(expr) == "6 * x"


def test_float_coefficients_stay_floats(params):
    x, _, _ = params
    assert repr(2.0 * x) == "2.0 * x"
    assert repr(0.5 * x + 0.5 * x) == "x"


def test_integer_overflow_falls_back_to_a_float(params):
    x, _, _ = params
    huge = 2**62
    product = (huge * x) * 4
    assert isinstance(product.coefficient(), float)
    assert product.coefficient() == pytest.approx(2.0**64)


def test_an_integer_beyond_64_bits_becomes_a_float():
    assert Constant(2**70).value == pytest.approx(2.0**70)
    assert isinstance(Constant(2**70).value, float)


def test_an_integer_beyond_the_float_range_is_rejected():
    huge = 10**400
    with pytest.raises(OverflowError):
        Constant(huge)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (np.float32(0.5), 0.5),
        (np.int8(-3), -3),
        (np.bool_(True), 1),
        (np.complex128(2 + 0j), 2.0),
        (np.uint64(7), 7),
    ],
)
def test_numpy_scalars_become_plain_numbers(raw, expected):
    value = Constant(raw).value
    assert value == expected
    assert type(value) in {int, float}


def test_a_numpy_complex_with_an_imaginary_part_stays_complex():
    assert Constant(np.complex128(1 + 2j)).value == 1 + 2j


def test_a_negligible_imaginary_part_is_dropped():
    assert Constant(3 + 1e-15j).value == 3.0
    assert isinstance(Constant(3 + 1e-15j).value, float)


def test_signed_zeros_are_the_same_constant():
    assert Constant(-0.0) == Constant(0)
    assert hash(Constant(-0.0)) == hash(Constant(0.0))


def test_nan_constants_are_structurally_equal():
    nan = float("nan")
    first, second = Constant(nan), Constant(-nan)
    assert first == second
    assert hash(first) == hash(second)
    assert math.isnan(Constant(nan).evaluate())


def test_nan_coefficients_sort_and_compare_consistently(binaries):
    a, b, _ = binaries
    nan = float("nan")
    assert nan * a + nan * b == nan * b + nan * a
    assert (nan * a + 2 * b).args == (2 * b + nan * a).args


def test_infinite_coefficients_are_kept(binaries):
    a, _, _ = binaries
    expr = math.inf * a
    assert repr(expr) == "inf * a"
    assert expr.evaluate({a: 1}) == math.inf


def test_complex_coefficients(binaries):
    a, b, _ = binaries
    expr = (1 + 2j) * a - 1j * b
    assert repr(expr) == "-1j * b + (1+2j) * a"
    assert expr.evaluate({a: 1, b: 1}) == 1 + 1j


def test_opposite_complex_coefficients_cancel(binaries):
    a, _, _ = binaries
    assert 1j * a - 1j * a == Constant(0)


def test_imaginary_coefficients_that_square_to_a_real_collapse(params):
    x, _, _ = params
    expr = (1j * x) * (1j * x)
    assert expr == -(x**2)
    # 1j * 1j is the complex (-1+0j), which collapses to the float -1.0
    assert expr.coefficient() == -1
    assert isinstance(expr.coefficient(), float)


def test_floating_point_cancellation_follows_float_arithmetic(params):
    x, _, _ = params
    # 0.1 + 0.2 - 0.3 is not exactly zero in floating point, and the coefficient keeps that residue
    residue = 0.1 * x + 0.2 * x - 0.3 * x
    assert residue != Constant(0)
    assert residue.coefficient() == pytest.approx(0.1 + 0.2 - 0.3)
    assert 0.5 * x + 0.25 * x - 0.75 * x == Constant(0)


# --------------------------------------------------------------------------- constant powers
@pytest.mark.parametrize(
    ("base", "exponent", "expected"),
    [
        (2, 10, 1024),
        (2, -1, 0.5),
        (4, 0.5, 2.0),
        (-2, 3, -8),
        (0, 0, 1),
        (1.5, 2, 2.25),
        (1j, 2, -1),
    ],
)
def test_constant_powers_match_python(base, exponent, expected):
    result = Constant(base) ** exponent
    assert isinstance(result, Constant)
    assert result.value == pytest.approx(expected)


def test_a_negative_base_under_a_fractional_power_is_complex():
    result = (Constant(-8) ** (1 / 3)).value
    assert isinstance(result, complex)
    assert result == pytest.approx((-8) ** (1 / 3))


def test_a_complex_power_matches_python():
    result = (Constant(1 + 1j) ** 0.5).value
    assert result == pytest.approx((1 + 1j) ** 0.5)
    assert (Constant(2) ** Constant(1j)).value == pytest.approx(2**1j)


def test_a_tiny_base_under_a_negative_power_is_a_division_by_zero():
    tiny = Constant(1e-15)
    with pytest.raises(ValueError, match="Division by zero"):
        _ = tiny**-1


# --------------------------------------------------------------------------- division
def test_division_by_a_number_scales_by_its_reciprocal(params):
    x, _, _ = params
    assert x / 4 == 0.25 * x
    assert x / True == x
    assert x / np.float64(2.0) == 0.5 * x


@pytest.mark.parametrize("divisor", [0, 0.0, 1e-15, np.float64(0.0), False])
def test_division_by_a_vanishing_number_is_rejected(params, divisor):
    x, _, _ = params
    with pytest.raises(ValueError, match="Division by zero"):
        _ = x / divisor


def test_division_by_a_complex_number_uses_a_negative_power(params):
    x, _, _ = params
    assert x / 2j == -0.5j * x


def test_division_by_an_expression_is_a_negative_power(params):
    x, y, _ = params
    assert x / y == x * y**-1
    assert 1 / x == x**-1
    assert x / x == Constant(1)
    assert (x * y) / y == x


def test_division_by_a_numpy_integer_goes_through_a_power(params):
    x, _, _ = params
    assert x / np.int64(4) == 0.25 * x


def test_floor_division_is_rejected_both_ways(params):
    x, _, _ = params
    with pytest.raises(NotSupportedOperation):
        _ = x // 2
    with pytest.raises(NotSupportedOperation):
        _ = 2 // x


# --------------------------------------------------------------------------- operands Python must handle
@pytest.mark.parametrize("other", [None, "x", [1], {1: 2}, object()])
def test_non_numeric_operands_raise_type_error(params, other):
    x, _, _ = params
    with pytest.raises(TypeError):
        _ = x + other
    with pytest.raises(TypeError):
        _ = other * x


@pytest.mark.parametrize(
    "op",
    [
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__pow__",
        "__rpow__",
    ],
)
def test_every_operator_defers_on_unknown_operands(params, op):
    x, _, _ = params
    assert getattr(x, op)(object()) is NotImplemented


def test_in_place_operators_do_not_mutate_shared_expressions(params):
    x, y, z = params
    original = x + y
    alias = original
    alias += z
    alias *= 2
    alias -= x
    alias /= 2
    assert original == x + y
    assert alias == ((x + y + z) * 2 - x) / 2


# --------------------------------------------------------------------------- canonical form
def test_reversed_operands_give_the_same_expression(params):
    x, _, _ = params
    assert 1 + x == x + 1
    assert 2 - x == -x + 2
    assert 3 * x == x * 3
    assert 2**x == Pow.build(Constant(2), x)


def test_sums_and_products_are_associative(params):
    x, y, z = params
    assert (x + y) + z == x + (y + z)
    assert (x * y) * z == x * (y * z)


def test_nested_sums_and_products_flatten(params):
    x, y, z = params
    assert Add.build([x, Add.build([y, Add.build([z, 1])])]).to_list() == [Constant(1), x, y, z]
    assert Mul.build([x, Mul.build([y, Mul.build([z, 2])])]).to_list() == [Constant(2), x, y, z]


def test_builders_accept_any_iterable(params):
    x, y, _ = params
    assert Add.build(term for term in (x, y)) == x + y
    assert Mul.build([x, 2, 3.0]) == 6.0 * x


def test_builders_reject_non_numeric_items(params):
    x, _, _ = params
    items = [x, "y"]
    with pytest.raises(TypeError):
        Add.build(items)


def test_a_single_operand_collapses(params):
    x, _, _ = params
    assert Add.build([x]) is x
    assert Mul.build([x]) is x
    assert Add.build([Constant(4)]) == Constant(4)


def test_constants_sort_first_then_variables_by_label_then_compounds(params):
    x, y, _ = params
    expr = Sin(x) + x**2 + 2 * x * y + y + x + 7
    assert [type(term) for term in expr.args] == [Constant, Parameter, Parameter, Pow, Mul, Sin]
    assert repr(expr) == "7 + x + y + x**2 + 2 * x * y + Sin(x)"


def test_labels_sort_as_strings():
    x2, x10, upper = BinaryVariable("x2"), BinaryVariable("x10"), BinaryVariable("X")
    assert [str(v) for v in (x2 + x10 + upper).args] == ["X", "x10", "x2"]


def test_unicode_labels_sort_by_code_point():
    alpha, zed, accented = BinaryVariable("λ"), BinaryVariable("z"), BinaryVariable("é")
    assert [str(v) for v in (alpha + zed + accented).args] == ["z", "é", "λ"]
    assert repr(alpha * accented) == "é * λ"


def test_like_terms_combine_across_variable_objects_with_the_same_label():
    first, second = BinaryVariable("q"), BinaryVariable("q")
    assert first + second == 2 * first
    assert first * second == first


def test_variables_are_identified_by_label_alone():
    assert BinaryVariable("q") == SpinVariable("q")
    assert BinaryVariable("q") != BinaryVariable("r")
    assert hash(BinaryVariable("q")) == hash(Parameter("q", 0.0))


def test_a_spin_is_not_idempotent():
    s = SpinVariable("s")
    assert s * s == s**2
    assert s * s != s


@pytest.mark.parametrize(("exponent", "collapses"), [(2, True), (2.0, True), (5, True), (0.5, False), (-1, False)])
def test_a_binary_collapses_only_under_positive_integer_powers(binaries, exponent, collapses):
    a, _, _ = binaries
    assert (a**exponent == a) is collapses


def test_powers_merge_only_for_integer_exponents(params):
    x, _, _ = params
    assert (x**2) ** -1 == x**-2
    assert (x**-1) ** -1 == x
    assert (x**2) ** 0.5 != x
    assert (x**0.5) ** 2 != x


def test_symbolic_exponents_add_up(params):
    x, y, z = params
    assert x**y * x**z == x ** (y + z)
    assert x**y * x ** (-y) == Constant(1)
    assert x**y * x == x ** (y + 1)


def test_one_to_any_power_and_anything_to_the_zero(params):
    x, y, _ = params
    assert Constant(1) ** y == Constant(1)
    assert x**0 == Constant(1)
    assert (x + y) ** 0 == Constant(1)
    assert x**0.0 == Constant(1)


def test_zero_times_anything_is_zero(params):
    x, _, _ = params
    assert 0 * Sin(x) == Constant(0)
    assert 0.0 * x**x == Constant(0)
    assert isinstance((0 * x).value, int)


def test_subtracting_a_sum_only_cancels_once_expanded(params):
    x, y, _ = params
    # Subtracting a sum scales it by -1, and Mul does not distribute
    difference = (x + y + 3) - (y + x)
    assert difference != Constant(3)
    assert difference.expand() == Constant(3)
    assert ((x + 1) - (x + 1)).expand() == Constant(0)


def test_equal_expressions_hash_equally_and_work_as_keys(params):
    x, y, _ = params
    table = {x * y: "product", x + y: "sum", Sin(x): "sine"}
    assert table[y * x] == "product"
    assert table[y + x] == "sum"
    assert table[Sin(x + 0)] == "sine"
    assert len({x + y, y + x, x + y + 0}) == 1


def test_expressions_never_equal_plain_numbers_or_other_types(params):
    x, _, _ = params
    assert Constant(2) != 2
    assert (x + 1) != "x + 1"
    assert x * 2 is not None


def test_distinct_structures_are_unequal(params):
    x, y, _ = params
    assert x * y != x + y
    assert Sin(x) != Cos(x)
    assert x**2 != 2 * x
    assert x**y != y**x


# --------------------------------------------------------------------------- running sums
def test_a_running_sum_matches_building_all_at_once(binaries):
    rng = random.Random(7)
    a, b, c = binaries
    terms = [rng.choice([a, b, c, a * b, b * c, Constant(1)]) * rng.choice([1, -1, 2, 0.5, 1j]) for _ in range(300)]
    running = Constant(0)
    for term in terms:
        running += term
    assert running == Add.build(terms)
    assert repr(running) == repr(Add.build(terms))


def test_a_running_sum_does_not_depend_on_the_order(binaries):
    a, b, c = binaries
    terms = [2 * a, b, -a, c, 3 * a * b, -b, 1, a * b, c]
    expected = sum(terms)
    for order in itertools.islice(itertools.permutations(terms), 0, 2000, 37):
        assert sum(order) == expected


def test_a_large_sum_can_be_built_incrementally():
    variables = [BinaryVariable(f"v{i}") for i in range(2000)]
    total = sum(variables)
    assert len(total.args) == 2000
    assert total.evaluate(dict.fromkeys(variables, 1)) == 2000.0


# --------------------------------------------------------------------------- evaluation
def test_evaluation_always_returns_a_float_for_real_results(binaries):
    a, b, _ = binaries
    assert isinstance((a + b).evaluate({a: 1, b: 0}), float)
    assert isinstance(Constant(3).evaluate(), float)
    assert isinstance((a * 2).evaluate({a: 1}), float)


def test_evaluation_accepts_numpy_values(params):
    x, y, _ = params
    assert (2 * x + y).evaluate({x: np.float64(1.5), y: np.float64(0.5)}) == 3.5


def test_evaluation_accepts_bit_lists_for_variables(binaries):
    a, b, _ = binaries
    assert (a + b).evaluate({a: [1], b: [1]}) == 2.0


def test_a_missing_variable_cannot_be_evaluated(binaries):
    a, b, _ = binaries
    expr = a + b
    env = {a: 1}
    with pytest.raises(EvaluationError):
        expr.evaluate(env)


def test_a_binary_outside_its_domain_cannot_be_evaluated(binaries):
    a, _, _ = binaries
    expr = a + 1
    env = {a: 2}
    with pytest.raises(EvaluationError):
        expr.evaluate(env)


def test_spins_evaluate_from_zero_and_one_bits():
    s = SpinVariable("s")
    expr = 3 * s + 1
    assert expr.evaluate({s: 0}) == -2.0
    assert expr.evaluate({s: 1}) == 4.0
    assert expr.evaluate({s: [1]}) == 4.0


def test_parameters_use_their_value_unless_overridden(params):
    x, y, _ = params
    expr = x * y
    assert expr.evaluate() == 2.0
    assert expr.evaluate({x: 5.0}) == 10.0
    assert expr.evaluate(None) == 2.0


def test_parameter_changes_are_seen_by_existing_expressions(params):
    x, y, _ = params
    expr = x + y
    x.set_value(10.0)
    assert expr.evaluate() == 12.0


def test_a_variable_repeated_through_the_tree_evaluates_consistently(params):
    x, _, _ = params
    expr = x + x**2 + Sin(x) * x
    at = 0.3
    assert expr.evaluate({x: at}) == pytest.approx(at + at**2 + math.sin(at) * at)


def test_evaluating_a_function_of_a_complex_value_is_rejected(params):
    x, _, _ = params
    expr = Sin(1j * x)
    with pytest.raises(ValueError, match="Only real values"):
        expr.evaluate({x: 1.0})


def test_log_and_sqrt_outside_their_domain_give_nan(params):
    x, _, _ = params
    with np.errstate(invalid="ignore", divide="ignore"):
        assert math.isnan(Log(x).evaluate({x: -1.0}))
        assert math.isnan(Sqrt(x).evaluate({x: -1.0}))
        assert Log(x).evaluate({x: 0.0}) == -math.inf


def test_zero_to_a_negative_power_fails_at_evaluation(params):
    x, _, _ = params
    expr = 1 / x
    env = {x: 0.0}
    with pytest.raises(ValueError, match="Division by zero"):
        expr.evaluate(env)


def test_a_negative_base_under_a_fractional_power_evaluates_to_a_complex(params):
    x, _, _ = params
    assert (x**0.5).evaluate({x: -4.0}) == pytest.approx(2j)


def test_evaluation_agrees_with_python_on_random_polynomials():
    rng = random.Random(3)
    symbols = [Parameter(f"t{i}", 0.0) for i in range(4)]
    for _ in range(50):
        values = {symbol: rng.uniform(-2, 2) for symbol in symbols}
        expr, reference = Constant(0), 0.0
        for _ in range(rng.randint(1, 6)):
            coeff = rng.uniform(-3, 3)
            factors = rng.sample(symbols, rng.randint(1, 3))
            powers = [rng.randint(1, 3) for _ in factors]
            term = Constant(coeff)
            value = coeff
            for symbol, power in zip(factors, powers):
                term *= symbol**power
                value *= values[symbol] ** power
            expr += term
            reference += value
        assert expr.evaluate(values) == pytest.approx(reference)
        assert expr.expand().evaluate(values) == pytest.approx(reference)


# --------------------------------------------------------------------------- expand
def test_expanding_a_binomial_gives_binomial_coefficients(params):
    x, _, _ = params
    expanded = ((x + 1) ** 10).expand()
    assert expanded.as_coefficients_dict()[x**5] == math.comb(10, 5)
    assert expanded.get_constant() == 1
    assert expanded.degree == 10


def test_expanding_binaries_uses_idempotence(binaries):
    a, b, _ = binaries
    assert ((a + 1) ** 2).expand() == 3 * a + 1
    assert ((a + b) ** 2).expand() == a + b + 2 * a * b


def test_expanding_spins_does_not(binaries):
    s = SpinVariable("s")
    assert ((s + 1) ** 2).expand() == s**2 + 2 * s + 1


def test_expanding_a_product_of_sums_with_cancellation(params):
    x, y, _ = params
    assert ((x + y) * (x - y)).expand() == x**2 - y**2
    assert ((x + 1) * (x - 1) - x**2).expand() == Constant(-1)


def test_expanding_is_idempotent(params):
    x, y, z = params
    expr = (x + y) * (y + z) ** 2 + Sin(x * (y + 1))
    once = expr.expand()
    assert once.expand() == once


def test_expanding_leaves_symbolic_and_negative_powers_alone(params):
    x, y, _ = params
    assert ((x + y) ** y).expand() == (x + y) ** y
    assert ((x + y) ** -2).expand() == (x + y) ** -2


def test_expanding_inside_a_negative_power(params):
    x, y, _ = params
    assert (((x + 1) * (y + 1)) ** -1).expand() == (x * y + x + y + 1) ** -1


# --------------------------------------------------------------------------- derivatives
def test_differentiating_by_an_absent_symbol_gives_zero(params):
    x, y, _ = params
    assert (x**3 + Sin(x)).derivative(y) == Constant(0)


def test_second_derivatives(params):
    x, _, _ = params
    assert (x**3).derivative(x).derivative(x) == 6 * x
    assert Sin(x).derivative(x).derivative(x) == -Sin(x)


def test_derivative_of_a_product_of_three(params):
    x, y, z = params
    assert (x * y * z).derivative(y) == x * z


def test_derivative_of_an_exponential_of_a_symbol(params):
    x, _, _ = params
    expr = 2**x
    assert expr.derivative(x).evaluate({x: 1.5}) == pytest.approx(2**1.5 * math.log(2))


@pytest.mark.parametrize(
    "factory",
    [
        lambda x, y: Exp(x * y) * Cos(x),
        lambda x, y: Log(x**2 + 1) / (y + x),
        lambda x, y: Sqrt(x**2 + y**2),
        lambda x, y: Tan(x / 3) ** y,
        lambda x, y: x ** Sin(x),
        lambda x, y: Inv(1 + Exp(-x)),
    ],
)
def test_derivatives_match_finite_differences(params, factory):
    x, y, _ = params
    expr = factory(x, y)
    at, h = {x: 0.7, y: 1.3}, 1e-6
    numeric = (expr.evaluate({**at, x: 0.7 + h}) - expr.evaluate({**at, x: 0.7 - h})) / (2 * h)
    assert expr.derivative(x).evaluate(at) == pytest.approx(numeric, rel=1e-5)


def test_derivative_by_a_binary_variable(binaries):
    a, b, _ = binaries
    assert (3 * a * b + a).derivative(a) == 3 * b + 1


def test_derivative_by_a_non_symbol_is_zero(params):
    x, _, _ = params
    assert (x**2).derivative(Constant(2)) == Constant(0)


# --------------------------------------------------------------------------- substitution
def test_substitution_is_simultaneous(params):
    x, y, _ = params
    assert (x - y).substitute({x: y, y: x}) == y - x


def test_substitution_matches_constants_inside_a_sum(params):
    x, _, _ = params
    assert (x + 2).substitute({Constant(2): 5}) == x + 5


def test_substituting_numbers_folds_functions_and_powers(params):
    x, y, _ = params
    assert Sin(x).substitute({x: 0}) == Constant(0.0)
    assert (x**y).substitute({x: 2, y: 3}) == Constant(8)
    assert (x / y).substitute({y: 4}) == 0.25 * x


def test_substituting_numpy_values(params):
    x, _, _ = params
    assert (2 * x).substitute({x: np.float64(1.5)}) == Constant(3.0)


def test_substituting_into_a_zero_division_is_rejected(params):
    x, y, _ = params
    expr = x / y
    mapping = {y: 0}
    with pytest.raises(ValueError, match="Division by zero"):
        expr.substitute(mapping)


def test_substituting_a_whole_product_only_matches_that_product(params):
    x, y, z = params
    assert (x * y * z).substitute({x * y: 2}) == x * y * z
    assert (x * y + z).substitute({x * y: 2}) == z + 2


def test_substitution_leaves_the_original_untouched(params):
    x, y, _ = params
    expr = x + y
    expr.substitute({x: 0})
    assert expr == x + y


# --------------------------------------------------------------------------- to_binary
def test_spins_encode_once_however_often_they_appear():
    s = SpinVariable("s")
    binary = (s + s * s + Sin(s)).to_binary()
    assert {symbol.label for symbol in binary.free_symbols()} == {"s(0)"}


def test_parameters_become_their_value_in_binary_form(params):
    x, _, _ = params
    b = BinaryVariable("b")
    assert (x * b).to_binary() == 1.0 * b


def test_integer_variables_encode_and_evaluate_consistently():
    v = Variable("v", domain=Domain.POSITIVE_INTEGER, bounds=(0, 5))
    binary = (2 * v + 1).to_binary()
    bits = sorted(binary.free_symbols(), key=lambda symbol: symbol.label)
    assert all(isinstance(bit, BinaryVariable) for bit in bits)
    assert binary.evaluate(dict.fromkeys(bits, 0)) == 1.0


def test_binary_form_of_a_binary_expression_is_unchanged(binaries):
    a, b, _ = binaries
    expr = a * b + 3
    assert expr.to_binary() == expr


# --------------------------------------------------------------------------- accessors
def test_degree_of_assorted_expressions(params):
    x, y, _ = params
    assert (x**2.0).degree == 2
    assert ((x + 1) ** 3).degree == 3
    assert (x * (y + x**2)).degree == 3
    assert Constant(1j).degree == 0
    assert (x**0).degree == 0


def test_degree_rejects_non_polynomials(params):
    x, y, _ = params
    for expr in (x**y, x**-1, Exp(x), (x + 1) ** 0.5):
        with pytest.raises(NonPolynomialError):
            _ = expr.degree


def test_monomial_factors_of_assorted_monomials(params):
    x, y, _ = params
    # Variables sort before powers
    assert (3 * x**2 * y).monomial_factors() == [(y, 1), (x, 2)]
    assert Sin(x).monomial_factors() == [(Sin(x), 1)]
    assert (2 * (x + y)).monomial_factors() == [(x + y, 1)]
    assert Constant(5).monomial_factors() == []


def test_coefficients_of_assorted_expressions(params):
    x, y, _ = params
    assert (x**2).as_coefficients_dict() == {x**2: 1}
    assert Sin(x).as_coefficients_dict() == {Sin(x): 1}
    assert (-2 * x * y).as_coefficients_dict() == {x * y: -2}
    assert (x + 1j).as_coefficients_dict() == {x: 1}


def test_constant_terms_of_assorted_expressions(params):
    x, _, _ = params
    assert (2 * x).get_constant() == 0
    assert (x + 1j).get_constant() == 1j
    assert (x**2).get_constant() == 0
    assert Constant(2.5).get_constant() == 2.5


def test_free_symbols_of_assorted_expressions(params):
    x, y, z = params
    assert Constant(1).free_symbols() == set()
    assert (x**y).free_symbols() == {x, y}
    assert Sin(z * x).free_symbols() == {x, z}
    assert (x + x * y + x**2).variables() == [x, y]


def test_membership_tests_symbols_only(params):
    x, y, z = params
    expr = x * y + 2
    assert x in expr
    assert z not in expr
    assert Constant(2) not in expr
    assert "x" not in expr


def test_to_list_of_a_power_and_a_function(params):
    x, _, _ = params
    assert (x**2).to_list() == [x**2]
    assert Exp(x).to_list() == [Exp(x)]


def test_node_payloads_stay_read_only_after_slots(params):
    x, y, _ = params
    expr = x + y
    with pytest.raises(AttributeError):
        expr.extra = 1


# --------------------------------------------------------------------------- printing
@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda x, y: x - 0.5 * y, "x - 0.5 * y"),
        (lambda x, y: -1 + x, "-1 + x"),
        (lambda x, y: x**-1, "x**-1"),
        (lambda x, y: x ** (-y), "x**(-y)"),
        (lambda x, y: Sin(x + y), "Sin(x + y)"),
        (lambda x, y: Sin(x) ** 2, "Sin(x)**2"),
        (lambda x, y: (x + y) * (x - y), "(x + y) * (x - y)"),
        (lambda x, y: -(x * y), "-x * y"),
        (lambda x, y: 2 ** (x + 1), "2**(1 + x)"),
        (lambda x, y: Constant(1e20) * x, "1e+20 * x"),
        # Only a real negative coefficient prints as a subtraction
        (lambda x, y: x - 1j * y, "x + -1j * y"),
    ],
)
def test_printing(params, factory, expected):
    x, y, _ = params
    assert repr(factory(x, y)) == expected


def test_str_matches_repr(params):
    x, y, _ = params
    expr = 2 * x - y
    assert str(expr) == repr(expr)


# --------------------------------------------------------------------------- functions
def test_functions_fold_numpy_and_python_constants():
    assert Exp(0) == Constant(1.0)
    assert Cos(np.float64(0.0)) == Constant(1.0)
    assert Abs(-3) == Constant(3.0)


def test_a_function_of_a_complex_constant_is_rejected():
    with pytest.raises(ValueError, match="Only real values"):
        Sin(1j)


def test_a_function_rejects_non_numeric_operands():
    with pytest.raises(TypeError, match="Sin expects an Expression or number"):
        Sin("x")


def test_the_function_base_cannot_be_instantiated(params):
    x, _, _ = params
    with pytest.raises(TypeError):
        Function(x)


def test_a_custom_function(params):
    x, _, _ = params

    class Cube(Function):
        NAME = "cube"

        @staticmethod
        def _numeric(value):
            return float(value**3)

        def _outer_derivative(self, operand):
            return 3 * operand**2

    assert Cube(2) == Constant(8.0)
    expr = Cube(x + 1)
    assert repr(expr) == "Cube(1 + x)"
    assert expr.evaluate({x: 1.0}) == 8.0
    assert expr.derivative(x) == 3 * (x + 1) ** 2
    assert Function._REGISTRY["cube"] is Cube


def test_deeply_nested_functions(params):
    x, _, _ = params
    expr = x
    for _ in range(200):
        expr = Sin(expr)
    value = 0.5
    for _ in range(200):
        value = math.sin(value)
    assert expr.evaluate({x: 0.5}) == pytest.approx(value)
    assert expr == copy.deepcopy(expr)


def test_inv_rejects_zero_and_non_numbers():
    with pytest.raises(ValueError, match="Division by zero"):
        Inv(0)
    with pytest.raises(TypeError):
        Inv("x")


# --------------------------------------------------------------------------- copying and serialization
@pytest.mark.parametrize(
    "factory",
    [
        lambda x, s, b: Constant(1 + 2j) * x,
        lambda x, s, b: x**s + Sqrt(b),
        lambda x, s, b: Abs(Tan(x * b)) - 1e-300,
        lambda x, s, b: (s + b) ** -1,
        lambda x, s, b: Constant(float("nan")) * b,
        lambda x, s, b: Constant(2**40) * x * s * b,
    ],
)
def test_copies_and_serializations_round_trip(factory):
    x, s, b = Parameter("x", 0.5), SpinVariable("s"), BinaryVariable("b")
    expr = factory(x, s, b)
    assert copy.copy(expr) == expr
    assert copy.deepcopy(expr) == expr
    assert pickle.loads(pickle.dumps(expr)) == expr
    restored = deserialize(serialize(expr))
    assert restored == expr
    assert repr(restored) == repr(expr)


def test_a_shallow_copy_is_a_new_object_sharing_the_value(params):
    x, y, _ = params
    expr = x + y
    clone = copy.copy(expr)
    assert clone is not expr
    assert clone == expr
    assert hash(clone) == hash(expr)


def test_a_deep_copy_copies_the_variables(params):
    x, y, _ = params
    clone = copy.deepcopy(x + y)
    copied = next(symbol for symbol in clone.free_symbols() if symbol.label == "x")
    assert copied is not x
    assert copied == x


def test_a_deserialized_parameter_keeps_its_value(params):
    x, _, _ = params
    x.set_value(4.0)
    restored = deserialize(serialize(2 * x))
    assert restored.evaluate() == 8.0


def test_a_restored_variable_hashes_like_the_original():
    b = BinaryVariable("b")
    restored = pickle.loads(pickle.dumps(b))
    assert hash(restored) == hash(b)
    assert {b: 1}[restored] == 1


# --------------------------------------------------------------------------- algebraic laws on random expressions
def _random_expression(rng, symbols, depth=3):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([*symbols, Constant(rng.randint(-3, 3)), Constant(rng.choice([0.5, -1.5, 2.0]))])
    left = _random_expression(rng, symbols, depth - 1)
    right = _random_expression(rng, symbols, depth - 1)
    op = rng.choice(["+", "-", "*", "pow", "sin"])
    # Sin of a constant folds to an arbitrary float, which would make sums depend on their grouping
    if op == "sin" and not left.free_symbols():
        op = "+"
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "pow":
        return left ** rng.randint(0, 3)
    return Sin(left)


def test_random_expressions_obey_the_algebraic_laws():
    rng = random.Random(11)
    symbols = [Parameter(f"r{i}", 0.0) for i in range(3)]
    for _ in range(300):
        a, b, c = (_random_expression(rng, symbols) for _ in range(3))
        assert a + b == b + a
        assert a * b == b * a
        assert (a + b) + c == a + (b + c)
        assert (a * b) * c == a * (b * c)
        assert hash(a + b) == hash(b + a)
        assert (a - a).expand() == Constant(0)
        assert a + 0 == a
        assert a * 1 == a
        assert isinstance(a, Expression)


def test_random_expressions_evaluate_the_same_after_expanding_and_copying():
    rng = random.Random(5)
    symbols = [Parameter(f"r{i}", 0.0) for i in range(3)]
    for _ in range(200):
        expr = _random_expression(rng, symbols)
        values = {symbol: rng.uniform(0.5, 1.5) for symbol in symbols}
        value = expr.evaluate(values)
        assert expr.expand().evaluate(values) == pytest.approx(value, rel=1e-9, abs=1e-9)
        assert pickle.loads(pickle.dumps(expr)).evaluate(values) == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_complex_arithmetic_rounds_exactly_like_python():
    # Fusing a * b - c * d into one FMA would round differently from Python, e.g. 7.56 for 7.559999999999999
    value = -3.4 + 2j
    assert (Constant(value) ** 2).value == value**2
    assert (Constant(value) * Constant(value)).value == value * value
    assert ((Constant(value) + BinaryVariable("b")) ** 2).expand().get_constant() == value * value
