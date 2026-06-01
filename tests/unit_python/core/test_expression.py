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

import pytest

from qilisdk.core.exceptions import NonPolynomialError
from qilisdk.core.expression import Constant, Cos, Exp, Expression, Log, Pow, Sin
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
