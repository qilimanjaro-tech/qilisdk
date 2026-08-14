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

"""Tests for :mod:`qilisdk.core.comparison`."""

from copy import copy

import pytest

from qilisdk.core.comparison import (
    EQ,
    GEQ,
    GT,
    LEQ,
    LT,
    NEQ,
    Comparison,
)
from qilisdk.core.expression import Add
from qilisdk.core.variables import BinaryVariable, Bitwise, Domain, Parameter, Variable
from qilisdk.utils.serialization import deserialize, serialize


def test_comparison_term_variables():
    b = BinaryVariable("b")
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    y = Variable("y", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = EQ(x + 2 * x * b, 3 * x * y)

    assert b in t.variables()
    assert x in t.variables()
    assert y in t.variables()


def test_Comparison_Term_degree():
    x = Variable("x", Domain.REAL)

    t = EQ(x**2, 3 * x)

    assert t.degree == 2

    t = EQ((2 * x + 1), (3 * x + 4))

    assert t.degree == 1

    t = LT(x * x * x, x**2)

    assert t.degree == 3

    y = Variable("y", Domain.REAL)

    t = GT(x * y, x)
    assert t.degree == 2

    _t = (4 + x**2) * x

    t = EQ(_t, x)
    assert t.degree == 3


def test_type_error_bool():
    x = Variable("x", Domain.REAL)
    t = EQ(x, 0)

    with pytest.raises(TypeError):
        _ = bool(t)

    t = EQ(x, x)

    with pytest.raises(TypeError):
        _ = bool(t)


def test_Comparison_Term_to_binary():
    b = BinaryVariable("b")
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = EQ(2 * x * b, 3 * b + 4)
    t_binary = EQ(2 * x.to_binary() * b, 3 * b + 4)

    assert t.to_binary().lhs == t_binary.lhs
    assert t.to_binary().rhs == t_binary.rhs

    t = EQ((4 * (2 * x + 2)).to_binary().expand(), 0)
    t_binary = EQ((4 * (2 * x.to_binary() + 2)).expand(), 0)
    assert t.lhs == t_binary.lhs
    assert t.rhs == t_binary.rhs

    # operands must be numbers or Expressions
    with pytest.raises(TypeError):
        _ = EQ("not an expression", 0)


def test_Comparison_Term_printing():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    y = Variable("y", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = EQ(2 * x, 0)
    expected_t = "2 * x == 0"

    assert repr(t) == expected_t

    t = EQ(2 * x, 1)
    expected_t = "2 * x == 1"

    assert repr(t) == expected_t

    t = LT(2 * x, 3 * y)
    expected_t = "-3 * y + 2 * x < 0"

    assert repr(t) == expected_t

    t = GT(1, x)
    expected_t = "-x > -1"

    assert repr(t) == expected_t

    t = LT(x + 0, 0)
    expected_t = "x < 0"

    assert repr(t) == expected_t

    t = GEQ(1 + x - 1, 2)
    expected_t = "x >= 2"

    assert repr(t) == expected_t

    t = LEQ(1 * x, -x)
    expected_t = "2 * x <= 0"

    assert repr(t) == expected_t

    t = EQ(Add.build(()), 0)
    expected_t = "0 == 0"

    assert repr(t) == expected_t

    t = NEQ((x + y) * 3, 3)
    expected_t = "3 * (x + y) != 3"

    assert repr(t) == expected_t

    t = EQ(2 * (x) ** 2, 2)
    expected_t = "2 * x**2 == 2"

    assert repr(t) == expected_t

    t = LT(2 * (x * y) + x, 5)
    expected_t = "x + 2 * x * y < 5"

    assert repr(t) == expected_t


def test_evaluating_with_complex_values_raises():
    p = Parameter("p", 1.0)
    for comparison in (LEQ(1j * p, 0), LEQ(0, 1j * p)):
        with pytest.raises(ValueError, match=r"complex values is not allowed"):
            comparison.evaluate({p: 1.0})


def test_a_negligible_imaginary_part_is_dropped():
    p = Parameter("p", 1.0)
    # (1j * p) ** 2 is real, it just arrives as a complex number.
    assert LEQ((1j * p) ** 2, 0).evaluate({p: 2.0})


def test_copy_and_yaml_round_trip():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    comparison = GEQ(2 * x, 3)

    duplicate = copy(comparison)
    assert duplicate == comparison
    assert hash(duplicate) == hash(comparison)

    restored = deserialize(serialize(comparison))
    assert isinstance(restored, Comparison)
    assert restored == comparison


def test_comparisons_of_different_operations_are_not_equal():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    assert LEQ(x, 3) != GEQ(x, 3)
    assert LEQ(x, 3) == LEQ(x, 3)
    assert LEQ(x, 3) != "not a comparison"
