# Copyright 2025 Qilimanjaro Quantum Tech
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

import math
from copy import copy

import pytest

from qilisdk.common.exceptions import EvaluationError, InvalidBoundsError, NotSupportedOperation, OutOfBoundsException
from qilisdk.common.variables import (
    HOBO,
    MAX_INT,
    MIN_INT,
    BinaryVar,
    ComparisonTerm,
    Domain,
    DomainWall,
    OneHot,
    Operation,
    SpinVar,
    Term,
    Variable,
)


def test_domain_check_value_and_bounds():
    # BINARY domain
    assert Domain.BINARY.check_value(0)
    assert Domain.BINARY.check_value(1)
    assert not Domain.BINARY.check_value(2)
    assert Domain.SPIN.check_value(-1)
    assert Domain.SPIN.check_value(1)
    assert not Domain.SPIN.check_value(0)
    assert Domain.INTEGER.check_value(5)
    assert not Domain.INTEGER.check_value(3.2)
    assert Domain.POSITIVE_INTEGER.check_value(0)
    assert not Domain.POSITIVE_INTEGER.check_value(-1)
    assert Domain.REAL.check_value(3.1)

    # min and max
    assert Domain.BINARY.min() == 0
    assert Domain.BINARY.max() == 1
    assert Domain.SPIN.min() == -1
    assert Domain.SPIN.max() == 1
    assert Domain.INTEGER.min() == MIN_INT
    assert Domain.INTEGER.max() == MAX_INT
    assert Domain.POSITIVE_INTEGER.min() == 0
    assert Domain.POSITIVE_INTEGER.max() == MAX_INT
    # REAL domain
    assert Domain.REAL.min() == -1e30
    assert Domain.REAL.max() == 1e30


def test_variable_bounds_validation():
    # valid bounds
    v = Variable("x", Domain.INTEGER, bounds=(1, 10))
    assert v.lower_bound == 1
    assert v.upper_bound == 10

    # invalid domain bound
    with pytest.raises(OutOfBoundsException):
        Variable("y", Domain.BINARY, bounds=(0, 2))

    with pytest.raises(OutOfBoundsException):
        Variable("y", Domain.POSITIVE_INTEGER, bounds=(-1, 2))

    # lower > upper
    with pytest.raises(InvalidBoundsError):
        Variable("z", Domain.INTEGER, bounds=(5, 3))


def test_set_bounds():
    v = Variable("x", Domain.POSITIVE_INTEGER)
    v.set_bounds(2, 5)
    assert v.bounds == (2, 5)
    with pytest.raises(OutOfBoundsException):
        v.set_bounds(-1, 5)
    with pytest.raises(InvalidBoundsError):
        v.set_bounds(2, 1)


def test_binaryvar_evaluate_and_copy():
    b = BinaryVar("b")
    assert b.num_binary_equivalent() == 1
    assert b.evaluate([0]) == 0
    assert b.evaluate([1]) == 1
    with pytest.raises(EvaluationError):
        b.evaluate([0, 1])
    b2 = copy(b)
    assert isinstance(b2, BinaryVar)
    assert b2.label == b.label
    with pytest.raises(NotImplementedError):
        b.update_variable(Domain.BINARY, (0, 1))


def test_spinvar_evaluate():
    s = SpinVar("s")
    assert s.num_binary_equivalent() == 1
    assert s.evaluate([1]) == 1
    assert s.evaluate([0]) == -1
    with pytest.raises(EvaluationError):
        s.evaluate([0, 1])


def test_arithmetic_and_comparisons():
    a = Variable("a", Domain.INTEGER, bounds=(0, 5))
    b = Variable("b", Domain.INTEGER, bounds=(0, 5))
    t = a + b * 2 - 3
    # t should be Term
    assert isinstance(t, Term)
    # test evaluation with values
    val = t.evaluate({a: [2], b: [1]})
    assert val == 2 + 1 * 2 - 3

    # test division by zero
    with pytest.raises(ValueError):  # noqa: PT011
        _ = a / 0
    # test unsupported rtruediv
    with pytest.raises(NotSupportedOperation):
        _ = 3 / a

    # power operations
    t2 = a**2
    assert isinstance(t2, Term)

    val = t2.evaluate({a: [2]})
    assert val == 2**2

    # negative power
    with pytest.raises(NotImplementedError):
        _ = a**-1

    # comparisons
    c = a < 3
    assert isinstance(c, ComparisonTerm)
    assert c.evaluate({a: [2]})
    assert not c.evaluate({a: [5]})
    # bool of comparison when constants
    c2 = 2 - a == 2 - a
    assert bool(c2)
    with pytest.raises(TypeError):
        _ = bool(a < b)


def test_term_constant_and_simplify():
    # constant term
    t = Term([1, 2, 3], Operation.ADD)
    # Simplify should keep as Term since len>1
    s = t._simplify()
    assert isinstance(s, Term)
    assert t == 6
    # constant with zero elements
    t0 = Term([], Operation.ADD)
    assert t0 == 0


def test_encoding_and_evaluate():
    var = Variable("v", Domain.INTEGER, bounds=(0, 2), encoding=OneHot)
    # should have 3 binary vars
    assert var.num_binary_equivalent() == 3
    # valid samples
    for i in range(3):
        binary = [1 if j == i else 0 for j in range(3)]
        assert OneHot.check_valid(binary)[0]
        val = var.evaluate(binary)
        assert val == i
    # invalid sample
    assert not OneHot.check_valid([0, 0, 0])[0]
    with pytest.raises(ValueError):  # noqa: PT011
        var.evaluate([0, 0, 0])

    var = Variable("v", Domain.REAL, bounds=(-1, 2), encoding=OneHot, precision=1e-1)
    # should have 3 binary vars
    assert var.num_binary_equivalent() == 31
    # valid samples
    for i in range(31):
        binary = OneHot._one_hot_encode(i, 31)
        assert OneHot.check_valid(binary)[0]
        val = var.evaluate(binary)
        assert math.isclose(val, i * 1e-1 - 1, abs_tol=1e-1)
    # # invalid sample
    assert not OneHot.check_valid([0, 0, 0])[0]
    with pytest.raises(ValueError):  # noqa: PT011
        var.evaluate([0, 0, 0])

    var = Variable("v", Domain.REAL, bounds=(-1, 2), encoding=DomainWall, precision=1e-1)
    # should have 3 binary vars
    assert var.num_binary_equivalent() == 30
    # valid samples
    for i in range(30):
        binary = DomainWall._domain_wall_encode(i, 30)
        val = var.evaluate(binary)
        assert math.isclose(val, i * 1e-1 - 1, abs_tol=1e-1)
    # # invalid sample
    assert not DomainWall.check_valid([0, 1, 0])[0]
    with pytest.raises(ValueError):  # noqa: PT011
        var.evaluate([0, 1, 0])

    x = Variable("x", Domain.REAL, (-1, 2), HOBO, precision=1e-1)
    assert x.evaluate(-1) == -1
    assert x.evaluate(-0.5) == -0.5
    assert x.evaluate([0]) == -1
    assert x.evaluate([1, 0, 1]) == -0.5


def test_hobo_num_binary_and_check_valid():
    var = Variable("v2", Domain.INTEGER, bounds=(0, 7), encoding=HOBO)
    # bounds difference =7 -> floor(log2(7))=2 -> +1 =3
    assert HOBO.num_binary_equivalent(var) == 3
    assert HOBO.check_valid([0, 1, 1])[0]
    assert var.evaluate([0, 1, 1]) == 6


def test_term_to_list_and_unfold_parentheses():
    # build term with parentheses
    a = Variable("a", Domain.INTEGER, bounds=(0, 5))
    b = Variable("b", Domain.INTEGER, bounds=(0, 5))
    t = (a + b) * 2
    lst = t.to_list()
    assert isinstance(lst, list)
    u = 2 * a + 2 * b
    assert isinstance(u, Term)
    assert t == u


def test_replace_and_update_range():
    a = Variable("x", Domain.INTEGER, bounds=(0, 1))
    b = Variable("y", Domain.INTEGER, bounds=(0, 1))
    c = Variable("z", Domain.INTEGER, bounds=(0, 1))

    c1 = a == b
    # replace x with a new variable z
    c1.replace_variables({a: c})
    # update bounds
    c1.update_variable_bounds({a: (-1, 2)})


def test_encoding_constraint_not_implemented():
    with pytest.raises(NotImplementedError):
        HOBO.encoding_constraint(Variable("v3", Domain.INTEGER, bounds=(0, 1)))
    # OneHot and DomainWall constraints produce ComparisonTerm
    var = Variable("v4", Domain.INTEGER, bounds=(0, 2), encoding=DomainWall)
    cons = DomainWall.encoding_constraint(var)
    assert isinstance(cons, ComparisonTerm)
