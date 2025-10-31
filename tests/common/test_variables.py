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

import numpy as np
import pytest

from qilisdk.core.exceptions import EvaluationError, InvalidBoundsError, NotSupportedOperation, OutOfBoundsException
from qilisdk.core.variables import (
    EQ,
    GEQ,
    GT,
    LEQ,
    LT,
    MAX_INT,
    MIN_INT,
    NEQ,
    BinaryVariable,
    Bitwise,
    ComparisonTerm,
    Domain,
    DomainWall,
    Equal,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    NotEqual,
    OneHot,
    Operation,
    Parameter,
    SpinVariable,
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


def test_comparison_operators():
    x = Variable("x", Domain.REAL)
    ct = LT(x, 5)
    assert ct.evaluate({x: 4})
    assert not ct.evaluate({x: 5})

    ct = LessThan(x, 5)
    assert ct.evaluate({x: 4})
    assert not ct.evaluate({x: 5})

    ct = LEQ(x, 5)
    assert ct.evaluate({x: 4})
    assert ct.evaluate({x: 5})
    assert not ct.evaluate({x: 6})

    ct = LessThanOrEqual(x, 5)
    assert ct.evaluate({x: 4})
    assert ct.evaluate({x: 5})
    assert not ct.evaluate({x: 6})

    ct = GT(x, 5)
    assert ct.evaluate({x: 6})
    assert not ct.evaluate({x: 5})

    ct = GreaterThan(x, 5)
    assert ct.evaluate({x: 6})
    assert not ct.evaluate({x: 5})

    ct = GEQ(x, 5)
    assert ct.evaluate({x: 6})
    assert ct.evaluate({x: 5})
    assert not ct.evaluate({x: 4})

    ct = GreaterThanOrEqual(x, 5)
    assert ct.evaluate({x: 6})
    assert ct.evaluate({x: 5})
    assert not ct.evaluate({x: 4})

    ct = EQ(x, 5)
    assert ct.evaluate({x: 5})
    assert not ct.evaluate({x: 6})
    assert not ct.evaluate({x: 5.1})

    ct = Equal(x, 5)
    assert ct.evaluate({x: 5})
    assert not ct.evaluate({x: 6})
    assert not ct.evaluate({x: 5.1})

    ct = NEQ(x, 5)
    assert ct.evaluate({x: 6})
    assert ct.evaluate({x: 5.1})
    assert not ct.evaluate({x: 5})

    ct = NotEqual(x, 5)
    assert ct.evaluate({x: 6})
    assert ct.evaluate({x: 5.1})
    assert not ct.evaluate({x: 5})


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


def test_variable_printing():
    label = "x_label"
    x = Variable(label, Domain.REAL)

    assert str(x) == label
    assert repr(x) == label


def test_binaryvar_evaluate_and_copy():
    b = BinaryVariable("b")
    assert b.num_binary_equivalent() == 1
    assert b.evaluate([0]) == 0
    assert b.evaluate([1]) == 1
    assert b.evaluate(0.0) == 0
    assert b.evaluate(1) == 1
    with pytest.raises(EvaluationError):
        b.evaluate([0, 1])

    with pytest.raises(EvaluationError):
        b.evaluate(2)
    b2 = copy(b)
    assert isinstance(b2, BinaryVariable)
    assert b2.label == b.label
    with pytest.raises(NotImplementedError):
        b.update_variable(Domain.BINARY, (0, 1))


def test_spinvar_evaluate():
    s = SpinVariable("s")
    assert s.num_binary_equivalent() == 1
    assert s.evaluate([1]) == 1
    assert s.evaluate([0]) == -1
    assert s.evaluate(1) == 1
    assert s.evaluate(0) == -1
    assert s.evaluate(-1) == -1
    with pytest.raises(EvaluationError):
        s.evaluate([0, 1])

    with pytest.raises(EvaluationError):
        s.evaluate(2)
    b2 = copy(s)
    assert isinstance(b2, SpinVariable)
    assert b2.label == s.label
    with pytest.raises(NotImplementedError):
        s.update_variable(Domain.BINARY, (0, 1))


def test_arithmetic_and_comparisons():
    a = Variable("a", Domain.INTEGER, bounds=(0, 5))
    b = Variable("b", Domain.INTEGER, bounds=(0, 5))

    t = a + b

    assert t * a == a.__rmul__(t)  # noqa: PLC2801

    t = -a

    assert t.evaluate({a: 1}) == -1
    assert t == -1 * a

    with pytest.raises(NotImplementedError):
        t = a / b

    t = a / 2

    assert t == a * 0.5

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

    with pytest.raises(NotSupportedOperation):
        _ = 3 // a

    # power operations
    t2 = a**2
    assert isinstance(t2, Term)
    val = t2.evaluate({a: [2]})
    assert val == 2**2

    t2 = a**0
    assert len(t2.variables()) == 0
    assert t2.evaluate({}) == 1

    t2 = a**1
    assert isinstance(t2, Term)
    val = t2.evaluate({a: [2]})
    assert val == 2

    # negative power
    with pytest.raises(NotImplementedError):
        _ = a**-1

    # comparisons
    c = LT(a, 3)
    assert isinstance(c, ComparisonTerm)
    assert c.evaluate({a: [2]})
    assert not c.evaluate({a: [5]})
    # bool of comparison when constants
    c2 = EQ(2 - a, 2 - a)
    zero = Term([], Operation.ADD)
    assert c2.lhs == zero
    assert c2.rhs == zero
    with pytest.raises(TypeError):
        _ = bool(a < b)

    assert a != 0

    t = a + b + 2
    assert 2 - t == -a - b

    assert -t == -1 * t
    assert 0 * t == Term([], Operation.MUL)

    t = Term([], Operation.ADD)

    assert t * 2 == Term([], Operation.MUL)
    assert 2 * t == Term([], Operation.MUL)

    t2 = -t * 10
    assert t2 == Term([], Operation.MUL)


def test_term_constant_and_simplify():
    # constant term
    t = Term([1, 2, 3], Operation.ADD)
    # Simplify should keep as Term since len>1
    s = t._simplify()
    assert isinstance(s, Term)
    ct = EQ(t, 6)
    zero = Term([0], Operation.ADD)
    assert ct.lhs == zero
    assert ct.rhs == zero
    # constant with zero elements
    t0 = Term([], Operation.ADD)
    assert t0 == zero


def test_hobo_num_binary_and_check_valid():
    var = Variable("v2", Domain.INTEGER, bounds=(0, 7), encoding=Bitwise)
    # bounds difference =7 -> floor(log2(7))=2 -> +1 =3
    assert Bitwise.num_binary_equivalent(var) == 3
    assert Bitwise.check_valid([0, 1, 1])[0]
    assert var.check_valid([0, 1, 1])[0]
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


def test_encoding_constraint_not_implemented():
    with pytest.raises(NotImplementedError):
        Bitwise.encoding_constraint(Variable("v3", Domain.INTEGER, bounds=(0, 1)))
    # OneHot and DomainWall constraints produce ComparisonTerm
    var = Variable("v4", Domain.INTEGER, bounds=(0, 2), encoding=DomainWall)
    cons = DomainWall.encoding_constraint(var)
    assert isinstance(cons, ComparisonTerm)


##############################
# Encoding Tests
##############################


def test_encoding():
    assert Bitwise.name == "Bitwise"

    assert Bitwise._bitwise_encode(5, 3) == [1, 0, 1]
    assert Bitwise._bitwise_encode(5, 4) == [1, 0, 1, 0]

    assert OneHot.name == "One-Hot"

    assert OneHot._one_hot_encode(5, 6) == [0, 0, 0, 0, 0, 1]
    assert OneHot._one_hot_encode(5, 7) == [0, 0, 0, 0, 0, 1, 0]
    with pytest.raises(ValueError, match=r"the input value \(5\) must be in range \[0, 4\]"):
        OneHot._one_hot_encode(5, 5)

    assert DomainWall.name == "Domain Wall"

    assert DomainWall._domain_wall_encode(5, 6) == [1, 1, 1, 1, 1, 0]
    assert DomainWall._domain_wall_encode(5, 7) == [1, 1, 1, 1, 1, 0, 0]
    with pytest.raises(ValueError, match=r"the input value \(5\) must be in range \[0, 4\]"):
        DomainWall._domain_wall_encode(5, 4)


def test_one_hot_find_zero():
    x = Variable("x", Domain.INTEGER, (0, 10), OneHot)
    assert OneHot._find_zero(x) == 0

    x = Variable("x", Domain.INTEGER, (-3, 10), OneHot)
    assert OneHot._find_zero(x) == 3

    x = Variable("x", Domain.INTEGER, (-10, 10), OneHot)
    assert OneHot._find_zero(x) == 10

    x = Variable("x", Domain.INTEGER, (0, 0), OneHot)
    assert OneHot._find_zero(x) == 0


def test_num_binary_equivalent():
    x = Variable("x", Domain.INTEGER, (0, 10), Bitwise)
    assert x.num_binary_equivalent() == 4

    x = Variable("x", Domain.REAL, (0, 10), Bitwise, precision=0.1)
    assert x.num_binary_equivalent() == 7


def test_invalid_bit_string():
    # OneHot

    x = Variable("x", Domain.INTEGER, (-10, 10), OneHot)

    with pytest.raises(ValueError, match=r"invalid binary string"):
        x.evaluate([1, 0, 1])

    with pytest.raises(ValueError, match=r"invalid binary string"):
        x.evaluate([0, 0, 0])

    x = Variable("x", Domain.INTEGER, (0, 2), OneHot)
    x.evaluate([0, 0, 1])
    with pytest.raises(ValueError, match=r"expected 3 variables but received 5"):
        x.evaluate([1, 0, 0, 0, 0])

    # Bitwise

    x = Variable("x", Domain.INTEGER, (0, 2), Bitwise)
    x.evaluate([0, 0])
    with pytest.raises(ValueError, match=r"expected 2 variables but received 3"):
        x.evaluate([1, 0, 0])

    # DomainWall
    x = Variable("x", Domain.INTEGER, (0, 10), DomainWall)

    with pytest.raises(ValueError, match=r"invalid binary string"):
        x.evaluate([1, 0, 1])

    x = Variable("x", Domain.INTEGER, (0, 2), DomainWall)
    x.evaluate([1, 1])
    with pytest.raises(ValueError, match=r"expected 2 variables but received 5"):
        x.evaluate([1, 0, 0, 0, 0])

    with pytest.raises(ValueError, match=r"invalid binary string"):
        x.evaluate([0, 1, 0])


def test_invalid_variable_evaluate():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 3))
    with pytest.raises(ValueError, match=r"The value -1 is invalid for the domain Positive Integer Domain"):
        x.evaluate(-1)
    with pytest.raises(ValueError, match=r"The value 4 is outside the defined bounds \(0, 3\)"):
        x.evaluate(4)

    x.update_variable(Domain.INTEGER, (-1, 3))
    with pytest.raises(ValueError, match=r"The value 1.1 is invalid for the domain Integer Domain"):
        x.evaluate(1.1)
    with pytest.raises(ValueError, match=r"The value 4 is outside the defined bounds \(-1, 3\)"):
        x.evaluate(4)

    x.update_variable(Domain.REAL, (-1, 3))
    with pytest.raises(ValueError, match=r"The value 4 is outside the defined bounds \(-1, 3\)"):
        x.evaluate(4)

    x.update_variable(Domain.BINARY, (0, 1))
    with pytest.raises(ValueError, match=r"The value 2 is invalid for the domain Binary Domain"):
        x.evaluate(2)


def test_encoding_and_evaluate():
    # #######################  OneHot #######################

    x = Variable("x", Domain.INTEGER, (-10, 10), OneHot)

    assert x.evaluate([1, 0, 0, 0]) == -10

    x = Variable("x", Domain.REAL, (0, 10), OneHot, precision=1)

    assert x.evaluate([1, 0, 0, 0]) == 0
    assert x.evaluate([1]) == 0

    assert x.term == sum(i * x[i] for i in range(x.num_binary_equivalent()))
    x.set_precision(1e-1)
    assert x.term == sum(i * x[i] for i in range(x.num_binary_equivalent())) * 1e-1

    var = Variable("v", Domain.INTEGER, bounds=(0, 2), encoding=OneHot)
    # should have 3 binary vars
    assert var.num_binary_equivalent() == 3
    # valid samples
    for i in range(3):
        binary = [1 if j == i else 0 for j in range(3)]
        assert OneHot.check_valid(binary)[0]
        assert var.check_valid(binary)[0]
        val = var.evaluate(binary)
        assert val == i
    # invalid sample
    assert not OneHot.check_valid([0, 0, 0])[0]
    assert not var.check_valid([0, 0, 0])[0]

    var = Variable("v", Domain.REAL, bounds=(-1, 2), encoding=OneHot, precision=1e-1)
    # should have 3 binary vars
    assert var.num_binary_equivalent() == 31
    # valid samples
    for i in range(31):
        binary = OneHot._one_hot_encode(i, 31)
        assert OneHot.check_valid(binary)[0]
        assert var.check_valid(binary)[0]
        val = var.evaluate(binary)
        assert math.isclose(val, i * 1e-1 - 1, abs_tol=1e-1)
    # # invalid sample
    assert not OneHot.check_valid([0, 0, 0])[0]
    assert not var.check_valid([0, 0, 0])[0]

    # #######################  DomainWall #######################

    var = Variable("v", Domain.REAL, bounds=(-1, 2), encoding=DomainWall, precision=1e-1)
    # should have 3 binary vars
    assert var.num_binary_equivalent() == 30
    # valid samples
    for i in range(30):
        binary = DomainWall._domain_wall_encode(i, 30)
        val = var.evaluate(binary)
        assert math.isclose(val, i * 1e-1 - 1, abs_tol=1e-1)

    assert var.evaluate([1]) == -0.9
    assert var.evaluate([1, 1]) == -0.8
    assert var.evaluate([0]) == -1

    # # invalid sample
    assert not DomainWall.check_valid([0, 1, 0])[0]
    assert not var.check_valid([0, 1, 0])[0]

    var = Variable("v", Domain.INTEGER, bounds=(0, 2), encoding=DomainWall)
    assert var.term == sum(var)

    # #######################  DomainWall #######################

    x = Variable("x", Domain.REAL, (-1, 2), Bitwise, precision=1e-1)
    assert x.evaluate(-1) == -1
    assert x.evaluate(-0.5) == -0.5
    assert x.evaluate([0]) == -1
    assert x.evaluate([1, 0, 1]) == -0.5


def test_encoding_constraint():
    x = Variable("x", Domain.INTEGER, (0, 3), OneHot)
    manual_encoding_constraint = EQ(sum(x), 1)
    encoding_constraint = x.encoding_constraint()

    assert encoding_constraint.lhs == manual_encoding_constraint.lhs
    assert encoding_constraint.rhs == manual_encoding_constraint.rhs
    assert encoding_constraint.operation == manual_encoding_constraint.operation

    x = Variable("x", Domain.REAL, (0, 3), OneHot, precision=0.1)
    manual_encoding_constraint = EQ(sum(x), 1)
    encoding_constraint = x.encoding_constraint()

    assert encoding_constraint.lhs == manual_encoding_constraint.lhs
    assert encoding_constraint.rhs == manual_encoding_constraint.rhs
    assert encoding_constraint.operation == manual_encoding_constraint.operation

    x = Variable("x", Domain.INTEGER, (0, 3), DomainWall)
    manual_encoding_constraint = EQ(sum(x[i + 1] * (1 - x[i]) for i in range(len(x.bin_vars) - 1)), 0)
    encoding_constraint = x.encoding_constraint()

    assert encoding_constraint.lhs == manual_encoding_constraint.lhs
    assert encoding_constraint.rhs == manual_encoding_constraint.rhs
    assert encoding_constraint.operation == manual_encoding_constraint.operation

    x = Variable("x", Domain.REAL, (0, 3), DomainWall, precision=0.1)
    manual_encoding_constraint = EQ(sum(x[i + 1] * (1 - x[i]) for i in range(len(x.bin_vars) - 1)), 0)
    encoding_constraint = x.encoding_constraint()

    assert encoding_constraint.lhs == manual_encoding_constraint.lhs
    assert encoding_constraint.rhs == manual_encoding_constraint.rhs
    assert encoding_constraint.operation == manual_encoding_constraint.operation


##############################
# Extra Tests
##############################


def test_setting_bounds():
    x = Variable("x", Domain.INTEGER, (0, 3))
    x.set_bounds(None, None)
    assert x.lower_bound == Domain.INTEGER.min()
    assert x.upper_bound == Domain.INTEGER.max()
    with pytest.raises(OutOfBoundsException):
        x.set_bounds(None, 1.1)

    x = Variable("x", Domain.REAL, (0, 3))
    x.set_bounds(None, None)
    assert x.lower_bound == Domain.REAL.min()
    assert x.upper_bound == Domain.REAL.max()

    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 3))
    x.set_bounds(None, None)
    assert x.lower_bound == Domain.POSITIVE_INTEGER.min()
    assert x.upper_bound == Domain.POSITIVE_INTEGER.max()

    with pytest.raises(OutOfBoundsException):
        x.set_bounds(None, 1.1)

    x = Variable("x", Domain.BINARY, (0, 1))
    x.set_bounds(None, None)
    assert x.lower_bound == Domain.BINARY.min()
    assert x.upper_bound == Domain.BINARY.max()

    with pytest.raises(OutOfBoundsException):
        x.set_bounds(None, 2)


def test_to_binary():
    x = BinaryVariable("x")

    assert isinstance(x.to_binary(), Term)
    assert x.to_binary() == Term([x], Operation.ADD)

    x = SpinVariable("x")

    assert isinstance(x.to_binary(), Term)
    assert x.to_binary() == Term([x], Operation.ADD)


##############################
# Test Term
##############################


def test_Term_construction():
    b = BinaryVariable("b")
    x = Variable("x", Domain.REAL)

    t = Term([1, 1], operation=Operation.ADD)
    expected = Term([2], operation=Operation.ADD)
    assert t == expected

    t = Term([1, Term([1], operation=Operation.ADD)], operation=Operation.ADD)
    expected = Term([2], operation=Operation.ADD)
    assert t == expected

    t = Term([Term.CONST, Term.CONST], operation=Operation.ADD)
    expected = Term([2], operation=Operation.ADD)
    assert t == expected

    t = Term([Term.CONST, Term([Term.CONST], operation=Operation.ADD)], operation=Operation.ADD)
    expected = Term([2], operation=Operation.ADD)
    assert t == expected

    t = Term([b, b], operation=Operation.MUL)
    expected = Term([b], operation=Operation.MUL)
    assert t == expected

    t = Term([b, Term([b, b], operation=Operation.MUL)], operation=Operation.MUL)
    expected = Term([b], operation=Operation.MUL)
    assert t == expected

    t = Term([x, x], operation=Operation.ADD)
    expected = Term([x], operation=Operation.ADD)
    expected._elements[x] = 2
    assert t == expected

    t = Term([x, Term([x, b], operation=Operation.ADD)], operation=Operation.ADD)
    expected = Term([x, b], operation=Operation.ADD)
    expected._elements[x] = 2
    assert t == expected

    _t = Term([3, Term.CONST], operation=Operation.MUL)
    t = Term([Term.CONST, _t], operation=Operation.ADD)

    assert t == Term([4], Operation.ADD)

    _t = Term([3, x], operation=Operation.MUL)
    t = Term([x, _t], operation=Operation.ADD)

    assert t == Term([Term([4, x], Operation.MUL)], operation=Operation.ADD)

    with pytest.raises(ValueError, match=r"Term accepts object of types Term or Variable but an object of type"):
        Term(["s"], operation=Operation.ADD)


def test_Term_degree():
    x = Variable("x", Domain.REAL)

    t = x**2 + 3 * x

    assert t.degree == 2

    t = (2 * x + 1) * (3 * x + 4)

    assert t.degree == 2

    t = x * x * x + x**2

    assert t.degree == 3

    y = Variable("y", Domain.REAL)

    t = x * y + x
    assert t.degree == 2

    t = Term([Term([4, x**2], Operation.ADD), x], operation=Operation.MUL)

    assert t.degree == 3


def test_Term_to_binary():
    b = BinaryVariable("b")
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = 2 * x * b + 3 * b + 4
    t_binary = 2 * x.to_binary() * b + 3 * b + 4

    assert t.to_binary() == t_binary

    t = Term([4, Term([2 * x, 2], Operation.ADD)], Operation.MUL)
    t_binary = 4 * (2 * x.to_binary() + 2)
    assert t.to_binary()._unfold_parentheses() == t_binary

    t = Term([], operation=Operation.DIV)
    with pytest.raises(ValueError, match=r"Can not evaluate any operation that is not Addition of Multiplication"):
        t.to_binary()

    t = Term([], operation=Operation.ADD)
    t._elements[""] = 1
    with pytest.raises(ValueError, match=r"Evaluating term with elements of type <class '.*?'> is not supported\."):
        t.to_binary()


def test_apply_operation_on_constants():
    t = Term([4, 5], operation=Operation.ADD)
    assert t[Term.CONST] == 9

    t = Term([4, 5], operation=Operation.SUB)
    assert t[Term.CONST] == -1

    t = Term([4, 5], operation=Operation.MUL)
    assert t[Term.CONST] == 20

    t = Term([4, 5], operation=Operation.DIV)
    assert t[Term.CONST] == 0.8

    t = Term([4, 5, 9, 7, 13], operation=Operation.ADD)
    assert t[Term.CONST] == 38

    t = Term([4, 5, 9, 7, 13], operation=Operation.SUB)
    assert t[Term.CONST] == -30

    t = Term([4, 5, 9, 7, 13], operation=Operation.MUL)
    assert t[Term.CONST] == 16380

    t = Term([4, 5, 9, 7, 13], operation=Operation.DIV)
    assert np.isclose(t[Term.CONST], 0.00097680097)


def test_Term_variables():
    b = BinaryVariable("b")
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    y = Variable("y", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = x + 2 * x * b + 3 * x * y

    assert b in t.variables()
    assert x in t.variables()
    assert y in t.variables()


def test_term_pop_error():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    t = 2 * x + 3
    t.pop(x)
    with pytest.raises(KeyError, match=r'item ".*?" not found in the term\.'):
        t.pop(x)


def test_unfold_parentheses():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = (2 * x + 1) * (3 * x + 2)
    expected_t = 6 * x**2 + 7 * x + 2

    assert t == expected_t

    t = (2 * x + 1) ** 2
    expected_t = 4 * x**2 + 4 * x + 1

    assert t == expected_t

    t = (2 * x**2 + 1) ** 2
    expected_t = 4 * x**4 + 4 * x**2 + 1

    assert t == expected_t

    t = (2 * x + 1) * x**2
    expected_t = 2 * x**3 + x**2

    assert t == expected_t

    t = x + 2

    assert t._unfold_parentheses() == t


def test_Term_evaluate():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    y = Variable("y", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = 2 * x + 3

    assert t.evaluate({x: 3}) == 9

    with pytest.raises(
        ValueError, match=r"Can not evaluate term because the value of the variable .*? is not provided\."
    ):
        t.evaluate({})

    t = 2 * x * y + 2 * x + 3 * y

    assert t.evaluate({x: 2, y: 1}) == 11
    assert t.evaluate({x: 1, y: 2}) == 12


def test_get_constant():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    y = Variable("y", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = Term([x], operation=Operation.MUL)
    assert t.get_constant() == 1

    t = Term([x], operation=Operation.ADD)
    assert t.get_constant() == 0

    t = 3 * x
    assert t.get_constant() == 3

    t = 3 * x + 2
    assert t.get_constant() == 2

    t = 3 * x + 2 * y
    assert t.get_constant() == 0

    t = 3 * x + 2 * y + 4
    assert t.get_constant() == 4

    t = 3 * x - 3 * x
    assert t.get_constant() == 0


def test_Term_printing():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    y = Variable("y", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = 2 * x
    expected_t = "(2) * x"
    assert repr(t) == expected_t

    t = 2 * x + 1
    expected_t = "(2) * x + (1)"
    assert repr(t) == expected_t

    t = 2 * x + 3 * y
    expected_t = "(2) * x + (3) * y"
    assert repr(t) == expected_t

    t = 1 + x
    expected_t = "x + (1)"
    assert repr(t) == expected_t

    t = x + 0
    expected_t = "x"
    assert repr(t) == expected_t

    t = x + (1 - 1)
    expected_t = "x"
    assert repr(t) == expected_t

    t = 1 * x
    expected_t = "x"
    assert repr(t) == expected_t

    t = Term([], Operation.ADD)
    expected_t = "0"
    assert repr(t) == expected_t

    t = (x + y) * 3
    expected_t = "(3.0) * x + (3.0) * y"
    assert repr(t) == expected_t

    t = 2 * (x) ** 2
    expected_t = "(2) * (x^2)"
    assert repr(t) == expected_t

    t = 2 * (x * y) + x
    expected_t = "(2) * (x * y) + x"
    assert repr(t) == expected_t

    t = x * y * 1
    expected_t = "x * y"
    assert repr(t) == expected_t

    t = 1 * x * y
    expected_t = "x * y"
    assert repr(t) == expected_t

    t = x * y * 2
    expected_t = "x * y * (2)"
    assert repr(t) == expected_t


def test_Term_division():
    x = Variable("x", Domain.REAL)
    t = 2 * x + 2
    with pytest.raises(NotImplementedError):
        t / x
    with pytest.raises(ValueError, match=r"Division by zero is not allowed"):
        t / 0
    with pytest.raises(NotSupportedOperation):
        2 / t
    with pytest.raises(NotSupportedOperation):
        2 // t

    t /= 2
    assert t == (x + 1)


def test_Term_power():
    x = Variable("x", Domain.REAL)
    t = 2 * x**2

    assert t**3 == 8 * x**6

    t = 2 * x**2 + 2

    assert t**3 == (8.0) * (x**6) + (24.0) * (x**4) + (24.0) * (x**2) + (8.0)

    t = Term([2], Operation.SUB)

    with pytest.raises(NotImplementedError):
        t**2


##############################
# Test Comparison Term
##############################


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

    _t = Term([Term([4, x**2], Operation.ADD), x], operation=Operation.MUL)

    t = EQ(_t, x)
    assert t.degree == 3


def test_type_error_bool():
    x = Variable("x", Domain.REAL)
    t = EQ(x, 0)

    with pytest.raises(TypeError):
        bool(t)

    t = EQ(x, x)

    with pytest.raises(TypeError):
        bool(t)


def test_Comparison_Term_to_binary():
    b = BinaryVariable("b")
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = EQ(2 * x * b, 3 * b + 4)
    t_binary = EQ(2 * x.to_binary() * b, 3 * b + 4)

    assert t.to_binary().lhs == t_binary.lhs
    assert t.to_binary().rhs == t_binary.rhs

    t = EQ(Term([4, Term([2 * x, 2], Operation.ADD)], Operation.MUL).to_binary()._unfold_parentheses(), 0)
    t_binary = EQ(4 * (2 * x.to_binary() + 2), 0)
    assert t.lhs == t_binary.lhs
    assert t.rhs == t_binary.rhs

    _t = Term([], operation=Operation.ADD)
    _t._elements[""] = 1
    with pytest.raises(ValueError, match=r"Term accepts object of types Term or Variable but an object of type "):
        t = EQ(_t, 0)


def test_Comparison_Term_printing():
    x = Variable("x", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)
    y = Variable("y", Domain.POSITIVE_INTEGER, (0, 8), Bitwise)

    t = EQ(2 * x, 0)
    expected_t = "(2) * x == 0"

    assert repr(t) == expected_t

    t = EQ(2 * x, 1)
    expected_t = "(2) * x == (1)"

    assert repr(t) == expected_t

    t = LT(2 * x, 3 * y)
    expected_t = "(2) * x + (-3.0) * y < 0"

    assert repr(t) == expected_t

    t = GT(1, x)
    expected_t = "(-1) * x > (-1)"

    assert repr(t) == expected_t

    t = LT(x + 0, 0)
    expected_t = "x < 0"

    assert repr(t) == expected_t

    t = GEQ(x + (1 - 1), 2)
    expected_t = "x >= (2)"

    assert repr(t) == expected_t

    t = LEQ(1 * x, -x)
    expected_t = "(2.0) * x <= 0"

    assert repr(t) == expected_t

    t = EQ(Term([], Operation.ADD), 0)
    expected_t = "0 == 0"

    assert repr(t) == expected_t

    t = NEQ((x + y) * 3, 3)
    expected_t = "(3.0) * x + (3.0) * y != (3)"

    assert repr(t) == expected_t

    t = EQ(2 * (x) ** 2, 2)
    expected_t = "(2) * (x^2) == (2)"

    assert repr(t) == expected_t

    t = LT(2 * (x * y) + x, 5)
    expected_t = "(2) * (x * y) + x < (5)"

    assert repr(t) == expected_t


####################
# Parameter
####################


def test_parameter_value():
    p = Parameter("p", 2, domain=Domain.POSITIVE_INTEGER, bounds=(2, 4))
    assert p.value == 2
    assert p.domain == Domain.POSITIVE_INTEGER
    assert p.bounds == (2, 4)

    p.set_value(3)
    assert p.value == 3

    p.set_bounds(None, 6)
    assert p.bounds == (0, 6)

    p.update_variable(domain=Domain.INTEGER)
    assert p.domain == Domain.INTEGER

    with pytest.raises(
        ValueError, match=r"Parameter value provided \(0.5\) doesn't correspond to the parameter's domain \(INTEGER\)"
    ):
        Parameter("p", 0.5, Domain.INTEGER, (0, 10))

    with pytest.raises(ValueError, match=r"The current value of the parameter \(9\) is outside the bounds \(2, 4\)"):
        Parameter("p", 9, Domain.INTEGER, (2, 4))

    p.set_bounds(2, 9)
    p.set_value(9)
    with pytest.raises(ValueError, match=r"The current value of the parameter \(9\) is outside the bounds \(2, 4\)"):
        p.set_bounds(2, 4)

    p.update_variable(Domain.REAL, (0, 10))
    p.set_value(0.5)
    with pytest.raises(
        ValueError,
        match=r"The provided domain \(INTEGER\) is incompatible with the current parameter value \(0.5\)",
    ):
        p.update_variable(Domain.INTEGER, (0, 10))
