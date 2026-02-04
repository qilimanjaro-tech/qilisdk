from __future__ import annotations

from uuid import UUID

import pytest

from qilisdk.qprogram.variables import (
    QProgramDomain,
    QProgramVariable,
    VariableExpression,
    requires_domain,
)


class DummyRepresenter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def represent_scalar(self, tag: str, value: str) -> tuple[str, str, str]:
        self.calls.append((tag, value))
        return ("scalar", tag, value)


class DummyNode:
    def __init__(self, value: str) -> None:
        self.value = value


class DummyWithDomain:
    def __init__(self) -> None:
        self.calls: list[object] = []

    @requires_domain("var", QProgramDomain.Time)
    def method(self, *, var: QProgramVariable | None = None) -> QProgramVariable | None:
        self.calls.append(var)
        return var


def test_requires_domain_adds_default_when_missing() -> None:
    dummy = DummyWithDomain()
    result = dummy.method()
    assert result is None
    assert dummy.calls == [None]


def test_requires_domain_validates_domain() -> None:
    dummy = DummyWithDomain()
    good = QProgramVariable("ok", domain=QProgramDomain.Time)
    assert dummy.method(var=good) is good

    bad = QProgramVariable("bad", domain=QProgramDomain.Frequency)
    with pytest.raises(ValueError, match=r"Expected domain QProgramDomain.Time"):
        dummy.method(var=bad)


def test_requires_domain_preserves_default_with_positional_args() -> None:
    sentinel = object()

    @requires_domain("var", QProgramDomain.Time)
    def sample_method(a: int, b: int, *, var: object = sentinel) -> object:
        return var

    assert sample_method(1, 2) is sentinel


def test_variable_properties_and_repr() -> None:
    var = QProgramVariable("time", domain=QProgramDomain.Time)
    assert var.label == "time"
    assert var.domain is QProgramDomain.Time
    assert isinstance(var.uuid, UUID)
    assert "label=time" in repr(var)
    assert "domain=QProgramDomain.Time" in repr(var)


def test_variable_operators_create_expressions() -> None:
    var = QProgramVariable("time", domain=QProgramDomain.Time)
    add_expr = var + 1
    radd_expr = 1 + var
    sub_expr = var - 2
    rsub_expr = 2 - var

    assert isinstance(add_expr, VariableExpression)
    assert add_expr.left is var
    assert add_expr.right == 1
    assert add_expr.operator == "+"
    assert radd_expr.left == 1
    assert radd_expr.right is var
    assert radd_expr.operator == "+"
    assert sub_expr.left is var
    assert sub_expr.right == 2
    assert sub_expr.operator == "-"
    assert rsub_expr.left == 2
    assert rsub_expr.right is var
    assert rsub_expr.operator == "-"

    assert "(" in repr(add_expr)
    assert "+" in repr(add_expr)


def test_variable_expression_requires_time_domain() -> None:
    var = QProgramVariable("scalar", domain=QProgramDomain.Scalar)
    with pytest.raises(NotImplementedError, match=r"QProgramDomain.Time"):
        _ = var + 1


def test_variable_expression_infers_domain_from_right() -> None:
    var = QProgramVariable("time", domain=QProgramDomain.Time)
    expr = 1 + var
    assert expr.domain is QProgramDomain.Time


def test_variable_expression_rejects_constants() -> None:
    with pytest.raises(ValueError, match=r"Cannot infer domain"):
        VariableExpression(1, "+", 2)


def test_variable_expression_operator_chaining() -> None:
    var = QProgramVariable("time", domain=QProgramDomain.Time)
    expr = var + 1
    add_expr = expr + 2
    radd_expr = 2 + expr
    sub_expr = expr - 3
    rsub_expr = 3 - expr

    assert add_expr.left is expr
    assert add_expr.right == 2
    assert add_expr.operator == "+"
    assert radd_expr.left == 2
    assert radd_expr.right is expr
    assert radd_expr.operator == "+"
    assert sub_expr.left is expr
    assert sub_expr.right == 3
    assert sub_expr.operator == "-"
    assert rsub_expr.left == 3
    assert rsub_expr.right is expr
    assert rsub_expr.operator == "-"


def test_extract_variables_and_constants() -> None:
    var = QProgramVariable("time", domain=QProgramDomain.Time)
    other = QProgramVariable("other", domain=QProgramDomain.Time)
    expr_left = var + 1
    expr_right = 1 + var
    nested = var + (other - 5)

    assert expr_left.extract_variables() is var
    assert expr_right.extract_variables() is var
    assert expr_left.extract_constants() == 1
    assert expr_right.extract_constants() == 1
    assert nested.extract_constants() == 5


def test_extract_variables_raises_when_missing() -> None:
    var = QProgramVariable("time", domain=QProgramDomain.Time)
    expr = var + 1
    expr.left = 1
    expr.right = 2
    with pytest.raises(ValueError, match=r"No Variable instance found"):
        expr.extract_variables()


def test_extract_constants_raises_when_missing() -> None:
    var = QProgramVariable("time", domain=QProgramDomain.Time)
    other = QProgramVariable("other", domain=QProgramDomain.Time)
    expr = var + other
    with pytest.raises(ValueError, match=r"No Variable instance found"):
        expr.extract_constants()
