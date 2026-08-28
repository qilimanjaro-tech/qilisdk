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

"""Comparisons between expressions.

A :class:`Comparison` relates two :class:`~qilisdk.core.expression.Expression` operands with one of
the six :class:`ComparisonOperation` relations. It is not an ``Expression`` itself: it is the
relation type that :class:`~qilisdk.core.model.Constraint` is built from. Build one with the
:func:`LT`/:func:`LEQ`/:func:`EQ`/:func:`NEQ`/:func:`GT`/:func:`GEQ` helpers rather than by hand.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from qilisdk.utils.hashing import hash as qili_hash
from qilisdk.yaml import yaml

from .expression import Constant, Expression, _coerce
from .types import Number, QiliEnum, RealNumber

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .variables import BaseVariable

__all__ = [
    "EQ",
    "GEQ",
    "GT",
    "LEQ",
    "LT",
    "NEQ",
    "Comparison",
    "ComparisonOperation",
    "Equal",
    "GreaterThan",
    "GreaterThanOrEqual",
    "LessThan",
    "LessThanOrEqual",
    "NotEqual",
]


def _assert_comparable(value: Number) -> RealNumber:
    """Check that an evaluated side of a comparison is a real number.

    ``Expression.evaluate`` already collapses a negligible imaginary part to a float, so anything
    still complex here has a real imaginary component and cannot be ordered.

    Returns:
        RealNumber: the value, unchanged.

    Raises:
        ValueError: if the value is complex.
    """
    if isinstance(value, complex):
        raise ValueError("evaluating inequality constraints with complex values is not allowed")
    return value


@yaml.register_class
class ComparisonOperation(QiliEnum):
    LT = "<"
    LEQ = "<="
    EQ = "=="
    NEQ = "!="
    GT = ">"
    GEQ = ">="


def LT(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> Comparison:
    """'Less Than' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        Comparison: a comparison term with the structure lhs < rhs.
    """
    return Comparison(lhs=lhs, rhs=rhs, operation=ComparisonOperation.LT)


LessThan = LT


def LEQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> Comparison:
    """'Less Than or equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        Comparison: a comparison term with the structure lhs <= rhs.
    """
    return Comparison(lhs=lhs, rhs=rhs, operation=ComparisonOperation.LEQ)


LessThanOrEqual = LEQ


def EQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> Comparison:
    """'Equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        Comparison: a comparison term with the structure lhs == rhs.
    """
    return Comparison(lhs=lhs, rhs=rhs, operation=ComparisonOperation.EQ)


Equal = EQ


def NEQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> Comparison:
    """'Not Equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        Comparison: a comparison term with the structure lhs != rhs.
    """
    return Comparison(lhs=lhs, rhs=rhs, operation=ComparisonOperation.NEQ)


NotEqual = NEQ


def GT(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> Comparison:
    """'Greater Than' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        Comparison: a comparison term with the structure lhs > rhs.
    """
    return Comparison(lhs=lhs, rhs=rhs, operation=ComparisonOperation.GT)


GreaterThan = GT


def GEQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> Comparison:
    """'Greater Than or equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        Comparison: a comparison term with the structure lhs >= rhs.
    """
    return Comparison(lhs=lhs, rhs=rhs, operation=ComparisonOperation.GEQ)


GreaterThanOrEqual = GEQ


@yaml.register_class
class Comparison:
    """A comparison (equality or inequality) between two :class:`Expression` operands.

    The relation is normalized at construction to ``lhs - rhs <op> 0`` with the additive constant
    moved to the right-hand side (so ``lhs`` carries no constant and ``rhs`` is that constant).
    """

    def __init__(
        self,
        lhs: RealNumber | Expression,
        rhs: RealNumber | Expression,
        operation: ComparisonOperation,
    ) -> None:
        """Initialize a new comparison term.

        Args:
            lhs (RealNumber | Expression): the left hand side of the comparison.
            rhs (RealNumber | Expression): the right hand side of the comparison.
            operation (ComparisonOperation): the comparison operation.

        Raises:
            TypeError: if an operand is neither a number nor an :class:`Expression`.
        """
        lhs_expr = _coerce(lhs)
        rhs_expr = _coerce(rhs)
        if lhs_expr is None or rhs_expr is None:
            raise TypeError("Comparison operands must be numbers or Expressions.")
        term = lhs_expr - rhs_expr
        const = term.get_constant()
        self._lhs: Expression = term - Constant(const)
        self._rhs: Expression = Constant(-const)
        self._operation = operation

    @property
    def operation(self) -> ComparisonOperation:
        """The comparison operation."""
        return self._operation

    @property
    def lhs(self) -> Expression:
        """The left hand side of the comparison term."""
        return self._lhs

    @property
    def rhs(self) -> Expression:
        """The right hand side of the comparison term."""
        return self._rhs

    def variables(self) -> list[BaseVariable]:
        """Collect the unique variables in the comparison term.

        Returns:
            list[BaseVariable]: the variables, sorted by label.
        """
        var = set()
        var.update(self._lhs.variables())
        var.update(self._rhs.variables())
        return sorted(var, key=lambda x: x.label)

    @property
    def degree(self) -> int:
        """The maximum degree of the two sides of the comparison term."""
        return max(self.rhs.degree, self.lhs.degree)

    def to_binary(self) -> Comparison:
        """Encode the continuous variables of both sides into binary.

        Returns:
            Comparison: the comparison term with both sides encoded into binary.
        """
        return Comparison(lhs=self.lhs.to_binary(), rhs=self.rhs.to_binary(), operation=self.operation)

    def _apply_comparison_operation(self, v1: RealNumber, v2: RealNumber) -> bool:
        if self.operation is ComparisonOperation.EQ:
            return v1 == v2
        if self.operation is ComparisonOperation.GEQ:
            return v1 >= v2
        if self.operation is ComparisonOperation.GT:
            return v1 > v2
        if self.operation is ComparisonOperation.LEQ:
            return v1 <= v2
        if self.operation is ComparisonOperation.LT:
            return v1 < v2
        if self.operation is ComparisonOperation.NEQ:
            return v1 != v2
        raise ValueError(f"Unsupported Operation of type {self.operation.value}")

    def evaluate(self, var_values: Mapping[BaseVariable, RealNumber | list[int]]) -> bool:
        """Evaluate the comparison given a set of variable values.

        Args:
            var_values (Mapping[BaseVariable, RealNumber | list[int]]): the variable assignment.

        Returns:
            bool: the result of the comparison.

        Raises:
            ValueError: if evaluation yields a complex value.
        """
        lhs = _assert_comparable(self._lhs.evaluate(var_values))
        rhs = _assert_comparable(self._rhs.evaluate(var_values))
        return self._apply_comparison_operation(lhs, rhs)

    def __getstate__(self) -> dict:
        return {"_lhs": self._lhs, "_rhs": self._rhs, "_operation": self._operation}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __copy__(self) -> Comparison:
        return Comparison(rhs=copy.copy(self.rhs), lhs=copy.copy(self.lhs), operation=self.operation)

    def __repr__(self) -> str:
        return f"{str(self.lhs).strip()} {self.operation.value} {str(self.rhs).strip()}"

    __str__ = __repr__

    def __bool__(self) -> bool:
        raise TypeError(
            "Symbolic Constraint Term objects do not have an inherent truth value. "
            "Use a method like .evaluate() to obtain a Boolean value."
        )

    def __hash__(self) -> int:
        return qili_hash(self._lhs, self.operation.value, self._rhs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Comparison):
            return False
        return hash(self) == hash(other)
