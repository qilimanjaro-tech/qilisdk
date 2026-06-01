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

"""Decision variables, parameters, encodings, and comparison relations.

The arithmetic core (``Expression`` and the operator/function nodes) lives in
:mod:`qilisdk.core.expression`; this module defines the *leaves* of that tree -- the named
:class:`BaseVariable` family (:class:`Parameter`, :class:`Variable`, :class:`BinaryVariable`,
:class:`SpinVariable`) -- together with the :class:`Encoding` strategies that lower continuous
variables to binary, and :class:`ComparisonTerm`, the (non-``Expression``) relation type produced by
the :func:`LT`/:func:`LEQ`/:func:`EQ`/:func:`NEQ`/:func:`GT`/:func:`GEQ` helpers.
"""

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np
from loguru import logger

from qilisdk.core.exceptions import EvaluationError, InvalidBoundsError, OutOfBoundsException
from qilisdk.settings import get_settings
from qilisdk.utils.hashing import hash as qili_hash
from qilisdk.yaml import yaml

# Re-export the expression algebra so existing ``from qilisdk.core.variables import ...`` keeps working.
from .expression import (
    Add,
    Constant,
    Cos,
    Exp,
    Expression,
    Function,
    Log,
    Mul,
    Pow,
    Sin,
    Sqrt,
    Tan,
    _coerce,
)
from .types import Number, QiliEnum, RealNumber

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "EQ",
    "GEQ",
    "GT",
    "LEQ",
    "LT",
    "NEQ",
    "Add",
    "BaseVariable",
    "BinaryVariable",
    "Bitwise",
    "ComparisonOperation",
    "ComparisonTerm",
    "Constant",
    "Cos",
    "Domain",
    "DomainWall",
    "Encoding",
    "Equal",
    "Exp",
    "Expression",
    "Function",
    "GreaterThan",
    "GreaterThanOrEqual",
    "LessThan",
    "LessThanOrEqual",
    "Log",
    "Mul",
    "NotEqual",
    "OneHot",
    "Parameter",
    "Pow",
    "Sin",
    "SpinVariable",
    "Sqrt",
    "Tan",
    "Variable",
]

MAX_INT = np.iinfo(np.int64).max
MIN_INT = np.iinfo(np.int64).min
LARGE_BOUND = 100


def LT(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> ComparisonTerm:
    """'Less Than' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs < rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.LT)


LessThan = LT


def LEQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> ComparisonTerm:
    """'Less Than or equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs <= rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.LEQ)


LessThanOrEqual = LEQ


def EQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> ComparisonTerm:
    """'Equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs == rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.EQ)


Equal = EQ


def NEQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> ComparisonTerm:
    """'Not Equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs != rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.NEQ)


NotEqual = NEQ


def GT(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> ComparisonTerm:
    """'Greater Than' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs > rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.GT)


GreaterThan = GT


def GEQ(lhs: RealNumber | Expression, rhs: RealNumber | Expression) -> ComparisonTerm:
    """'Greater Than or equal to' mathematical operation.

    Args:
        lhs (RealNumber | Expression): the left hand side of the comparison term.
        rhs (RealNumber | Expression): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs >= rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.GEQ)


GreaterThanOrEqual = GEQ


def _extract_number(label: str) -> int:
    """Extract the trailing ``(<number>)`` index from a variable label.

    Returns:
        int: the parsed index, or 0 if the label has no trailing ``(<number>)``.
    """
    pattern = re.compile(r"\((\d+)\)$")
    matches = pattern.search(label)
    if matches is not None:
        return int(matches.group(1))
    return 0


@yaml.register_class(shared=True)
class Domain(QiliEnum):
    INTEGER = "Integer Domain"
    POSITIVE_INTEGER = "Positive Integer Domain"
    REAL = "Real Domain"
    BINARY = "Binary Domain"
    SPIN = "Spin Domain"

    def check_value(self, value: Number) -> bool:
        """Whether ``value`` is valid for this domain.

        Args:
            value (Number): the value to be evaluated.

        Returns:
            bool: True if the value provided is valid, False otherwise.
        """
        if self == Domain.BINARY:
            return isinstance(value, Number) and value in {0, 1}
        if self == Domain.SPIN:
            return isinstance(value, Number) and value in {-1, 1}
        if self == Domain.REAL:
            return isinstance(value, (int, float))
        if self == Domain.INTEGER:
            return isinstance(value, int)
        if self == Domain.POSITIVE_INTEGER:
            return isinstance(value, int) and value >= 0
        return False

    def min(self) -> float:
        """Return the smallest value allowed by this domain.

        Returns:
            float: the minimum value allowed of a given domain.
        """
        if self in {Domain.BINARY, Domain.POSITIVE_INTEGER}:
            return 0
        if self == Domain.SPIN:
            return -1
        if self == Domain.INTEGER:
            return MIN_INT
        return -1e30

    def max(self) -> float:
        """Return the largest value allowed by this domain.

        Returns:
            float: the maximum value allowed for a given domain.
        """
        if self in {Domain.BINARY, Domain.SPIN}:
            return 1
        if self in {Domain.POSITIVE_INTEGER, Domain.INTEGER}:
            return MAX_INT
        return 1e30


@yaml.register_class
class ComparisonOperation(QiliEnum):
    LT = "<"
    LEQ = "<="
    EQ = "=="
    NEQ = "!="
    GT = ">"
    GEQ = ">="


@yaml.register_class
class Encoding(ABC):
    """Abstract variable encoding: how a continuous variable is represented in binary variables."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The encoding's name."""

    @staticmethod
    @abstractmethod
    def encode(var: Variable, precision: float = 1e-2) -> Expression:
        """Return an expression of binary variables representing ``var`` in this encoding."""

    @staticmethod
    @abstractmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        """Return a constraint that ensures the encoding is respected."""

    @staticmethod
    @abstractmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        """Decode a binary assignment into the value of the continuous variable."""

    @staticmethod
    @abstractmethod
    def num_binary_equivalent(var: Variable, precision: float = 1e-2) -> int:
        """Number of binary variables needed to encode ``var``."""

    @staticmethod
    @abstractmethod
    def check_valid(value: list[int] | int) -> tuple[bool, int]:
        """Whether ``value`` is a valid sample in this encoding (and the encoding error)."""


def _check_output(var: Variable, output: Number) -> RealNumber:
    """Parse an evaluation output into a real number within ``var``'s domain.

    Args:
        var (Variable): the variable being evaluated.
        output (Number): the raw evaluation output.

    Returns:
        RealNumber: the output coerced to a valid value within the variable's domain.

    Raises:
        ValueError: if the output is not real or violates the domain.
    """
    if isinstance(output, RealNumber):
        out = float(output)
    elif isinstance(output, complex) and abs(output.imag) < get_settings().atol:
        out = float(output.real)
    else:
        raise ValueError(f"Evaluation answer ({output}) is outside the variable domain ({var.domain}).")

    out = int(out) if var.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER} else out

    if not var.domain.check_value(out):
        raise ValueError(f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}")

    return out


@yaml.register_class
class Bitwise(Encoding):
    """Bitwise (binary) variable encoding."""

    name = "Bitwise"

    @staticmethod
    def _bitwise_encode(x: int, N: int) -> list[int]:
        """Encode the integer ``x`` using ``N`` bits (little-endian).

        Returns:
            list[int]: the little-endian binary digits of ``x``.
        """
        return list(reversed([int(b) for b in format(x, f"0{N}b")]))

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Expression:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        abs_bound = np.abs(bounds[1] - bounds[0])
        n_binary = int(np.floor(np.log2(abs_bound if abs_bound != 0 else 1)))
        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary + 1)]

        term: Expression = Constant(0)
        for i in range(n_binary):
            term += 2**i * binary_vars[i]
        term += (np.abs(bounds[1] - bounds[0]) + 1 - 2**n_binary) * binary_vars[-1]
        term += bounds[0]
        return term * var.precision if var.domain is Domain.REAL else term

    @staticmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        term = Bitwise.encode(var, precision)
        binary_var = sorted(term.variables(), key=lambda x: _extract_number(x.label))

        binary_list = Bitwise._bitwise_encode(value, len(binary_var)) if isinstance(value, Number) else value

        if not Bitwise.check_valid(binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the Bitwise encoding.")

        if len(binary_list) < len(binary_var):
            for _ in range(len(binary_var) - len(binary_list)):
                binary_list.append(0)
        elif len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict: dict[BaseVariable, list[int]] = {binary_var[i]: [binary_list[i]] for i in range(len(binary_list))}

        out = _check_output(var, term.evaluate(binary_dict))

        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        raise NotImplementedError("Bitwise encoding constraints are not supported at the moment")

    @staticmethod
    def num_binary_equivalent(var: Variable, precision: float = 1e-2) -> int:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.floor(np.log2(np.abs(bounds[1] - bounds[0]))))

        return n_binary + 1

    @staticmethod
    def check_valid(value: list[int] | int) -> tuple[bool, int]:
        return True, 0


@yaml.register_class
class OneHot(Encoding):
    """One-Hot variable encoding."""

    name = "One-Hot"

    @staticmethod
    def _one_hot_encode(x: int, N: int) -> list[int]:
        """One-hot encode integer ``x`` in range ``[0, N-1]``.

        Returns:
            list[int]: a length-``N`` list that is 1 at index ``x`` and 0 elsewhere.

        Raises:
            ValueError: if ``x`` is outside ``[0, N-1]``.
        """
        if not (0 <= x < N):
            raise ValueError(f"the input value ({x}) must be in range [0, {N - 1}]")
        return [1 if i == x else 0 for i in range(N)]

    @staticmethod
    def _find_zero(var: Variable) -> int:
        binary_var = var.bin_vars
        term = var.term
        for i in range(var.num_binary_equivalent()):
            if binary_var[i] not in term.free_symbols():
                return i
        return 0

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Expression:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]

        term = Add.build(tuple((bounds[0] + i) * binary_vars[i] for i in range(n_binary)))

        return term * var.precision if var.domain is Domain.REAL else term

    @staticmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        term = OneHot.encode(var, precision)
        binary_var = sorted(term.variables(), key=lambda x: _extract_number(x.label))

        binary_list = OneHot._one_hot_encode(value, len(binary_var) + 1) if isinstance(value, int) else value

        if not OneHot.check_valid(binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the one hot encoding.")

        # after encoding we will have one less variable than the binary list, because the first variable is multiplied
        # by 0 so it is removed from the term.
        if len(binary_list) < len(binary_var) + 1:
            for _ in range(len(binary_var) - len(binary_list) + 1):
                binary_list.append(0)
        elif len(binary_list) != len(binary_var) + 1:
            raise ValueError(f"expected {len(binary_var) + 1} variables but received {len(binary_list)}")

        zero_index = OneHot._find_zero(var)
        binary_dict: dict[BaseVariable, list[int]] = {}
        for i in range(var.num_binary_equivalent()):
            if i < zero_index:
                binary_dict[binary_var[i]] = [binary_list[i]]
            if i > zero_index:
                binary_dict[binary_var[i - 1]] = [binary_list[i]]

        out = _check_output(var, term.evaluate(binary_dict))

        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]
        return ComparisonTerm(lhs=sum(binary_vars, Constant(0)), rhs=1, operation=ComparisonOperation.EQ)

    @staticmethod
    def num_binary_equivalent(var: Variable, precision: float = 1e-2) -> int:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        return n_binary

    @staticmethod
    def check_valid(value: list[int] | int) -> tuple[bool, int]:
        binary_list = OneHot._one_hot_encode(value, value) if isinstance(value, int) else value
        num_ones = binary_list.count(1)
        return num_ones == 1, (num_ones - 1) ** 2


@yaml.register_class
class DomainWall(Encoding):
    """Domain-wall variable encoding."""

    name = "Domain Wall"

    @staticmethod
    def _domain_wall_encode(x: int, N: int) -> list[int]:
        """Domain-wall encode integer ``x`` in range ``[0, N]``.

        Returns:
            list[int]: ``x`` ones followed by ``N - x`` zeros.

        Raises:
            ValueError: if ``x`` is outside ``[0, N]``.
        """
        if not (0 <= x <= N):
            raise ValueError(f"the input value ({x}) must be in range [0, {N}]")
        return [1] * x + [0] * (N - x)

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Expression:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]

        term: Expression = Constant(0)
        for i in range(n_binary):
            term += binary_vars[i]
        term += bounds[0]

        return term * var.precision if var.domain is Domain.REAL else term

    @staticmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        term = DomainWall.encode(var, precision)
        binary_var = sorted(term.variables(), key=lambda x: _extract_number(x.label))

        binary_list: list[int] = (
            DomainWall._domain_wall_encode(value, len(binary_var)) if isinstance(value, RealNumber) else value
        )

        if not DomainWall.check_valid(binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the domain wall encoding.")

        if len(binary_list) < len(binary_var):
            for _ in range(len(binary_var) - len(binary_list)):
                binary_list.append(0)
        elif len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict: dict[BaseVariable, list[int]] = {binary_var[i]: [binary_list[i]] for i in range(len(binary_list))}

        out = _check_output(var, term.evaluate(binary_dict))

        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]
        return ComparisonTerm(
            lhs=sum((binary_vars[i + 1] * (1 - binary_vars[i]) for i in range(len(binary_vars) - 1)), Constant(0)),
            rhs=0,
            operation=ComparisonOperation.EQ,
        )

    @staticmethod
    def num_binary_equivalent(var: Variable, precision: float = 1e-2) -> int:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        return n_binary

    @staticmethod
    def check_valid(value: list[int] | int) -> tuple[bool, int]:
        binary_list = DomainWall._domain_wall_encode(value, value) if isinstance(value, int) else value
        value = sum(binary_list[i + 1] * (1 - binary_list[i]) for i in range(len(binary_list) - 1))
        return value == 0, value


# Variables ###


class BaseVariable(Expression, ABC):
    """Abstract base class for symbolic named leaves (decision variables and parameters)."""

    def __init__(self, label: str, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        """Initialize a new variable.

        Args:
            label (str): The name of the variable.
            domain (Domain): The domain of the values this variable can take.
            bounds (tuple[float | None, float | None], optional): the (lower, upper) bounds, both
                included. ``None`` selects the domain's extreme. Defaults to (None, None).

        Raises:
            OutOfBoundsException: a bound does not respect the variable's domain.
            InvalidBoundsError: the lower bound is greater than the upper bound.
        """
        self._label = label
        self._domain = domain

        lower_bound, upper_bound = bounds
        if lower_bound is None:
            lower_bound = domain.min()
        if upper_bound is None:
            upper_bound = domain.max()

        if not self.domain.check_value(upper_bound):
            raise OutOfBoundsException(
                f"the upper bound ({upper_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if not self.domain.check_value(lower_bound):
            raise OutOfBoundsException(
                f"the lower bound ({lower_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if lower_bound > upper_bound:
            raise InvalidBoundsError("lower bound can't be larger than the upper bound.")
        self._bounds = (lower_bound, upper_bound)
        self._hash_cache: int | None = None

    @property
    def bounds(self) -> tuple[float, float]:
        """The (lower, upper) bounds of the variable."""
        return self._bounds

    @property
    def lower_bound(self) -> float:
        """The lower bound of the variable."""
        return self._bounds[0]

    @property
    def upper_bound(self) -> float:
        """The upper bound of the variable."""
        return self._bounds[1]

    @property
    def label(self) -> str:
        """The label (name) of the variable."""
        return self._label

    @property
    def domain(self) -> Domain:
        """The domain of values the variable can take."""
        return self._domain

    def set_bounds(self, lower_bound: float | None, upper_bound: float | None) -> None:
        """Set the bounds of the variable.

        Args:
            lower_bound (float | None): The lower bound (``None`` -> domain minimum).
            upper_bound (float | None): The upper bound (``None`` -> domain maximum).

        Raises:
            OutOfBoundsException: a bound does not respect the variable's domain.
            InvalidBoundsError: the lower bound is greater than the upper bound.
        """
        self._hash_cache = None
        if lower_bound is None:
            lower_bound = self._domain.min()
        if upper_bound is None:
            upper_bound = self._domain.max()
        if not self.domain.check_value(lower_bound):
            raise OutOfBoundsException(
                f"the lower bound ({lower_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if not self.domain.check_value(upper_bound):
            raise OutOfBoundsException(
                f"the upper bound ({upper_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if lower_bound > upper_bound:
            raise InvalidBoundsError(
                f"the lower bound ({lower_bound}) should not be greater than the upper bound ({upper_bound})"
            )
        self._bounds = (lower_bound, upper_bound)

    @abstractmethod
    def num_binary_equivalent(self) -> int:
        """Number of binary variables needed to represent this variable in its encoding."""

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        """Replace this variable's domain and bounds.

        Args:
            domain (Domain): The updated domain of the variable.
            bounds (tuple[float | None, float | None]): The updated bounds. Defaults to (None, None).
        """
        self._hash_cache = None
        self._domain = domain
        self.set_bounds(bounds[0], bounds[1])

    # ------------------------------------------------------------------ Expression interface
    def free_symbols(self) -> set[BaseVariable]:
        return {self}

    @property
    def degree(self) -> int:
        return 1

    def diff(self, symbol: BaseVariable) -> Expression:
        return Constant(1) if self == symbol else Constant(0)

    def _sort_key(self) -> tuple:
        return (1, self._label)

    def _compute_hash(self) -> int:
        return qili_hash(self._label)

    def __repr__(self) -> str:
        return f"{self._label}"

    def __str__(self) -> str:
        return f"{self._label}"


@yaml.register_class
class BinaryVariable(BaseVariable):
    """Binary decision variable restricted to ``{0, 1}``.

    Example:
        .. code-block:: python

            from qilisdk.core.variables import BinaryVariable

            x = BinaryVariable("x")
    """

    def __init__(self, label: str) -> None:
        super().__init__(label=label, domain=Domain.BINARY)

    @property
    def is_idempotent_under_mul(self) -> bool:
        return True

    def num_binary_equivalent(self) -> int:  # noqa: PLR6301
        return 1

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> RealNumber:
        env = env if env is not None else {}
        if self not in env:
            raise EvaluationError(f"No value was provided to evaluate the binary variable {self}.")
        value = env[self]
        if isinstance(value, (int, float)):
            if value in {1.0, 0.0}:
                return int(value)
            if not self.domain.check_value(value):
                raise EvaluationError(f"Evaluating a Binary variable with a value {value} that is outside the domain.")
            return value
        if not isinstance(value, list):
            raise EvaluationError(f"Evaluating a Binary variable with an unsupported value {value!r}.")
        if len(value) != 1:
            raise EvaluationError("Evaluating a Binary variable with a binary list of more than one item.")
        return value[0]

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        raise NotImplementedError

    def __copy__(self) -> BinaryVariable:
        return BinaryVariable(label=self.label)


@yaml.register_class
class SpinVariable(BaseVariable):
    """Spin decision variable restricted to ``{-1, 1}``."""

    def __init__(self, label: str) -> None:
        super().__init__(label=label, domain=Domain.SPIN, bounds=(-1, 1))

    def num_binary_equivalent(self) -> int:  # noqa: PLR6301
        return 1

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        raise NotImplementedError

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> RealNumber:
        env = env if env is not None else {}
        if self not in env:
            raise EvaluationError(f"No value was provided to evaluate the spin variable {self}.")
        value = env[self]
        if isinstance(value, Number):
            if not self.domain.check_value(value) and value != 0:
                raise EvaluationError(f"Evaluating a Spin variable with a value {value} that is outside the domain.")
            return -1 if value in {0, -1} else 1
        if len(value) != 1:
            raise EvaluationError("Evaluating a Spin variable with a list of more than one item.")
        return -1 if value[0] in {0, -1} else 1

    def __copy__(self) -> SpinVariable:
        return SpinVariable(label=self.label)


@yaml.register_class
class Variable(BaseVariable):
    """Generic (possibly continuous) optimization variable with a configurable binary encoding.

    Example:
        .. code-block:: python

            from qilisdk.core.variables import Domain, Variable

            price = Variable("price", domain=Domain.REAL, bounds=(0, 10))
            binary_term = price.to_binary()
    """

    def __init__(
        self,
        label: str,
        domain: Domain,
        bounds: tuple[float | None, float | None] = (None, None),
        encoding: type[Encoding] = Bitwise,
        precision: float = 1e-2,
    ) -> None:
        """Initialize a new generic variable.

        Args:
            label (str): The name of the variable.
            domain (Domain): The domain of the values this variable can take.
            bounds (tuple[float | None, float | None], optional): the (lower, upper) bounds, both
                included. ``None`` selects the domain's extreme. Defaults to (None, None).
            encoding (type[Encoding], optional): the binary encoding. Defaults to Bitwise.
            precision (float, optional): the floating point precision for REAL variables. Defaults to 1e-2.
        """
        super().__init__(label=label, domain=domain, bounds=bounds)
        self._encoding = encoding
        self._precision = precision
        self._term: Expression | None = None
        self._bin_vars: list[BaseVariable] = []

    @property
    def encoding(self) -> type[Encoding]:
        return self._encoding

    @property
    def precision(self) -> float:
        return self._precision

    @property
    def term(self) -> Expression:
        if self._term is None:
            if self.bounds[1] > LARGE_BOUND or self.bounds[0] < -LARGE_BOUND:
                logger.warning(
                    f"Encoding variable {self.label} which has the bounds {self.bounds}"
                    + "is very expensive and may take a very long time."
                )
            self._term = self.to_binary()
        return self._term

    @property
    def bin_vars(self) -> list[BaseVariable]:
        if self._term is None:
            self.to_binary()
        return self._bin_vars

    def set_precision(self, precision: float) -> None:
        self._precision = precision
        self._term = None

    def __copy__(self) -> Variable:
        return Variable(label=self.label, domain=self.domain, bounds=self.bounds, encoding=self._encoding)

    def __getitem__(self, item: int) -> BaseVariable:
        if self._term is None:
            self.to_binary()
        return self._bin_vars[item]

    def update_variable(
        self,
        domain: Domain,
        bounds: tuple[float | None, float | None] = (None, None),
        encoding: type[Encoding] | None = None,
    ) -> None:
        self._encoding = encoding if encoding is not None else self._encoding
        self._term = None
        return super().update_variable(domain, bounds)

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> RealNumber:
        env = env if env is not None else {}
        if self not in env:
            raise EvaluationError(f"No value was provided to evaluate the variable {self}.")
        value = env[self]
        if isinstance(value, (int, float)):
            if not self.domain.check_value(value):
                raise ValueError(f"The value {value} is invalid for the domain {self.domain.value}")
            if value < self.lower_bound or value > self.upper_bound:
                raise ValueError(f"The value {value} is outside the defined bounds {self.bounds}")
            return value
        if not isinstance(value, list):
            raise EvaluationError(f"Evaluating variable {self} with an unsupported value {value!r}.")
        return self.encoding.evaluate(self, value, self._precision)

    def to_binary(self) -> Expression:
        if self._term is None:
            term = self.encoding.encode(self, precision=self._precision)
            self._term = copy.copy(term)
            self._bin_vars = [BinaryVariable(f"{self.label}({i})") for i in range(self.num_binary_equivalent())]
            self._bin_vars = sorted(self._bin_vars, key=lambda x: _extract_number(x.label))
        return self._term

    def num_binary_equivalent(self) -> int:
        """Number of binary variables needed to encode the continuous variable.

        Returns:
            int: the number of binary variables in the variable's encoding.
        """
        return self.encoding.num_binary_equivalent(self, precision=self._precision)

    def check_valid(self, binary_list: list[int]) -> tuple[bool, int]:
        """Check whether ``binary_list`` is a valid sample in the variable's encoding.

        Returns:
            tuple[bool, int]: whether the sample is valid, and the encoding error.
        """
        return self.encoding.check_valid(binary_list)

    def encoding_constraint(self) -> ComparisonTerm:
        """Return a constraint that ensures the variable's encoding is respected."""
        return self.encoding.encoding_constraint(self, precision=self._precision)


@yaml.register_class(shared=True)
class Parameter(BaseVariable):
    """Symbolic scalar used to parametrize expressions while remaining differentiable.

    Example:
        .. code-block:: python

            from qilisdk.core.variables import Parameter

            theta = Parameter("theta", value=0.5)
            theta.set_value(0.75)
    """

    def __init__(
        self,
        label: str,
        value: RealNumber,
        domain: Domain = Domain.REAL,
        bounds: tuple[float | None, float | None] = (None, None),
        trainable: bool = True,
    ) -> None:
        super().__init__(label=label, domain=domain, bounds=bounds)

        if not self.domain.check_value(value):
            raise ValueError(
                f"Parameter value provided ({value}) doesn't correspond to the parameter's domain ({self.domain.name})"
            )
        self._value = value
        self._trainable = trainable
        self.set_bounds(bounds[0], bounds[1])

    @property
    def is_parameter(self) -> bool:
        return True

    @property
    def value(self) -> RealNumber:
        return self._value

    @property
    def is_trainable(self) -> bool:
        return self._trainable

    def check_value(self, value: RealNumber) -> None:
        if not self.domain.check_value(value):
            raise ValueError(
                f"Parameter value provided ({value}) doesn't correspond to the parameter's domain ({self.domain.name})"
            )
        if value > self.bounds[1] or value < self.bounds[0]:
            raise ValueError(f"The value provided ({value}) is outside the bound of the parameter {self.bounds}")

    def set_value(self, value: RealNumber) -> None:
        self.check_value(value)
        self._value = cast("RealNumber", value.item()) if isinstance(value, np.generic) else value

    def num_binary_equivalent(self) -> int:  # noqa: PLR6301
        """A parameter has no binary representation.

        Returns:
            int: always 0; parameters are not encoded into binary variables.
        """
        return 0

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> RealNumber:
        """Evaluate the parameter, using the value from ``env`` if present else its stored value.

        Args:
            env (Mapping[BaseVariable, Number | list[int]] | None): an optional assignment.

        Returns:
            RealNumber: the parameter's value.
        """
        env = env if env is not None else {}
        if self in env:
            value = env[self]
            if not isinstance(value, RealNumber):
                raise NotImplementedError("Evaluating the value of a parameter with a list is not supported.")
            self.check_value(value)
            return value
        return self.value

    def to_binary(self) -> Expression:
        """Return the constant representation of the parameter."""
        return Constant(self.value)

    def set_bounds(self, lower_bound: float | None, upper_bound: float | None) -> None:
        upper_bound = upper_bound if upper_bound is not None else self.domain.max()
        lower_bound = lower_bound if lower_bound is not None else self.domain.min()
        if self.value > upper_bound or self.value < lower_bound:
            raise ValueError(
                f"The current value of the parameter ({self.value}) is outside the bounds ({lower_bound}, {upper_bound})"
            )
        super().set_bounds(lower_bound, upper_bound)

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        if len(bounds) != 2:  # noqa: PLR2004
            raise ValueError(
                "Invalid bounds provided: the bounds need to be a tuple with the format (lower_bound, upper_bound)"
            )
        if domain.check_value(self.value):
            self._domain = domain
        else:
            raise ValueError(
                f"The provided domain ({domain.name}) is incompatible with the current parameter value ({self.value})"
            )
        self.set_bounds(lower_bound=bounds[0], upper_bound=bounds[1])

    __hash__ = Expression.__hash__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Expression):
            return hash(self) == hash(other)
        if isinstance(other, (float, int)):
            return self.value == other
        return False

    def __le__(self, other: object) -> bool:
        if isinstance(other, (float, int)):
            return self.value <= other
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (float, int)):
            return self.value < other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, (float, int)):
            return self.value >= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, (float, int)):
            return self.value > other
        return NotImplemented


@yaml.register_class
class ComparisonTerm:
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
            raise TypeError("ComparisonTerm operands must be numbers or Expressions.")
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

    def to_binary(self) -> ComparisonTerm:
        """Encode the continuous variables of both sides into binary.

        Returns:
            ComparisonTerm: the comparison term with both sides encoded into binary.
        """
        return ComparisonTerm(lhs=self.lhs.to_binary(), rhs=self.rhs.to_binary(), operation=self.operation)

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
        lhs = self._lhs.evaluate(var_values)
        rhs = self._rhs.evaluate(var_values)
        if isinstance(lhs, complex):
            if abs(lhs.imag) > get_settings().atol:
                raise ValueError("evaluating inequality constraints with complex values is not allowed")
            lhs = lhs.real
        if isinstance(rhs, complex):
            if abs(rhs.imag) > get_settings().atol:
                raise ValueError("evaluating inequality constraints with complex values is not allowed")
            rhs = rhs.real
        return self._apply_comparison_operation(lhs, rhs)

    def __getstate__(self) -> dict:
        return {"_lhs": self._lhs, "_rhs": self._rhs, "_operation": self._operation}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __copy__(self) -> ComparisonTerm:
        return ComparisonTerm(rhs=copy.copy(self.rhs), lhs=copy.copy(self.lhs), operation=self.operation)

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
        if not isinstance(other, ComparisonTerm):
            return False
        return hash(self) == hash(other)
