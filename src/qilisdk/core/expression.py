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

"""A symbolic expression tree (AST) for qilisdk.

The public entry point is :class:`Expression`, the base of every node:

* leaves -- :class:`Constant` and the variable family (``Parameter``, ``Variable``,
  ``BinaryVariable``, ``SpinVariable``) defined in :mod:`qilisdk.core.variables`;
* operator nodes -- :class:`Add`, :class:`Mul`, :class:`Pow`;
* unary maths functions -- :class:`Function` and its concrete subclasses (``Sin``, ``Cos``,
  ``Exp``, ``Log``, ``Tan``, ``Sqrt``).

Construction *canonicalizes* (a cheap, total normalization: flattening, combining like terms/powers,
folding constants, eliminating identities, ordering operands deterministically). Canonical form is
the sole definition of ``==``/``hash`` -- equal expressions are structurally identical and equality
is order-independent for ``+`` and ``*`` (``x + y == y + x``). Semantic rewrites -- :meth:`expand` and
:meth:`derivative` -- are explicit and never participate in equality.

The tree itself lives in C++ (``expression_module``): every operator and constant node holds a handle
to its C++ node, and the classes here are thin wrappers over it. Variables are the exception, they
stay plain Python objects that the C++ tree references.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import expression_module as _cpp  # ty:ignore
import numpy as np

from qilisdk.core.exceptions import NonPolynomialError, NotSupportedOperation
from qilisdk.settings import get_settings
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .types import Number, RealNumber
    from .variables import BaseVariable, Parameter

_TOL = get_settings().atol

# The C++ node kinds that the trusting constructors build directly
_KIND_POW = 2
_KIND_MUL = 3
_KIND_ADD = 4


def _coerce(obj: object) -> Expression | None:
    """Coerce a value into an :class:`Expression`.

    Returns:
        Expression | None: ``obj`` if it is already an ``Expression``, a :class:`Constant` wrapping a
        number, or ``None`` if ``obj`` is neither.
    """
    if isinstance(obj, Expression):
        return obj
    try:
        return Constant(obj)  # ty:ignore[invalid-argument-type]
    except TypeError:
        return None


class Expression:
    """Base of every node in the expression tree."""

    __slots__ = ()

    # Variables keep no node of their own, so this default marks them for the C++ side
    _node: ClassVar[object] = None

    # ---- markers ----
    # Plain class attributes rather than properties, for speed

    #: Whether ``self * self == self`` (true only for ``BinaryVariable``).
    is_idempotent_under_mul: ClassVar[bool] = False

    #: Whether this expression is a :class:`~qilisdk.core.variables.Parameter` leaf.
    is_parameter: ClassVar[bool] = False

    # ---- core ----
    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> Number:
        """Numerically evaluate the expression given an assignment of symbols to values.

        Returns:
            Number: a ``float`` for real results, otherwise a ``complex``.
        """
        return _cpp.evaluate(self, env)

    def free_symbols(self) -> set[BaseVariable]:
        """The set of named leaves (variables/parameters) occurring in the expression.

        Returns:
            set[BaseVariable]: the variables and parameters.
        """
        return set(_cpp.free_symbols(self))

    @property
    def degree(self) -> int:
        """Polynomial degree. Raises :class:`NonPolynomialError` for non-polynomial expressions."""
        return _cpp.degree(self)

    def derivative(self, symbol: BaseVariable) -> Expression:
        """Symbolic derivative with respect to ``symbol``.

        Returns:
            Expression: the derivative.
        """
        return _cpp.derivative(self, symbol)

    def expand(self) -> Expression:
        """Distribute products over sums.

        Returns:
            Expression: a canonical sum-of-monomials equal to this expression.
        """
        return _cpp.expand(self)

    def substitute(self, mapping: Mapping[Expression, Expression | Number]) -> Expression:
        """Structurally replace sub-expressions according to ``mapping``.

        Returns:
            Expression: the expression with substitutions applied.
        """
        return _cpp.substitute(self, mapping)

    def to_binary(self) -> Expression:
        """Encode every continuous ``Variable`` into binary variables (no-op for most nodes).

        Returns:
            Expression: an equivalent expression over binary variables.
        """
        return _cpp.to_binary(self)

    def free_parameters(self) -> set[Parameter]:
        """Collect the parameters occurring in the expression.

        Returns:
            set[Parameter]: the free parameters.
        """
        return {symbol for symbol in self.free_symbols() if symbol.is_parameter}  # ty:ignore[invalid-return-type]

    def variables(self) -> list[BaseVariable]:
        """Collect the named leaves of the expression.

        Returns:
            list[BaseVariable]: the variables and parameters, sorted by label.
        """
        return _cpp.variables(self)

    def is_parameterized(self) -> bool:
        """Whether every named leaf is a :class:`~qilisdk.core.variables.Parameter`.

        Returns:
            bool: ``True`` if the expression contains only parameters (or no variables).
        """
        return _cpp.is_parameterized(self)

    def get_constant(self) -> Number:
        """Return the additive constant of the expression.

        Returns:
            Number: the constant term (0 unless the expression is a sum or a constant).
        """
        return _cpp.get_constant(self)

    def as_coefficients_dict(self) -> dict[Expression, Number]:
        """Map each (non-constant) monomial to its numeric coefficient.

        Returns:
            dict[Expression, Number]: a mapping from monomial to coefficient.
        """
        return _cpp.as_coefficients_dict(self)

    def monomial_factors(self) -> list[tuple[Expression, int]]:
        """Decompose a single monomial into its factors.

        Returns:
            list[tuple[Expression, int]]: the ``(base, integer_power)`` factors.
        """
        return _cpp.monomial_factors(self)

    def to_list(self) -> list[Expression]:
        """The node's operands as a list: the summands of a sum, the factors of a product.

        A leaf has no operands, so it yields itself. This is the readable spelling of ``.args``.

        Returns:
            list[Expression]: the operands of this node.
        """
        return _cpp.to_list(self)

    # ---- identity ----
    def __hash__(self) -> int:
        return _cpp.hash(self)

    def __eq__(self, other: object) -> bool:
        return _cpp.equal(self, other)

    def __contains__(self, item: object) -> bool:
        """Whether ``item`` is one of the named leaves (variables/parameters) of the expression.

        Returns:
            bool: ``True`` if ``item`` appears as a free symbol in the expression.
        """
        return isinstance(item, Expression) and item in self.free_symbols()

    def __repr__(self) -> str:
        return _cpp.repr(self)

    # ---- arithmetic ----
    def __add__(self, other: object) -> Expression:
        return _cpp.add(self, other)

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other: object) -> Expression:
        return _cpp.sub(self, other)

    def __rsub__(self, other: object) -> Expression:
        return _cpp.rsub(self, other)

    __isub__ = __sub__

    def __neg__(self) -> Expression:
        return _cpp.neg(self)

    def __mul__(self, other: object) -> Expression:
        return _cpp.mul(self, other)

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other: object) -> Expression:
        return _cpp.truediv(self, other)

    __itruediv__ = __truediv__

    def __rtruediv__(self, other: object) -> Expression:
        return _cpp.rtruediv(self, other)

    def __floordiv__(self, other: object) -> Expression:
        raise NotSupportedOperation("Floor division is not supported for symbolic expressions")

    __rfloordiv__ = __floordiv__

    def __pow__(self, other: object) -> Expression:
        return _cpp.pow(self, other)

    def __rpow__(self, other: object) -> Expression:
        return _cpp.rpow(self, other)


class _Node(Expression):
    """A constant, operator or function node, backed by a C++ node in ``_node``.

    Nodes are immutable, so a copy shares the C++ node with the original.
    """

    # The weak reference lets a node hand back its live wrapper rather than a new one each time
    __slots__ = ("__weakref__", "_node")

    def __copy__(self) -> Expression:
        clone = object.__new__(type(self))
        clone._node = self._node
        return clone


@yaml.register_class
class Constant(_Node):
    """A numeric literal leaf. Replaces the old ``Term.CONST`` sentinel."""

    __slots__ = ()

    def __init__(self, value: Number) -> None:
        self._node = _cpp.constant_node(value)

    @property
    def value(self) -> Number:
        """The numeric literal. Read-only: nodes are immutable so the cached hash stays valid."""
        return _cpp.value(self)

    def __getstate__(self) -> dict:
        return {"_value": self.value}

    def __setstate__(self, state: dict) -> None:
        self._node = _cpp.constant_node(state["_value"])


@yaml.register_class
class Add(_Node):
    """A canonical n-ary sum. Operands are deterministically ordered and combine like terms."""

    __slots__ = ()

    def __init__(self, args: tuple[Expression, ...]) -> None:
        # Trusting constructor: ``args`` must already be canonical (flattened, like-terms combined,
        # at most one Constant, length >= 2, deterministically sorted). Use ``build`` to normalize.
        self._node = _cpp.node(_KIND_ADD, args)

    @property
    def args(self) -> tuple[Expression, ...]:
        """The summands. Read-only: nodes are immutable so the cached hash stays valid."""
        return _cpp.args(self)

    @classmethod
    def build(cls, raw: Iterable[Expression]) -> Expression:
        """Build the canonical sum of ``raw``.

        Returns:
            Expression: the sum, collapsed to a single operand or a constant where possible.
        """
        return _cpp.build_add(raw)

    def __getstate__(self) -> dict:
        return {"_args": self.args}

    def __setstate__(self, state: dict) -> None:
        self._node = _cpp.node(_KIND_ADD, state["_args"])


@yaml.register_class
class Mul(_Node):
    """A canonical n-ary product. Collects like powers and folds the numeric coefficient.

    ``Mul`` does **not** distribute over sums; use :meth:`expand` for that.
    """

    __slots__ = ()

    def __init__(self, args: tuple[Expression, ...]) -> None:
        # Trusting constructor: ``args`` must already be canonical. Use ``build`` to normalize.
        self._node = _cpp.node(_KIND_MUL, args)

    @property
    def args(self) -> tuple[Expression, ...]:
        """The factors. Read-only: nodes are immutable so the cached hash stays valid."""
        return _cpp.args(self)

    @classmethod
    def build(cls, raw: Iterable[Expression]) -> Expression:
        """Build the canonical product of ``raw``.

        Returns:
            Expression: the product, collapsed to a single factor or a constant where possible.
        """
        return _cpp.build_mul(raw)

    def coefficient(self) -> Number:
        """The numeric coefficient of the product.

        Returns:
            Number: the leading constant factor, or 1 if there is none.
        """
        return _cpp.coefficient(self)

    def monomial(self) -> Expression:
        """The product without its numeric coefficient.

        Returns:
            Expression: the remaining factors.
        """
        return _cpp.monomial(self)

    def __getstate__(self) -> dict:
        return {"_args": self.args}

    def __setstate__(self, state: dict) -> None:
        self._node = _cpp.node(_KIND_MUL, state["_args"])


@yaml.register_class
class Pow(_Node):
    """A power ``base ** exp``. The exponent may be a numeric or symbolic :class:`Expression`."""

    __slots__ = ()

    def __init__(self, base: Expression, exp: Expression) -> None:
        # Trusting constructor: use ``build`` to normalize.
        self._node = _cpp.node(_KIND_POW, (base, exp))

    @property
    def base(self) -> Expression:
        """The base. Read-only: nodes are immutable so the cached hash stays valid."""
        return _cpp.args(self)[0]

    @property
    def exp(self) -> Expression:
        """The exponent. Read-only: nodes are immutable so the cached hash stays valid."""
        return _cpp.args(self)[1]

    @classmethod
    def build(cls, base: Expression, exp: Expression) -> Expression:
        """Build the canonical power ``base ** exp``.

        Returns:
            Expression: the power, folded or simplified where possible.
        """
        return _cpp.build_pow(base, exp)

    def __getstate__(self) -> dict:
        return {"_base": self.base, "_exp": self.exp}

    def __setstate__(self, state: dict) -> None:
        self._node = _cpp.node(_KIND_POW, (state["_base"], state["_exp"]))


class Function(_Node, ABC):
    """Abstract base for unary maths functions (``sin``, ``cos``, ``exp``, ...).

    A concrete function declares -- all at the *class* level so serialization carries only the
    operand -- a stable ``NAME``, a numpy numeric kernel :meth:`_numeric`, and the local outer
    derivative :meth:`_outer_derivative` (the chain rule is applied by :meth:`derivative`).
    """

    __slots__ = ()

    NAME: ClassVar[str] = ""

    def __new__(cls, arg: object) -> Expression:
        # A constant operand folds straight to a Constant
        return _cpp.function(cls, arg)

    def __init__(self, arg: object) -> None:
        """The node is already built by ``__new__``."""

    @property
    def arg(self) -> Expression:
        """The operand. Read-only: nodes are immutable so the cached hash stays valid."""
        return _cpp.args(self)[0]

    @staticmethod
    @abstractmethod
    def _numeric(value: RealNumber) -> Number:
        """Numeric kernel applied to an already-evaluated, real operand."""

    @abstractmethod
    def _outer_derivative(self, operand: Expression) -> Expression:
        """The outer derivative ``f'(operand)`` as an expression (chain rule applied by :meth:`derivative`)."""

    def __getnewargs__(self) -> tuple[Expression]:
        # __new__ needs the operand, so copy and pickle must hand it back.
        return (self.arg,)

    def __getstate__(self) -> dict:
        # The operand travels in __getnewargs__ alone: sending it twice would make deepcopy exponential in the depth
        return {}

    def __setstate__(self, state: dict) -> None:
        """The node is already rebuilt by ``__new__`` from :meth:`__getnewargs__`."""

    @classmethod
    def to_yaml(cls, representer, node):  # ruff: ignore[missing-type-function-argument, missing-return-type-class-method]
        return representer.represent_mapping(cls.yaml_tag, {"arg": node.arg})  # ty:ignore[unresolved-attribute]

    @classmethod
    def from_yaml(cls, constructor, node):  # ruff: ignore[missing-type-function-argument, missing-return-type-class-method]
        mapping = constructor.construct_mapping(node, deep=True)
        return cls(mapping["arg"])


@yaml.register_class
class Sin(Function):
    """Sine of an expression."""

    NAME = "sin"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.sin(value))

    def _outer_derivative(self, operand: Expression) -> Expression:  # ruff: ignore[no-self-use]
        return Cos(operand)


@yaml.register_class
class Cos(Function):
    """Cosine of an expression."""

    NAME = "cos"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.cos(value))

    def _outer_derivative(self, operand: Expression) -> Expression:  # ruff: ignore[no-self-use]
        return -Sin(operand)


@yaml.register_class
class Exp(Function):
    """Exponential of an expression."""

    NAME = "exp"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.exp(value))

    def _outer_derivative(self, operand: Expression) -> Expression:  # ruff: ignore[no-self-use]
        return Exp(operand)


@yaml.register_class
class Log(Function):
    """Natural logarithm of an expression."""

    NAME = "log"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.log(value))

    def _outer_derivative(self, operand: Expression) -> Expression:  # ruff: ignore[no-self-use]
        return Pow.build(operand, Constant(-1))


@yaml.register_class
class Tan(Function):
    """Tangent of an expression."""

    NAME = "tan"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        if abs(np.cos(value)) < _TOL:
            raise ValueError("Tangent is not defined for values where cosine is zero.")
        return float(np.tan(value))

    def _outer_derivative(self, operand: Expression) -> Expression:  # ruff: ignore[no-self-use]
        return Pow.build(Cos(operand), Constant(-2))


@yaml.register_class
class Sqrt(Function):
    """Square root of an expression."""

    NAME = "sqrt"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.sqrt(value))

    def _outer_derivative(self, operand: Expression) -> Expression:  # ruff: ignore[no-self-use]
        return Constant(0.5) * Pow.build(operand, Constant(-0.5))


@yaml.register_class
class Abs(Function):
    """Absolute value of an expression.

    ``Abs`` is not differentiable at zero and there is no ``sign`` node to express its derivative
    away from zero, so :meth:`Expression.derivative` raises on it.
    """

    NAME = "abs"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(abs(value))

    def _outer_derivative(self, operand: Expression) -> Expression:
        raise NotSupportedOperation(f"The derivative of {type(self).__name__} is not supported.")


def Inv(arg: object) -> Expression:
    """Build the reciprocal ``1 / arg``.

    This is a thin wrapper over :class:`Pow` rather than a node of its own, so ``Inv(x)`` and
    ``x ** -1`` are the same expression.

    Returns:
        Expression: the reciprocal of ``arg``.

    Raises:
        TypeError: if ``arg`` is neither an :class:`Expression` nor a number.
    """
    operand = _coerce(arg)
    if operand is None:
        raise TypeError(f"Inv expects an Expression or number, got {type(arg).__name__}")
    return Pow.build(operand, Constant(-1))


_cpp.initialize(Constant, Add, Mul, Pow, Expression, Log, NonPolynomialError, _TOL)
