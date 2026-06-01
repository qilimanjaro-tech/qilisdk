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

The public entry point is :class:`Expression`, the abstract base of every node:

* leaves -- :class:`Constant` and the variable family (``Parameter``, ``Variable``,
  ``BinaryVariable``, ``SpinVariable``) defined in :mod:`qilisdk.core.variables`;
* operator nodes -- :class:`Add`, :class:`Mul`, :class:`Pow`;
* unary maths functions -- :class:`Function` and its concrete subclasses (``Sin``, ``Cos``,
  ``Exp``, ``Log``, ``Tan``, ``Sqrt``).

Construction *canonicalizes* (a cheap, total normalization: flattening, combining like terms/powers,
folding constants, eliminating identities, ordering operands deterministically). Canonical form is
the sole definition of ``==``/``hash`` -- equal expressions are structurally identical and equality
is order-independent for ``+`` and ``*`` (``x + y == y + x``). Semantic rewrites -- :meth:`expand`,
:meth:`simplify`, :meth:`diff` -- are explicit and never participate in equality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from qilisdk.core.exceptions import NonPolynomialError, NotSupportedOperation
from qilisdk.settings import get_settings
from qilisdk.utils.hashing import hash as qili_hash
from qilisdk.yaml import yaml

from .types import Number, RealNumber

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .variables import BaseVariable, Parameter

_TOL = get_settings().atol


def _float_if_real(value: Number) -> Number:
    """Collapse a complex value with negligible imaginary part to a real value.

    Returns:
        Number: the real value if the imaginary part is negligible, otherwise ``value`` unchanged.
    """
    if isinstance(value, RealNumber):
        return value
    if isinstance(value, complex) and abs(value.imag) < _TOL:
        return value.real
    return value


def _assert_real(value: Number) -> RealNumber:
    """Coerce ``value`` to a real number.

    Returns:
        RealNumber: the real value.

    Raises:
        ValueError: if ``value`` has a non-negligible imaginary part.
    """
    real = _float_if_real(value)
    if isinstance(real, RealNumber):
        return real
    raise ValueError(f"Only real values are allowed but {real} was provided.")


def _finalize(value: Number) -> Number:
    """Normalize a numeric evaluation result.

    Returns:
        Number: a ``float`` for (near-)real values, otherwise the complex value unchanged.
    """
    if isinstance(value, RealNumber):
        return float(value)
    if isinstance(value, complex) and abs(value.imag) < _TOL:
        return float(value.real)
    return value


def _coerce(obj: object) -> Expression | None:
    """Coerce a value into an :class:`Expression`.

    Returns:
        Expression | None: ``obj`` if it is already an ``Expression``, a :class:`Constant` wrapping a
        number, or ``None`` if ``obj`` is neither.
    """
    if isinstance(obj, Expression):
        return obj
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, bool):
        obj = int(obj)
    if isinstance(obj, (int, float, complex)):
        return Constant(obj)
    return None


def _is_int_const(expr: Expression) -> bool:
    return isinstance(expr, Constant) and isinstance(expr.value, RealNumber) and float(expr.value).is_integer()


def _int_exponent(expr: Expression) -> int | None:
    """Return the integer value of a constant integer exponent, or ``None`` if it is not one."""
    if isinstance(expr, Constant) and isinstance(expr.value, RealNumber) and float(expr.value).is_integer():
        return int(expr.value)
    return None


def _is_pos_int_const(expr: Expression) -> bool:
    value = _int_exponent(expr)
    return value is not None and value > 0


def _peel_coeff(expr: Expression) -> tuple[Expression, Number]:
    """Split a monomial into its base and numeric coefficient.

    Returns:
        tuple[Expression, Number]: the monomial with its numeric coefficient stripped, and that coefficient.
    """
    if isinstance(expr, Mul):
        return expr.monomial(), expr.coefficient()
    return expr, 1


def _scale(base: Expression, coeff: Number) -> Expression:
    """Re-attach a numeric coefficient to a monomial.

    Returns:
        Expression: the product ``coeff * base``.
    """
    return Mul.build((Constant(coeff), base))


def _mul_expand(left: Expression, right: Expression) -> Expression:
    """Distribute the product of two (possibly ``Add``) expressions.

    Returns:
        Expression: the distributed product as a canonical sum.
    """
    left_terms = left.args if isinstance(left, Add) else (left,)
    right_terms = right.args if isinstance(right, Add) else (right,)
    return Add.build(tuple(Mul.build((x, y)) for x in left_terms for y in right_terms))


class Expression(ABC):
    """Abstract base of every node in the expression tree."""

    _TOL = _TOL

    # ------------------------------------------------------------------ markers
    @property
    def is_idempotent_under_mul(self) -> bool:
        """Whether ``self * self == self`` (true only for ``BinaryVariable``)."""
        return False

    @property
    def is_parameter(self) -> bool:
        """Whether this expression is a :class:`~qilisdk.core.variables.Parameter` leaf."""
        return False

    # ------------------------------------------------------------------ abstract core
    @abstractmethod
    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> Number:
        """Numerically evaluate the expression given an assignment of symbols to values."""

    @abstractmethod
    def free_symbols(self) -> set[BaseVariable]:
        """The set of named leaves (variables/parameters) occurring in the expression."""

    @property
    @abstractmethod
    def degree(self) -> int:
        """Polynomial degree. Raises :class:`NonPolynomialError` for non-polynomial expressions."""

    @abstractmethod
    def diff(self, symbol: BaseVariable) -> Expression:
        """Symbolic derivative with respect to ``symbol``."""

    @abstractmethod
    def _sort_key(self) -> tuple:
        """A total-order key used to order operands deterministically and define equality."""

    @abstractmethod
    def _compute_hash(self) -> int: ...

    # ------------------------------------------------------------------ shared semantics
    def simplify(self) -> Expression:
        """Return a semantically-equal but possibly simpler expression (opt-in, not used by ``==``)."""
        return self

    def expand(self) -> Expression:
        """Distribute products over sums.

        Returns:
            Expression: a canonical sum-of-monomials equal to this expression.
        """
        return self

    def substitute(self, mapping: Mapping[Expression, Expression | Number]) -> Expression:
        """Structurally replace sub-expressions according to ``mapping``.

        Returns:
            Expression: the expression with substitutions applied.
        """
        if self in mapping:
            replacement = _coerce(mapping[self])
            return replacement if replacement is not None else self
        return self

    def to_binary(self) -> Expression:
        """Encode every continuous ``Variable`` into binary variables (no-op for most nodes).

        Returns:
            Expression: an equivalent expression over binary variables.
        """
        return self

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
        return sorted(self.free_symbols(), key=lambda symbol: symbol.label)

    def is_parameterized(self) -> bool:
        """Whether every named leaf is a :class:`~qilisdk.core.variables.Parameter`.

        Returns:
            bool: ``True`` if the expression contains only parameters (or no variables).
        """
        return all(symbol.is_parameter for symbol in self.free_symbols())

    def get_constant(self) -> Number:  # noqa: PLR6301
        """Return the additive constant of the expression.

        Returns:
            Number: the constant term (0 unless the expression is a sum or a constant).
        """
        return 0

    def as_coefficients_dict(self) -> dict[Expression, Number]:
        """Map each (non-constant) monomial to its numeric coefficient.

        Returns:
            dict[Expression, Number]: a mapping from monomial to coefficient.
        """
        return {self: 1}

    def monomial_factors(self) -> list[tuple[Expression, int]]:
        """Decompose a single monomial into its factors.

        Returns:
            list[tuple[Expression, int]]: the ``(base, integer_power)`` factors.
        """
        return [(self, 1)]

    # ------------------------------------------------------------------ identity
    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = self._compute_hash()
        return self._hash_cache

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expression) and hash(self) == hash(other)

    def __contains__(self, item: object) -> bool:
        """Whether ``item`` is one of the named leaves (variables/parameters) of the expression.

        Returns:
            bool: ``True`` if ``item`` appears as a free symbol in the expression.
        """
        return isinstance(item, Expression) and item in self.free_symbols()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_hash_cache", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._hash_cache = None

    # ------------------------------------------------------------------ arithmetic
    def __add__(self, other: object) -> Expression:
        rhs = _coerce(other)
        return NotImplemented if rhs is None else Add.build((self, rhs))

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other: object) -> Expression:
        rhs = _coerce(other)
        return NotImplemented if rhs is None else Add.build((self, Mul.build((Constant(-1), rhs))))

    def __rsub__(self, other: object) -> Expression:
        lhs = _coerce(other)
        return NotImplemented if lhs is None else Add.build((lhs, Mul.build((Constant(-1), self))))

    __isub__ = __sub__

    def __neg__(self) -> Expression:
        return Mul.build((Constant(-1), self))

    def __mul__(self, other: object) -> Expression:
        rhs = _coerce(other)
        return NotImplemented if rhs is None else Mul.build((self, rhs))

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            if abs(other) < self._TOL:
                raise ValueError("Division by zero is not allowed")
            return Mul.build((Constant(1.0 / other), self))
        rhs = _coerce(other)
        return NotImplemented if rhs is None else Mul.build((self, Pow.build(rhs, Constant(-1))))

    __itruediv__ = __truediv__

    def __rtruediv__(self, other: object) -> Expression:
        lhs = _coerce(other)
        return NotImplemented if lhs is None else Mul.build((lhs, Pow.build(self, Constant(-1))))

    def __floordiv__(self, other: object) -> Expression:
        raise NotSupportedOperation("Floor division is not supported for symbolic expressions")

    __rfloordiv__ = __floordiv__

    def __pow__(self, other: object) -> Expression:
        exp = _coerce(other)
        return NotImplemented if exp is None else Pow.build(self, exp)

    def __rpow__(self, other: object) -> Expression:
        base = _coerce(other)
        return NotImplemented if base is None else Pow.build(base, self)


@yaml.register_class
class Constant(Expression):
    """A numeric literal leaf. Replaces the old ``Term.CONST`` sentinel."""

    def __init__(self, value: Number) -> None:
        raw: Any = value
        if isinstance(raw, np.generic):
            raw = raw.item()
        if isinstance(raw, bool):
            raw = int(raw)
        self.value: Number = _float_if_real(raw)
        self._hash_cache: int | None = None

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> Number:
        return _finalize(self.value)

    def free_symbols(self) -> set[BaseVariable]:  # noqa: PLR6301
        return set()

    @property
    def degree(self) -> int:
        return 0

    def diff(self, symbol: BaseVariable) -> Expression:  # noqa: PLR6301
        return Constant(0)

    def get_constant(self) -> Number:
        return self.value

    def as_coefficients_dict(self) -> dict[Expression, Number]:  # noqa: PLR6301
        return {}

    def monomial_factors(self) -> list[tuple[Expression, int]]:  # noqa: PLR6301
        return []

    def _sort_key(self) -> tuple:
        value = self.value
        if isinstance(value, complex):
            return (0, float(value.real), float(value.imag))
        return (0, float(value), 0.0)

    def _compute_hash(self) -> int:
        return qili_hash("Constant", self.value)

    def __repr__(self) -> str:
        return repr(_float_if_real(self.value))


@yaml.register_class
class Add(Expression):
    """A canonical n-ary sum. Operands are deterministically ordered and combine like terms."""

    def __init__(self, args: tuple[Expression, ...]) -> None:
        # Trusting constructor: ``args`` must already be canonical (flattened, like-terms combined,
        # at most one Constant, length >= 2, deterministically sorted). Use ``_build`` to normalize.
        self.args: tuple[Expression, ...] = tuple(args)
        self._hash_cache: int | None = None

    @classmethod
    def build(cls, raw: tuple[Expression, ...]) -> Expression:
        coefficients: dict[Expression, Number] = {}
        const: Number = 0

        def accumulate(expr: Expression, scale: Number) -> None:
            nonlocal const
            if isinstance(expr, Constant):
                const += scale * expr.value
            elif isinstance(expr, Add):
                for term in expr.args:
                    accumulate(term, scale)
            else:
                base, coeff = _peel_coeff(expr)
                coefficients[base] = coefficients.get(base, 0) + scale * coeff

        for term in raw:
            accumulate(term, 1)

        terms: list[Expression] = []
        for base, raw_coeff in coefficients.items():
            coeff = _float_if_real(raw_coeff)
            if coeff == 0:
                continue
            terms.append(base if coeff == 1 else _scale(base, coeff))
        const = _float_if_real(const)
        if const != 0:
            terms.append(Constant(const))

        if not terms:
            return Constant(0)
        if len(terms) == 1:
            return terms[0]
        terms.sort(key=lambda term: term._sort_key())  # noqa: SLF001
        return cls(tuple(terms))

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> Number:
        env = env if env is not None else {}
        total: Number = 0
        for term in self.args:
            total += term.evaluate(env)
        return _finalize(total)

    def free_symbols(self) -> set[BaseVariable]:
        symbols: set[BaseVariable] = set()
        for term in self.args:
            symbols |= term.free_symbols()
        return symbols

    @property
    def degree(self) -> int:
        return max((term.degree for term in self.args), default=0)

    def diff(self, symbol: BaseVariable) -> Expression:
        return Add.build(tuple(term.diff(symbol) for term in self.args))

    def expand(self) -> Expression:
        return Add.build(tuple(term.expand() for term in self.args))

    def simplify(self) -> Expression:
        return Add.build(tuple(term.simplify() for term in self.args))

    def substitute(self, mapping: Mapping[Expression, Expression | Number]) -> Expression:
        if self in mapping:
            return super().substitute(mapping)
        return Add.build(tuple(term.substitute(mapping) for term in self.args))

    def to_binary(self) -> Expression:
        return Add.build(tuple(term.to_binary() for term in self.args))

    def get_constant(self) -> Number:
        for term in self.args:
            if isinstance(term, Constant):
                return term.value
        return 0

    def as_coefficients_dict(self) -> dict[Expression, Number]:
        coefficients: dict[Expression, Number] = {}
        for term in self.args:
            if isinstance(term, Constant):
                continue
            base, coeff = _peel_coeff(term)
            coefficients[base] = coeff
        return coefficients

    def _sort_key(self) -> tuple:
        return (4, tuple(term._sort_key() for term in self.args))  # noqa: SLF001

    def _compute_hash(self) -> int:
        return qili_hash("Add", self.args)

    def __repr__(self) -> str:
        return " + ".join(repr(term) for term in self.args)


@yaml.register_class
class Mul(Expression):
    """A canonical n-ary product. Collects like powers and folds the numeric coefficient.

    ``Mul`` does **not** distribute over sums; use :meth:`expand` for that.
    """

    def __init__(self, args: tuple[Expression, ...]) -> None:
        # Trusting constructor: ``args`` must already be canonical. Use ``_build`` to normalize.
        self.args: tuple[Expression, ...] = tuple(args)
        self._hash_cache: int | None = None

    @classmethod
    def build(cls, raw: tuple[Expression, ...]) -> Expression:
        coefficient: Number = 1
        powers: dict[Expression, Expression] = {}

        def accumulate(expr: Expression) -> None:
            nonlocal coefficient
            if isinstance(expr, Constant):
                coefficient *= expr.value
            elif isinstance(expr, Mul):
                for factor in expr.args:
                    accumulate(factor)
            else:
                base, exponent = (expr.base, expr.exp) if isinstance(expr, Pow) else (expr, Constant(1))
                powers[base] = exponent if base not in powers else powers[base] + exponent

        for factor in raw:
            accumulate(factor)

        coefficient = _float_if_real(coefficient)
        if coefficient == 0:
            return Constant(0)

        factors: list[Expression] = []
        for base, raw_exponent in powers.items():
            exponent = (
                Constant(1) if (base.is_idempotent_under_mul and _is_pos_int_const(raw_exponent)) else raw_exponent
            )
            if isinstance(exponent, Constant) and exponent.value == 0:
                continue
            factors.append(
                base if (isinstance(exponent, Constant) and exponent.value == 1) else Pow.build(base, exponent)
            )

        out: list[Expression] = [Constant(coefficient)] if coefficient != 1 else []
        out.extend(factors)
        if not out:
            return Constant(1)
        if len(out) == 1:
            return out[0]
        out.sort(key=lambda factor: factor._sort_key())  # noqa: SLF001
        return cls(tuple(out))

    def coefficient(self) -> Number:
        for factor in self.args:
            if isinstance(factor, Constant):
                return factor.value
        return 1

    def monomial(self) -> Expression:
        rest = tuple(factor for factor in self.args if not isinstance(factor, Constant))
        if not rest:
            return Constant(1)
        if len(rest) == 1:
            return rest[0]
        return Mul(rest)

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> Number:
        env = env if env is not None else {}
        total: Number = 1
        for factor in self.args:
            total *= factor.evaluate(env)
        return _finalize(total)

    def free_symbols(self) -> set[BaseVariable]:
        symbols: set[BaseVariable] = set()
        for factor in self.args:
            symbols |= factor.free_symbols()
        return symbols

    @property
    def degree(self) -> int:
        return sum(factor.degree for factor in self.args)

    def diff(self, symbol: BaseVariable) -> Expression:
        terms: list[Expression] = []
        for index in range(len(self.args)):
            factors = (*self.args[:index], self.args[index].diff(symbol), *self.args[index + 1 :])
            terms.append(Mul.build(factors))
        return Add.build(tuple(terms))

    def expand(self) -> Expression:
        result: Expression = Constant(1)
        for factor in self.args:
            result = _mul_expand(result, factor.expand())
        return result

    def simplify(self) -> Expression:
        return Mul.build(tuple(factor.simplify() for factor in self.args))

    def substitute(self, mapping: Mapping[Expression, Expression | Number]) -> Expression:
        if self in mapping:
            return super().substitute(mapping)
        return Mul.build(tuple(factor.substitute(mapping) for factor in self.args))

    def to_binary(self) -> Expression:
        return Mul.build(tuple(factor.to_binary() for factor in self.args))

    def as_coefficients_dict(self) -> dict[Expression, Number]:
        return {self.monomial(): self.coefficient()}

    def monomial_factors(self) -> list[tuple[Expression, int]]:
        factors: list[tuple[Expression, int]] = []
        for factor in self.args:
            if isinstance(factor, Constant):
                continue
            factors.extend(factor.monomial_factors())
        return factors

    def _sort_key(self) -> tuple:
        return (3, tuple(factor._sort_key() for factor in self.args))  # noqa: SLF001

    def _compute_hash(self) -> int:
        return qili_hash("Mul", self.args)

    def __repr__(self) -> str:
        parts = []
        for factor in self.args:
            parts.append(f"({factor!r})" if isinstance(factor, Add) else repr(factor))
        return " * ".join(parts)


@yaml.register_class
class Pow(Expression):
    """A power ``base ** exp``. The exponent may be a numeric or symbolic :class:`Expression`."""

    def __init__(self, base: Expression, exp: Expression) -> None:
        # Trusting constructor: use ``_build`` to normalize.
        self.base: Expression = base
        self.exp: Expression = exp
        self._hash_cache: int | None = None

    @classmethod
    def build(cls, base: Expression, exp: Expression) -> Expression:
        if isinstance(exp, Constant):
            if exp.value == 1:
                return base
            if exp.value == 0:
                return Constant(1)
            if isinstance(base, Constant):
                return Constant(base.value**exp.value)
            if isinstance(base, Pow):
                inner = _int_exponent(base.exp)
                outer = _int_exponent(exp)
                if inner is not None and outer is not None:
                    return cls.build(base.base, Constant(inner * outer))
            if base.is_idempotent_under_mul and _is_pos_int_const(exp):
                return base
        if isinstance(base, Constant) and base.value == 1:
            return Constant(1)
        return cls(base, exp)

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> Number:
        env = env if env is not None else {}
        return _finalize(self.base.evaluate(env) ** self.exp.evaluate(env))

    def free_symbols(self) -> set[BaseVariable]:
        return self.base.free_symbols() | self.exp.free_symbols()

    @property
    def degree(self) -> int:
        exponent = _int_exponent(self.exp)
        if exponent is not None and exponent >= 0:
            return self.base.degree * exponent
        raise NonPolynomialError(f"Expression {self!r} is not a polynomial; its degree is undefined.")

    def diff(self, symbol: BaseVariable) -> Expression:
        base, exp = self.base, self.exp
        if symbol not in exp.free_symbols():
            # d/dx b**c = c * b**(c-1) * b'
            return Mul.build((exp, Pow.build(base, Add.build((exp, Constant(-1)))), base.diff(symbol)))
        # general case: b**e * (e' * ln(b) + e * b'/b)
        chain = Add.build(
            (
                Mul.build((exp.diff(symbol), Log(base))),
                Mul.build((exp, base.diff(symbol), Pow.build(base, Constant(-1)))),
            )
        )
        return Mul.build((self, chain))

    def expand(self) -> Expression:
        base = self.base.expand()
        exponent = _int_exponent(self.exp)
        if exponent is not None and exponent > 0:
            result = base
            for _ in range(exponent - 1):
                result = _mul_expand(result, base)
            return result
        return Pow.build(base, self.exp)

    def simplify(self) -> Expression:
        return Pow.build(self.base.simplify(), self.exp.simplify())

    def substitute(self, mapping: Mapping[Expression, Expression | Number]) -> Expression:
        if self in mapping:
            return super().substitute(mapping)
        return Pow.build(self.base.substitute(mapping), self.exp.substitute(mapping))

    def to_binary(self) -> Expression:
        return Pow.build(self.base.to_binary(), self.exp.to_binary())

    def monomial_factors(self) -> list[tuple[Expression, int]]:
        exponent = _int_exponent(self.exp)
        if exponent is not None and exponent > 0:
            return [(self.base, exponent)]
        raise NonPolynomialError(f"Expression {self!r} is not a monomial with an integer power.")

    def _sort_key(self) -> tuple:
        return (2, self.base._sort_key(), self.exp._sort_key())  # noqa: SLF001

    def _compute_hash(self) -> int:
        return qili_hash("Pow", self.base, self.exp)

    def __repr__(self) -> str:
        base = f"({self.base!r})" if isinstance(self.base, (Add, Mul)) else repr(self.base)
        exp = f"({self.exp!r})" if isinstance(self.exp, (Add, Mul)) else repr(self.exp)
        return f"{base}**{exp}"


class Function(Expression, ABC):
    """Abstract base for unary maths functions (``sin``, ``cos``, ``exp``, ...).

    A concrete function declares -- all at the *class* level so serialization carries only the
    operand -- a stable ``NAME``, a numpy numeric kernel :meth:`_numeric`, and the local outer
    derivative :meth:`_derivative` (the chain rule is applied by :meth:`diff`).
    """

    NAME: ClassVar[str] = ""
    _REGISTRY: ClassVar[dict[str, type[Function]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "NAME", ""):
            Function._REGISTRY[cls.NAME] = cls

    def __new__(cls, arg: object) -> Expression:
        operand = _coerce(arg)
        if operand is None:
            raise TypeError(f"{cls.__name__} expects an Expression or number, got {type(arg).__name__}")
        if isinstance(operand, Constant):
            return Constant(cls._numeric(_assert_real(operand.value)))
        return super().__new__(cls)

    def __init__(self, arg: object) -> None:
        operand = _coerce(arg)
        self.arg: Expression = operand  # ty:ignore[invalid-assignment]
        self._hash_cache: int | None = None

    @staticmethod
    @abstractmethod
    def _numeric(value: RealNumber) -> Number:
        """Numeric kernel applied to an already-evaluated, real operand."""

    @abstractmethod
    def _derivative(self, operand: Expression) -> Expression:
        """The outer derivative ``f'(operand)`` as an expression (chain rule applied by :meth:`diff`)."""

    def evaluate(self, env: Mapping[BaseVariable, Number | list[int]] | None = None) -> Number:
        env = env if env is not None else {}
        return _finalize(self._numeric(_assert_real(self.arg.evaluate(env))))

    def free_symbols(self) -> set[BaseVariable]:
        return self.arg.free_symbols()

    @property
    def degree(self) -> int:
        if self.arg.free_symbols():
            raise NonPolynomialError(f"Expression {self!r} is not a polynomial; its degree is undefined.")
        return 0

    def diff(self, symbol: BaseVariable) -> Expression:
        return Mul.build((self._derivative(self.arg), self.arg.diff(symbol)))

    def expand(self) -> Expression:
        return type(self)(self.arg.expand())

    def simplify(self) -> Expression:
        return type(self)(self.arg.simplify())

    def substitute(self, mapping: Mapping[Expression, Expression | Number]) -> Expression:
        if self in mapping:
            return super().substitute(mapping)
        return type(self)(self.arg.substitute(mapping))

    def to_binary(self) -> Expression:
        return type(self)(self.arg.to_binary())

    def _sort_key(self) -> tuple:
        return (5, self.NAME, self.arg._sort_key())  # noqa: SLF001

    def _compute_hash(self) -> int:
        return qili_hash(self.NAME, self.arg)

    def __copy__(self) -> Expression:
        return type(self)(self.arg)

    def __repr__(self) -> str:
        return f"{self.NAME}({self.arg!r})"

    @classmethod
    def to_yaml(cls, representer, node):  # noqa: ANN001, ANN206
        return representer.represent_mapping(cls.yaml_tag, {"arg": node.arg})  # ty:ignore[unresolved-attribute]

    @classmethod
    def from_yaml(cls, constructor, node):  # noqa: ANN001, ANN206
        mapping = constructor.construct_mapping(node, deep=True)
        return cls(mapping["arg"])


@yaml.register_class
class Sin(Function):
    """Sine of an expression."""

    NAME = "sin"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.sin(value))

    def _derivative(self, operand: Expression) -> Expression:  # noqa: PLR6301
        return Cos(operand)


@yaml.register_class
class Cos(Function):
    """Cosine of an expression."""

    NAME = "cos"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.cos(value))

    def _derivative(self, operand: Expression) -> Expression:  # noqa: PLR6301
        return -Sin(operand)


@yaml.register_class
class Exp(Function):
    """Exponential of an expression."""

    NAME = "exp"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.exp(value))

    def _derivative(self, operand: Expression) -> Expression:  # noqa: PLR6301
        return Exp(operand)


@yaml.register_class
class Log(Function):
    """Natural logarithm of an expression."""

    NAME = "log"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.log(value))

    def _derivative(self, operand: Expression) -> Expression:  # noqa: PLR6301
        return Pow.build(operand, Constant(-1))


@yaml.register_class
class Tan(Function):
    """Tangent of an expression."""

    NAME = "tan"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.tan(value))

    def _derivative(self, operand: Expression) -> Expression:  # noqa: PLR6301
        return Pow.build(Cos(operand), Constant(-2))


@yaml.register_class
class Sqrt(Function):
    """Square root of an expression."""

    NAME = "sqrt"

    @staticmethod
    def _numeric(value: RealNumber) -> Number:
        return float(np.sqrt(value))

    def _derivative(self, operand: Expression) -> Expression:  # noqa: PLR6301
        return Constant(0.5) * Pow.build(operand, Constant(-0.5))
