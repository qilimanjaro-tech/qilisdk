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

from __future__ import annotations

import functools
from typing import Any, Callable, ParamSpec, Self, TypeVar
from uuid import UUID, uuid4

from qilisdk.core.types import QiliEnum
from qilisdk.yaml import yaml

P = ParamSpec("P")
R = TypeVar("R")


@yaml.register_class
class QProgramDomain(QiliEnum):
    """QProgramDomain class."""

    Scalar = "Scalar"
    Time = "Time"
    Frequency = "Frequency"
    Phase = "Phase"
    Voltage = "Voltage"
    Flux = "Flux"


def requires_domain(parameter: str, domain: QProgramDomain) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to denote that a parameter requires a variable of a specific domain.

    Args:
        parameter (str): The parameter name.
        domain (QProgramDomain): The variable's domain.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: Decorator that enforces the variable's domain.
    """

    def decorator_function(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            mutable_kwargs: dict[str, Any] = dict(kwargs)
            # Check if the parameter is not inside kwargs, for optional parameters
            if parameter not in mutable_kwargs and len(args) == 1:
                mutable_kwargs[parameter] = None
            # Get the argument by name
            param_value = mutable_kwargs.get(parameter) if parameter in mutable_kwargs else None

            if isinstance(param_value, QProgramVariable) and param_value.domain != domain:
                raise ValueError(f"Expected domain {domain} for {parameter}, but got {param_value.domain}")
            return func(*args, **mutable_kwargs)

        return wrapper

    return decorator_function


@yaml.register_class
class QProgramVariable:
    """Variable class used to define variables inside a QProgram."""

    def __init__(self, label: str, domain: QProgramDomain = QProgramDomain.Scalar) -> None:
        self._uuid: UUID = uuid4()
        self._label: str = label
        self._domain: QProgramDomain = domain

    def __repr__(self) -> str:
        return f"QProgramVariable(uuid={self.uuid!r}, label={self.label}, domain={self.domain})"

    def __hash__(self) -> int:
        return hash(self._uuid)

    def __eq__(self, other: object) -> bool:
        return other is not None and isinstance(other, QProgramVariable) and self._uuid == other._uuid

    def __add__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(self, "+", other)

    def __radd__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(other, "+", self)

    def __sub__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(self, "-", other)

    def __rsub__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(other, "-", self)

    @property
    def uuid(self) -> UUID:
        """Get the uuid of the variable

        Returns:
            UUID: The uuid of the variable
        """
        return self._uuid

    @property
    def label(self) -> str:
        """Get the label of the variable

        Returns:
            str: The label of the variable
        """
        return self._label

    @property
    def domain(self) -> QProgramDomain:
        """Get the domain of the variable

        Returns:
            QProgramDomain: The domain of the variable
        """
        return self._domain


@yaml.register_class
class IntVariable(QProgramVariable, int):
    """Integer variable. This class is used to define a variable of type int, such that Python recognizes this class
    as an integer."""

    def __new__(cls, _: str = "", __: QProgramDomain = QProgramDomain.Scalar) -> Self:
        # Create a new float instance
        instance = int.__new__(cls, 0)
        return instance

    def __init__(self, label: str = "", domain: QProgramDomain = QProgramDomain.Scalar) -> None:
        QProgramVariable.__init__(self, label, domain)


@yaml.register_class
class FloatVariable(QProgramVariable, float):
    """Float variable. This class is used to define a variable of type float, such that Python recognizes this class
    as a float."""

    def __new__(cls, _: str = "", __: QProgramDomain = QProgramDomain.Scalar) -> Self:
        # Create a new int instance
        instance = float.__new__(cls, 0.0)
        return instance

    def __init__(self, label: str = "", domain: QProgramDomain = QProgramDomain.Scalar) -> None:
        QProgramVariable.__init__(self, label, domain)


@yaml.register_class
class VariableExpression(QProgramVariable):
    """An expression combining Variables and/or constants."""

    def __init__(self, left: QProgramVariable | int, operator: str, right: QProgramVariable | int) -> None:
        self.left: QProgramVariable | int = left
        self.operator: str = operator
        self.right: QProgramVariable | int = right
        domain = self._infer_domain(left, right)
        if domain != QProgramDomain.Time:
            raise NotImplementedError("Variable Expressions are only supported for QProgramDomain.Time.")
        self._domain = domain
        super().__init__(label="", domain=self._domain)

    @staticmethod
    def _infer_domain(left: QProgramVariable | int, right: QProgramVariable | int) -> QProgramDomain:
        if isinstance(left, QProgramVariable):
            return left.domain
        if isinstance(right, QProgramVariable):
            return right.domain
        raise ValueError("Cannot infer domain from constants.")

    def __repr__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

    def __add__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(self, "+", other)

    def __radd__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(other, "+", self)

    def __sub__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(self, "-", other)

    def __rsub__(self, other: QProgramVariable | int) -> VariableExpression:
        return VariableExpression(other, "-", self)

    def extract_variables(self) -> QProgramVariable:
        """Recursively extract all Variable instances used in this expression.

        Returns:
            Variable: A Variable instance found in the expression.

        Raises:
            ValueError: If no Variable instance is found in the expression.
        """

        def _collect(expr: QProgramVariable | int) -> QProgramVariable | None:
            if isinstance(expr, VariableExpression):
                result = _collect(expr.left)
                if result is not None:
                    return result
                return _collect(expr.right)
            if isinstance(expr, QProgramVariable):
                return expr
            return None

        result = _collect(self)
        if result is None:
            raise ValueError(f"No Variable instance found in expression: {self}")
        return result

    def extract_constants(self) -> int:
        """Recursively extract all constants used in this expression.

        Returns:
            int: A constant integer found in the expression.

        Raises:
            ValueError: If no constant is found in the expression.
        """

        def _collect(expr: QProgramVariable | int) -> int | None:
            if isinstance(expr, VariableExpression):
                result = _collect(expr.left)
                if result is not None:
                    return result
                return _collect(expr.right)
            if isinstance(expr, int) and not isinstance(expr, IntVariable):
                return expr
            return None

        result = _collect(self)
        if result is None:
            raise ValueError(f"No Variable instance found in expression: {self}")
        return result
