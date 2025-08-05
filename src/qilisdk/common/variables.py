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

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence, TypeVar

import numpy as np
from loguru import logger

from qilisdk.common.exceptions import EvaluationError, InvalidBoundsError, NotSupportedOperation, OutOfBoundsException
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from ruamel.yaml.nodes import ScalarNode
    from ruamel.yaml.representer import RoundTripRepresenter

Number = int | float
GenericVar = TypeVar("GenericVar", bound="Variable")
CONST_KEY = "_const_"
MAX_INT = np.iinfo(np.int64).max
MIN_INT = np.iinfo(np.int64).min
LARGE_BOUND = 100


def LT(lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term) -> ComparisonTerm:
    """'Less Than' mathematical operation

    Args:
        lhs (Number | BaseVariable | Term): the left hand side of the comparison term.
        rhs (Number | BaseVariable | Term): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs < rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.LT)


LessThan = LT


def LEQ(lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term) -> ComparisonTerm:
    """'Less Than or equal to' mathematical operation

    Args:
        lhs (Number | BaseVariable | Term): the left hand side of the comparison term.
        rhs (Number | BaseVariable | Term): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs <= rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.LEQ)


LessThanOrEqual = LEQ


def EQ(lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term) -> ComparisonTerm:
    """'Equal to' mathematical operation

    Args:
        lhs (Number | BaseVariable | Term): the left hand side of the comparison term.
        rhs (Number | BaseVariable | Term): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs == rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.EQ)


Equal = EQ


def NEQ(lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term) -> ComparisonTerm:
    """'Not Equal to' mathematical operation

    Args:
        lhs (Number | BaseVariable | Term): the left hand side of the comparison term.
        rhs (Number | BaseVariable | Term): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs != rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.NEQ)


NotEqual = NEQ


def GT(lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term) -> ComparisonTerm:
    """'Greater Than' mathematical operation

    Args:
        lhs (Number | BaseVariable | Term): the left hand side of the comparison term.
        rhs (Number | BaseVariable | Term): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs > rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.GT)


GreaterThan = GT


def GEQ(lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term) -> ComparisonTerm:
    """'Greater Than or equal to' mathematical operation

    Args:
        lhs (Number | BaseVariable | Term): the left hand side of the comparison term.
        rhs (Number | BaseVariable | Term): the right hand side of the comparison term.

    Returns:
        ComparisonTerm: a comparison term with the structure lhs >= rhs.
    """
    return ComparisonTerm(lhs=lhs, rhs=rhs, operation=ComparisonOperation.GEQ)


GreaterThanOrEqual = GEQ


def _extract_number(label: str) -> int:
    """Extracts the number from the variable's label.

    Args:
        label (str): variable label that follows the format <label>(<number>).

    Returns:
        int: the number in the label.
    """
    pattern = re.compile(r"\((\d+)\)$")
    matches = pattern.search(label)
    if matches is not None:
        return int(matches.group(1))
    return 0


@yaml.register_class
class Domain(str, Enum):
    INTEGER = "Integer Domain"
    POSITIVE_INTEGER = "Positive Integer Domain"
    REAL = "Real Domain"
    BINARY = "Binary Domain"
    SPIN = "Spin Domain"

    def check_value(self, value: Number) -> bool:
        """checks if the provided value is valid for a given domain

        Args:
            value (int | float): the value to be evaluated.

        Returns:
            bool: True if the value provided is valid, False otherwise.
        """
        if self == Domain.BINARY:
            return isinstance(value, int) and value in {0, 1}
        if self == Domain.SPIN:
            return isinstance(value, int) and value in {-1, 1}
        if self == Domain.REAL:
            return isinstance(value, (int, float))
        if self == Domain.INTEGER:
            return isinstance(value, int)
        if self == Domain.POSITIVE_INTEGER:
            return isinstance(value, int) and value >= 0
        return False

    def min(self) -> float:
        """
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
        """
        Returns:
            float: the maximum value allowed for a given domain.
        """
        if self in {Domain.BINARY, Domain.SPIN}:
            return 1
        if self in {Domain.POSITIVE_INTEGER, Domain.INTEGER}:
            return MAX_INT
        return 1e30

    @classmethod
    def to_yaml(cls, representer: RoundTripRepresenter, node: Domain) -> ScalarNode:
        """
        Method to be called automatically during YAML serialization.

        Returns:
            ScalarNode: The YAML scalar node representing the Domain.
        """
        return representer.represent_scalar("!Domain", f"{node.value}")

    @classmethod
    def from_yaml(cls, _, node: ScalarNode) -> Domain:
        """
        Method to be called automatically during YAML deserialization.

        Returns:
            Domain: The Domain instance created from the YAML node value.
        """
        return cls(node.value)


@yaml.register_class
class Operation(str, Enum):
    MUL = "*"
    ADD = "+"
    DIV = "/"
    SUB = "-"

    @classmethod
    def to_yaml(cls, representer: RoundTripRepresenter, node: Operation) -> ScalarNode:
        """
        Method to be called automatically during YAML serialization.

        Returns:
            ScalarNode: The YAML scalar node representing the Operation.
        """
        return representer.represent_scalar("!Operation", f"{node.value}")

    @classmethod
    def from_yaml(cls, _, node: ScalarNode) -> Operation:
        """
        Method to be called automatically during YAML deserialization.

        Returns:
            Operation: The Operation instance created from the YAML node value.
        """
        return cls(node.value)


@yaml.register_class
class ComparisonOperation(str, Enum):
    LT = "<"
    LEQ = "<="
    EQ = "=="
    NEQ = "!="
    GT = ">"
    GEQ = ">="

    @classmethod
    def to_yaml(cls, representer: RoundTripRepresenter, node: ComparisonOperation) -> ScalarNode:
        """
        Method to be called automatically during YAML serialization.

        Returns:
            ScalarNode: The YAML scalar node representing the ComparisonOperation.
        """
        return representer.represent_scalar("!ComparisonOperation", f"{node.value}")

    @classmethod
    def from_yaml(cls, _, node: ScalarNode) -> ComparisonOperation:
        """
        Method to be called automatically during YAML deserialization.

        Returns:
            ComparisonOperation: The ComparisonOperation instance created from the YAML node value.
        """
        return cls(node.value)


@yaml.register_class
class Encoding(ABC):
    """Represents an abstract variable encoding class.

    The Encoding is defined on the variable bases, and it defines how the continuous variables are encoded into binary
    variables.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Encoding's name

        Returns:
            str: The name of the encoding.
        """

    @staticmethod
    @abstractmethod
    def encode(var: Variable, precision: float = 1e-2) -> Term:
        """Given a continuous variable return a Term that only consists of
            binary variables that represent the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): The continuous variable to be encoded
            precision (int): the precision to be considered for real variables (Only applies if
                                the variable domain is Domain.Real)

        Returns:
            Term: a term that only contains binary variables
        """

    @staticmethod
    @abstractmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        """Given a continuous variable return a Constraint Term that ensures that the encoding is respected.

        Args:
            var (ContinuousVar): The continuous variable to be encoded
            precision (float): the precision to be considered for real variables (Only applies if
                                the variable domain is Domain.Real)

        Returns:
            Constraint Term: a constraint term that ensures the encoding is respected.
        """

    @staticmethod
    @abstractmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            value (list[int] | int): a list of binary values or an integer value.
            precision (float): the precision to be considered for real variables (Only applies if
                                the variable domain is Domain.Real)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """

    @staticmethod
    @abstractmethod
    def num_binary_equivalent(var: "Variable", precision: float = 1e-2) -> int:
        """Give a continuous variable return the number of binary variables needed to encode it.

        Args:
            var (ContinuousVar): the continuous variable.
            precision (float): the precision to be considered for real variables (Only applies if
                                the variable domain is Domain.Real)

        Returns:
            int: the number of binary variables needed to encode it.
        """

    @staticmethod
    @abstractmethod
    def check_valid(value: list[int] | int) -> tuple[bool, int]:
        """checks if the binary list sample is a valid sample in this encoding.

        Args:
            value (list[int] | int):  a list of binary values or an integer value.

        Returns:
            tuple[bool, int]: the boolean is True if the sample is a valid encoding,
                                while the int is the error in the encoding.
        """


@yaml.register_class
class Bitwise(Encoding):
    """Represents a Bitwise variable encoding class."""

    name = "Bitwise"

    @staticmethod
    def _bitwise_encode(x: int, N: int) -> list[int]:
        """encode the integer x in Bitwise encoding.

        Args:
            x (int): the integer to be encoded.
            N (int): the number of bits to encode x.

        Returns:
            list[int]: a binary list representing the Bitwise encoding of the integer x.
        """
        return list(reversed([int(b) for b in format(x, f"0{N}b")]))

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Term:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        abs_bound = np.abs(bounds[1] - bounds[0])
        n_binary = int(np.floor(np.log2(abs_bound if abs_bound != 0 else 1)))
        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary + 1)]

        term = sum(2**i * binary_vars[i] for i in range(n_binary))
        term += (np.abs(bounds[1] - bounds[0]) + 1 - 2**n_binary) * binary_vars[-1]
        term += bounds[0]
        return term * var.precision if var.domain is Domain.REAL else term

    @staticmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        term = Bitwise.encode(var, precision)
        binary_var = sorted(
            term.variables(),
            key=lambda x: _extract_number(x.label),
        )

        binary_list = Bitwise._bitwise_encode(value, len(binary_var)) if isinstance(value, Number) else value

        if not Bitwise.check_valid(binary_list)[0]:
            raise ValueError(
                f"invalid binary string {binary_list} with the Bitwise encoding."
            )  # can not be reached in the case of Bitwise encoding.

        if len(binary_list) < len(binary_var):
            for _ in range(len(binary_var) - len(binary_list)):
                binary_list.append(0)
        elif len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict: dict[BaseVariable, list[int]] = {binary_var[i]: [binary_list[i]] for i in range(len(binary_list))}

        out = term.evaluate(binary_dict)

        out = int(out) if var.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER} else out

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )  # not sure this line can be reached.
        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        raise NotImplementedError("Bitwise encoding constraints are not supported at the moment")

    @staticmethod
    def num_binary_equivalent(var: "Variable", precision: float = 1e-2) -> int:
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
    """Represents a One-Hot variable encoding class."""

    name = "One-Hot"

    @staticmethod
    def _one_hot_encode(x: int, N: int) -> list[int]:
        """One-hot encode integer x in range [0, N-1].

        Args:
            x (int): the value to be encoded
            N (int): the number of bits to encode x.

        Raises:
            ValueError: if x is larger than N or less than 0

        Returns:
            list[int]: a binary list representing the one hot encoding of the integer x.
        """
        if not (0 <= x < N):
            raise ValueError(f"the input value ({x}) must be in range [0, {N - 1}]")
        return [1 if i == x else 0 for i in range(N)]

    @staticmethod
    def _find_zero(var: Variable) -> int:
        binary_var = var.bin_vars
        term = var.term
        for i in range(var.num_binary_equivalent()):
            if binary_var[i] not in term:
                return i
        return 0

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Term:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]

        term = Term([(bounds[0] + i) * binary_vars[i] for i in range(n_binary)], Operation.ADD)

        return term * var.precision if var.domain is Domain.REAL else term

    @staticmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        term = OneHot.encode(var, precision)
        binary_var = sorted(
            term.variables(),
            key=lambda x: _extract_number(x.label),
        )

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

        out = term.evaluate(binary_dict)

        out = int(out) if var.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER} else out

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )  # not sure this line can be reached.

        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]
        return ComparisonTerm(lhs=sum(binary_vars), rhs=1, operation=ComparisonOperation.EQ)

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
    """Represents a Domain-wall variable encoding class."""

    name = "Domain Wall"

    @staticmethod
    def _domain_wall_encode(x: int, N: int) -> list[int]:
        """domain wall encode integer x in range [0, N-1].

        Args:
            x (int): the value to be encoded
            N (int): the number of bits to encode x.

        Raises:
            ValueError: if x is larger than N or less than 0

        Returns:
            list[int]: a binary list representing the domain wall encoding of the integer x.
        """
        if not (0 <= x <= N):
            raise ValueError(f"the input value ({x}) must be in range [0, {N}]")
        return [1] * x + [0] * (N - x)

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Term:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]

        term = Term([0], Operation.ADD)
        for i in range(n_binary):
            term += binary_vars[i]

        term += bounds[0]

        return term * var.precision if var.domain is Domain.REAL else term

    @staticmethod
    def evaluate(var: Variable, value: list[int] | int, precision: float = 1e-2) -> float:
        term = DomainWall.encode(var, precision)
        binary_var = term.variables()
        binary_var = sorted(
            term.variables(),
            key=lambda x: _extract_number(x.label),
        )

        binary_list = DomainWall._domain_wall_encode(value, len(binary_var)) if isinstance(value, int) else value

        if not DomainWall.check_valid(binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the domain wall encoding.")

        if len(binary_list) < len(binary_var):
            for _ in range(len(binary_var) - len(binary_list)):
                binary_list.append(0)
        elif len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict: dict[BaseVariable, list[int]] = {binary_var[i]: [binary_list[i]] for i in range(len(binary_list))}

        out = term.evaluate(binary_dict)

        out = int(out) if var.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER} else out

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )  # not sure if this line is reachable.
        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        binary_vars = [BinaryVariable(var.label + f"({i})") for i in range(n_binary)]
        return ComparisonTerm(
            lhs=sum(binary_vars[i + 1] * (1 - binary_vars[i]) for i in range(len(binary_vars) - 1)),
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


class BaseVariable(ABC):
    """Represents an abstract structure of any variable that can be included in the optimization ``Model``
    (It's the ``Model``'s minimum mathematical expression).

    ``Variable``'s are used to construct ``Term``s that can then be used to build ``Model``s.
    """

    def __init__(self, label: str, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        """initialize a new Variable object

        Args:
            label (str): The name of the variable.
            domain (Domain): The domain of the values this variable can take.
            bounds (tuple[float  |  None, float  |  None], optional): the bounds on the variable's values.
                                                The bounds follow the structure (lower_bound, Upper_bound) both
                                                included. Defaults to (None, None).
                                                Note: if None is selected then the lowest/highest possible value of the
                                                variable's domain is chosen.

        Raises:
            OutOfBoundsException: the lower bound or the upper bound don't correspond to the variable domain.
            InvalidBoundsError: the lower bound is higher than the upper bound.
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
                f"the lower bound ({upper_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if not self.domain.check_value(lower_bound):
            raise OutOfBoundsException(
                f"the upper bound ({lower_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if lower_bound > upper_bound:
            raise InvalidBoundsError("lower bound can't be larger than the upper bound.")
        self._bounds = (lower_bound, upper_bound)

    @property
    def bounds(self) -> tuple[float, float]:
        """Property that stores a tuple representing the bounds of the values a variable is allowed to take.º

        Returns:
            tuple(float, float): The lower and upper bound of the variable.
        """
        return self._bounds

    @property
    def lower_bound(self) -> float:
        """The lower bound of the variable.

        Returns:
            float: the value of the lower bound.
        """
        return self._bounds[0]

    @property
    def upper_bound(self) -> float:
        """The upper bound of the variable.

        Returns:
            float: the value of the upper bound.
        """
        return self._bounds[1]

    @property
    def label(self) -> str:
        """the label (name) of the variable.

        Returns:
            string: the name of the variable.
        """
        return self._label

    @property
    def domain(self) -> Domain:
        """The domain of values that the variable is allowed to take.

        Returns:
            Domain: The domain of the values the variable can take.
        """
        return self._domain

    def set_bounds(self, lower_bound: float | None, upper_bound: float | None) -> None:
        """set the bounds of the variable.

        Args:
            lower_bound (float | None): The lower bound (if None the lowest allowed bound in the variable domain is
            selected). Defaults to None.
            upper_bound (float | None): The upper bound (if None the highest allowed bound in the variable domain is
            selected). Defaults to None.
        Raises:
            OutOfBoundsException: the lower bound or the upper bound don't correspond to the variable domain.
            InvalidBoundsError: the lower bound is higher than the upper bound.
        """
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
        """Returns:
        int: the number of binary variables that are needed to represent this variable in the given encoding.
        """

    @abstractmethod
    def evaluate(self, value: list[int] | Number) -> float:
        """Evaluates the value of the variable given a binary string or a number.

        Args:
            value (list[int] | int | float): the value used to evaluate the variable.
                If the value provided is binary list (list[int]) then the value of the variable is evaluated based on
                its binary representation. This representation is constructed using the encoding, bounds and domain
                of the variable. To check the binary representation of a variable you can check the method `to_binary()`

        Returns:
            float: the evaluated vale of the variable.
        """

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None]) -> None:
        """Replaces the information of the variable with those coming from the dictionary
        if the variable label is in the dictionary

        Args:
            var_dict (dict): A dictionary that holds the labels of the variables to be
                            changed alongside the new values they should take
        """

        self._domain = domain
        self.set_bounds(bounds[0], bounds[1])

    @abstractmethod
    def to_binary(self) -> Term:
        """Returns the binary representation of a variable.º

        Returns:
            Term: the binary representation of a variable.
        """

    def __repr__(self) -> str:
        return f"{self._label}"

    def __str__(self) -> str:
        return f"{self._label}"

    def __add__(self, other: Number | BaseVariable | Term) -> Term:
        if isinstance(other, Term):
            return other + self

        return Term(elements=[self, other], operation=Operation.ADD)

    __radd__ = __add__
    __iadd__ = __add__

    def __mul__(self, other: Number | BaseVariable | Term) -> Term:
        if isinstance(other, Term):
            return other * self

        return Term(elements=[self, other], operation=Operation.MUL)

    def __rmul__(self, other: Number | BaseVariable | Term) -> Term:
        if isinstance(other, Term):
            return other * self

        return Term(elements=[other, self], operation=Operation.MUL)

    __imul__ = __mul__

    def __sub__(self, other: Number | BaseVariable | Term) -> Term:
        return self + -1 * other

    def __rsub__(self, other: Number | BaseVariable | Term) -> Term:
        return -1 * self + other

    __isub__ = __sub__

    def __neg__(self) -> Term:
        return -1 * self

    def __truediv__(self, other: Number) -> Term:
        if not isinstance(other, Number):
            raise NotImplementedError("Only division by numbers is currently supported")

        if other == 0:
            raise ValueError("Division by zero is not allowed")

        other = 1 / other
        return self * other

    __itruediv__ = __truediv__

    def __rtruediv__(self, other: Number | BaseVariable | Term) -> Term:
        raise NotSupportedOperation("Only division by numbers is currently supported")

    def __rfloordiv__(self, other: Number | BaseVariable | Term) -> Term:
        raise NotSupportedOperation("Only division by numbers is currently supported")

    def __pow__(self, a: int) -> Term:
        out: BaseVariable | Term = copy.copy(self)

        if a < 0:
            raise NotImplementedError("Negative Power is not Supported.")

        if a == 0:
            return Term(elements=[1], operation=Operation.ADD)

        for _ in range(a - 1):
            out *= copy.copy(self)

        if isinstance(out, BaseVariable):
            out = Term(elements=[out], operation=Operation.ADD)
        return out

    def __hash__(self) -> int:
        return hash((self._label, self._domain.value, self._bounds))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseVariable):
            return False
        return hash(self) == hash(other)


@yaml.register_class
class BinaryVariable(BaseVariable):
    """Represents Binary Variable structure."""

    def __init__(self, label: str) -> None:
        super().__init__(label=label, domain=Domain.BINARY)

    def num_binary_equivalent(self) -> int:  # noqa: PLR6301
        return 1

    def evaluate(self, value: list[int] | Number) -> float:
        if isinstance(value, Number):
            if value in {1.0, 0.0}:
                return int(value)
            if not self.domain.check_value(value):
                raise EvaluationError(f"Evaluating a Binary variable with a value {value} that is outside the domain.")
            return value  # I don't think this line is reachable
        if len(value) != 1:
            raise EvaluationError("Evaluating a Binary variable with a binary list of more than one item.")
        return value[0]

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None]) -> None:
        raise NotImplementedError

    def to_binary(self) -> Term:
        return Term([self], Operation.ADD)

    def __copy__(self) -> BinaryVariable:
        return BinaryVariable(label=self.label)


@yaml.register_class
class SpinVariable(BaseVariable):
    """Represents Spin Variable structure."""

    def __init__(self, label: str) -> None:
        super().__init__(label=label, domain=Domain.SPIN, bounds=(-1, 1))

    def num_binary_equivalent(self) -> int:  # noqa: PLR6301
        return 1

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None]) -> None:
        raise NotImplementedError

    def evaluate(self, value: list[int] | Number) -> float:
        if isinstance(value, Number):
            if not self.domain.check_value(value) and value != 0:
                raise EvaluationError(f"Evaluating a Spin variable with a value {value} that is outside the domain.")
            return -1 if value in {0, -1} else 1
        if len(value) != 1:
            raise EvaluationError("Evaluating a Spin variable with a list of more than one item.")
        return -1 if value[0] in {0, -1} else 1

    def to_binary(self) -> Term:
        return Term([self], Operation.ADD)

    def __copy__(self) -> SpinVariable:
        return SpinVariable(label=self.label)


@yaml.register_class
class Variable(BaseVariable):
    """Represents General Variable structure (Continuous, binary, or Spin).
    Note: For Binary or Spin variables it's recommended to use the BinaryVar and SpinVar objects."""

    def __init__(
        self,
        label: str,
        domain: Domain,
        bounds: tuple[float | None, float | None] = (None, None),
        encoding: type[Encoding] = Bitwise,
        precision: float = 1e-2,
    ) -> None:
        """_summary_

        Args:
            label (str): The name of the variable.
            domain (Domain): The domain of the values this variable can take.
            bounds (tuple[float  |  None, float  |  None], optional): the bounds on the values of the variable The bounds
                    have the structure (lower_bound, Upper_bound) both values included. Defaults to (None, None).
                    Note: if None is selected then the lowest/highest possible value of the variable's domain is chosen.
            encoding (type[Encoding], optional): _description_. Defaults to Bitwise.
            precision (float, optional): The floating point precision for REAL variables. Defaults to 1e-2.
        """
        super().__init__(label=label, domain=domain, bounds=bounds)
        self._encoding = encoding
        self._precision = 1e-2
        self._term: Term | None = None
        self._bin_vars: list[BaseVariable] = []
        self._precision = precision

    @property
    def encoding(self) -> type[Encoding]:
        return self._encoding

    @property
    def precision(self) -> float:
        return self._precision

    @property
    def term(self) -> Term:
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
        self, domain: Domain, bounds: tuple[float | None, float | None], encoding: type[Encoding] | None = None
    ) -> None:
        self._encoding = encoding if encoding is not None else self._encoding
        self._term = None
        return super().update_variable(domain, bounds)

    def evaluate(self, value: list[int] | Number) -> float:
        if isinstance(value, Number):
            if not self.domain.check_value(value):
                raise ValueError(f"The value {value} is invalid for the domain {self.domain.value}")
            if value < self.lower_bound or value > self.upper_bound:
                raise ValueError(f"The value {value} is outside the defined bounds {self.bounds}")
            return value
        return self.encoding.evaluate(self, value, self._precision)

    def to_binary(self) -> Term:
        if self._term is None:
            term = self.encoding.encode(self, precision=self._precision)
            self._term = copy.copy(term)
            self._bin_vars = [BinaryVariable(f"{self.label}({i})") for i in range(self.num_binary_equivalent())]
            self._bin_vars = sorted(
                self._bin_vars,
                key=lambda x: _extract_number(x.label),
            )
        return self._term

    def num_binary_equivalent(self) -> int:
        """
        Returns:
            int: the number of binary variables needed to encode the continuous variable.
        """
        return self.encoding.num_binary_equivalent(self, precision=self._precision)

    def check_valid(self, binary_list: list[int]) -> tuple[bool, int]:
        """checks if the binary list sample is a valid sample in the variable's encoding.

        Args:
            binary_list (list[int] | int):  a list of binary values or an integer value.

        Returns:
            tuple[bool, int]: the boolean is True if the sample is a valid encoding,
                                and the integer is the error in the encoding.
        """
        return self.encoding.check_valid(binary_list)

    def encoding_constraint(self) -> ComparisonTerm:
        """Given a continuous variable return a Comparison Term that ensures that the encoding is respected.

        Returns:
            ComparisonTerm: a Comparison Term that ensures the encoding is respected.
        """
        return self.encoding.encoding_constraint(self, precision=self._precision)


# Terms ###


@yaml.register_class
class Term:
    """Represents a mathematical Term (e.g. 3x*y, 2x, ...).

    And they are built from:
    - ``Variable``'s: The decision variables of the model (x, y, ...).
    - Other ``Term``'s: Allowing for complex expressions to be constructed.
    """

    CONST = Variable(CONST_KEY, Domain.REAL)

    def __init__(self, elements: Sequence[BaseVariable | Term | Number], operation: Operation) -> None:
        """initialize a new term object.

        Args:
            elements (Sequence[BaseVariable  |  Term  |  Number]): a list of elements in the term.
            operation (Operation): the mathematical operation between these elements.

        Raises:
            ValueError: if the items inside elements are not from the listed types (BaseVariable  |  Term  |  Number).
        """
        self._operation = operation
        self._elements: dict[BaseVariable | Term, Number] = {}  # The list of elements in the term.
        # key: the term or variable | value: the coefficient corresponding to that value.
        for e in elements:
            if isinstance(e, BaseVariable):
                if e in self:
                    if self._is_constant(e):
                        self[e] = self._apply_operation_on_constants([self[e], 1])
                    elif isinstance(e, BinaryVariable) and self.operation == Operation.MUL:
                        self[e] = 1
                    else:
                        self[e] += 1
                else:
                    self[e] = 1
            elif isinstance(e, Number):
                if self.CONST in self:
                    self[self.CONST] = self._apply_operation_on_constants([self[self.CONST], e])
                else:
                    self[self.CONST] = e
            elif isinstance(e, Term):
                if len(e) == 0:
                    if self.CONST in self:
                        self[self.CONST] = self._apply_operation_on_constants([self[self.CONST], 0])
                    else:
                        self[self.CONST] = 0
                elif e.operation == self._operation:
                    for key in e:
                        if key in self:
                            if isinstance(key, BaseVariable) and self._is_constant(key):
                                self[key] = self._apply_operation_on_constants([self[key], e[key]])
                            elif isinstance(key, BinaryVariable) and self.operation == Operation.MUL:
                                self[key] = 1
                            else:
                                self[key] += e[key]
                        else:
                            self[key] = e[key]
                else:
                    e_copy = copy.copy(e)
                    coeff = 1.0
                    if e_copy.operation == Operation.MUL and self.CONST in e_copy:
                        coeff = e_copy.pop(self.CONST)
                    simple_e = e_copy._simplify()  # noqa: SLF001
                    simple_e = self.CONST if isinstance(simple_e, Term) and len(simple_e) == 0 else simple_e
                    if simple_e in self:
                        if isinstance(simple_e, BaseVariable) and self._is_constant(simple_e):
                            self[simple_e] = self._apply_operation_on_constants([self[simple_e], coeff])
                        else:
                            self[simple_e] += coeff
                    else:
                        self[simple_e] = coeff
            else:
                raise ValueError(
                    f"Term accepts object of types Term or Variable but an object of type {e.__class__()} was given"
                )
        self._remove_zeros()

    @property
    def operation(self) -> Operation:
        """
        Returns:
            Operation: the operation between the term's elements.
        """
        return self._operation

    @property
    def degree(self) -> int:
        """
        Returns:
            int: the highest degree in the term.
        """
        degree = 0
        if self.operation == Operation.MUL:
            for element in self:
                if isinstance(element, Term):
                    degree += element.degree
                elif isinstance(element, BaseVariable) and not self._is_constant(element):
                    degree += int(self[element])
            return degree

        for element in self:
            if isinstance(element, Term):
                degree = max(degree, element.degree)
            elif isinstance(element, BaseVariable) and not self._is_constant(element):
                degree = max(degree, 1)
        return degree

    def to_binary(self) -> Term:
        """Returns the term in binary format. That is encoding all continuous variables into
            binary according to the encoding defined in the variable.

        Raises:
            ValueError: The term contains operations that are not addition or multiplication.
            ValueError: the term contains an element that is not a Term or a BaseVariable.

        Returns:
            Term: the term after transforming all the variables into binary.
        """
        if self.operation not in {Operation.ADD, Operation.MUL}:
            raise ValueError("Can not evaluate any operation that is not Addition of Multiplication")
        out_list: list[BaseVariable | Term | Number] = []
        for e in self:
            if isinstance(e, Term):
                out_list.append(self[e] * e.to_binary())
            elif isinstance(e, BaseVariable):
                if self._is_constant(e):
                    out_list.append(self[e])
                elif isinstance(e, Variable):
                    x = e.to_binary()
                    if self.operation == Operation.MUL:
                        out_list.append(x ** int(self[e]))
                    else:
                        out_list.append(self[e] * x)
                else:
                    out_list.append(self[e] * e)
            else:
                raise ValueError(f"Evaluating term with elements of type {e.__class__} is not supported.")

        return Term(out_list, self.operation)

    def _apply_operation_on_constants(self, const_list: list[Number]) -> Number:
        out = const_list[0]
        for i in range(1, len(const_list)):
            if self.operation is Operation.ADD:
                out += float(const_list[i])
            elif self.operation is Operation.SUB:
                out -= float(const_list[i])
            elif self.operation is Operation.MUL:
                out *= float(const_list[i])
            elif self.operation is Operation.DIV:
                out /= float(const_list[i])

        return out

    def variables(self) -> list[BaseVariable]:
        """Returns the unique list of variables in the Term

        Returns:
            list[Variable]: The unique list of variables in the Term.
        """
        var = set()
        for e in self:
            if isinstance(e, BaseVariable) and not self._is_constant(e):
                var.add(e)
            elif isinstance(e, Term):
                var.update(e.variables())
        return sorted(var, key=lambda x: x.label)

    def _simplify(self) -> Term | BaseVariable:
        """Simplify the term object.

        Returns:
            (Term | BaseVariable): the simplified term.
        """
        if len(self) == 1:
            item = next(iter(self._elements.keys()))
            if self._elements[item] == 1:
                return item
        return self

    def pop(self, item: BaseVariable | Term) -> Number:
        """Remove an item from the term.

        Args:
            item (BaseVariable | Term): the item to be removed.

        Raises:
            KeyError: if item is not in the term.

        Returns:
            Number: the coefficient of the removed item.
        """
        try:
            return self._elements.pop(item)
        except KeyError as e:
            raise KeyError(f'item "{item}" not found in the term.') from e

    def _is_constant(self, variable: BaseVariable) -> bool:
        """Checks if the variable is a constant variable as defined by the Term class.

        Args:
            variable (BaseVariable): the variable to be checked.

        Returns:
            bool: True if the variable is a constant, False otherwise.
        """
        return variable == self.CONST

    def to_list(self) -> list[BaseVariable | Term | Number]:
        """Exports the current term into a list of its elements.

        Returns:
            list[BaseVariable | Term | Number]: A list of the elements inside the term.
        """
        out_list: list[BaseVariable | Term | Number] = []
        for e in self:
            if isinstance(e, BaseVariable) and self._is_constant(e):
                out_list.append(self[e])
            elif self.operation == Operation.MUL:
                for _ in range(int(self[e])):
                    out_list.append(e)
            else:
                out_list.append(self[e] * e if self[e] != 1 else e)
        return out_list

    def _unfold_parentheses(self) -> Term:
        """Simplifies any parentheses in the term expression.

        Returns:
            Term: A new term with a more simplified form.
        """
        out = copy.copy(self)
        if out.operation != Operation.MUL:
            return out

        parentheses: list[tuple[Term, Number]] = []

        for e in out:
            if isinstance(e, Term) and e.operation == Operation.ADD:
                parentheses.append((copy.copy(e), out[e]))

        for term, _ in parentheses:
            out.pop(term)

        if len(out) == 0 and len(parentheses) != 0:
            out = Term([1], Operation.ADD)

        for _term, coeff in parentheses:
            term = copy.copy(_term)
            if coeff > 1:
                term **= int(coeff)
            final_out = []
            for t in term:
                final_out.append(t * out * term[t])
            out = Term(final_out, Operation.ADD)

        return out

    def _remove_zeros(self) -> None:
        """Simplifies any un-necessary zeros from terms."""
        to_be_popped = []
        if self.operation == Operation.MUL and self.CONST in self and self[self.CONST] == 0:
            l = len(self)
            for _ in range(l):
                self._elements.popitem()
        for e in self:
            if self[e] == 0:
                to_be_popped.append(e)
        for p in to_be_popped:
            self._elements.pop(p)

    def evaluate(self, var_values: Mapping[BaseVariable, list[int] | Number]) -> float:
        """Evaluates the term given a set of values for the variables in the term.

        Args:
            var_values (Mapping[BaseVariable, list[int]  |  Number]): the values of the variables in the term.
                If the value provided is binary list (list[int]) then the value of the variable is evaluated based on
                its binary representation. This representation is constructed using the encoding, bounds and domain
                of the variable. To check the binary representation of a variable you can check the method `to_binary()`

        Raises:
            ValueError: if not all variables in the term are provided a value.

        Returns:
            float: the result from evaluating the term.
        """
        var_hash_dict = {hash(v): var_values[v] for v in var_values}
        for var in self.variables():
            if hash(var) not in var_hash_dict:
                raise ValueError(f"Can not evaluate term because the value of the variable {var} is not provided.")
        output = 0.0 if self.operation in {Operation.ADD, Operation.SUB} else 1.0
        for e in self:
            if isinstance(e, Term):
                output = self._apply_operation_on_constants([output, e.evaluate(var_values) * self[e]])
            elif isinstance(e, BaseVariable):
                if e == self.CONST:
                    output = self._apply_operation_on_constants([output, self[e]])
                elif self.operation == Operation.MUL:
                    output = self._apply_operation_on_constants([output, e.evaluate(var_hash_dict[hash(e)]) ** self[e]])
                else:
                    output = self._apply_operation_on_constants([output, e.evaluate(var_hash_dict[hash(e)]) * self[e]])
        return output

    def get_constant(self) -> Number:
        """
        Returns:
            Number: The constant value of the term.
        """
        if self.CONST in self:
            return self[self.CONST]
        return 0 if self.operation in {Operation.ADD, Operation.SUB} else 1

    def __copy__(self) -> Term:
        return Term(copy.copy(self.to_list()), self.operation)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "0"
        output_string = ""
        const = self.get_constant()
        keys = list(self._elements.keys())

        if (
            (self.operation in {Operation.ADD, Operation.SUB} and const == 0)
            or (self.operation in {Operation.MUL, Operation.DIV} and const == 1)
        ) and Term.CONST in keys:
            keys.remove(Term.CONST)

        for i, e in enumerate(keys):
            if isinstance(e, Term):
                term_str = str(e).strip()
                if len(term_str) > 0:
                    if term_str[0] == "(" and term_str[-1] == ")":
                        term_str = term_str.removeprefix("(").removesuffix(")")
                    output_string += f"({term_str}) " if self[e] == 1 else f"({self[e]}) * ({term_str}) "
            elif isinstance(e, BaseVariable):
                if self._is_constant(e):
                    if self.operation in {Operation.ADD, Operation.SUB} and self[e] == 0:
                        continue
                    if self.operation in {Operation.MUL, Operation.DIV} and self[e] == 1:
                        continue
                    output_string += f"({self[e]}) "
                elif (self.operation is Operation.MUL or self.operation is Operation.DIV) and self[e] > 1:
                    output_string += f"({e}^{self[e]}) "
                else:
                    output_string += f"{e} " if self[e] == 1 else f"({self[e]}) * {e} "
            else:
                output_string += f"{e} "
            if i < len(keys) - 1:
                output_string += f"{self.operation.value} "

        return output_string.strip()

    __str__ = __repr__

    def __getitem__(self, item: BaseVariable | Term) -> Number:
        return self._elements[item]

    def __setitem__(self, key: BaseVariable | Term, item: Number) -> None:
        self._elements[key] = item

    def __iter__(self) -> Iterator[BaseVariable | Term]:
        yield from self._elements

    def __contains__(self, item: BaseVariable | Term) -> bool:
        return item in self._elements

    __next__ = __iter__

    def __len__(self) -> int:
        return len(self._elements)

    def __add__(self, other: Number | BaseVariable | Term) -> Term:
        out = self.to_list() if self.operation == Operation.ADD else [copy.copy(self)]
        out.append(other)
        return Term(out, Operation.ADD)

    __iadd__ = __add__

    def __radd__(self, other: Number | BaseVariable | Term) -> Term:
        out = self.to_list() if self.operation == Operation.ADD else [copy.copy(self)]
        out.insert(0, other)
        return Term(out, Operation.ADD)

    def __mul__(self, other: Number | BaseVariable | Term) -> Term:
        out = self.to_list() if self.operation == Operation.MUL else [copy.copy(self)]
        if len(out) == 0:
            out = [0]
        out.append(other)
        return Term(out, Operation.MUL)._unfold_parentheses()

    __imul__ = __mul__

    def __rmul__(self, other: Number | BaseVariable | Term) -> Term:
        out = self.to_list() if self.operation == Operation.MUL else [copy.copy(self)]
        if len(out) == 0:
            out = [0]
        out.insert(0, other)
        return Term(out, Operation.MUL)._unfold_parentheses()

    def __neg__(self) -> Term:
        return -1 * self

    def __sub__(self, other: Number | BaseVariable | Term) -> Term:
        return self + -1 * other

    def __rsub__(self, other: Number | BaseVariable | Term) -> Term:
        return -1 * self + other

    __isub__ = __sub__

    def __truediv__(self, other: Number) -> Term:
        if not isinstance(other, Number):
            raise NotImplementedError("Only division by numbers is currently supported")

        if other == 0:
            raise ValueError("Division by zero is not allowed")

        other = 1 / other
        return self * other

    __itruediv__ = __truediv__

    def __rtruediv__(self, other: Number | BaseVariable | Term) -> Term:
        raise NotSupportedOperation("Only division by numbers is currently supported")

    def __rfloordiv__(self, other: Number | BaseVariable | Term) -> Term:
        raise NotSupportedOperation("Only division by numbers is currently supported")

    def __pow__(self, a: int) -> Term:
        if self.operation == Operation.ADD:
            out = copy.copy(self)
            for _ in range(a - 1):
                out_list = []
                for element in self:
                    out_list.append(out * copy.copy(element) * self[element])
                out = Term(out_list, Operation.ADD)
            return out
        if self.operation == Operation.MUL:
            out = copy.copy(self)
            for element in out:
                if element is Term.CONST:
                    out[element] **= a
                else:
                    out[element] *= a
            return out
        raise NotImplementedError(
            "The power operation for terms that are not addition or multiplication is not supported."
        )

    def __hash__(self) -> int:
        return hash((frozenset(self._elements.items()), self.operation))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return False
        return hash(self) == hash(other)


@yaml.register_class
class ComparisonTerm:
    """Represents a mathematical comparison Term, that can be an equality or an inequality between two ``Term``s
    (e.g. x+y>0, x>2, ...).

    They are built from a left and a right hand part, each of which can contain:
    - ``Variable``'s: The decision variables of the model (x, y, ...).
    - Other ``Term``'s: Allowing for complex expressions to be constructed (x+y, ...)
    """

    def __init__(
        self, lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term, operation: ComparisonOperation
    ) -> None:
        """Initializes a new comparison term.

        Args:
            lhs (Number | BaseVariable | Term): the left hand side of the comparison term.
            rhs (Number | BaseVariable | Term): the right hand side of the comparison term.
            operation (ComparisonOperation): the comparison operations between the left and right hand sides.
        """
        term = lhs - rhs
        if not isinstance(term, Term):
            term = Term([term], Operation.ADD)  # I don't think this line is reachable
        const = -1 * term.pop(Term.CONST) if Term.CONST in term else 0
        self._lhs = term
        self._rhs = Term([const], Operation.ADD)
        self._operation = operation

    @property
    def operation(self) -> ComparisonOperation:
        """
        Returns:
            ComparisonOperation: the comparison operation between the left and right hand sides.
        """
        return self._operation

    @property
    def lhs(self) -> Term:
        """
        Returns:
            Term: the left hand side of the comparison term.
        """
        return self._lhs

    @property
    def rhs(self) -> Term:
        """
        Returns:
            Term: the right hand side of the comparison term.
        """
        return self._rhs

    def variables(self) -> list[BaseVariable]:
        """Returns the unique list of variables in the Term

        Returns:
            list[Variable]: The unique list of variables in the Term.
        """
        lhs_var = self._lhs.variables()
        rhs_var = self._rhs.variables()

        var = set()
        var.update(lhs_var)
        var.update(rhs_var)

        return sorted(var, key=lambda x: x.label)

    @property
    def degree(self) -> int:
        """
        Returns:
            int: the maximum degree in the left and right hand sides of the comparison term.
        """
        return max(self.rhs.degree, self.lhs.degree)

    def to_list(self) -> list:
        """Exports the comparison term into a list. The elements of the right hand side are first moved to the left hand
        side before the generation of the list. Therefore, you can assume that the right hand side will be zero.

        Returns:
            list: a list constructed from all the elements in the left and right hand sides of the comparison term.
        """
        logger.info(
            "to_list(): The elements of output list assume the comparison term has been transformed "
            + f"from (lhs {self.operation.value} rhs) to (lhs - rhs {self.operation.value} 0).",
        )
        out = self.lhs.to_list()
        out.extend((-1 * self.rhs).to_list())
        return out

    def to_binary(self) -> ComparisonTerm:
        """Returns the comparison term in binary format. That is encoding all continuous variables into
            binary according to the encoding defined in the variable.

        Returns:
            ComparisonTerm: the comparison term after transforming all the variables into binary.
        """
        return ComparisonTerm(rhs=self.rhs.to_binary(), lhs=self.lhs.to_binary(), operation=self.operation)

    def _apply_comparison_operation(self, v1: Number, v2: Number) -> bool:
        """Compare two arguments.

        Args:
            v1 (Number): the left hand side value.
            v2 (Number): the right hand side value.

        Raises:
            ValueError: if the comparison term's operation is invalid.

        Returns:
            bool: the result of the comparison between v1 and v2 assuming the
            comparison operation of the comparison term object.
        """
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

    def evaluate(self, var_values: Mapping[BaseVariable, Number | list[int]]) -> bool:
        """Evaluates the comparison term given a set of values for the variables in the term.

        Args:
            var_values (Mapping[BaseVariable, list[int]  |  Number]): the values of the variables in the comparison term.

        Returns:
            bool: the result from evaluating the comparison term.
        """
        return self._apply_comparison_operation(self._lhs.evaluate(var_values), self._rhs.evaluate(var_values))

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
