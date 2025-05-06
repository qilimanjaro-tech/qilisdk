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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator, Mapping, Sequence, TypeVar
from warnings import warn

import numpy as np

from qilisdk.analog.exceptions import NotSupportedOperation

Number = int | float
GenericVar = TypeVar("GenericVar", bound="Variable")
CONST_KEY = "_const_"
MAX_INT = np.iinfo(np.int64).max
MIN_INT = np.iinfo(np.int64).max
LARGE = 100


class Side(Enum):
    RIGHT = "right"
    LEFT = "left"


class Domain(str, Enum):
    INTEGER = "Integer Domain"
    POSITIVE_INTEGER = "Positive Integer Domain"
    REAL = "Real Domain"
    BINARY = "Binary Domain"
    SPIN = "Spin Domain"

    def check_value(self, value: float) -> bool:
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
        if self in {Domain.BINARY, Domain.POSITIVE_INTEGER}:
            return 0
        if self == Domain.SPIN:
            return -1
        if self == Domain.INTEGER:
            return MIN_INT
        return -1e30

    def max(self) -> float:
        if self in {Domain.BINARY, Domain.SPIN}:
            return 1
        if self in {Domain.POSITIVE_INTEGER, Domain.INTEGER}:
            return MAX_INT
        return 1e30


class Operation(Enum):
    MUL = "*"
    ADD = "+"
    DIV = "/"
    SUB = "-"


class ComparisonOperation(Enum):
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="


class Encoding(ABC):
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
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is Domain.Real)

        Returns:
            Term: a term that only contains binary variables
        """

    @staticmethod
    @abstractmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        """Given a continuous variable return a Constraint Term that ensures that the encoding is respected.

        Args:
            var (ContinuousVar): The continuous variable to be encoded
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is Domain.Real)

        Returns:
            Constraint Term: a constraint term that ensures the encoding is respected.
        """

    @staticmethod
    @abstractmethod
    def evaluate(var: Variable, binary_list: list[int], precision: float = 1e-2) -> float:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is Domain.Real)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """

    @staticmethod
    def num_binary_equivalent(var: "Variable", precision: float = 1e-2) -> int:
        """Give a continuous variable return the number of binary variables needed to encode it.

        Args:
            var (ContinuousVar): the continuous variable.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is Domain.Real)

        Returns:
            int: the number of binary variables needed to encode it.
        """
        raise NotImplementedError("This is an abstract class and is not meant to be executed.")

    @staticmethod
    def check_valid(binary_list: list[int]) -> tuple[bool, int]:
        """checks if the binary list sample is a valid sample in this encoding.

        Args:
            binary_list (list[int]):  a list of binary values.

        Returns:
            tuple[bool, int]: the boolean is True if the sample is a valid encoding, while the int is the error in the encoding.
        """
        raise NotImplementedError("This is an abstract class and is not meant to be executed.")

    @staticmethod
    @abstractmethod
    def term_equals_to(var: Variable, number: int, precision: float = 1e-2) -> Term:
        """returns a term that is 1 if the variable is equal to the number, else 0.

        Args:
            var (ContinuousVar): the continuous variable.
            number (int): the number to equate the variable to.
        """


class HOBO(Encoding):
    @property
    def name(self) -> str:
        return "HOBO"

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Term:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        abs_bound = np.abs(bounds[1] - bounds[0])
        n_binary = int(np.floor(np.log2(abs_bound if abs_bound != 0 else 1)))

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary + 1)]

        term = sum(2**i * binary_vars[i] for i in range(n_binary))

        term += (np.abs(bounds[1] - bounds[0]) + 1 - 2**n_binary) * binary_vars[-1]

        term += bounds[0]

        return term

    @staticmethod
    def evaluate(var: Variable, binary_list: list[int], precision: float = 1e-2) -> float:
        # TODO (ameer): allow to map continuous values to binary string of given format.
        if not HOBO.check_valid(binary_list=binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the HOBO encoding.")
        term = HOBO.encode(var)
        binary_var = term.variables()

        if len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict: dict[BaseVariable, list[int]] = {binary_var[i]: [binary_list[i]] for i in range(len(binary_list))}

        if var.domain is Domain.REAL:
            term *= precision

        out = term.evaluate(binary_dict, precision=precision)

        out = int(out) if var.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER} else out

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )
        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is Domain.Real)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """
        raise NotImplementedError("HOBO encoding constraints are not supported at the moment")

    @staticmethod
    def num_binary_equivalent(var: "Variable", precision: float = 1e-2) -> int:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.floor(np.log2(np.abs(bounds[1] - bounds[0]))))

        return n_binary + 1

    @staticmethod
    def check_valid(binary_list: list[int]) -> tuple[bool, int]:
        return True, 0

    @staticmethod
    def term_equals_to(var: Variable, number: int, precision: float = 1e-2) -> Term:
        encoded_num: list[int] = []

        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        abs_bound = np.abs(bounds[1] - bounds[0])
        n_binary = int(np.floor(np.log2(abs_bound if abs_bound != 0 else 1)))

        aux_num = number

        aux_num -= int(bounds[0])

        overflow_val = np.abs(bounds[1] - bounds[0]) + 1 - 2**n_binary

        if aux_num >= overflow_val:
            encoded_num.insert(0, 1)
            aux_num -= overflow_val
        else:
            encoded_num.insert(0, 0)

        for i in range(n_binary, 0, -1):
            if aux_num >= 2**i:
                encoded_num.insert(0, 1)
                aux_num -= 2**i
            else:
                encoded_num.insert(0, 0)

        out_term = Term([1], Operation.MUL)

        for i in range(var.num_binary_equivalent()):
            out_term *= var[i] if encoded_num[i] == 0 else (1 - var[i])

        return out_term


class OneHot(Encoding):
    @property
    def name(self) -> str:
        return "ONE HOT"

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Term:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]

        term = Term([(bounds[0] + i) * binary_vars[i] for i in range(n_binary)], Operation.ADD)

        return term

    @staticmethod
    def evaluate(var: Variable, binary_list: list[int], precision: float = 1e-2) -> float:
        if not OneHot.check_valid(binary_list=binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the one hot encoding.")

        term = OneHot.encode(var)
        binary_var = term.variables()

        if len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict: dict[BaseVariable, list[int]] = {binary_var[i]: [binary_list[i]] for i in range(len(binary_list))}

        if var.domain is Domain.REAL:
            term *= precision

        out = term.evaluate(binary_dict, precision=precision)

        out = int(out) if var.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER} else out

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )

        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]
        return ComparisonTerm(1, sum(binary_vars), ComparisonOperation.EQ)

    @staticmethod
    def num_binary_equivalent(var: Variable, precision: float = 1e-2) -> int:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        return n_binary

    @staticmethod
    def check_valid(binary_list: list[int]) -> tuple[bool, int]:
        num_ones = binary_list.count(1)
        return num_ones == 1, (num_ones - 1) ** 2

    @staticmethod
    def term_equals_to(var: "Variable", number: int, precision: float = 1e-2) -> Term:
        out_term = Term([1], Operation.MUL)
        if var.domain is Domain.REAL:
            for i in range(var.num_binary_equivalent()):
                out_term *= (1 - var[i]) if i != number / precision else var[i]
        else:
            for i in range(var.num_binary_equivalent()):
                out_term *= (1 - var[i]) if i != number else var[i]
        return out_term


class DomainWall(Encoding):
    @property
    def name(self) -> str:
        return "Domain Wall"

    @staticmethod
    def encode(var: Variable, precision: float = 1e-2) -> Term:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]

        term = Term([0], Operation.ADD)
        for i in range(n_binary):
            term += binary_vars[i]

        term += bounds[0]

        return term

    @staticmethod
    def evaluate(var: Variable, binary_list: list[int], precision: float = 1e-2) -> float:
        if not DomainWall.check_valid(binary_list=binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the domain wall encoding.")
        term = DomainWall.encode(var)
        binary_var = term.variables()

        if len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict: dict[BaseVariable, list[int]] = {binary_var[i]: [binary_list[i]] for i in range(len(binary_list))}

        if var.domain is Domain.REAL:
            term *= precision

        out = term.evaluate(binary_dict, precision)

        out = int(out) if var.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER} else out

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )
        return out

    @staticmethod
    def encoding_constraint(var: Variable, precision: float = 1e-2) -> ComparisonTerm:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]
        return ComparisonTerm(
            0,
            sum(binary_vars[i + 1] * (1 - binary_vars[i]) for i in range(len(binary_vars) - 1)),
            ComparisonOperation.EQ,
        )

    @staticmethod
    def num_binary_equivalent(var: Variable, precision: float = 1e-2) -> int:
        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        return n_binary

    @staticmethod
    def check_valid(binary_list: list[int]) -> tuple[bool, int]:
        value = sum(binary_list[i + 1] * (1 - binary_list[i]) for i in range(len(binary_list) - 1))
        return value == 0, value

    @staticmethod
    def term_equals_to(var: Variable, number: int, precision: float = 1e-2) -> Term:
        encoded_num: list[int] = []
        aux_number = number

        for i in range(var.num_binary_equivalent()):
            if i <= number:
                encoded_num.append(1)
            else:
                encoded_num.append(0)

        bounds = var.bounds
        if var.domain is Domain.REAL:
            bounds = (bounds[0] / precision, bounds[1] / precision)

        aux_number -= int(bounds[0])

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        num_sum = 0
        for i in range(n_binary):
            if num_sum < aux_number:
                encoded_num.append(1)
                num_sum += 1
            else:
                encoded_num.append(0)

        if num_sum < aux_number:
            raise ValueError(f"There are not enough qubits to encode the number {number}")

        out_term = Term([1], Operation.MUL)

        for i in range(var.num_binary_equivalent()):
            out_term *= var[i] if encoded_num[i] == 0 else (1 - var[i])

        return out_term


# Variables ###


class BaseVariable:
    """This class represents the general structure of any variable that can be included in the model."""

    def __init__(self, label: str, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        """initialize a new Variable object

        Args:
            label (str): The name of the variable.
            domain (Domain): The domain of the values this variable can take.
            bounds (tuple[float  |  None, float  |  None], optional): the bounds on the values of the variable
                                                The bounds have the structure (lower_bound, Upper_bound) both values
                                                included. Defaults to (None, None).
                                                Note: if None is selected then the lowest/highest possible value of the
                                                variable's domain is chosen.

        Raises:
            ValueError: the lower bound or the upper bound don't correspond to the variable domain.
            ValueError: the lower bound is higher than the upper bound.
        """
        self._label = label
        self._domain = domain

        lower_bound, upper_bound = bounds
        if lower_bound is None:
            lower_bound = domain.min()
        if upper_bound is None:
            upper_bound = domain.max()

        if not self.domain.check_value(upper_bound):
            raise ValueError(
                f"the lower bound ({upper_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if not self.domain.check_value(lower_bound):
            raise ValueError(
                f"the upper bound ({lower_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if lower_bound > upper_bound:
            raise ValueError("lower bound can't be larger than the upper bound.")
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
            ValueError: the lower bound or the upper bound don't correspond to the variable domain.
            ValueError: the lower bound is higher than the upper bound.
        """
        if lower_bound is None:
            lower_bound = self._domain.min()
        if upper_bound is None:
            upper_bound = self._domain.max()
        if not self.domain.check_value(lower_bound):
            raise ValueError(
                f"the lower bound ({lower_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if not self.domain.check_value(upper_bound):
            raise ValueError(
                f"the upper bound ({upper_bound}) does not respect the domain of the variable ({self.domain})"
            )
        if lower_bound > upper_bound:
            raise ValueError(
                f"the lower bound ({lower_bound}) should not be greater than the upper bound ({upper_bound})"
            )
        self._bounds = (lower_bound, upper_bound)

    def num_binary_equivalent(self) -> int:
        raise NotImplementedError

    def evaluate(self, binary_list: list[int], precision: float = 1e-2) -> float:
        raise NotImplementedError

    def compare(self, other: BaseVariable) -> bool:
        """Checks if two Variable objects are equal based on their hash values.

        Args:
            other (`Variable`): the `Variable` object to be compared with.

        Returns:
            bool: a boolean value that indicates whether the hash value of the current `Variable` object (`self`) is
            equal to the hash value of the `other` `Variable` object passed as an argument.
        """
        return hash(self) == hash(other)

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None]) -> None:
        """Replaces the information of the variable with those coming from the dictionary
        if the variable label is in the dictionary

        Args:
            var_dict (dict): A dictionary that holds the labels of the variables to be
                            changed alongside the new values they should take
        """

        self._domain = domain
        self.set_bounds(bounds[0], bounds[1])

    def __copy__(self) -> BaseVariable:
        return BaseVariable(label=self.label, domain=self.domain)

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

    def __lt__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.LT)

    def __le__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.LE)

    def __eq__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:  # type: ignore[override]
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.EQ)

    def __ne__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:  # type: ignore[override]
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.NE)

    def __gt__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.GT)

    def __ge__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.GE)

    def __hash__(self) -> int:
        return hash((self._label, self._domain.value, self._bounds))


class BinaryVar(BaseVariable):
    def __init__(self, label: str) -> None:
        super().__init__(label=label, domain=Domain.BINARY)

    def num_binary_equivalent(self) -> int:  # noqa: PLR6301
        return 1

    def evaluate(self, binary_list: list[int], precision: float = 1e-2) -> float:  # noqa: PLR6301
        if len(binary_list) != 1:
            raise ValueError("Evaluating a Binary variable with a binary list of more than one item.")
        return binary_list[0]

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None]) -> None:
        raise NotImplementedError

    def __copy__(self) -> BinaryVar:
        return BinaryVar(label=self.label)


class SpinVar(BaseVariable):
    def __init__(self, label: str) -> None:
        super().__init__(label=label, domain=Domain.SPIN, bounds=(-1, 1))

    def num_binary_equivalent(self) -> int:  # noqa: PLR6301
        return 1

    def update_variable(self, domain: Domain, bounds: tuple[float | None, float | None]) -> None:
        raise NotImplementedError

    def evaluate(self, binary_list: list[int], precision: float = 1e-2) -> float:  # noqa: PLR6301
        if len(binary_list) != 1:
            raise ValueError("Evaluating a Spin variable with a binary list of more than one item.")
        return -1 if binary_list[0] == 0 else 1


class Variable(BaseVariable):

    def __init__(
        self, label: str, domain: Domain, bounds: tuple[float | None, float | None], encoding: type[Encoding] = HOBO
    ) -> None:
        super().__init__(label=label, domain=domain, bounds=bounds)
        self._encoding = encoding
        self._precision = 1e-2
        self._term: Term | None = None
        self._bin_vars: list[BaseVariable] = []

    @property
    def encoding(self) -> type[Encoding]:
        return self._encoding

    @property
    def term(self) -> Term:
        if self._term is None:
            if self.bounds[1] > LARGE or self.bounds[0] < -LARGE:
                warn(
                    f"Encoding variable {self.label} which has the bounds {self.bounds}"
                    + "is very expensive and may take a very long time."
                )
            self._term = self.encode()
            self._bin_vars = self._term.variables()
        return self._term

    @property
    def bin_vars(self) -> list[BaseVariable]:
        if self._term is None:
            self._term = self.encode()
            self._bin_vars = self._term.variables()
        return self._bin_vars

    def __copy__(self) -> Variable:
        return Variable(label=self.label, domain=self.domain, bounds=self.bounds, encoding=self._encoding)

    def __getitem__(self, item: int) -> BaseVariable:
        if self._term is None:
            self._term = self.encode()
            self._bin_vars = self._term.variables()
        return self._bin_vars[item]

    def update_variable(
        self, domain: Domain, bounds: tuple[float | None, float | None], encoding: type[Encoding] | None = None
    ) -> None:
        self._encoding = encoding if encoding is not None else self._encoding
        return super().update_variable(domain, bounds)

    def evaluate(self, binary_list: list[int], precision: float = 1e-2) -> float:
        if len(binary_list) == 1:
            if not self.domain.check_value(binary_list[0]):
                raise ValueError(f"The value {binary_list} is invalid for the domain {self.domain.value}")
            if self.domain == Domain.REAL:
                return binary_list[0] * precision
            return binary_list[0]
        return self.encoding.evaluate(self, binary_list, precision=precision)

    def encode(self, precision: float = 1e-2) -> Term:
        self._precision = precision
        term = self.encoding.encode(self, precision=self._precision)
        self._term = copy.copy(term)
        self._bin_vars = self._term.variables()
        return term

    def num_binary_equivalent(self, precision: float = 1e-2) -> int:
        return self.encoding.num_binary_equivalent(self, precision=precision)

    def check_valid(self, binary_list: list[int]) -> tuple[bool, int]:
        return self.encoding.check_valid(binary_list)

    def encoding_constraint(self, precision: float = 1e-2) -> ComparisonTerm:
        return self.encoding.encoding_constraint(self, precision=precision)

    def term_equals_to(self, number: int, precision: float = 1e-2) -> Term:
        return self.encoding.term_equals_to(self, number, precision)


# Terms ###


class Term:
    CONST = BaseVariable(CONST_KEY, Domain.REAL)

    def __init__(self, elements: Sequence[BaseVariable | Term | Number], operation: Operation) -> None:
        self._operation = operation
        self._elements: dict[int, Number] = {}
        self._map: dict[int, BaseVariable | Term] = {}
        for e in elements:
            if isinstance(e, BaseVariable):
                if e in self:
                    if self.is_constant(e):
                        self[e] = self._apply_operation_on_constants([self[e], 1])
                    elif isinstance(e, BinaryVar) and self.operation == Operation.MUL:
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
                if e.operation == self._operation:
                    for key in e:
                        if key in self:
                            if isinstance(key, BaseVariable) and self.is_constant(key):
                                self[key] = self._apply_operation_on_constants([self[key], e[key]])
                            else:
                                self[key] += e[key]
                        else:
                            self[key] = e[key]
                else:
                    e_copy = copy.copy(e)
                    if len(e_copy) > 0:
                        coeff = 1.0
                        if e_copy.operation == Operation.MUL and self.CONST in e_copy:
                            coeff = e_copy.pop(self.CONST)
                        simple_e = e_copy.simplify()
                        simple_e = self.CONST if isinstance(simple_e, Term) and len(simple_e) == 0 else simple_e
                        if simple_e in self:
                            self[simple_e] += coeff  # self._apply_operation_on_constants([self[simple_e], coeff])
                        else:
                            self[simple_e] = coeff
                    elif len(e_copy) == 0:
                        if self.CONST in self:
                            self[self.CONST] = self._apply_operation_on_constants([self[self.CONST], 0])
                        else:
                            self[self.CONST] = 0
            else:
                raise ValueError(
                    f"Term accepts object of types Term or Variable but an object of type {e.__class__()} was given"
                )
        self.remove_zeros()

    @property
    def operation(self) -> Operation:
        return self._operation

    @property
    def degree(self) -> int:
        degree = 0
        if self.operation == Operation.MUL:
            for element in self:
                if isinstance(element, Term):
                    degree += element.degree
                elif isinstance(element, BaseVariable) and not self.is_constant(element):
                    degree += int(self[element])
            return degree

        for element in self:
            if isinstance(element, Term):
                degree = max(degree, element.degree)
            elif isinstance(element, BaseVariable) and not self.is_constant(element):
                degree = max(degree, 1)
        return degree

    def replace_variables(self, var_dict: dict[BaseVariable, BaseVariable]) -> None:
        for var, replacement_var in var_dict.items():
            if var in self and isinstance(var, Variable):
                if isinstance(replacement_var, Variable):
                    var.update_variable(
                        domain=replacement_var.domain, bounds=replacement_var.bounds, encoding=replacement_var.encoding
                    )
                else:
                    raise ValueError(
                        f"Can't update the variable {var} because the {replacement_var} is not of the same type as {var}"
                    )

    def to_binary(self) -> Term:
        if self.operation not in {Operation.ADD, Operation.MUL}:
            raise ValueError("Can not evaluate any operation that is not Addition of Multiplication")
        out_list: list[BaseVariable | Term | Number] = []
        for e in self:
            if isinstance(e, Term):
                out_list.append(self[e] * e.to_binary())
            elif isinstance(e, BaseVariable):
                if self.is_constant(e):
                    out_list.append(self[e])
                elif isinstance(e, Variable):
                    x = e.encode()
                    if self.operation == Operation.MUL:
                        out_list.append(x ** int(self[e]))
                    else:
                        out_list.append(self[e] * x)
                else:
                    out_list.append(self[e] * e)
            else:
                raise ValueError(f"Evaluating term with elements of type {e.__class__} is not supported.")

        return Term(out_list, self.operation)

    def update_negative_variables_range(self, var_dict: dict[BaseVariable, tuple[float | None, float | None]]) -> None:
        for var, bounds in var_dict.items():
            if var in self and isinstance(var, Variable):
                var.update_variable(domain=var.domain, bounds=bounds, encoding=var.encoding)

    def _apply_operation_on_constants(self, const_list: list[Number]) -> Number:
        out = 0.0 if self.operation in {Operation.ADD, Operation.SUB} else 1.0
        for c in const_list:
            if self.operation is Operation.ADD:
                out += float(c)
            elif self.operation is Operation.SUB:
                out -= float(c)
            elif self.operation is Operation.MUL:
                out *= float(c)
            elif self.operation is Operation.DIV:
                out /= float(c)

        return out

    def variables(self) -> list[BaseVariable]:
        """Returns the unique list of variables in the Term

        Returns:
            list[Variable]: The unique list of variables in the Term.
        """
        var = set()
        for e in self:
            if isinstance(e, BaseVariable) and not self.is_constant(e):
                var.add(e)
            elif isinstance(e, Term):
                var.update(e.variables())
        return list(var)

    def simplify(self) -> Term | BaseVariable:
        if len(self) == 1:
            item = next(iter(self._elements.keys()))
            if self._elements[item] == 1:
                return self._map[item]
        return self

    def pop(self, item: BaseVariable | Term) -> Number:
        return self._elements.pop(hash(item))

    def is_constant(self, variable: BaseVariable) -> bool:
        return variable.compare(self.CONST)

    def to_list(self) -> list[BaseVariable | Term | Number]:
        out_list: list[BaseVariable | Term | Number] = []
        for e in self:
            if isinstance(e, BaseVariable) and self.is_constant(e):
                out_list.append(self[e])
            elif self.operation == Operation.MUL:
                for _ in range(int(self[e])):
                    out_list.append(e)
            else:
                out_list.append(self[e] * e if self[e] != 1 else e)
        return out_list

    def unfold_parentheses(self) -> Term:
        out = copy.copy(self)
        if out.operation != Operation.MUL:
            return out

        parentheses: list[tuple[Term, Number]] = []

        for e in out:
            if isinstance(e, Term) and e.operation == Operation.ADD:
                parentheses.append((copy.copy(e), out[e]))

        for term, _ in parentheses:
            out.pop(term)

        for term, coeff in parentheses:
            if coeff > 1:
                term **= int(coeff)
            final_out = []
            for t in term:
                final_out.append(t * out * term[t])
            out = Term(final_out, Operation.ADD)

        return out

    def remove_zeros(self) -> None:
        to_be_popped = []
        if self.operation == Operation.MUL and self.CONST in self and self[self.CONST] == 0:
            l = len(self)
            for _ in range(l):
                self._elements.popitem()
        for e in self:
            if self[e] == 0:
                to_be_popped.append(hash(e))
        for p in to_be_popped:
            self._elements.pop(p)

    def evaluate(self, var_values: Mapping[BaseVariable, list[int]], precision: float = 1e-2) -> float:
        for var in self.variables():
            if var not in var_values:
                raise ValueError(f"Can not evaluate term because the value of the variable {var} is not provided.")
        output = 0.0 if self.operation in {Operation.ADD, Operation.SUB} else 1.0
        for e in self:
            if isinstance(e, Term):
                output = self._apply_operation_on_constants([output, e.evaluate(var_values, precision) * self[e]])
            elif isinstance(e, BaseVariable):
                if e.compare(self.CONST):
                    output = self._apply_operation_on_constants([output, self[e]])
                elif self.operation == Operation.MUL:
                    output = self._apply_operation_on_constants([output, e.evaluate(var_values[e]) ** self[e]])
                else:
                    output = self._apply_operation_on_constants([output, e.evaluate(var_values[e]) * self[e]])
        return output

    def get_constant(self) -> Number:
        if self.CONST in self:
            return self[self.CONST]
        return 0 if self.operation in {Operation.ADD, Operation.SUB} else 1

    def __copy__(self) -> Term:
        return Term(copy.copy(self.to_list()), self.operation)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "0"
        output_string = ""
        for i, e in enumerate(self):
            if isinstance(e, Term):
                term_str = str(e).strip()
                if len(term_str) > 0:
                    if term_str[0] == "(" and term_str[-1] == ")":
                        term_str = term_str.removeprefix("(").removesuffix(")")
                    output_string += f"({term_str}) " if self[e] == 1 else f"({self[e]}) * ({term_str}) "
            elif isinstance(e, BaseVariable):
                if self.is_constant(e):
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
            if i < len(self) - 1:
                output_string += f"{self.operation.value} "

        return output_string

    __str__ = __repr__

    def __getitem__(self, item: BaseVariable | Term) -> Number:
        return self._elements[hash(item)]

    def __setitem__(self, key: BaseVariable | Term, item: Number) -> None:
        self._map[hash(key)] = key
        self._elements[hash(key)] = item

    def __hash__(self) -> int:
        return hash((frozenset(self._elements.items()), self.operation))

    def __iter__(self) -> Iterator[BaseVariable | Term]:
        for e in self._elements:
            yield self._map[e]

    def __contains__(self, item: BaseVariable | Term) -> bool:
        item_hash = hash(item)
        return any(item_hash == e for e in self._elements)

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
        out.append(other)
        return Term(out, Operation.MUL).unfold_parentheses()

    __imul__ = __mul__

    def __rmul__(self, other: Number | BaseVariable | Term) -> Term:
        out = self.to_list() if self.operation == Operation.MUL else [copy.copy(self)]
        out.insert(0, other)
        return Term(out, Operation.MUL).unfold_parentheses()

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
                out[element] += a
            return out
        raise NotImplementedError(
            "The power operation for terms that are not addition or multiplication is not supported."
        )

    def __lt__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.LT)

    def __le__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.LE)

    def __eq__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:  # type: ignore[override]
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.EQ)

    def __ne__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:  # type: ignore[override]
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.NE)

    def __gt__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.GT)

    def __ge__(self, other: Number | BaseVariable | Term) -> ComparisonTerm:
        return ComparisonTerm(lhs=self, rhs=other, operation=ComparisonOperation.GE)


class ComparisonTerm:
    def __init__(
        self, lhs: Number | BaseVariable | Term, rhs: Number | BaseVariable | Term, operation: ComparisonOperation
    ) -> None:
        term = lhs - rhs
        if not isinstance(term, Term):
            term = Term([term], Operation.ADD)
        const = -1 * term.pop(Term.CONST) if Term.CONST in term else 0
        self._lhs = term
        self._rhs = Term([const], Operation.ADD)
        self._operation = operation

    @property
    def operation(self) -> ComparisonOperation:
        return self._operation

    @property
    def lhs(self) -> Term:
        return self._lhs

    @property
    def rhs(self) -> Term:
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

        return list(var)

    def degree(self) -> int:
        return max(self.rhs.degree, self.lhs.degree)

    def to_list(self) -> list:
        out = self.rhs.to_list()
        out.extend(self.lhs.to_list())
        return out

    def to_binary(self) -> ComparisonTerm:
        return ComparisonTerm(rhs=self.rhs.to_binary(), lhs=self.lhs.to_binary(), operation=self.operation)

    def _apply_comparison_operation(self, v1: Number, v2: Number) -> bool:
        if self.operation is ComparisonOperation.EQ:
            return v1 == v2
        if self.operation is ComparisonOperation.GE:
            return v1 >= v2
        if self.operation is ComparisonOperation.GT:
            return v1 > v2
        if self.operation is ComparisonOperation.LE:
            return v1 <= v2
        if self.operation is ComparisonOperation.LT:
            return v1 < v2
        if self.operation is ComparisonOperation.NE:
            return v1 != v2
        raise ValueError(f"Unsupported Operation of type {self.operation.value}")

    def evaluate(self, var_values: dict[BaseVariable, list[int]], precision: float = 1e-2) -> bool:
        return self._apply_comparison_operation(
            self._lhs.evaluate(var_values, precision), self._rhs.evaluate(var_values, precision)
        )

    def replace_variables(self, var_dict: dict[BaseVariable, BaseVariable]) -> None:
        self.lhs.replace_variables(var_dict)
        self.rhs.replace_variables(var_dict)

    def update_negative_variables_range(self, var_dict: dict[BaseVariable, tuple[float | None, float | None]]) -> None:
        if isinstance(self.lhs, Term):
            self.lhs.update_negative_variables_range(var_dict)
        if isinstance(self.rhs, Term):
            self.rhs.update_negative_variables_range(var_dict)

    def __copy__(self) -> ComparisonTerm:
        return ComparisonTerm(rhs=copy.copy(self.rhs), lhs=copy.copy(self.lhs), operation=self.operation)

    def __repr__(self) -> str:
        return f"{self.lhs!s} {self.operation.value} {self.rhs!s}"

    __str__ = __repr__

    def __bool__(self) -> bool:
        if self.rhs.degree == self.lhs.degree == 0:
            return self.rhs.get_constant() == self.lhs.get_constant()
        raise TypeError(
            "Symbolic Constraint Term objects do not have an inherent truth value. "
            "Use a method like .evaluate() to obtain a Boolean value."
        )
