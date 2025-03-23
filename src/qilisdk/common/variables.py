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
from abc import ABC
from enum import Enum
from math import prod

import numpy as np

# Utils ###
Number = int | float


def _check_and_convert(c: object) -> Variable | Term:
    if isinstance(c, (Variable, Term)):
        return c
    if isinstance(c, (float, int)) or (isinstance(c, np.generic) and np.issubdtype(c, np.number)):
        return ConstantVar(c)
    raise ValueError(f"{c} type is not supported.")


def _multiply_operations(op1: Operation, op2: Operation) -> Operation:
    if op1 not in {Operation.ADD, Operation.SUB} or op2 not in {Operation.ADD, Operation.SUB}:
        raise ValueError("only subtraction and additions are supported")
    if op1 is Operation.SUB and op2 is Operation.SUB:
        return Operation.ADD
    if op1 is Operation.SUB or op2 is Operation.SUB:
        return Operation.SUB
    return Operation.ADD


def _apply_operation(v1: Variable, op: Operation, v2: Variable) -> Term:
    if op is Operation.ADD:
        return v1 + v2
    if op is Operation.SUB:
        return v1 - v2
    if op is Operation.MUL:
        return v1 * v2
    if op is Operation.DIV:
        return v1 / v2
    return None


def apply_comparison(v1, op, v2):
    if op is ComparisonOperators.EQ:
        return v1 == v2
    if op is ComparisonOperators.GE:
        return v1 >= v2
    if op is ComparisonOperators.GT:
        return v1 > v2
    if op is ComparisonOperators.LE:
        return v1 <= v2
    if op is ComparisonOperators.LT:
        return v1 < v2
    if op is ComparisonOperators.NE:
        return v1 != v2
    return None


def apply_operation_on_constants(const_list: list[tuple[Operation, int, ConstantVar]], operation: Operation):
    total_const = None  # 0 if (operation is Operation.SUB or operation is Operation.ADD) else 1
    min_i = 10000

    for op, i, con in const_list:
        v = con.value
        # if op in [Operation.ADD, Operation.SUB]:
        #     v = apply_operation(0, op, con.value)
        if total_const is None:
            total_const = 0 if (op is Operation.SUB or op is Operation.ADD) else 1
        total_const = _apply_operation(total_const, op, v)
        min_i = min(min_i, i)
    if operation is Operation.SUB and min_i > 0:
        total_const *= -1
    return min_i, ConstantVar(total_const)


def compare_vars(v1: "Variable", v2: "Variable"):
    if v1.label != v2.label:
        return False
    if v1.__class__ is not v2.__class__:
        return False
    if v1.bounds != v2.bounds:
        return False
    if isinstance(v1, ContinuousVar) and v1.encoding is not v2.encoding:
        return False
    return not (isinstance(v1, ConstantVar) and v1.value is not v2.value)


class Side(Enum):
    RIGHT = "right"
    LEFT = "left"


class Domain(str, Enum):
    Integer = "Integer Domain"
    PositiveInteger = "Positive Integer Domain"
    Real = "Real Domain"
    Binary = "Binary Domain"
    Spin = "Spin Domain"

    def check_value(self, value: float) -> bool:
        if self == Domain.Binary:
            return isinstance(value, int) and value in {0, 1}
        if self == Domain.Spin:
            return isinstance(value, int) and value in {-1, 1}
        if self == Domain.Real:
            return isinstance(value, (int, float))
        if self == Domain.Integer:
            return isinstance(value, int)
        if self == Domain.PositiveInteger:
            return isinstance(value, int) and value >= 0
        return False

    def min(self) -> float:
        if self in {Domain.Binary, Domain.PositiveInteger}:
            return 0
        if self == Domain.Spin:
            return -1
        return -1e30

    def max(self) -> float:
        if self in {Domain.Binary, Domain.Spin}:
            return 1
        return 1e30


class Operation(Enum):
    MUL = "*"
    ADD = "+"
    DIV = "/"
    SUB = "-"


class ComparisonOperators(Enum):
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="


# Encoding ###


class Encoding(ABC):
    __name = ""

    @property
    def name(self):
        return self.__name

    @staticmethod
    def encode(var: "ContinuousVar", precision: int = 100) -> "Term":
        """Given a continuous variable return a Term that only consists of
            binary variables that represent the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): The continuous variable to be encoded
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            Term: a term that only contains binary variables
        """
        raise NotImplementedError("This is an abstract class and is not meant to be executed.")

    @staticmethod
    def encoding_constraint(var: "ContinuousVar", precision: int = 100) -> ConstraintTerm | None:
        """Given a continuous variable return a Constraint Term that ensures that the encoding is respected.

        Args:
            var (ContinuousVar): The continuous variable to be encoded
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            Constraint Term: a constraint term that ensures the encoding is respected.
        """
        raise NotImplementedError("This is an abstract class and is not meant to be executed.")

    @staticmethod
    def evaluate(var: "ContinuousVar", binary_list: list[int], precision: int = 100) -> float:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """
        raise NotImplementedError("This is an abstract class and is not meant to be executed.")

    @staticmethod
    def num_binary_equivalent(var: "ContinuousVar", precision: int = 100) -> int:
        """Give a continuous variable return the number of binary variables needed to encode it.

        Args:
            var (ContinuousVar): the continuous variable.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

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
    def term_equals_to(var: "ContinuousVar", number: int, precision: int = 100):
        """returns a term that is 1 if the variable is equal to the number, else 0.

        Args:
            var (ContinuousVar): the continuous variable.
            number (int): the number to equate the variable to.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This is an abstract class and is not meant to be executed.")


class HOBO(Encoding):
    __name = "HOBO"

    @property
    def name(self):
        return self.__name

    @staticmethod
    def encode(var: "ContinuousVar", precision: int = 100) -> Term:
        """Given a continuous variable return a Term that only consists of
            binary variables that represent the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): The continuous variable to be encoded
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            Term: a term that only contains binary variables
        """
        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        abs_bound = np.abs(bounds[1] - bounds[0])
        n_binary = int(np.floor(np.log2(abs_bound if abs_bound != 0 else 1)))

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary + 1)]

        term = sum(2**i * binary_vars[i] for i in range(n_binary))

        term += (np.abs(bounds[1] - bounds[0]) + 1 - 2**n_binary) * binary_vars[-1]

        term += bounds[0]

        return term, binary_vars

    @staticmethod
    def evaluate(var: "ContinuousVar", binary_list: list[int], precision: int = 100) -> float:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """
        if not HOBO.check_valid(binary_list=binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the HOBO encoding.")
        term, binary_var = HOBO.encode(var)

        if len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict = {binary_var[i].label: binary_list[i] for i in range(len(binary_list))}

        if var.domain is RealDomain:
            term *= 1 / precision

        out = term.evaluate(binary_dict)

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )
        return out

    @staticmethod
    def encoding_constraint(var: "ContinuousVar", precision: int = 100) -> ConstraintTerm | None:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """
        return None

    @staticmethod
    def num_binary_equivalent(var: "ContinuousVar", precision: int = 100) -> int:
        """Give a continuous variable return the number of binary variables needed to encode it.

        Args:
            var (ContinuousVar): the continuous variable.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            int: the number of binary variables needed to encode it.
        """

        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.floor(np.log2(np.abs(bounds[1] - bounds[0]))))

        return n_binary + 1

    @staticmethod
    def check_valid(binary_list: list[int]) -> tuple[bool, int]:
        return True, 0

    @staticmethod
    def term_equals_to(var: "ContinuousVar", number: int, precision: int = 100):
        """returns a term that is 1 if the variable is equal to the number, else 0.

        Args:
            var (ContinuousVar): the continuous variable.
            number (int): the number to equate the variable to.

        Raises:
            NotImplementedError: _description_
        """

        encoded_num = []

        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        abs_bound = np.abs(bounds[1] - bounds[0])
        n_binary = int(np.floor(np.log2(abs_bound if abs_bound != 0 else 1)))

        aux_num = number

        aux_num -= bounds[0]

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

        return prod(var[i] if encoded_num[i] == 0 else (1 - var[i]) for i in range(var.num_binary_equivalent()))


class one_hot(Encoding):
    __name = "ONE HOT"

    @property
    def name(self):
        return self.__name

    @staticmethod
    def encode(var: "ContinuousVar", precision: int = 100) -> "Term":
        """Given a continuous variable return a Term that only consists of
            binary variables that represent the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): The continuous variable to be encoded
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            Term: a term that only contains binary variables
        """

        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]

        term = sum((bounds[0] + i) * binary_vars[i] for i in range(n_binary))

        # term += bounds[0]

        return term, binary_vars

    @staticmethod
    def evaluate(var: "ContinuousVar", binary_list: list[int], precision: int = 100) -> float:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """
        if not one_hot.check_valid(binary_list=binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the one hot encoding.")

        term, binary_var = one_hot.encode(var)

        if len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict = {binary_var[i].label: binary_list[i] for i in range(len(binary_list))}

        if var.domain is RealDomain:
            term *= 1 / precision

        out = term.evaluate(binary_dict)

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )

        return out

    @staticmethod
    def encoding_constraint(var: "ContinuousVar", precision: int = 100) -> ConstraintTerm | None:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """

        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]
        return sum(binary_vars) == 1

    @staticmethod
    def num_binary_equivalent(var: "ContinuousVar", precision: int = 100) -> int:
        """Give a continuous variable return the number of binary variables needed to encode it.

        Args:
            var (ContinuousVar): the continuous variable.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            int: the number of binary variables needed to encode it.
        """

        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        return n_binary

    @staticmethod
    def check_valid(binary_list: list[int]) -> tuple[bool, int]:
        num_ones = binary_list.count(1)
        return num_ones == 1, (num_ones - 1) ** 2

    @staticmethod
    def term_equals_to(var: "ContinuousVar", number: int, precision: int = 100):
        """returns a term that is 1 if the variable is equal to the number, else 0.

        Args:
            var (ContinuousVar): the continuous variable.
            number (int): the number to equate the variable to.

        Raises:
            NotImplementedError: _description_
        """

        if var.domain is RealDomain:
            return prod((1 - var[i]) if i != number * precision else var[i] for i in range(var.num_binary_equivalent()))
        return prod((1 - var[i]) if i != number else var[i] for i in range(var.num_binary_equivalent()))


class domain_wall(Encoding):
    __name = "Domain Wall"

    @property
    def name(self):
        return self.__name

    @staticmethod
    def encode(var: "ContinuousVar", precision: int = 100) -> "Term":
        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]

        term = sum(binary_vars[i] for i in range(n_binary))

        term += bounds[0]

        return term, binary_vars

    @staticmethod
    def evaluate(var: "ContinuousVar", binary_list: list[int], precision: int = 100) -> float:
        if not domain_wall.check_valid(binary_list=binary_list)[0]:
            raise ValueError(f"invalid binary string {binary_list} with the domain wall encoding.")
        term, binary_var = domain_wall.encode(var)

        if len(binary_list) != len(binary_var):
            raise ValueError(f"expected {len(binary_var)} variables but received {len(binary_list)}")

        binary_dict = {binary_var[i].label: binary_list[i] for i in range(len(binary_list))}

        if var.domain is RealDomain:
            term *= 1 / precision

        out = term.evaluate(binary_dict)

        if not var.domain.check_value(out):
            raise ValueError(
                f"The value {out} violates the domain {var.domain.__class__.__name__} of the variable {var}"
            )
        return out

    @staticmethod
    def encoding_constraint(var: "ContinuousVar", precision: int = 100) -> ConstraintTerm | None:
        """Given a binary string, evaluate the value of the continuous variable in the given encoding.

        Args:
            var (ContinuousVar): the variable to be evaluated
            binary_list (list[int]): a list of binary values.
            precision (int): the precision to be considered for real variables (Only applies if the variable domain is RealDomain)

        Returns:
            float: the value of the continuous variable given the specified binary values.
        """

        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.abs(bounds[1] - bounds[0])) + 1

        binary_vars = [BinaryVar(var.label + f"({i})") for i in range(n_binary)]
        return sum(binary_vars[i + 1] * (1 - binary_vars[i]) for i in range(len(binary_vars) - 1)) == 0

    @staticmethod
    def num_binary_equivalent(var: "ContinuousVar", precision: int = 100) -> int:
        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        n_binary = int(np.abs(bounds[1] - bounds[0]))

        return n_binary

    @staticmethod
    def check_valid(binary_list: list[int]) -> tuple[bool, int]:
        value = sum(binary_list[i + 1] * (1 - binary_list[i]) for i in range(len(binary_list) - 1))
        return value == 0, value

    @staticmethod
    def term_equals_to(var: "ContinuousVar", number: int, precision: int = 100):
        """returns a term that is 1 if the variable is equal to the number, else 0.

        Args:
            var (ContinuousVar): the continuous variable.
            number (int): the number to equate the variable to.
        """

        encoded_num = []
        aux_number = number

        for i in range(var.num_binary_equivalent()):
            if i <= number:
                encoded_num.append(1)
            else:
                encoded_num.append(0)

        bounds = var.bounds
        if var.domain is RealDomain:
            bounds = (precision * bounds[0], precision * bounds[1])

        aux_number -= bounds[0]

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

        return prod(var[i] if encoded_num[i] == 0 else (1 - var[i]) for i in range(var.num_binary_equivalent()))


# Variables ###


class Variable:
    """This class represents the general structure of any variable that can be included in the model."""

    def __init__(self, label: str, domain: Domain, bounds: tuple[float | None, float | None] = (None, None)) -> None:
        self._label = label
        self._domain = domain

        lower_bound, upper_bound = bounds
        if lower_bound is None:
            lower_bound = domain.min()
        if upper_bound is None:
            upper_bound = domain.max()
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

    def set_bounds(self, bounds: tuple[float, float]) -> None:
        if not self.domain.check_value(bounds[0]):
            raise ValueError(
                f"the lower bound ({bounds[0]}) does not respect the domain of the variable ({self.domain})"
            )
        if not self.domain.check_value(bounds[1]):
            raise ValueError(
                f"the upper bound ({bounds[1]}) does not respect the domain of the variable ({self.domain})"
            )
        if bounds[0] > bounds[1]:
            raise ValueError(f"the lower bound ({bounds[0]}) should not be greater than the upper bound ({bounds[1]})")
        self._bounds = bounds

    def num_binary_equivalent(self):
        raise NotImplementedError

    def variables(self):  # noqa: ANN201
        # TODO (ameer): Fix the problem in model line 800 instead of having this method.
        yield self

    def replace_variables(self, var_dict):
        """Replaces the information of the variable with those coming from the dictionary
        if the variable label is in the dictionary

        Args:
            var_dict (dict): A dictionary that holds the labels of the variables to be
                             changed alongside the new values they should take
        """
        if self.label in var_dict.keys():
            self._domain = var_dict[self.label].domain
            self.bounds = var_dict[self.label].bounds

    def __copy__(self) -> Variable:
        return Variable(label=self.label, domain=self.domain)

    def __repr__(self) -> str:
        return f"{self._label}"

    def __str__(self) -> str:
        return f"{self._label}"

    def __add__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Term):
            return other + self

        if isinstance(other, Number):
            other = ConstantVar(other)

        out = Term(elements=[self, other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __mul__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Term):
            return other * self

        if isinstance(other, Number):
            other = ConstantVar(other)

        return Term(elements=[self, other], operation=Operation.MUL)

    def __truediv__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Number):
            other = ConstantVar(other)

        if isinstance(other, ConstantVar):
            # if other.value == 1:
            #     return self
            if other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other

        return Term(elements=[self, other], operation=Operation.DIV)

    def __sub__(self, other: Number | Variable | Term):
        if isinstance(other, Number):
            other = ConstantVar(other)

        out = Term(elements=[self, -1 * other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __iadd__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Term):
            return other + self

        if isinstance(other, Number):
            other = ConstantVar(other)

        # if isinstance(other, ConstantVar) and other.value == 0:
        #     return self
        out = Term(elements=[self, other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __imul__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Term):
            return other * self

        if isinstance(other, Number):
            other = ConstantVar(other)

        if isinstance(other, ConstantVar):
            # if other.value == 1:
            #     return self
            # if other.value == 0:
            #     return 0
            return Term(elements=[other, self], operation=Operation.MUL)

        return Term(elements=[self, other], operation=Operation.MUL)

    def __itruediv__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Number):
            other = ConstantVar(other)

        if isinstance(other, ConstantVar):
            # if other.value == 1:
            #     return self
            if other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other

        return Term(elements=[self, other], operation=Operation.DIV)

    def __isub__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Number):
            other = ConstantVar(other)

        # if isinstance(other, ConstantVar):
        #     if other.value == 0:
        #         return self

        out = Term(elements=[self, -1 * other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __radd__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Term):
            return other + self
        if isinstance(other, Number):
            other = ConstantVar(other)

        # if isinstance(other, ConstantVar):
        #     if other.value == 0:
        #         return self

        out = Term(elements=[other, self], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __rmul__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Term):
            return other * self

        if isinstance(other, Number):
            other = ConstantVar(other)

        # if isinstance(other, ConstantVar):
        #     if other.value == 1:
        #         return self
        #     if other.value == 0:
        #         return 0

        return Term(elements=[other, self], operation=Operation.MUL)

    def __rtruediv__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Number):
            other = ConstantVar(other)
        return Term(elements=[other, self], operation=Operation.DIV)

    def __rfloordiv__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Number):
            other = ConstantVar(other)
        return Term(elements=[other, self], operation=Operation.DIV)

    def __rsub__(self, other: Number | Variable | Term) -> Term:
        if isinstance(other, Number):
            other = ConstantVar(other)
        # if isinstance(other, ConstantVar):
        #     if other.value == 0:
        #         return self

        out = Term(elements=[other, -1 * self], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __pow__(self, a):
        if not isinstance(a, int):
            raise ValueError("only integer powers are allowed.")

        out = copy.copy(self)

        for _ in range(a - 1):
            out *= out
        return out

    def __lt__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.LT)

    def __le__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.LE)

    def __eq__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.EQ)

    def __ne__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.NE)

    def __gt__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.GT)

    def __ge__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.GE)


class BinaryVar(Variable):
    def __init__(self, label: str) -> None:
        super().__init__(label=label, domain=Domain.Binary)

    def num_binary_equivalent(self):
        return 1

    def evaluate(self, binary_list, precision=100):
        return binary_list[self.label]

    def __copy__(self):
        return BinaryVar(label=self.label)

    # def __mul__(self, other: Number | Variable | Term) -> Term:
    #     if isinstance(other, BinaryVar) and compare_vars(self, other):
    #         return self
    #     return super().__mul__(other)

    # def __imul__(self, other: Number | Variable | Term) -> Term:
    #     if isinstance(other, BinaryVar):
    #         if compare_vars(self, other):
    #             return self
    #     return super().__imul__(other)

    # def __rmul__(self, other):
    #     if isinstance(other, BinaryVar):
    #         if compare_vars(self, other):
    #             return self
    #     return super().__rmul__(other)


class SpinVar(Variable):
    def __init__(self, label: str):
        super().__init__(label=label, domain=Domain.Spin, bounds=(-1, 1))

    def num_binary_equivalent(self):
        return 1


# class ContinuousVar(Variable):
#     def __init__(self, label: str, domain: Domain, bounds: tuple[int, int], encoding: Encoding = HOBO):
#         super().__init__(label=label, domain=domain, bounds=bounds)

#         if not issubclass(encoding, Encoding):
#             raise ValueError("only encodings specified by the Encoding class are allowed.")
#         self._encoding = encoding

#         self.__term, self.__bin_vars = self.encode()

#     @property
#     def encoding(self):
#         return self._encoding

#     @property
#     def term(self):
#         return self.__term

#     @property
#     def bin_vars(self):
#         return self.__bin_vars

#     def __copy__(self):
#         return ContinuousVar(label=self.label, domain=self.domain, bounds=self.bounds, encoding=self._encoding)

#     def __getitem__(self, item):
#         return self.__bin_vars[item]

#     def evaluate(self, binary_list, precision=100):
#         return self.encoding.evaluate(self, binary_list, precision=precision)

#     def encode(self, precision=100):
#         self.__term, self.__bin_vars = self.encoding.encode(self, precision=precision)
#         return self.__term, self.__bin_vars

#     def num_binary_equivalent(self, precision=100):
#         return self.encoding.num_binary_equivalent(self, precision=precision)

#     def check_valid(self, binary_list):
#         return self.encoding.check_valid(binary_list)

#     def encoding_constraint(self, precision: int = 100) -> ConstraintTerm | None:
#         return self.encoding.encoding_constraint(self, precision=precision)

#     def term_equals_to(self, number: int, precision: int = 100):
#         return self.encoding.term_equals_to(self, number, precision)


class ConstantVar(Variable):
    def __init__(self, value: Number):
        super().__init__(label=str(value), domain=Domain.Real)
        self._value = value

    @property
    def value(self):
        return self._value

    def num_binary_equivalent(self):
        return 0

    def __copy__(self):
        return ConstantVar(self.value)

    def __repr__(self):
        return f"{np.round(self.value, 5)}" if self.value >= 0 else f"({np.round(self.value, 5)})"

    def __str__(self):
        return f"{np.round(self.value, 5)}" if self.value >= 0 else f"({np.round(self.value, 5)})"

    # def __add__(self, other):
    #     other = _check_and_convert(other)

    #     if isinstance(other, ConstantVar):
    #         if other.value == 0:
    #             return self
    #         val = self.value + other.value
    #         return ConstantVar(val)

    #     return super().__add__(other)

    # def __mul__(self, other):
    #     other = _check_and_convert(other)

    #     if isinstance(other, ConstantVar):
    #         if other.value == 1:
    #             return self
    #         if other.value == 0:
    #             return 0
    #         val = self.value * other.value
    #         return ConstantVar(val)
    #     if self.value == 1:
    #         return other
    #     if self.value == 0:
    #         return 0
    #     return super().__mul__(other)

    # def __truediv__(self, other):
    #     if isinstance(other, ConstantVar):
    #         val = self.value / other.value
    #         return ConstantVar(val)
    #     return super().__truediv__(other)

    # def __sub__(self, other):
    #     other = _check_and_convert(other)
    #     if isinstance(other, ConstantVar):
    #         if other.value == 0:
    #             return self
    #         val = self.value - other.value
    #         return ConstantVar(val)
    #     return super().__sub__(other)

    # def __iadd__(self, other):
    #     other = _check_and_convert(other)

    #     if isinstance(other, ConstantVar):
    #         if other.value == 0:
    #             return self
    #         val = self.value + other.value
    #         return ConstantVar(val)
    #     if self.value == 0:
    #         return other
    #     return super().__iadd__(other)

    # def __imul__(self, other):
    #     other = _check_and_convert(other)

    #     if isinstance(other, ConstantVar):
    #         if other.value == 1:
    #             return self
    #         if other.value == 0:
    #             return 0
    #         val = self.value * other.value
    #         return ConstantVar(val)
    #     if self.value == 1:
    #         return other
    #     if self.value == 0:
    #         return 0
    #     return super().__imul__(other)

    # def __itruediv__(self, other):
    #     if isinstance(other, ConstantVar):
    #         val = self.value / other.value
    #         return ConstantVar(val)
    #     return super().__itruediv__(other)

    # def __isub__(self, other):
    #     other = _check_and_convert(other)
    #     if isinstance(other, ConstantVar):
    #         if other.value == 0:
    #             return self
    #         val = self.value - other.value
    #         return ConstantVar(val)
    #     return super().__isub__(other)

    # def __radd__(self, other):
    #     other = _check_and_convert(other)

    #     if isinstance(other, ConstantVar):
    #         if other.value == 0:
    #             return self
    #         val = other.value + self.value
    #         return ConstantVar(val)

    #     return super().__radd__(other)

    # def __rmul__(self, other):
    #     other = _check_and_convert(other)

    #     if isinstance(other, ConstantVar):
    #         if other.value == 1:
    #             return self
    #         if other.value == 0:
    #             return 0
    #         val = other.value * self.value
    #         return ConstantVar(val)
    #     if self.value == 1:
    #         return other
    #     if self.value == 0:
    #         return 0
    #     return super().__rmul__(other)

    # def __rtruediv__(self, other):
    #     if isinstance(other, ConstantVar):
    #         val = other.value / self.value
    #         return ConstantVar(val)

    #     return super().__rtruediv__(other)

    # def __rfloordiv__(self, other):
    #     if isinstance(other, ConstantVar):
    #         val = other.value / self.value
    #         return ConstantVar(val)
    #     return super().__rfloordiv__(other)

    # def __rsub__(self, other):
    #     other = _check_and_convert(other)
    #     if isinstance(other, ConstantVar):
    #         if other.value == 0:
    #             return self
    #         val = other.value - self.value
    #         return ConstantVar(val)
    #     return super().__rsub__(other)


# Terms ###


class Term:
    def __init__(self, elements: list[Variable | Term], operation: Operation) -> None:
        self._elements = list(elements)
        self._operation = operation

    @property
    def elements(self):
        return self._elements

    @property
    def operation(self):
        return self._operation

    def variables(self):
        for e in self.elements:
            if isinstance(e, ConstantVar):
                pass
            elif isinstance(e, Variable):
                yield e
            else:
                yield from e.variables()

    def to_list(self):
        output = []
        if len(self.elements) > 1:
            for i in range(len(self.elements) - 1):
                if isinstance(self.elements[i], Term):
                    output.append(self.elements[i].to_list())
                else:
                    output.append(self.elements[i])
                output.append(self.operation)

            if isinstance(self.elements[-1], Term):
                output.append(self.elements[-1].to_list())
            else:
                output.append(self.elements[-1])
        else:
            output = [self.elements[0]] if len(self.elements) > 0 else []
        return output

    def degree(self):
        if self.operation is Operation.MUL:
            degree = 0
            for t in self.elements:
                if isinstance(t, ConstantVar):
                    continue
                if isinstance(t, Variable):
                    degree += 1

        elif self.operation is Operation.ADD:
            degree = 0
            for t in self.elements:
                if isinstance(t, Term):
                    aux_degree = t.degree()
                    degree = max(degree, aux_degree)

                elif isinstance(t, ConstantVar):
                    continue
                elif isinstance(t, Variable):
                    degree = max(1, degree)
        return degree

    def replace_variables(self, var_dict):
        for i, _ in enumerate(self.elements):
            self.elements[i].replace_variables(var_dict)

    def evaluate(self, var_values, precision=100):
        if self.operation not in {Operation.ADD, Operation.MUL}:
            raise ValueError("Can not evaluate any operation that is not Addition of Multiplication")
        out = 0 if self.operation is Operation.ADD else 1
        for e in self.elements:
            if isinstance(e, Term):
                out = _apply_operation(out, self.operation, e.evaluate(var_values))
            elif isinstance(e, ConstantVar):
                out = _apply_operation(out, self.operation, e.value)
            elif isinstance(e, Variable):
                v = var_values[e.label]
                if isinstance(v, list):
                    if isinstance(e, ContinuousVar):
                        out = _apply_operation(out, self.operation, e.evaluate(v, precision=precision))
                    else:
                        raise ValueError("Providing a list of values for the a Binary variable.")
                else:
                    if not e.domain.check_value(v):
                        raise ValueError(
                            f"value {v} doesn't respect the domain of the variable {e} ({e.domain.__class__.__name__()})"
                        )
                    out = _apply_operation(out, self.operation, v)
            else:
                raise ValueError(f"Evaluating term with elements of type {e.__class__} is not supported.")

        return out

    def to_binary(self):
        """
        Replaces the variable in the expression for the term indicated in the dictionary
        """

        if self.operation not in [Operation.ADD, Operation.MUL]:
            raise ValueError("Can not evaluate any operation that is not Addition of Multiplication")
        out = 0 if self.operation is Operation.ADD else 1
        for e in self.elements:
            if isinstance(e, Term):
                out = _apply_operation(out, self.operation, e.to_binary())
            elif isinstance(e, ConstantVar):
                out = _apply_operation(out, self.operation, e.value)
            elif isinstance(e, Variable):
                if isinstance(e, ContinuousVar):
                    x, _ = e.encode()
                    out = _apply_operation(out, self.operation, x)
                else:
                    out = _apply_operation(out, self.operation, e)
            else:
                raise ValueError(f"Evaluating term with elements of type {e.__class__} is not supported.")

        return out

    def update_variables_precision(self, var_dict: dict):
        """
        var_dict = {'<var_name>' : {'var' : <var>, 'precision': #number}}

        Args:
            var_dict (dict): _description_
        """

        if self.operation is Operation.ADD:
            for i, e in enumerate(self.elements):
                if isinstance(e, Term):
                    if e.operation is Operation.MUL:
                        for var in e.variables():
                            if var.label in var_dict:
                                e /= var_dict[var.label]["precision"]
                                self.elements[i] = e
                elif isinstance(e, Variable):
                    if e.label in var_dict:
                        e /= var_dict[e.label]["precision"]
                        self.elements[i] = e
        elif self.operation is Operation.MUL:
            exp = copy.copy(self)
            for var in self.variables():
                if var.label in var_dict:
                    exp /= var_dict[var.label]["precision"]
                    self._elements = exp.elements

    def update_negative_variables_range(self, var_dict: dict):
        """
        var_dict = {'<var_name>' : {'var' : <var>, 'precision': #number, 'original_bounds': (#number, #number)}}

        Args:
            var_dict (dict): _description_
        """
        out = copy.copy(self)

        if out.operation is Operation.ADD:
            for i, e in enumerate(out.elements):
                if isinstance(e, Term):
                    if e.operation is Operation.MUL:
                        for var in e.variables():
                            if var.label in var_dict:
                                out += var_dict[var.label]["precision"]
                elif isinstance(e, Variable):
                    if e.label in var_dict:
                        out += var_dict[e.label]["precision"]
        self._elements = out.elements

    def collect_constants(self):
        constants = []
        pop_list = []
        for i, e in enumerate(self.elements):
            if isinstance(e, Term):
                if e.operation in [Operation.ADD, Operation.SUB] and self.operation in [Operation.ADD, Operation.SUB]:
                    constants.extend(e.collect_constants())
            elif isinstance(e, ConstantVar):
                constants.append((self.operation if i != 0 else Operation.ADD, i, e))
                pop_list.append(i)

        for i in sorted(pop_list, reverse=True):
            self.elements.pop(i)
        return constants

    def simplify_constants(self, maintain_index: bool = False) -> Term:
        constants = []
        pop_list = []
        out = Term(elements=self.elements, operation=self.operation)
        for i, e in enumerate(out.elements):
            # collect constants
            if isinstance(e, Term):
                if e.operation in {Operation.ADD, Operation.SUB} and out.operation in {Operation.ADD, Operation.SUB}:
                    constants.extend(e.collect_constants())
                else:
                    out.elements[i] = out.elements[i].simplify_constants()
                    if isinstance(out.elements[i], ConstantVar):
                        constants.append((out.operation if i != 0 else Operation.ADD, i, e))
                        pop_list.append(i)
            elif isinstance(e, ConstantVar):
                constants.append((out.operation if i != 0 else Operation.ADD, i, e))
                pop_list.append(i)

        # remove constants from the term
        for i in sorted(pop_list, reverse=True):
            out.elements.pop(i)

        # operate on the constants
        i, out_const = apply_operation_on_constants(constants, out.operation)
        if out_const.value == 0:
            if out.operation in {Operation.ADD, Operation.SUB}:
                # if len(out.elements) == 1:
                #     return out.elements[0]
                return out
            if out.operation == Operation.MUL:
                return 0
        elif out_const.value == 1:
            if out.operation == Operation.MUL:
                return out

        if len(constants) > 0:
            out.elements.insert(i if maintain_index else 0, out_const)

        # if len(out.elements) == 1:
        #     return out.elements[0]

        return out

    def parse_parentheses(self, parent_operation=Operation.ADD):
        parsed_list = []
        op = _multiply_operations(self.operation, parent_operation)
        for i, element in enumerate(self.elements):
            if isinstance(element, Term) and element.operation in [Operation.ADD, Operation.SUB]:
                if i == 0:
                    parsed_list.extend(element.parse_parentheses())
                else:
                    parsed_list.extend(element.parse_parentheses(parent_operation=op))

            elif i == 0:
                parsed_list.append((Operation.ADD, element))
            else:
                parsed_list.append((op, element))
        return parsed_list

    def unfold_parentheses(self, other):
        self_elements = []
        self_is_parentheses = False
        other_elements = []
        other_is_parentheses = False

        if self.operation in [Operation.ADD, Operation.SUB]:
            self_elements = self.parse_parentheses()
            self_is_parentheses = True
        else:
            self_elements = [self]
        if isinstance(other, Term):
            if other.operation in [Operation.ADD, Operation.SUB]:
                other_elements = other.parse_parentheses()
                other_is_parentheses = True
            else:
                other_elements = [other]
        else:
            other_elements = [other]

        output = 0
        if self_is_parentheses and other_is_parentheses:
            # two parentheses
            for op1, el1 in self_elements:
                for op2, el2 in other_elements:
                    output = _apply_operation((output), _multiply_operations(op1, op2), (el1 * el2))
        elif self_is_parentheses:
            for op1, el1 in self_elements:
                for mul in other_elements:
                    output = _apply_operation((output), op1, (el1 * mul))
        elif other_is_parentheses:
            for op2, el2 in other_elements:
                for mul in self_elements:
                    output = _apply_operation((output), op2, (el2 * mul))

        return output

    def create_hashable_term_name(self):
        """
        Assumptions:
            1. the operation is a multiplication
        """
        if self.operation is not Operation.MUL:
            raise ValueError(f"only terms with operation = {Operation.MUL.name} are allowed to be hashed")

        coeff = 1
        var_list = []

        for t in self.elements:
            if isinstance(t, ConstantVar):
                coeff = t.value
            elif isinstance(t, Variable):
                var_list.append(t)

        var_list = sorted(var_list, key=lambda v: v.label, reverse=False)
        hash_name = ""
        for i in var_list:
            hash_name += i.label

        return hash_name, coeff, var_list

    def simplify_variable_coefficients(self) -> Term:
        """
        Assumptions:
          1. The operation is an addition
          2. only takes into account terms that are from the form: coefficient * variables
        """
        if self.operation is None or self.operation not in {Operation.ADD, Operation.SUB}:
            return self

        out = Term(self.elements, self.operation)

        hash_list = {}
        pop_list = []
        for i, e in enumerate(out.elements):
            if isinstance(e, Term) and e.operation is Operation.MUL:
                name, coeff, var_list = e.create_hashable_term_name()
                if name not in hash_list:
                    hash_list[name] = [coeff, var_list]
                else:
                    hash_list[name][0] += coeff
                pop_list.append(i)
            elif isinstance(e, Variable) and not isinstance(e, ConstantVar):
                if e.label not in hash_list:
                    hash_list[e.label] = [1, [e]]
                else:
                    hash_list[e.label][0] += 1
                pop_list.append(i)
        pop_list = sorted(pop_list, reverse=True)

        for i in pop_list:
            out.elements.pop(i)

        for k, v in hash_list.items():
            term = _check_and_convert(v[0])
            for var in v[1]:
                term *= var

            term = _check_and_convert(term)
            out.elements.append(term)

        return out

    def simplify_binary(self):
        var_dict = {}
        for e in self.elements:
            if e.label not in var_dict:
                var_dict[e.label] = []

            var_dict[e.label].append(e)

        out = 1
        for k, v in var_dict.items():
            if isinstance(v[0], BinaryVar):
                out *= v[0]
            else:
                for e in v:
                    out *= e
        return out

    def simplify(self):
        out = Term(self.elements, self.operation)
        if self.operation is Operation.ADD:
            out = out.simplify_constants()
            if isinstance(out, Term) and out.operation is Operation.ADD:
                out = out.simplify_variable_coefficients()
            if isinstance(out, Term):
                for element in out.elements:
                    if isinstance(element, Term):
                        element = element.simplify()
        if self.operation is Operation.MUL:
            out = out.simplify_binary()
            if isinstance(out, Term):
                out = out.simplify_constants()

        return out

    def __copy__(self):
        elements = []
        for e in self.elements:
            elements.append(copy.copy(e))

        return Term(elements=elements, operation=self.operation)

    def __repr__(self):
        output_string = ""
        if len(self.elements) > 1:
            for i in range(len(self.elements) - 1):
                if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                    self.elements[i], Term
                ):
                    output_string += f"({self.elements[i]})"
                else:
                    output_string += f"{self.elements[i]}"
                output_string += f"{self.operation.value}"
            if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                self.elements[-1], Term
            ):
                output_string += f"({self.elements[-1]})"
            else:
                output_string += f"{self.elements[-1]}"
        else:
            output_string = f"{self.elements[0]}" if len(self.elements) > 0 else ""
        return output_string
        # if (self.operation is Operation.MUL and (isinstance(self.lhs, Term) or isinstance(self.rhs, Term))) or (
        #     self.operation is Operation.DIV and (isinstance(self.lhs, Term) or isinstance(self.rhs, Term))
        # ):
        #     return f"({str(self.lhs)}) {self.operation.value} ({str(self.rhs)})"
        # return f"{str(self.lhs)} {self.operation.value} {str(self.rhs)}"

    def __str__(self):
        output_string = ""
        if len(self.elements) > 1:
            for i in range(len(self.elements) - 1):
                if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                    self.elements[i], Term
                ):
                    output_string += f"({self.elements[i]})"
                else:
                    output_string += f"{self.elements[i]}"

                output_string += f"{self.operation.value}"

            if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                self.elements[-1], Term
            ):
                output_string += f"({self.elements[-1]})"
            else:
                output_string += f"{self.elements[-1]}"
        else:
            output_string = f"{self.elements[0]}" if len(self.elements) > 0 else ""
        return output_string

    def __add__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 0:
                return out
        if out.operation is Operation.ADD:
            if isinstance(other, Term) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                out.elements.append(other)

            out = out.simplify_constants()
            if isinstance(out, Term) and out.operation is Operation.ADD:
                out = out.simplify_variable_coefficients()
            return out
        if isinstance(other, Term) and other.operation is Operation.ADD:
            other.elements.insert(0, out)
            other = other.simplify_constants()
            if isinstance(other, Term) and other.operation is Operation.ADD:
                other = other.simplify_variable_coefficients()
            return other

        out = Term(elements=[out, other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __mul__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 1:
                return out
            if other.value == 0:
                return 0

        if out.operation is Operation.MUL:
            if isinstance(other, Term) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                if isinstance(other, Term) and other.operation in [Operation.ADD, Operation.SUB]:
                    return out.unfold_parentheses(other)
                out.elements.append(other)
            out = out.simplify_binary()
            if isinstance(out, Term):
                out = out.simplify_constants()
            return out

        if out.operation is Operation.DIV:  # ASSUMPTION : division only consists of two elements
            out._elements[0] *= other
            return out

        if out.operation in [Operation.ADD, Operation.SUB]:
            return out.unfold_parentheses(other)
        return Term(elements=[out, other], operation=Operation.MUL)

    def __truediv__(self, other):
        other = _check_and_convert(other)

        if isinstance(other, ConstantVar):
            if other.value == 1:
                return self
            if other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other

        return Term(elements=[self, _check_and_convert(other)], operation=Operation.DIV)

    def __sub__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 0:
                return out
        return out + (-1 * other)

    def __iadd__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 0:
                return out
        if out.operation is Operation.ADD:
            if isinstance(other, Term) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                out.elements.append(other)
            out = out.simplify_constants()

            if isinstance(out, Term) and out.operation is Operation.ADD:
                out = out.simplify_variable_coefficients()
            return out
        if isinstance(other, Term) and other.operation is Operation.ADD:
            other.elements.insert(0, out)
            other = other.simplify_constants()
            if isinstance(other, Term) and other.operation is Operation.ADD:
                other = other.simplify_variable_coefficients()
            return other

        out = Term(elements=[out, other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __imul__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 1:
                return out
            if other.value == 0:
                return 0

        if out.operation is Operation.MUL:
            if isinstance(other, Term) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                if isinstance(other, Term) and other.operation in [Operation.ADD, Operation.SUB]:
                    return out.unfold_parentheses(other)
                out.elements.append(other)
            out = out.simplify_constants()
            return out

        if out.operation is Operation.DIV:
            out._elements[0] *= other
            return out
        if out.operation in [Operation.ADD, Operation.SUB]:
            return out.unfold_parentheses(other)
        return Term(elements=[out, other], operation=Operation.MUL)

    def __itruediv__(self, other):
        other = _check_and_convert(other)

        if isinstance(other, ConstantVar):
            if other.value == 1:
                return self
            if other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other
        return Term(elements=[self, _check_and_convert(other)], operation=Operation.DIV)

    def __isub__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 0:
                return out
        return out + (-1 * other)

    def __radd__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 0:
                return out
        if out.operation is Operation.ADD:
            if isinstance(other, Term) and other.operation == out.operation:
                other.elements.extend(out.elements)
                return other
            t = Term(elements=[other, *out.elements], operation=Operation.ADD)
            t = t.simplify_constants()

            if isinstance(out, Term) and out.operation is Operation.ADD:
                t = t.simplify_variable_coefficients()
            return t
        if isinstance(other, Term) and other.operation is Operation.ADD:
            other.elements.append(out)
            other.simplify_constants()
            if isinstance(other, Term) and other.operation is Operation.ADD:
                other = other.simplify_variable_coefficients()
            return other

        out = Term(elements=[other, out], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Term) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __rmul__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 1:
                return out
            if other.value == 0:
                return 0
        if out.operation is Operation.MUL:
            if isinstance(other, Term) and other.operation == out.operation:
                other.elements.extend(out.elements)
                return other
            if isinstance(other, Term) and other.operation in [Operation.ADD, Operation.SUB]:
                return out.unfold_parentheses(other)
            t = Term(elements=[other, *out.elements], operation=Operation.MUL)
            t = t.simplify_constants()
            return t

        if out.operation in [Operation.ADD, Operation.SUB]:
            return out.unfold_parentheses(other)
        return Term(elements=[other, out], operation=Operation.MUL)

    def __rtruediv__(self, other):
        return Term(elements=[_check_and_convert(other), self], operation=Operation.DIV)

    def __rfloordiv__(self, other):
        return Term(elements=[_check_and_convert(other), self], operation=Operation.DIV)

    def __rsub__(self, other):
        other = _check_and_convert(other)
        out = Term(self.elements, self.operation)

        if isinstance(other, ConstantVar):
            if other.value == 0:
                return out
        return other + (-1 * out)

    def __pow__(self, a):
        if not isinstance(a, int):
            raise ValueError("only integer powers are allowed.")

        out = Term(self.elements, self.operation)
        for _ in range(a - 1):
            out *= out
        return out

    def __lt__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.LT)

    def __le__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.LE)

    def __eq__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.EQ)

    def __ne__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.NE)

    def __gt__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.GT)

    def __ge__(self, other):
        return ConstraintTerm(lhs=self, rhs=_check_and_convert(other), operation=ComparisonOperators.GE)


class ConstraintTerm:
    def __init__(self, rhs: Variable | Term, lhs: Variable | Term, operation: ComparisonOperators) -> None:
        self._lhs = _check_and_convert(lhs)
        if isinstance(self._lhs, Term):
            self._lhs = self._lhs.simplify()

        self._rhs = _check_and_convert(rhs)
        if isinstance(self._rhs, Term):
            self._rhs = self._rhs.simplify()

        if not isinstance(self._rhs, ConstantVar):
            self._lhs -= self._rhs
            self._rhs = ConstantVar(0)
            self._lhs = self._lhs.simplify()
        if isinstance(self._lhs, Term):
            constants = self._lhs.collect_constants()
        else:
            constants = []
        if len(constants) > 1:
            raise RuntimeError("Simplification step failed.")
        if len(constants) == 1:
            self._rhs -= constants[0][2]

        if not isinstance(operation, ComparisonOperators):
            raise ValueError(
                f"parameter operation expected type {ComparisonOperators} but received type {operation.__class__}"
            )

        self._op = operation

    @property
    def rhs(self):
        return self._rhs

    @property
    def lhs(self):
        return self._lhs

    @property
    def operation(self):
        return self._op

    def variables(self):
        if isinstance(self.lhs, ConstantVar):
            pass
        elif isinstance(self.lhs, Variable):
            yield self.lhs
        else:
            yield from self.lhs.variables()
        if isinstance(self.rhs, ConstantVar):
            pass
        elif isinstance(self.rhs, Variable):
            yield self.rhs
        else:
            yield from self.rhs.variables()

    def degree(self):
        degree = 0
        if isinstance(self.rhs, Term):
            aux_degree = self.rhs.degree()
            degree = max(degree, aux_degree)
        elif isinstance(self.rhs, Variable) and not isinstance(self.rhs, ConstantVar):
            degree = max(degree, 1)

        if isinstance(self.lhs, Term):
            aux_degree = self.lhs.degree()
            degree = max(degree, aux_degree)
        elif isinstance(self.lhs, Variable) and not isinstance(self.lhs, ConstantVar):
            degree = max(degree, 1)

        return degree

    def to_list(self):
        out_list = [None, None, None]
        if isinstance(self.lhs, Term):
            out_list[0] = self.lhs.to_list()
        else:
            out_list[0] = self.lhs
        out_list[1] = self.operation
        if isinstance(self.rhs, Term):
            out_list[2] = self.rhs.to_list()
        else:
            out_list[2] = self.rhs

        return out_list

    def to_binary(self):
        if isinstance(self.lhs, Term):
            lhs = self.lhs.to_binary()
        elif isinstance(self.lhs, ContinuousVar):
            lhs, _ = self.lhs.encode()
        else:
            lhs = self.lhs

        if isinstance(self.rhs, Term):
            rhs = self.rhs.to_binary()
        elif isinstance(self.rhs, ContinuousVar):
            rhs, _ = self.rhs.encode()
        else:
            rhs = self.rhs

        return ConstraintTerm(rhs=rhs, lhs=lhs, operation=self.operation)

    def evaluate(self, var_values, precision=100, return_values=False):
        lhs = 0
        rhs = 0

        if isinstance(self.lhs, Term):
            lhs = self.lhs.evaluate(var_values=var_values, precision=precision)
        elif isinstance(self.lhs, ConstantVar):
            lhs = self.lhs.value
        elif isinstance(self.lhs, Variable):
            v = var_values[self.lhs.label]
            if isinstance(v, list):
                if isinstance(self.lhs, ContinuousVar):
                    lhs = self.lhs.evaluate(v, precision=precision)
                else:
                    raise ValueError("Providing a list of values for the a Binary variable.")
            else:
                if not self.lhs.domain.check_value(v):
                    raise ValueError(
                        f"value {v} doesn't respect the domain of the variable {self.lhs} ({self.lhs.domain.__class__.__name__()})"
                    )
                lhs = v

        if isinstance(self.rhs, Term):
            rhs = self.rhs.evaluate(var_values=var_values, precision=precision)
        elif isinstance(self.rhs, ConstantVar):
            rhs = self.rhs.value
        elif isinstance(self.rhs, Variable):
            v = var_values[self.rhs.label]
            if isinstance(v, list):
                if isinstance(self.rhs, ContinuousVar):
                    rhs = self.rhs.evaluate(v, precision=precision)
                else:
                    raise ValueError("Providing a list of values for the a Binary variable.")
            else:
                if not self.rhs.domain.check_value(v):
                    raise ValueError(
                        f"value {v} doesn't respect the domain of the variable {self.rhs} ({self.rhs.domain.__class__.__name__()})"
                    )
                rhs = v

        if return_values:
            return apply_comparison(lhs, self.operation, rhs), lhs, self.operation, rhs
        return apply_comparison(lhs, self.operation, rhs)

    def replace_variables(self, var_dict):
        self.lhs.replace_variables(var_dict)
        self.rhs.replace_variables(var_dict)

    def update_variables_precision(self, var_dict):
        if isinstance(self.lhs, Term):
            self.lhs.update_variables_precision(var_dict)
        if isinstance(self.rhs, Term):
            self.rhs.update_variables_precision(var_dict)

    def update_negative_variables_range(self, var_dict):
        if isinstance(self.lhs, Term):
            self.lhs.update_negative_variables_range(var_dict)
        if isinstance(self.rhs, Term):
            self.rhs.update_negative_variables_range(var_dict)

    def __copy__(self):
        return ConstraintTerm(rhs=copy.copy(self.rhs), lhs=copy.copy(self.lhs), operation=self.operation)

    def __repr__(self):
        return f"{self.lhs!s} {self.operation.value} {self.rhs!s}"

    def __str__(self):
        return f"{self.lhs!s} {self.operation.value} {self.rhs!s}"
