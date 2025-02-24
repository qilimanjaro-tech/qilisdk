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
import operator
from abc import ABC
from collections import defaultdict
from numbers import Complex
from typing import ClassVar, Generator

import numpy as np

from .exceptions import InvalidHamiltonianOperation, NotSupportedOperation


class PauliOperator(ABC):
    """Abstract Representation of a generic Pauli operator

    Args:
        - qubit (int): the qubit that the operator will be acting on
        - name (str): the name of the Pauli operator
        - matrix : the matrix representation of the Pauli operator

    Attributes:
        - qubit (int): the qubit that the operator will be acting on
        - name (str): the name of the Pauli operator
        - matrix : the matrix representation of the Pauli operator
    """

    _NAME: ClassVar[str]
    _MATRIX: ClassVar[np.ndarray]

    def __init__(self, qubit: int) -> None:
        self._qubit = qubit

    @property
    def qubit(self) -> int:
        return self._qubit

    @property
    def name(self) -> str:
        """The name of the Pauli operator.

        Returns:
            str: the name of the pauli operator.
        """
        return self._NAME

    @property
    def matrix(self) -> np.ndarray:
        """The matrix representation of the Pauli operator.

        Returns:
            list[list[complex]]: the matrix representation of the pauli operator.
        """
        return self._MATRIX

    def parse(self) -> Generator[tuple[int, list[PauliOperator]], None, None]:
        """Yields the operator in the format (1, [<operator>])."""
        yield 1, [self]

    def __copy__(self) -> PauliOperator:
        return type(self)(self.qubit)

    def __repr__(self) -> str:
        return f"{self.name}({self.qubit})"

    def __str__(self) -> str:
        return f"{self.name}({self.qubit})"

    # Arithmetic Operators
    def to_hamiltonian(self) -> Hamiltonian:
        """Helper function to convert to Hamiltonian representation.

        Returns:
            Hamiltonian: a hamiltonian with the pauli operator stored in it.
        """
        return Hamiltonian({((self.qubit, self.name),): 1})

    def __add__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() + other

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() - other

    def __rsub__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return other - self.to_hamiltonian()

    __isub__ = __sub__

    def __mul__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() * other

    def __rmul__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        if isinstance(other, (int, float)):
            if other == 0:
                return 0
            if other == 1:
                return self

        return other * self.to_hamiltonian()

    __imul__ = __mul__

    def __truediv__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() / other

    def __rtruediv__(self, _: complex | PauliOperator | Hamiltonian):  # noqa: ANN204
        raise NotSupportedOperation("Division by operators is not supported")

    __itruediv__ = __truediv__


class Z(PauliOperator):
    """The Pauli Z operator"""

    _NAME: ClassVar[str] = "Z"
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, -1]], dtype=complex)

    def __init__(self, qubit: int) -> None:
        """constructs a new Pauli Z operator

        Args:
            qubit (int): the qubit that the operator will act on.
        """
        super().__init__(qubit=qubit)


class X(PauliOperator):
    """The Pauli X operator"""

    _NAME: ClassVar[str] = "X"
    _MATRIX: ClassVar[np.ndarray] = np.array([[0, 1], [1, 0]], dtype=complex)

    def __init__(self, qubit: int) -> None:
        """Constructs a new Pauli X operator

        Args:
            - qubit (int): the qubit that the operator will act on.
        """
        super().__init__(qubit=qubit)


class Y(PauliOperator):
    """The Pauli Y operator"""

    _NAME: ClassVar[str] = "Y"
    _MATRIX: ClassVar[np.ndarray] = np.array([[0, 1j], [1j, 0]], dtype=complex)

    def __init__(self, qubit: int) -> None:
        """Constructs a new Pauli Y operator

        Args:
            qubit (int): the qubit that the operator will act on.
        """
        super().__init__(qubit=qubit)

    def __copy__(self) -> Y:
        return Y(qubit=self.qubit)


class I(PauliOperator):  # noqa: E742
    """The Identity operator"""

    _NAME: ClassVar[str] = "I"
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, 1]], dtype=complex)

    def __init__(self, qubit: int) -> None:
        """Create a new Identity operator

        Args:
            qubit (int): the qubit that the operator will act on.
        """
        super().__init__(qubit=qubit)


class Hamiltonian:
    """
    Assumes elements have the following encoding:
        {
            ((qubit, pauli_operator), ...) : coefficient,
        }

        example:
        {
            ((0, 'Z'), (1, 'Y')): 1,
            ((1, 'X'),): 1j,
        }
    """

    _PAULI_MAP: ClassVar[dict[str, type[PauliOperator]]] = {"Z": Z, "X": X, "Y": Y, "I": I}
    _PAULI_PRODUCT_TABLE: ClassVar[dict[tuple[str, str], tuple[complex, type[PauliOperator]]]] = {
        ("X", "X"): (1, I),
        ("X", "Y"): (1j, Z),
        ("X", "Z"): (-1j, Y),
        ("Y", "X"): (-1j, Z),
        ("Y", "Y"): (1, I),
        ("Y", "Z"): (1j, X),
        ("Z", "X"): (1j, Y),
        ("Z", "Y"): (-1j, X),
        ("Z", "Z"): (1, I),
    }

    def __init__(self, elements: dict[tuple[tuple[int, str], ...], complex] | None = None) -> None:
        self._elements = defaultdict(complex)
        if elements:
            for key, val in elements.items():
                self._elements[key] += val

    @property
    def nqubits(self) -> int:
        """Returns the number of qubits

        Raises:
            InvalidHamiltonianOperation: if the hamiltonian has no terms.

        Returns:
            int: the number of qubits
        """
        n_qubits = -1
        for key in self._elements:
            for qid, _ in key:
                n_qubits = max(n_qubits, qid)

        if n_qubits == -1:
            raise InvalidHamiltonianOperation("Can't compute the number of qubits if the hamiltonian has no terms")
        return n_qubits

    @property
    def elements(self) -> dict[tuple[tuple[int, str], ...], complex]:
        """Returns the dictionary of the elements

        Returns:
            dict: a dictionary of the hamiltonian elements and their coefficient
        """
        return self._elements

    def variables(self) -> Generator[PauliOperator, None, None]:
        """A generator object that returns all the pauli operators in the Hamiltonian.
            Note: the pauli operators repeat in case they appear more than once in the Hamiltonian.
        Yields:
            PauliOperator: A pauli operator object.
        """
        for key in self.elements:
            for qid, pauli in key:
                yield Hamiltonian._PAULI_MAP[pauli](qid)

    def parse(self) -> Generator[tuple[complex, list[PauliOperator]], None, None]:
        """A generator that parses the Hamiltonian object term by term.

        Yields:
            tuple[complex, list[PauliOperators]]: the coefficient and the list
            of pauli operators of the term.
        """
        for key, value in self.elements.items():
            yield value, [Hamiltonian._PAULI_MAP[op](qid) for qid, op in key]

    def simplify(self) -> Hamiltonian:
        """Simplify the hamiltonian expression by removing values close to zero.

        Returns:
            Hamiltonian: The simplified Hamiltonian.
        """
        keys_to_remove = [key for key, value in self.elements.items() if np.real_if_close(value) == 0]
        for key in keys_to_remove:
            del self.elements[key]

        identities_to_accumulate = [
            (key, value)
            for key, value in self.elements.items()
            if len(key) == 1 and key[0][0] != 0 and key[0][1] == "I"
        ]
        for key, value in identities_to_accumulate:
            self.elements[(0, "I"),] += value
            del self.elements[key]

        return self

    def __getitem__(self, index: tuple[tuple[int, str], ...]) -> complex:
        return self.elements[index]

    def __setitem__(self, key: tuple[tuple[int, str], ...], value: complex) -> None:
        self.elements[key] = value

    def __copy__(self) -> Hamiltonian:
        return Hamiltonian(elements=self.elements.copy())

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        # Return "0" if there are no terms
        if not self.elements:
            return "0"

        def _format_coeff(c: complex) -> str:
            eps = 1e-14
            re, im = c.real, c.imag

            # 1) Purely real?
            if abs(im) < eps:
                # Check if real is integral
                re_int = np.round(re)
                if abs(re - re_int) < eps:
                    return str(int(re_int))  # e.g. '2' instead of '2.0'
                return str(re)  # e.g. '2.5'

            # 2) Purely imaginary?
            if abs(re) < eps:
                # Check if imaginary is integral
                im_int = np.round(im)
                if abs(im - im_int) < eps:
                    # e.g. 2 => '2j', -3 => '-3j'
                    return f"{int(im_int)}j"
                return f"{im}j"  # e.g. '2.5j'

            # 3) General complex with nonzero real & imag
            s = str(c)  # e.g. '(3+2j)'
            return s

        parts = []
        items = list(self.elements.items())

        for i, (operators, raw_coeff) in enumerate(items):
            # np.real_if_close can force near-real numbers to real floats,
            # but it's still of type `np.complex128` or Python `complex`.
            coeff = np.real_if_close(raw_coeff)

            is_first = i == 0

            # Prepare a base string for the numeric part *without any leading '+' or '-'
            base_str = _format_coeff(coeff)

            # Now handle sign logic:
            # For the first term, we only show a minus sign if `coeff` is negative or has a negative real part.
            # For subsequent terms, we show "+ ..." if positive, "- ..." if negative, etc.

            # 1) Figure out if the overall number is "negative" in a real sense:
            #    - If purely real and negative => negative
            #    - If purely imaginary and negative => negative
            #    - If truly complex with nonzero real+imag, we follow your original approach
            #      (treat real>0 => +, real<0 => -).
            re, im = coeff.real, coeff.imag
            # We'll do a simple sign check: if re < 0, we call it negative. If re==0 and im<0, negative.
            negative = im < 0 if abs(re) < 1e-14 else re < 0  # noqa: PLR2004

            # First term: no leading '+' if positive
            if is_first:
                if negative:
                    coeff_str = "-" if base_str == "-1" else base_str if base_str.startswith("-") else f"- {base_str}"
                elif base_str == "1":
                    coeff_str = ""  # implies +1
                else:
                    coeff_str = base_str
            # Subsequent terms: show '+' or '-'
            elif negative:
                # Remove leading '-' if present
                if base_str == "-1":
                    coeff_str = "-"
                elif base_str.startswith("-"):
                    # e.g. base_str = '-3+2j' => we want " - 3+2j" or just "- 3+2j" ?
                    coeff_str = f"- {base_str[1:]}" if len(base_str) > 1 else "-"
                else:
                    coeff_str = f"- {base_str}"
            else:
                # Positive
                coeff_str = "+" if base_str == "1" else f"+ {base_str}"

            # Build the operators string (e.g. "Z(0) Y(1)")
            ops_str = " ".join(f"{op}({qid})" for qid, op in operators)

            # Combine with a single space if both strings exist
            if coeff_str and ops_str:
                parts.append(f"{coeff_str} {ops_str}")
            else:
                parts.append(coeff_str + ops_str)

        return " ".join(parts)

    @staticmethod
    def _multiply_sets(
        set1: tuple[tuple[int, str], ...], set2: tuple[tuple[int, str], ...]
    ) -> tuple[complex, tuple[tuple[int, str], ...]]:
        list1 = list(set1)
        list2 = list(set2)
        list1.extend(list2)

        # Sort by qubit so we can group by qubit easily
        combined_list = sorted(list1, key=operator.itemgetter(0))

        sum_dict: dict[int, list[PauliOperator]] = {}
        for qid, op_name in combined_list:
            if qid not in sum_dict:
                sum_dict[qid] = []
            sum_dict[qid].append(Hamiltonian._PAULI_MAP[op_name](qid))

        simplified_list: list[tuple[int, str]] = []
        accumulated_phase = complex(1)

        for operators in sum_dict.values():
            op1 = operators[0]
            phase = complex(1)
            if len(operators) > 1:
                # Multiply all operators acting on the same qubit
                for op2 in operators[1:]:
                    aux_phase, op1 = Hamiltonian._multiply_pauli(op1, op2)
                    phase *= aux_phase

            if op1.name != "I":
                simplified_list.append((op1.qubit, op1.name))
            accumulated_phase *= phase

        if not simplified_list:
            simplified_list.append((0, "I"))

        return accumulated_phase, tuple(simplified_list)

    @staticmethod
    def _multiply_pauli(op1: PauliOperator, op2: PauliOperator) -> tuple[complex, PauliOperator]:
        """Multiply two Pauli Operators

        Raises:
            ValueError: If the two Operators are not acting on the same qubit
            NotImplementedError: Multiplication between the two operators is not supported.

        Returns:
            tuple[complex, "PauliOperator"]: A tuple containing the resulting
            coefficient and Pauli operator.
        """

        if op1.qubit != op2.qubit:
            raise ValueError("Operators must act on the same qubit")

        if isinstance(op1, I):
            return 1, op2
        if isinstance(op2, I):
            return 1, op1

        result = Hamiltonian._PAULI_PRODUCT_TABLE.get((op1.name, op2.name))
        if result is not None:
            coefficient, operator_cls = result
            if operator_cls is I:
                return coefficient, I(0)
            return coefficient, operator_cls(op1.qubit)

        raise NotImplementedError(f"Operation between operator {op1.name} and operator {op2.name} is not supported.")

    @staticmethod
    def _multiply_hamiltonians(h1: Hamiltonian, h2: Hamiltonian) -> Hamiltonian:
        out = Hamiltonian()
        for k1, v1 in h1.elements.items():
            for k2, v2 in h2.elements.items():
                phase, new_key = Hamiltonian._multiply_sets(k1, k2)
                out.elements[new_key] += phase * v1 * v2
        return out.simplify()

    def __add__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        out = copy.copy(self)
        if isinstance(other, Hamiltonian):
            for key, val in other.elements.items():
                out.elements[key] += val
        elif isinstance(other, PauliOperator):
            encoded = ((other.qubit, other.name),)
            out.elements[encoded] += 1
        elif isinstance(other, Complex):
            out.elements[(0, "I"),] += other
        else:
            raise InvalidHamiltonianOperation(f"Invalid addition between Hamiltonian and {other.__class__.__name__}.")
        return out.simplify()

    def __radd__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.__add__(other)

    def __sub__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self + (-1 * other)

    def __rsub__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        return other + (-1 * self)

    def __mul__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        if isinstance(other, Hamiltonian):
            return Hamiltonian._multiply_hamiltonians(self, other)
        if isinstance(other, PauliOperator):
            return Hamiltonian._multiply_hamiltonians(self, other.to_hamiltonian())
        if isinstance(other, Complex):
            out = copy.copy(self)
            for k in out.elements:
                out.elements[k] *= other
            return out.simplify()
        raise InvalidHamiltonianOperation(f"Invalid multiplication between Hamiltonian and {other.__class__.__name__}.")

    def __rmul__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        if isinstance(other, Hamiltonian):
            # multiplication is non-commutative for Pauli ops, so we must consider key order carefully
            return Hamiltonian._multiply_hamiltonians(other, self)
        if isinstance(other, PauliOperator):
            return Hamiltonian._multiply_hamiltonians(other.to_hamiltonian(), self)
        if isinstance(other, Complex):
            # scalar multiplication is commutative, so just do __mul__
            return self.__mul__(other)
        raise InvalidHamiltonianOperation(f"Invalid multiplication between Hamiltonian and {other.__class__.__name__}.")

    def __truediv__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        if not isinstance(other, Complex):
            raise InvalidHamiltonianOperation("Division of operators is not supported")

        return self * (1 / other)

    def __rtruediv__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        if not isinstance(other, Complex):
            raise InvalidHamiltonianOperation("Division of operators is not supported")

        return (1 / other) * self

    def __rfloordiv__(self, other: Complex | PauliOperator | Hamiltonian) -> Hamiltonian:
        if not isinstance(other, Complex):
            raise NotSupportedOperation("Division of operators is not supported")

        return (1 / other) * self

    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __itruediv__ = __truediv__
