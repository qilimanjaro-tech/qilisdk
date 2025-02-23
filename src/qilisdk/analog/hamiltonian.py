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

    def __add__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        return self.to_hamiltonian() + other

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        return self.to_hamiltonian() - other

    def __rsub__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        return other - self.to_hamiltonian()

    __isub__ = __sub__

    def __mul__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        return self.to_hamiltonian() * other

    def __rmul__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        if isinstance(other, (int, float)):
            if other == 0:
                return 0
            if other == 1:
                return self

        return other * self.to_hamiltonian()

    __imul__ = __mul__

    def __truediv__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
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
    _PAULI_MAP: ClassVar = {"Z": Z, "X": X, "Y": Y, "I": I}
    _PAULI_PRODUCT_TABLE: ClassVar = {
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

    def __init__(self, elements: dict[tuple[tuple[int, str], ...], complex]) -> None:
        self._elements = {}
        for operators, coeff in elements.items():
            self._elements[operators] = coeff

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
            for qid, operator in key:
                yield Hamiltonian._PAULI_MAP[operator](qid)

    def parse(self) -> Generator[tuple[complex, list[PauliOperator]], None, None]:
        """A generator that parses the Hamiltonian object term by term.

        Yields:
            tuple[complex, list[PauliOperators]]: the coefficient and the list
            of pauli operators of the term.
        """
        for key, value in self.elements.items():
            yield value, [Hamiltonian._PAULI_MAP[op](qid) for qid, op in key]

    def simplify(self) -> Hamiltonian:
        """Simplify the hamiltonian expression by removing values close to zero."""
        pop_keys = []
        for key, value in self.elements.items():
            if np.real_if_close(value) == 0:
                pop_keys.append(key)

        for key in pop_keys:
            self.elements.pop(key)

        return self

    def __copy__(self) -> Hamiltonian:
        return Hamiltonian(elements=self.elements.copy())

    def __repr__(self) -> str:
        out = ""
        for operators, _coeff in self.elements.items():
            coeff = np.real_if_close(_coeff)
            if out == "":  # noqa: PLC1901
                if isinstance(coeff, (complex)) or np.iscomplex(coeff):
                    out += f"({coeff}) "
                elif coeff != 1:
                    if coeff == -1:
                        out += "-"
                    else:
                        out += f"{coeff} "
            elif isinstance(coeff, (complex)) or np.iscomplex(coeff):
                out += f"+ ({coeff}) "
            elif coeff not in (1, -1):  # noqa: PLR6201
                out += f"+ {coeff} " if coeff > 0 else f"- {np.abs(coeff)} "
            else:
                out += "+ " if coeff > 0 else "- "

            for qid, o in operators:
                out += f"{o}({qid}) "
        return out

    def __str__(self) -> str:
        out = ""
        for operators, _coeff in self.elements.items():
            coeff = np.real_if_close(_coeff)
            if out == "":  # noqa: PLC1901
                if isinstance(coeff, (complex)) or np.iscomplex(coeff):
                    out += f"({coeff}) "
                elif coeff != 1:
                    if coeff == -1:
                        out += "-"
                    else:
                        out += f"{coeff} "
            elif isinstance(coeff, (complex)) or np.iscomplex(coeff):
                out += f"+ ({coeff}) "
            elif coeff not in (1, -1):  # noqa: PLR6201
                out += f"+ {coeff} " if coeff > 0 else f"- {np.abs(coeff)} "
            else:
                out += "+ " if coeff > 0 else "- "

            for qid, o in operators:
                out += f"{o}({qid}) "
        return out

    def __getitem__(self, index: tuple[tuple[int, str], ...]) -> complex:
        return self.elements[index]

    def __setitem__(self, key: tuple[tuple[int, str], ...], value: complex) -> None:
        self.elements[key] = value

    def __add__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        out = copy.copy(self)

        if isinstance(other, Hamiltonian):
            for key, value in other.elements.items():
                if key in out.elements:
                    out[key] += value
                else:
                    out[key] = value
        elif isinstance(other, PauliOperator):
            encoded = ((other.qubit, other.name),)
            if encoded in out.elements:
                out[encoded] += 1
            else:
                out[encoded] = 1
        elif isinstance(other, (int, float, complex)):
            if ((0, "I"),) in out.elements:
                out[(0, "I"),] += other
            else:
                out[(0, "I"),] = other
        else:
            raise InvalidHamiltonianOperation(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.simplify()
        return out

    def __mul__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        out = Hamiltonian({})

        if isinstance(other, Hamiltonian):
            # unfold parenthesis
            for key1 in self.elements:
                for key2 in other.elements:
                    phase, new_key = Hamiltonian._multiply_sets(key1, key2)
                    if new_key in out.elements:
                        out[new_key] += phase * self[key1] * other[key2]
                    else:
                        out[new_key] = phase * self[key1] * other[key2]
        elif isinstance(other, PauliOperator):
            key2 = ((other.qubit, other.name),)
            for key1 in self.elements:
                phase, new_key = Hamiltonian._multiply_sets(key1, key2)
                if new_key in out.elements:
                    out[new_key] += phase * self[key1]
                else:
                    out[new_key] = phase * self[key1]
        elif isinstance(other, Complex):
            out = copy.copy(self)
            for key in out.elements:
                out[key] *= other
            return out
        else:
            raise InvalidHamiltonianOperation(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.simplify()
        return out

    def __truediv__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        if not isinstance(other, Complex):
            raise InvalidHamiltonianOperation("Division of operators is not supported")

        return self * (1 / other)

    def __sub__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        return self + (-1 * other)

    def __radd__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        out = copy.copy(self)

        if isinstance(other, Hamiltonian):
            for key, value in other.elements.items():
                if key in out.elements:
                    out[key] += value
                else:
                    out[key] = value
        elif isinstance(other, PauliOperator):
            encoded = ((other.qubit, other.name),)
            if encoded in out.elements:
                out[encoded] += 1
            else:
                out[encoded] = 1
        elif isinstance(other, Complex):
            if ((0, "I"),) in out.elements:
                out[(0, "I"),] += other
            else:
                out[(0, "I"),] = other
        else:
            raise InvalidHamiltonianOperation(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.simplify()
        return out

    def __rmul__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        out = Hamiltonian({})

        if isinstance(other, Hamiltonian):
            # unfold parenthesis
            for key1 in self.elements:
                for key2 in other.elements:
                    phase, new_key = Hamiltonian._multiply_sets(key2, key1)
                    if new_key in out.elements:
                        out[new_key] += phase * self[key1] * other[key2]
                    else:
                        out[new_key] = phase * self[key1] * other[key2]
        elif isinstance(other, PauliOperator):
            key2 = ((other.qubit, other.name),)
            for key1 in self.elements:
                phase, new_key = Hamiltonian._multiply_sets(key2, key1)
                if new_key in out.elements:
                    out[new_key] += phase * self[key1]
                else:
                    out[new_key] = phase * self[key1]
        elif isinstance(other, Complex):
            out = copy.copy(self)
            for key in out.elements:
                out[key] *= other
            return out
        else:
            raise InvalidHamiltonianOperation(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.simplify()
        return out

    def __rtruediv__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        if not isinstance(other, Complex):
            raise InvalidHamiltonianOperation("Division of operators is not supported")

        return (1 / other) * self

    def __rfloordiv__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        if not isinstance(other, Complex):
            raise NotSupportedOperation("Division of operators is not supported")

        return (1 / other) * self

    def __rsub__(
        self, other: Complex | PauliOperator | Hamiltonian
    ) -> Hamiltonian:
        return other + (-1 * self)

    __iadd__ = __add__
    __imul__ = __mul__
    __itruediv__ = __truediv__
    __isub__ = __sub__
    
    @staticmethod
    def _multiply_sets(
        set1: tuple[tuple[int, str], ...], set2: tuple[tuple[int, str], ...]
    ) -> tuple[complex, tuple[tuple[int, str], ...]]:
        list1 = list(set1)
        list2 = list(set2)

        list1.extend(list2)

        combined_list = sorted(list1, key=lambda x: x[0])  # noqa: FURB118

        sum_dict: dict[int, list] = {}

        for qid, op in combined_list:
            if qid not in sum_dict:
                sum_dict[qid] = []
            sum_dict[qid].append(Hamiltonian._PAULI_MAP[op](qid))

        simplified_list = []
        accumulated_phase = 1 + 0j
        for v in sum_dict.values():
            op1 = v[0]
            phase = 1 + 0j
            if len(v) > 1:
                for i in range(1, len(v)):
                    aux_phase, op1 = Hamiltonian._multiply_pauli(op1, v[i])
                    phase *= aux_phase

            if op1.name != "I":
                simplified_list.append((op1.qubit, op1.name))
            accumulated_phase *= phase

        if len(simplified_list) == 0:
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
            coefficient, operator_fn = result
            if operator_fn is I:
                return coefficient, I(0)
            return coefficient, operator_fn(op1.qubit)

        raise NotImplementedError(f"Operation between operator {op1.name} and operator {op2.name} is not supported.")

