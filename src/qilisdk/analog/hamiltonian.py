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
from abc import ABC
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np
from scipy.sparse import csc_array, identity, kron, spmatrix

from qilisdk.common.variables import Parameter, Term
from qilisdk.yaml import yaml

from .exceptions import InvalidHamiltonianOperation

if TYPE_CHECKING:
    from collections.abc import Iterator


Number = int | float | complex


###############################################################################
# Flyweight Cache
###############################################################################
_OPERATOR_CACHE: dict[tuple[str, int], PauliOperator] = {}


def _get_pauli(name: str, qubit: int) -> PauliOperator:
    key = (name, qubit)
    if key in _OPERATOR_CACHE:
        return _OPERATOR_CACHE[key]

    if name == "Z":
        op = PauliZ(qubit)
    elif name == "X":
        op = PauliX(qubit)  # type: ignore[assignment]
    elif name == "Y":
        op = PauliY(qubit)  # type: ignore[assignment]
    elif name == "I":
        op = PauliI(qubit)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown Pauli operator name: {name}")

    _OPERATOR_CACHE[key] = op
    return op


###############################################################################
# Public Factory Functions
###############################################################################
def Z(qubit: int) -> Hamiltonian:
    return _get_pauli("Z", qubit).to_hamiltonian()


def X(qubit: int) -> Hamiltonian:
    return _get_pauli("X", qubit).to_hamiltonian()


def Y(qubit: int) -> Hamiltonian:
    return _get_pauli("Y", qubit).to_hamiltonian()


def I(qubit: int = 0) -> Hamiltonian:  # noqa: E743
    return _get_pauli("I", qubit).to_hamiltonian()


###############################################################################
# Abstract Base PauliOperator
###############################################################################
class PauliOperator(ABC):
    """
    A generic Pauli operator that acts on one qubit.
    Flyweight usage: do NOT instantiate directly—use X(q), Y(q), etc.
    """

    _NAME: ClassVar[str]
    _MATRIX: ClassVar[np.ndarray]

    # __slots__ = ("_qubit",)

    def __init__(self, qubit: int) -> None:
        self._qubit = qubit

    @property
    def qubit(self) -> int:
        return self._qubit

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def matrix(self) -> np.ndarray:
        return self._MATRIX

    def to_hamiltonian(self) -> Hamiltonian:
        """Convert this single operator to a Hamiltonian with one term.

        Returns:
            Hamiltonian: The converted Hamiltonian.
        """
        return Hamiltonian({(self,): 1})

    def __hash__(self) -> int:
        return hash((self._NAME, self._qubit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliOperator):
            return False
        return (self._NAME == other._NAME) and (self._qubit == other._qubit)

    def __repr__(self) -> str:
        return f"{self.name}({self.qubit})"

    def __str__(self) -> str:
        return f"{self.name}({self.qubit})"

    # ----------- Arithmetic Operators ------------

    def __add__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() + other

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() - other

    def __rsub__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        return other - self.to_hamiltonian()

    __isub__ = __sub__

    def __mul__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() * other

    def __rmul__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        return other * self.to_hamiltonian()

    __imul__ = __mul__

    def __truediv__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        return self.to_hamiltonian() / other

    def __rtruediv__(self, _: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        raise InvalidHamiltonianOperation("Division by operators is not supported")

    __itruediv__ = __truediv__


###############################################################################
# Concrete Flyweight Operator Classes
###############################################################################
@yaml.register_class
class PauliZ(PauliOperator):
    # __slots__ = ()
    _NAME: ClassVar[str] = "Z"
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, -1]], dtype=complex)


@yaml.register_class
class PauliX(PauliOperator):
    # __slots__ = ()
    _NAME: ClassVar[str] = "X"
    _MATRIX: ClassVar[np.ndarray] = np.array([[0, 1], [1, 0]], dtype=complex)


@yaml.register_class
class PauliY(PauliOperator):
    # __slots__ = ()
    _NAME: ClassVar[str] = "Y"
    _MATRIX: ClassVar[np.ndarray] = np.array([[0, -1j], [1j, 0]], dtype=complex)


@yaml.register_class
class PauliI(PauliOperator):
    # __slots__ = ()
    _NAME: ClassVar[str] = "I"
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, 1]], dtype=complex)


@yaml.register_class
class Hamiltonian:
    _EPS: float = 1e-14
    _PAULI_PRODUCT_TABLE: ClassVar[dict[tuple[str, str], tuple[complex, Callable[..., PauliOperator]]]] = {
        ("X", "X"): (1, PauliI),
        ("X", "Y"): (1j, PauliZ),
        ("X", "Z"): (-1j, PauliY),
        ("Y", "X"): (-1j, PauliZ),
        ("Y", "Y"): (1, PauliI),
        ("Y", "Z"): (1j, PauliX),
        ("Z", "X"): (1j, PauliY),
        ("Z", "Y"): (-1j, PauliX),
        ("Z", "Z"): (1, PauliI),
    }

    ZERO: int = 0

    def __init__(self, elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] | None = None) -> None:
        """A class to represent abstract Hamiltonian expressions.

        Args:
            elements (dict[tuple[PauliOperator, ...], complex] | None, optional): maps from a tuple of PauliOperator objects
                        to a complex coefficient. For example:

                        .. code-block:: text

                            {
                            (Z(0), Y(1)):  1,
                            (X(1),):       1j,
                            }

                        Defaults to None.

        Raises:
            ValueError: If the Term provided contains Generic variables and not only Parameters.
        """
        self._elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = defaultdict(complex)
        self._parameters: dict[str, Parameter] = {}
        if elements:
            for key, val in elements.items():
                if isinstance(val, Term):
                    for v in val.variables():
                        if isinstance(v, Parameter):
                            self._parameters[v.label] = v
                        else:
                            raise ValueError(
                                "Only Parameters are allowed to be used in hamiltonians. Generic Variables are not supported"
                            )
                elif isinstance(val, Parameter):
                    self._parameters[val.label] = val
                self._elements[key] += val
            self.simplify()

    @property
    def nqubits(self) -> int:
        """Number of qubits acting on the hamiltonian."""
        qubits = {op.qubit for key in self._elements for op in key}

        return max(qubits) + 1 if qubits else 0

    @property
    def elements(self) -> dict[tuple[PauliOperator, ...], complex]:
        """Returns the internal dictionary of elements (read-only)."""
        return {
            k: (v if isinstance(v, complex) else (v.evaluate({}) if isinstance(v, Term) else v.evaluate()))
            for k, v in self._elements.items()
        }

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total RealNumber of parameters required by all parameterized gates in the circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """
        return len(self._parameters)

    @property
    def parameters(self) -> dict[str, Parameter]:
        return self._parameters

    def get_parameter_values(self) -> list[float]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """
        return [param.value for param in self._parameters.values()]

    def get_parameter_names(self) -> list[str]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """
        return list(self._parameters.keys())

    def get_parameters(self) -> dict[str, float]:
        """
        Retrieve the parameter names and values from all parameterized gates in the circuit.

        Returns:
            dict[str, float]: A dictionary of the parameters with their current values.
        """
        return {label: param.value for label, param in self._parameters.items()}

    def set_parameter_values(self, values: list[float]) -> None:
        """
        Set new parameter values for all parameterized gates in the circuit.

        Args:
            values (list[float]): A list containing new parameter values to assign to the parameterized gates.

        Raises:
            ValueError: If the RealNumber of provided values does not match the expected RealNumber of parameters.
        """
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but Hamiltonian has {self.nparameters} parameters.")
        for i, parameter in enumerate(self._parameters.values()):
            parameter.set_value(values[i])

    def set_parameters(self, parameter_dict: dict[str, int | float]) -> None:
        """Set the parameter values by their label. No need to provide the full list of parameters.

        Args:
            parameter_dict (dict[str, RealNumber]): _description_

        Raises:
            ValueError: _description_
        """
        for label, param in parameter_dict.items():
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this hamiltonian.")
            self._parameters[label].set_value(param)

    def simplify(self) -> Hamiltonian:
        """Simplify the Hamiltonian expression by removing near-zero terms and accumulating constant terms.

        Returns:
            Hamiltonian: Simplified Hamiltonian
        """
        # 1) Remove near-zero
        keys_to_remove = [
            key for key, value in self._elements.items() if isinstance(value, complex) and abs(value) < Hamiltonian._EPS
        ]
        for key in keys_to_remove:
            del self._elements[key]

        # 2) Accumulate identities that do NOT act on qubit=0 => I(0)
        to_accumulate = [
            (key, value)
            for key, value in self._elements.items()
            if len(key) == 1 and key[0].name == "I" and key[0].qubit != 0
        ]
        for key, value in to_accumulate:
            del self._elements[key]
            self._elements[PauliI(0),] += value

        return self

    def _apply_operator_on_qubit(self, terms: list[PauliOperator]) -> spmatrix:
        """Get the matrix representation of a single term by taking the tensor product
        of operators acting on each qubit. For qubits with no operator in `terms`,
        the identity is used.

        Args:
            terms (list[PauliOperator]): A list of Pauli operators in the term.

        Returns:
            spmatrix: The full matrix representation of the term.
        """
        # Build a list of factors for each qubit
        factors = []
        for q in range(self.nqubits):
            # Look for an operator acting on qubit q
            op = next((t for t in terms if t.qubit == q), None)
            if op is not None:
                # Wrap the operator's matrix as a sparse matrix.
                factors.append(csc_array(np.array(op.matrix)))
            else:
                factors.append(identity(2, format="csc"))
        # Compute the tensor (Kronecker) product over all qubits.
        full_matrix = reduce(lambda A, B: kron(A, B, format="csc"), factors)
        return full_matrix

    def to_matrix(self) -> spmatrix:
        """Return the full matrix representation of the Hamiltonian by summing over all terms.

        Returns:
            spmatrix: The sparse matrix representation of the Hamiltonian.
        """
        dim = 2**self.nqubits
        # Initialize a zero matrix of the appropriate dimension.
        result = csc_array(np.zeros((dim, dim), dtype=complex))
        for coeff, term in self:
            result += coeff * self._apply_operator_on_qubit(term)
        return result

    def get_static_hamiltonian(self) -> Hamiltonian:
        out = Hamiltonian()
        for pauli, value in self.elements.items():
            aux: Hamiltonian | PauliOperator = pauli[0]
            for p in list(pauli)[1:]:
                aux *= p
            out += aux * value
        return out

    def __iter__(self) -> Iterator[tuple[complex, list[PauliOperator]]]:
        for key, value in self.elements.items():
            yield value, list(key)

    # ------- Equality & hashing --------

    def __eq__(self, other: object) -> bool:
        if other == Hamiltonian.ZERO:
            return bool(
                len(self._elements) == 0
                or (len(self._elements) == 1 and (PauliI(0),) in self._elements and self._elements[PauliI(0),] == 0)
            )
        if isinstance(other, Number):
            return bool(
                len(self._elements) == 1 and (PauliI(0),) in self._elements and self._elements[PauliI(0),] == other
            )
        if isinstance(other, PauliOperator):
            other = other.to_hamiltonian()
        if not isinstance(other, Hamiltonian):
            return False
        return dict(self._elements) == dict(other._elements)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        items_frozen = frozenset(self._elements.items())
        return hash(items_frozen)

    def __copy__(self) -> Hamiltonian:
        return Hamiltonian(elements=self._elements.copy())

    # ------- String representation --------

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        # Return "0" if there are no terms
        if not self._elements:
            return "0"

        def _format_coeff(c: complex) -> str:
            re, im = c.real, c.imag

            # 1) Purely real?
            if abs(im) < Hamiltonian._EPS:
                re_int = np.round(re)
                if abs(re - re_int) < Hamiltonian._EPS:
                    return str(int(re_int))  # e.g. '2' instead of '2.0'
                return str(re)  # e.g. '2.5'

            # 2) Purely imaginary?
            if abs(re) < Hamiltonian._EPS:
                im_int = np.round(im)
                if abs(im - im_int) < Hamiltonian._EPS:
                    return f"{int(im_int)}j"  # e.g. 2 => '2j', -3 => '-3j'
                return f"{im}j"  # e.g. '2.5j'

            # 3) General complex with nonzero real & imag
            s = str(c)  # e.g. '(3+2j)'
            return s

        # We want to place the single identity term (I(0),) at the front if it exists
        items = list(self.elements.items())
        try:
            i = next(idx for idx, (key, _) in enumerate(items) if len(key) == 1 and key[0] == (PauliI(0)))
            item = items.pop(i)
            items.insert(0, item)
        except StopIteration:
            pass

        parts = []
        for idx, (operator, coeff) in enumerate(items):
            base_str = _format_coeff(coeff)

            if idx == 0:
                # first term
                if len(operator) == 1 and operator[0].name == "I":
                    coeff_str = base_str
                elif base_str == "1":
                    coeff_str = ""
                elif base_str == "-1":
                    coeff_str = "-"
                else:
                    coeff_str = base_str
            elif base_str == "1":
                coeff_str = "+"
            elif base_str == "-1":
                coeff_str = "-"
            elif base_str.startswith("-"):
                coeff_str = f"- {base_str[1:]}"
            else:
                coeff_str = f"+ {base_str}"

            # Operators string
            ops_str = " ".join(str(op) for op in operator if op.name != "I")
            if coeff_str and ops_str:
                parts.append(f"{coeff_str} {ops_str}")
            else:
                parts.append(coeff_str + ops_str)

        return " ".join(parts)

    @classmethod
    def parse(cls, hamiltonian_str: str) -> Hamiltonian:
        hamiltonian_str = hamiltonian_str.strip()

        # 1) remove *all* spaces inside any ( … ) group (coefficients or indices)
        hamiltonian_str = re.sub(
            r"\(\s*([0-9A-Za-z.+\-j\s]+?)\s*\)",
            lambda m: "(" + re.sub(r"\s+", "", str(m.group(1))) + ")",
            hamiltonian_str,
        )

        # 2) collapse multiple spaces down to one (outside the parens now)
        hamiltonian_str = re.sub(r"\s+", " ", hamiltonian_str)

        # 3) ensure a single space between a closing “)” and the next operator token like X(0)/Y(1)/etc.
        hamiltonian_str = re.sub(r"\)\s*(?=[XYZI]\()", ") ", hamiltonian_str)

        # Special case: "0" => empty Hamiltonian
        if hamiltonian_str == "0":
            return cls({})

        elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = defaultdict(
            complex
        )  # TODO (ameer): the parsing doesn't support Term and Parameters

        # If there's no initial +/- sign, prepend '+ ' for easier splitting
        if not hamiltonian_str.startswith("+") and not hamiltonian_str.startswith("-"):
            hamiltonian_str = "+ " + hamiltonian_str

        # Replace " - " with " + - " so each term is split on " + "
        hamiltonian_str = hamiltonian_str.replace(" - ", " + - ")

        # Split on " + "
        tokens = hamiltonian_str.split(" + ")
        # Remove any empty tokens (can happen if the string started "+ ")
        tokens = [t.strip() for t in tokens if t.strip()]

        # Regex to match operator tokens like "Z(0)", "X(1)", "I(0)"
        operator_pattern = re.compile(r"([XYZI])\((\d+)\)")

        def parse_token(token: str) -> tuple[complex, list[PauliOperator]]:
            def looks_like_number(text: str) -> bool:
                # If it's empty, it's not a number
                if not text:
                    return False
                # If the first char is digit, '(', '.', '+', '-', or '0',
                # or if 'j' is present, assume it's numeric
                first = text[0]
                if first.isdigit() or first in {"(", ".", "+", "-"}:
                    return True
                return "j" in text

            sign = 1
            # Check leading sign
            if token.startswith("-"):
                sign = -1
                token = token[1:].strip()
            elif token.startswith("+"):
                # optional leading '+'
                token = token[1:].strip()

            words = token.split()
            if not words:
                # e.g. just "-" or "+"
                # means coefficient = ±1, no operators
                return complex(sign), []

            # Attempt to parse the first word as a numeric coefficient
            maybe_coeff = words[0]
            # Decide if 'maybe_coeff' is numeric or an operator
            if looks_like_number(maybe_coeff):
                # parse as a complex number
                coeff_str = maybe_coeff
                # If it's e.g. '(2.5+3j)', remove parentheses
                if coeff_str.startswith("(") and coeff_str.endswith(")"):
                    coeff_str = coeff_str[1:-1]
                coeff_val = complex(coeff_str) * sign
                words = words[1:]  # consume this word
            else:
                # No explicit coefficient => ±1
                coeff_val = complex(sign)

            # Now parse the remaining words as operators
            ops = []
            for w in words:
                match = operator_pattern.fullmatch(w)
                if not match:
                    raise ValueError(f"Unrecognized operator format: '{w}'")
                name, qubit_str = match.groups()
                qubit = int(qubit_str)
                op = _get_pauli(name, qubit)
                ops.append(op)

            return coeff_val, ops

        for token in tokens:
            coeff, op_list = parse_token(token)
            if not op_list:
                # purely scalar => store as (I(0),)
                elements[PauliI(0),] += coeff
            else:
                # Sort operators by qubit for canonical ordering
                op_list.sort(key=lambda op: op.qubit)
                elements[tuple(op_list)] += coeff

        hamiltonian = cls(elements)
        hamiltonian.simplify()
        return hamiltonian

    # ------- Internal multiplication helpers --------

    @staticmethod
    def _multiply_sets(
        set1: tuple[PauliOperator, ...], set2: tuple[PauliOperator, ...]
    ) -> tuple[complex, tuple[PauliOperator, ...]]:
        # Combine all operators into a single list
        combined = list(set1) + list(set2)

        # Group by qubit
        combined.sort(key=lambda op: op.qubit)
        sum_dict: dict[int, list[PauliOperator]] = defaultdict(list)
        for op in combined:
            sum_dict[op.qubit].append(op)

        accumulated_phase = complex(1)
        final_ops: list[PauliOperator] = []

        for qubit_ops in sum_dict.values():
            op1 = qubit_ops[0]
            phase = complex(1)
            # Multiply together all operators on the same qubit
            for op2 in qubit_ops[1:]:
                aux_phase, op1 = Hamiltonian._multiply_pauli(op1, op2)
                phase *= aux_phase
            if op1.name != "I":
                final_ops.append(op1)
            accumulated_phase *= phase

        # If everything simplified to identity, we store I(0)
        if not final_ops:
            final_ops = [PauliI(0)]

        # Sort again by qubit (to keep canonical form)
        final_ops.sort(key=lambda op: op.qubit)
        return accumulated_phase, tuple(final_ops)

    @staticmethod
    def _multiply_pauli(op1: PauliOperator, op2: PauliOperator) -> tuple[complex, PauliOperator]:
        if op1.qubit != op2.qubit:
            raise ValueError("Operators must act on the same qubit for multiplication.")

        # If either is identity, no phase
        if op1.name == "I":
            return (1, op2)
        if op2.name == "I":
            return (1, op1)

        # Look up the product in the table
        key = (op1.name, op2.name)
        result = Hamiltonian._PAULI_PRODUCT_TABLE.get(key)
        if result is None:
            raise InvalidHamiltonianOperation(f"Multiplying {op1} and {op2} not supported.")
        phase, op_cls = result

        # By convention, an I operator is always I(0) in this code
        if op_cls is PauliI:
            return phase, PauliI(0)
        # Otherwise, keep the same qubit
        return phase, op_cls(op1.qubit)

    # ------- Public arithmetic operators --------

    def __add__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        out = copy.copy(self)
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError("Term provided contains generic variables that are not Parameter.")
        out._add_inplace(other)
        return out.simplify()

    def __radd__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError("Term provided contains generic variables that are not Parameter.")
        return self.__add__(other)

    def __sub__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError("Term provided contains generic variables that are not Parameter.")
        out = copy.copy(self)
        out._sub_inplace(other)
        return out.simplify()

    def __rsub__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        # (other - self)
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError("Term provided contains generic variables that are not Parameter.")
        out = copy.copy(other if isinstance(other, Hamiltonian) else Hamiltonian() + other)
        out._sub_inplace(self)
        return out.simplify()

    def __mul__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError("Term provided contains generic variables that are not Parameter.")
        out = copy.copy(self)
        out._mul_inplace(other)
        return out.simplify()

    def __rmul__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError("Term provided contains generic variables that are not Parameter.")
        if isinstance(other, Hamiltonian):
            out = copy.copy(other)
            out._mul_inplace(self)
            return out.simplify()
        return self.__mul__(other)

    def __truediv__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        out = copy.copy(self)
        out._div_inplace(other)
        return out.simplify()

    def __rtruediv__(self, other: Number | PauliOperator | Hamiltonian) -> Hamiltonian:
        # (other / self)
        raise InvalidHamiltonianOperation("Division by operators is not supported")

    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __itruediv__ = __truediv__

    def _add_inplace(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> None:
        if isinstance(other, Hamiltonian):
            # If it's empty, do nothing
            if not other.elements:
                return
            # Otherwise, add each term
            for key, val in other._elements.items():  # noqa: SLF001
                self._elements[key] += val

            self._parameters.update(other.parameters)
        elif isinstance(other, PauliOperator):
            # Just add 1 to that single operator key
            self._elements[other,] += 1
        elif isinstance(other, (int, float, complex)):
            if other == 0:
                return
            # Add the scalar to (I(0),)
            self._elements[PauliI(0),] += other
        elif isinstance(other, (Term, Parameter)):
            if isinstance(other, Term):
                if not other.is_parameterized_term():
                    raise ValueError(
                        "Only Parameters are allowed to be used in hamiltonians. Generic Variables are not supported"
                    )
                self._parameters.update({v.label: v for v in other if isinstance(v, Parameter)})
            else:
                self._parameters[other.label] = other
            self._elements[PauliI(0),] += other
        else:
            raise InvalidHamiltonianOperation(f"Invalid addition between Hamiltonian and {other.__class__.__name__}.")

    def _sub_inplace(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> None:
        if isinstance(other, Hamiltonian):
            for key, val in other._elements.items():  # noqa: SLF001
                self._elements[key] -= val
            self._parameters.update(other._parameters)  # noqa: SLF001
        elif isinstance(other, PauliOperator):
            self._elements[other,] -= 1
        elif isinstance(other, (int, float, complex)):
            if other == 0:
                return
            self._elements[PauliI(0),] -= other
        elif isinstance(other, (Term, Parameter)):
            if isinstance(other, Term):
                if not other.is_parameterized_term():
                    raise ValueError(
                        "Only Parameters are allowed to be used in hamiltonians. Generic Variables are not supported"
                    )
                self._parameters.update({v.label: v for v in other if isinstance(v, Parameter)})
            else:
                self._parameters[other.label] = other
            self._elements[PauliI(0),] -= other
        else:
            raise InvalidHamiltonianOperation(
                f"Invalid subtraction between Hamiltonian and {other.__class__.__name__}."
            )

    def _mul_inplace(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> None:
        if isinstance(other, (int, float, complex)):
            # 0 short-circuit
            if other == 0:
                # everything becomes 0
                self._elements.clear()
                return None
            # 1 short-circuit
            if other == 1:
                return None
            # scale all coefficients
            for k in self._elements:
                self._elements[k] *= other
            return None

        if isinstance(other, (Term, Parameter)):
            if isinstance(other, Term):
                if not other.is_parameterized_term():
                    raise ValueError(
                        "Only Parameters are allowed to be used in hamiltonians. Generic Variables are not supported"
                    )
                self._parameters.update({v.label: v for v in other if isinstance(v, Parameter)})
            else:
                self._parameters[other.label] = other
            for k in self._elements:
                self._elements[k] *= other
            return None

        if isinstance(other, PauliOperator):
            # Convert single PauliOperator -> Hamiltonian with 1 key
            # Then do the single-key Hamiltonian path below
            other = other.to_hamiltonian()

        if isinstance(other, Hamiltonian):
            if not other.elements:
                # Multiply by "0" Hamiltonian => 0
                self._elements.clear()
                return None

            # Check if 'other' is purely scalar identity => short-circuit
            if len(other.elements) == 1:
                ((ops2, c2),) = other._elements.items()  # single item  # noqa: SLF001
                if len(ops2) == 1:
                    op2 = ops2[0]
                    if op2.name == "I" and op2.qubit == 0:
                        # effectively scalar c2
                        return self._mul_inplace(c2)

            # Otherwise, we do the general multiply
            new_dict: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = defaultdict(complex)
            for ops1, c1 in self._elements.items():
                for ops2, c2 in other._elements.items():  # noqa: SLF001
                    phase, new_ops = self._multiply_sets(ops1, ops2)
                    new_dict[new_ops] += phase * c1 * c2
            self._elements = new_dict
            self._parameters.update(other._parameters)  # noqa: SLF001

        else:
            raise InvalidHamiltonianOperation(
                f"Invalid multiplication between Hamiltonian and {other.__class__.__name__}."
            )
        return None

    def _div_inplace(self, other: Number | PauliOperator | Hamiltonian) -> None:
        # Only valid for scalars
        if not isinstance(other, (int, float, complex)):
            raise InvalidHamiltonianOperation("Division by operators is not supported")
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        self._mul_inplace(1 / other)
