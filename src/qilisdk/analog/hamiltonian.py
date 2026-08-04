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
from itertools import combinations, product
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix, kron, spmatrix

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.qtensor import QTensor
from qilisdk.core.types import Number
from qilisdk.core.variables import BaseVariable, Parameter, Term
from qilisdk.settings import get_settings
from qilisdk.utils.hashing import hash as qili_hash
from qilisdk.yaml import yaml

from .exceptions import InvalidHamiltonianOperation

if TYPE_CHECKING:
    from collections.abc import Iterator

_DIVISION_BY_OPERATORS_MESSAGE = "Division by operators is not supported"
_GENERIC_VARIABLE_IN_TERM_MESSAGE = "Term provided contains generic variables that are not Parameter."
_GENERIC_VARIABLE_IN_HAMILTONIAN_MESSAGE = (
    "Only Parameters are allowed to be used in hamiltonians. Generic Variables are not supported"
)


def _complex_dtype() -> np.dtype:
    return np.dtype(get_settings().complex_precision.dtype)


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
        op = PauliX(qubit)
    elif name == "Y":
        op = PauliY(qubit)
    elif name == "I":
        op = PauliI(qubit)
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
    A generic abstract Pauli operator that acts on one qubit.

    Example:
        .. code-block:: python

            from qilisdk.analog import PauliX

            op = PauliX(0)

    Flyweight usage: do NOT instantiate directly—use PauliX(q), PauliY(q), etc.

    Note: You can also use the factory functions X(q), Y(q), Z(q), I(q) to get a Hamiltonian object.
    """

    _NAME: ClassVar[str]
    _MATRIX: ClassVar[np.ndarray]
    _MATRIX_CACHE: ClassVar[dict[np.dtype, np.ndarray]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._MATRIX_CACHE = {}

    def __init__(self, qubit: int) -> None:
        # QSDK-05: reject negative qubit indices at construction (the upper bound
        # depends on the Hamiltonian / circuit and is enforced at execution).
        if qubit < 0:
            raise ValueError(f"Qubit index must be non-negative, got {qubit}.")
        self._qubit = qubit

    @property
    def qubit(self) -> int:
        return self._qubit

    @property
    def name(self) -> str:
        return self._NAME

    @classmethod
    def matrix_for_dtype(cls, dtype: np.dtype) -> np.ndarray:
        cached = cls._MATRIX_CACHE.get(dtype)
        if cached is None:
            cached = cls._MATRIX.astype(dtype, copy=False)
            cls._MATRIX_CACHE[dtype] = cached
        return cached

    @property
    def matrix(self) -> np.ndarray:
        return self.matrix_for_dtype(_complex_dtype())

    def to_hamiltonian(self) -> Hamiltonian:
        """Convert this single operator to a Hamiltonian with one term.

        Returns:
            Hamiltonian: The converted Hamiltonian.
        """
        return Hamiltonian({(self,): 1})

    def __hash__(self) -> int:
        return qili_hash(self._NAME, self._qubit)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Hamiltonian):
            return other == self
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
        raise InvalidHamiltonianOperation(_DIVISION_BY_OPERATORS_MESSAGE)

    __itruediv__ = __truediv__


###############################################################################
# Concrete Flyweight Operator Classes
###############################################################################
@yaml.register_class
class PauliZ(PauliOperator):
    _NAME: ClassVar[str] = "Z"
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, -1]], dtype=_complex_dtype())


@yaml.register_class
class PauliX(PauliOperator):
    _NAME: ClassVar[str] = "X"
    _MATRIX: ClassVar[np.ndarray] = np.array([[0, 1], [1, 0]], dtype=_complex_dtype())


@yaml.register_class
class PauliY(PauliOperator):
    _NAME: ClassVar[str] = "Y"
    _MATRIX: ClassVar[np.ndarray] = np.array([[0, -1j], [1j, 0]], dtype=_complex_dtype())


@yaml.register_class
class PauliI(PauliOperator):
    _NAME: ClassVar[str] = "I"
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, 1]], dtype=_complex_dtype())


_PAULI_CLASS_BY_NAME: dict[str, type[PauliOperator]] = {cls._NAME: cls for cls in (PauliI, PauliX, PauliY, PauliZ)}


# Wrapping a lattice direction into a ring only adds a new bond once it spans at least 3 sites;
# below that the wrap-around bond would duplicate one the open lattice already has.
_MIN_PERIODIC_EXTENT = 3


def _validate_nqubits(nqubits: int, minimum: int = 1, label: str = "") -> None:
    """
    Validate the number of qubits given for a certain Hamiltonian constructor.

    Args:
        nqubits (int): the qubit count to validate.
        minimum (int, optional): the smallest count the model can be built on, e.g. 2 for a model
            with two-qubit couplings. Defaults to 1.
        label (str, optional): the model name, used in the error message. Defaults to "".

    Raises:
        ValueError: if ``nqubits`` is not greater than zero.
        ValueError: if ``nqubits`` is below ``minimum``.
    """
    if nqubits <= 0:
        raise ValueError(f"nqubits must be greater than zero, got {nqubits}.")
    if nqubits < minimum:
        raise ValueError(f"{label} Hamiltonians need at least {minimum} qubits, got {nqubits}.")


def _uniform(generator: np.random.Generator, coefficient_range: tuple[float, float]) -> float:
    """
    Draw a single coefficient uniformly at random from a range.

    Args:
        generator (np.random.Generator): the random number generator to draw from.
        coefficient_range (tuple[float, float]): the range to draw from.

    Returns:
        float: the drawn coefficient.
    """
    return float(generator.uniform(low=coefficient_range[0], high=coefficient_range[1]))


def _validate_coefficient_range(coefficient_range: tuple[float, float], name: str) -> None:
    """
    Validate that a sampling range is well ordered.

    Args:
        coefficient_range (tuple[float, float]): the range to validate.
        name (str): the name of the range, used in the error message.

    Raises:
        ValueError: if the lower bound is greater than the upper bound.
    """
    if coefficient_range[0] > coefficient_range[1]:
        raise ValueError(f"{name} must be a (low, high) pair with low <= high, got {coefficient_range}.")


# Cache sparse single-qubit matrices once per dtype to avoid rebuilding them for every term.
_SINGLE_QUBIT_SPARSE: dict[tuple[str, np.dtype], csr_matrix] = {}


def _get_single_qubit_sparse_matrix(name: str) -> csr_matrix:
    dtype = _complex_dtype()
    key = (name, dtype)
    cached = _SINGLE_QUBIT_SPARSE.get(key)
    if cached is None:
        pauli_cls = _PAULI_CLASS_BY_NAME[name]
        cached = csr_matrix(pauli_cls.matrix_for_dtype(dtype), dtype=dtype)
        _SINGLE_QUBIT_SPARSE[key] = cached
    return cached


@yaml.register_class
class Hamiltonian(Parameterizable):
    """
    Represent a Hamiltonian expressed as a linear combination of Pauli operators.

    Example:
        .. code-block:: python

            from qilisdk.analog.hamiltonian import Hamiltonian, X, Z

            H = X(0) * X(1) + Z(1)
    """

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
        """
        Build a Hamiltonian from a mapping of Pauli operator products to coefficients.

        Args:
            elements (dict[tuple[PauliOperator, ...], complex | Term | Parameter], optional):
                Mapping from operator tuples to numerical coefficients or symbolic parameters. For example:

                .. code-block:: python

                    {
                        (Z(0), Y(1)): 1.0,
                        (X(1),): 1j,
                    }

                Defaults to None, which creates an empty Hamiltonian.

        Raises:
            ValueError: If the provided coefficients include generic variables instead of parameters.
        """
        super().__init__()
        self._elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = defaultdict(complex)
        if elements:
            for key, val in elements.items():
                if isinstance(val, Term):
                    for v in val.variables():
                        if isinstance(v, Parameter):
                            self._add_parameter(v.label, v)
                        else:
                            raise ValueError(_GENERIC_VARIABLE_IN_HAMILTONIAN_MESSAGE)
                elif isinstance(val, BaseVariable):
                    if isinstance(val, Parameter):
                        self._add_parameter(val.label, val)

                    else:
                        raise ValueError(_GENERIC_VARIABLE_IN_HAMILTONIAN_MESSAGE)
                self._elements[key] += val
            self.simplify()

    @property
    def nqubits(self) -> int:
        """Number of qubits on which the Hamiltonian acts."""
        qubits = {op.qubit for key in self._elements for op in key}

        return max(qubits) + 1 if qubits else 0

    @property
    def elements(self) -> dict[tuple[PauliOperator, ...], complex]:
        """Return the stored operator-coefficient mapping with symbolic terms evaluated."""
        return {
            k: (
                v if isinstance(v, complex) else (v.evaluate({}) if isinstance(v, Term) else v.evaluate())  # ty:ignore[unresolved-attribute]
            )
            for k, v in self._elements.items()
        }

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

    def _apply_operator_on_qubit(self, terms: list[PauliOperator], padding: int = 0) -> spmatrix:
        """Get the matrix representation of a single term by taking the tensor product
        of operators acting on each qubit. For qubits with no operator in `terms`,
        the identity is used.

        Args:
            terms (list[PauliOperator]): A list of Pauli operators in the term.

        Returns:
            spmatrix: The full matrix representation of the term.

        Raises:
            ValueError: If multiple operators act on the same qubit.
        """

        total_qubits = self.nqubits + padding
        if total_qubits == 0:
            return csr_matrix((1, 1), dtype=_complex_dtype())

        ordered_terms = sorted(terms, key=lambda op: op.qubit)

        # Check we don't have multiple operators on the same qubit
        qubit_indices = [op.qubit for op in ordered_terms]
        if len(qubit_indices) != len(set(qubit_indices)):
            raise ValueError("The list should not contain multiple operators acting on the same qubit.")

        identity_single = _get_single_qubit_sparse_matrix("I")
        idx = 0
        next_op = ordered_terms[0] if ordered_terms else None
        result: spmatrix | None = None

        for qubit in range(total_qubits):
            if next_op is not None and next_op.qubit == qubit:
                single = _get_single_qubit_sparse_matrix(next_op.name)
                idx += 1
                next_op = ordered_terms[idx] if idx < len(ordered_terms) else None
            else:
                single = identity_single

            result = single if result is None else kron(result, single, format="csr")

        # Added for type safety, but should never occur
        return result if result is not None else csr_matrix((2**total_qubits, 2**total_qubits), dtype=_complex_dtype())

    def to_matrix(self) -> spmatrix:
        """Return the full matrix representation of the Hamiltonian by summing over all terms.

        Returns:
            spmatrix: The sparse matrix representation of the Hamiltonian.
        """
        logger.debug("[Hamiltonian] Building sparse matrix representation over {} qubits", self.nqubits)
        dim = 2**self.nqubits
        # Initialize a zero matrix of the appropriate dimension.
        result = csr_matrix((dim, dim), dtype=_complex_dtype())
        for coeff, term in self:
            result += coeff * self._apply_operator_on_qubit(term)
        return result

    def to_qtensor(self, total_nqubits: int | None = None) -> QTensor:
        """Return the Hamiltonian as a ``QTensor`` built from the sparse matrix representation.

        Args:
            total_nqubits (int, optional): Specify the total number of qubits that this hamiltonian acts on. Defaults to None.

        Returns:
            QTensor: The QTensor object representation of the Hamiltonian.

        Raises:
            ValueError: If the total_nqubits provided is lower than the number of qubits effected by the hamiltonian.
        """

        nqubits = total_nqubits or self.nqubits
        padding = nqubits - self.nqubits
        if nqubits < self.nqubits:
            raise ValueError(
                f"The total number of qubits can't be less than the number of the qubits effected by this hamiltonian ({self.nqubits})"
            )

        logger.debug("[Hamiltonian] Building QTensor representation over {} qubits", nqubits)
        dim = 2 ** (nqubits)

        # Initialize a zero matrix of the appropriate dimension.
        result = csr_matrix((dim, dim), dtype=_complex_dtype())
        for coeff, term in self:
            result += coeff * self._apply_operator_on_qubit(term, padding=padding)
        return QTensor(result)

    def get_static_hamiltonian(self) -> Hamiltonian:
        """Return a Hamiltonian containing only constant coefficients."""
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
        if isinstance(other, QTensor):
            other = Hamiltonian.from_qtensor(other)
        if not isinstance(other, Hamiltonian):
            return False
        return dict(self._elements) == dict(other._elements)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return qili_hash(self._elements)

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

    def get_commuting_partitions(self) -> list[dict[tuple[PauliOperator, ...], complex | Term | Parameter]]:
        """
        Split the Hamiltonian into a list of partitions, each containing commuting terms.

        For now this is a greedy algorithm, but a smarter graph-coloring approach could be used later.

        Returns:
            list[dict[tuple[PauliOperator, ...], complex | Term | Parameter]]:
                A list of dictionaries, each representing a partition of the Hamiltonian containing commuting terms.
        """
        logger.debug("[Hamiltonian] Partitioning Hamiltonian into commuting groups")
        partitions: list[dict[tuple[PauliOperator, ...], complex | Term | Parameter]] = []

        # Check each term with each partition
        for term, coeff in self.elements.items():
            placed = False
            for partition in partitions:
                # Check if the term commutes with all terms in the current partition.
                # Both are single Pauli strings, so the parity check is exact and avoids
                # building intermediate Hamiltonians.
                if all(self._pauli_strings_commute(term, other_term) for other_term in partition):
                    # If so, add it to this partition
                    partition[term] = coeff
                    placed = True
                    break

            # Otherwise create a new partition for this term
            if not placed:
                partitions.append({term: coeff})

        return partitions

    @classmethod
    def transverse_field(cls, nqubits: int, x_coefficient: float = 1.0) -> Hamiltonian:
        """
        Build a uniform transverse field, :math:`\\sum_i h\\, X_i`.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                H = Hamiltonian.transverse_field(nqubits=2, x_coefficient=1.3)
                # 1.3 X(0) + 1.3 X(1)

        Args:
            nqubits (int): the number of qubits the Hamiltonian acts on.
            x_coefficient (float, optional): the field strength on every qubit. Defaults to 1.0.

        Returns:
            Hamiltonian: the transverse-field Hamiltonian.

        Raises:
            ValueError: if ``nqubits`` is not greater than zero.
        """
        _validate_nqubits(nqubits)
        logger.debug("[Hamiltonian] Building transverse field over {} qubits", nqubits)
        return cls({(_get_pauli("X", qubit),): x_coefficient for qubit in range(nqubits)})

    @classmethod
    def longitudinal_field(cls, nqubits: int, z_coefficient: float = 1.0) -> Hamiltonian:
        """
        Build a uniform longitudinal field, :math:`\\sum_i h\\, Z_i`.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                H = Hamiltonian.longitudinal_field(nqubits=2, z_coefficient=1.3)
                # 1.3 Z(0) + 1.3 Z(1)

        Args:
            nqubits (int): the number of qubits the Hamiltonian acts on.
            z_coefficient (float, optional): the field strength on every qubit. Defaults to 1.0.

        Returns:
            Hamiltonian: the longitudinal-field Hamiltonian.

        Raises:
            ValueError: if ``nqubits`` is not greater than zero.
        """
        _validate_nqubits(nqubits)
        logger.debug("[Hamiltonian] Building longitudinal field over {} qubits", nqubits)
        return cls({(_get_pauli("Z", qubit),): z_coefficient for qubit in range(nqubits)})

    @classmethod
    def ising(cls, nqubits: int, zz_coefficient: float = 1.0, z_coefficient: float = 0.0) -> Hamiltonian:
        """
        Build an all-to-all Ising Hamiltonian, :math:`\\sum_{i<j} J\\, Z_i Z_j + \\sum_i h\\, Z_i`.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                H = Hamiltonian.ising(nqubits=2, zz_coefficient=2.0)
                # 2 Z(0) Z(1)

        Args:
            nqubits (int): the number of qubits the Hamiltonian acts on. Must be at least 2.
            zz_coefficient (float, optional): the coupling on every pair of qubits. Defaults to 1.0.
            z_coefficient (float, optional): the longitudinal field on every qubit. Defaults to 0.0,
                which leaves the field out entirely.

        Returns:
            Hamiltonian: the Ising Hamiltonian.

        Raises:
            ValueError: if ``nqubits`` is less than 2.
        """
        _validate_nqubits(nqubits, minimum=2, label="Ising")
        logger.debug("[Hamiltonian] Building Ising model over {} qubits", nqubits)
        elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = {
            (_get_pauli("Z", first), _get_pauli("Z", second)): zz_coefficient
            for first, second in combinations(range(nqubits), 2)
        }
        elements.update({(_get_pauli("Z", qubit),): z_coefficient for qubit in range(nqubits)})
        return cls(elements)

    @classmethod
    def ising_chain(
        cls,
        nqubits: int,
        zz_coefficient: float = 1.0,
        z_coefficient: float = 0.0,
        periodic: bool = False,
    ) -> Hamiltonian:
        """
        Build a 1D nearest-neighbour Ising chain, :math:`\\sum_i J\\, Z_i Z_{i+1} + \\sum_i h\\, Z_i`.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                H = Hamiltonian.ising_chain(nqubits=3, zz_coefficient=2.0)
                # 2 Z(0) Z(1) + 2 Z(1) Z(2)

                # Closing the chain into a ring adds the bond between the two ends.
                H = Hamiltonian.ising_chain(nqubits=3, zz_coefficient=2.0, periodic=True)
                # 2 Z(0) Z(1) + 2 Z(1) Z(2) + 2 Z(0) Z(2)

        Args:
            nqubits (int): the number of qubits in the chain. Must be at least 2.
            zz_coefficient (float, optional): the coupling on every bond. Defaults to 1.0.
            z_coefficient (float, optional): the longitudinal field on every qubit. Defaults to 0.0,
                which leaves the field out entirely.
            periodic (bool, optional): whether to close the chain into a ring by coupling the last
                qubit back to the first. Ignored for fewer than 3 qubits, where the ring bond would
                duplicate the one bond the chain already has. Defaults to False.

        Returns:
            Hamiltonian: the Ising chain Hamiltonian.

        Raises:
            ValueError: if ``nqubits`` is less than 2.
        """
        _validate_nqubits(nqubits, minimum=2, label="Ising chain")
        logger.debug("[Hamiltonian] Building Ising chain over {} qubits (periodic={})", nqubits, periodic)
        bonds = [(qubit, qubit + 1) for qubit in range(nqubits - 1)]
        if periodic and nqubits >= _MIN_PERIODIC_EXTENT:
            bonds.append((0, nqubits - 1))
        elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = {
            (_get_pauli("Z", first), _get_pauli("Z", second)): zz_coefficient for first, second in bonds
        }
        elements.update({(_get_pauli("Z", qubit),): z_coefficient for qubit in range(nqubits)})
        return cls(elements)

    @classmethod
    def ising_grid(
        cls,
        rows: int,
        columns: int,
        zz_coefficient: float = 1.0,
        z_coefficient: float = 0.0,
        periodic: bool = False,
    ) -> Hamiltonian:
        """
        Build a 2D nearest-neighbour Ising model on a ``rows`` x ``columns`` square lattice.

        Qubits are numbered in row-major order, so the qubit at ``(row, column)`` has index
        ``row * columns + column``, and the Hamiltonian acts on ``rows * columns`` qubits. Each qubit
        is coupled to its neighbour to the right and its neighbour below, giving the usual square
        lattice.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                # A 2x2 lattice: qubits 0 1 on the top row, 2 3 on the bottom.
                H = Hamiltonian.ising_grid(rows=2, columns=2)
                # Z(0) Z(1) + Z(0) Z(2) + Z(1) Z(3) + Z(2) Z(3)

        Args:
            rows (int): the number of lattice rows.
            columns (int): the number of lattice columns.
            zz_coefficient (float, optional): the coupling on every bond. Defaults to 1.0.
            z_coefficient (float, optional): the longitudinal field on every qubit. Defaults to 0.0,
                which leaves the field out entirely.
            periodic (bool, optional): whether to wrap the lattice into a torus by coupling each edge
                back to the opposite one. Wrapping is skipped along any direction shorter than 3
                sites, where it would duplicate an existing bond. Defaults to False.

        Returns:
            Hamiltonian: the Ising grid Hamiltonian.

        Raises:
            ValueError: if ``rows`` or ``columns`` is not greater than zero.
            ValueError: if the lattice holds fewer than 2 qubits.
        """
        if rows <= 0:
            raise ValueError(f"rows must be greater than zero, got {rows}.")
        if columns <= 0:
            raise ValueError(f"columns must be greater than zero, got {columns}.")
        nqubits = rows * columns
        _validate_nqubits(nqubits, minimum=2, label="Ising grid")
        logger.debug("[Hamiltonian] Building {}x{} Ising grid (periodic={})", rows, columns, periodic)

        def index(row: int, column: int) -> int:
            return row * columns + column

        bonds: list[tuple[int, int]] = []
        for row in range(rows):
            for column in range(columns):
                if column + 1 < columns:
                    bonds.append((index(row, column), index(row, column + 1)))
                elif periodic and columns >= _MIN_PERIODIC_EXTENT:
                    bonds.append((index(row, 0), index(row, column)))
                if row + 1 < rows:
                    bonds.append((index(row, column), index(row + 1, column)))
                elif periodic and rows >= _MIN_PERIODIC_EXTENT:
                    bonds.append((index(0, column), index(row, column)))

        elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = {
            (_get_pauli("Z", first), _get_pauli("Z", second)): zz_coefficient for first, second in bonds
        }
        elements.update({(_get_pauli("Z", qubit),): z_coefficient for qubit in range(nqubits)})
        return cls(elements)

    @classmethod
    def transverse_field_ising(
        cls,
        nqubits: int,
        x_coefficient: float = 1.0,
        zz_coefficient: float = 1.0,
        z_coefficient: float = 0.0,
    ) -> Hamiltonian:
        """
        Build an all-to-all transverse-field Ising Hamiltonian.

        .. math::

            H = \\sum_{i<j} J\\, Z_i Z_j + \\sum_i h_x\\, X_i + \\sum_i h_z\\, Z_i

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                H = Hamiltonian.transverse_field_ising(nqubits=2, x_coefficient=1.3, zz_coefficient=-2)
                # 1.3 X(0) + 1.3 X(1) - 2 Z(0) Z(1)

        Args:
            nqubits (int): the number of qubits the Hamiltonian acts on. Must be at least 2.
            x_coefficient (float, optional): the transverse field on every qubit. Defaults to 1.0.
            zz_coefficient (float, optional): the coupling on every pair of qubits. Defaults to 1.0.
            z_coefficient (float, optional): the longitudinal field on every qubit. Defaults to 0.0,
                which leaves the field out entirely.

        Returns:
            Hamiltonian: the transverse-field Ising Hamiltonian.

        Raises:
            ValueError: if ``nqubits`` is less than 2.
        """
        _validate_nqubits(nqubits, minimum=2, label="Transverse-field Ising")
        logger.debug("[Hamiltonian] Building transverse-field Ising model over {} qubits", nqubits)
        elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = {
            (_get_pauli("X", qubit),): x_coefficient for qubit in range(nqubits)
        }
        elements.update(
            {
                (_get_pauli("Z", first), _get_pauli("Z", second)): zz_coefficient
                for first, second in combinations(range(nqubits), 2)
            }
        )
        elements.update({(_get_pauli("Z", qubit),): z_coefficient for qubit in range(nqubits)})
        return cls(elements)

    @classmethod
    def xy(cls, nqubits: int, xx_coefficient: float = 1.0, yy_coefficient: float | None = None) -> Hamiltonian:
        """
        Build an all-to-all XY Hamiltonian, :math:`\\sum_{i<j} J_x X_i X_j + J_y Y_i Y_j`.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                H = Hamiltonian.xy(nqubits=2, xx_coefficient=0.5)
                # 0.5 X(0) X(1) + 0.5 Y(0) Y(1)

        Args:
            nqubits (int): the number of qubits the Hamiltonian acts on. Must be at least 2.
            xx_coefficient (float, optional): the ``XX`` coupling on every pair of qubits. Defaults to 1.0.
            yy_coefficient (float, optional): the ``YY`` coupling on every pair of qubits. Defaults to
                None, which reuses ``xx_coefficient`` and so gives the isotropic XY model.

        Returns:
            Hamiltonian: the XY Hamiltonian.

        Raises:
            ValueError: if ``nqubits`` is less than 2.
        """
        _validate_nqubits(nqubits, minimum=2, label="XY")
        if yy_coefficient is None:
            yy_coefficient = xx_coefficient
        logger.debug("[Hamiltonian] Building XY model over {} qubits", nqubits)
        pairs = list(combinations(range(nqubits), 2))
        elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = {
            (_get_pauli("X", first), _get_pauli("X", second)): xx_coefficient for first, second in pairs
        }
        elements.update({(_get_pauli("Y", first), _get_pauli("Y", second)): yy_coefficient for first, second in pairs})
        return cls(elements)

    @classmethod
    def heisenberg(
        cls,
        nqubits: int,
        xx_coefficient: float = 1.0,
        yy_coefficient: float | None = None,
        zz_coefficient: float | None = None,
        z_coefficient: float = 0.0,
    ) -> Hamiltonian:
        """
        Build an all-to-all Heisenberg Hamiltonian.

        .. math::

            H = \\sum_{i<j} \\left( J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j \\right) + \\sum_i h\\, Z_i

        Leaving ``yy_coefficient`` and ``zz_coefficient`` at their defaults gives the isotropic XXX
        model; setting ``zz_coefficient`` alone gives the XXZ model; setting all three independently
        gives the fully anisotropic XYZ model.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                # Isotropic XXX model.
                H = Hamiltonian.heisenberg(nqubits=2, xx_coefficient=0.5)
                # 0.5 X(0) X(1) + 0.5 Y(0) Y(1) + 0.5 Z(0) Z(1)

                # XXZ model with an anisotropic ZZ coupling.
                H = Hamiltonian.heisenberg(nqubits=2, xx_coefficient=1.0, zz_coefficient=0.3)

        Args:
            nqubits (int): the number of qubits the Hamiltonian acts on. Must be at least 2.
            xx_coefficient (float, optional): the ``XX`` coupling on every pair of qubits. Defaults to 1.0.
            yy_coefficient (float, optional): the ``YY`` coupling on every pair of qubits. Defaults to
                None, which reuses ``xx_coefficient``.
            zz_coefficient (float, optional): the ``ZZ`` coupling on every pair of qubits. Defaults to
                None, which reuses ``xx_coefficient``.
            z_coefficient (float, optional): the longitudinal field on every qubit. Defaults to 0.0,
                which leaves the field out entirely.

        Returns:
            Hamiltonian: the Heisenberg Hamiltonian.

        Raises:
            ValueError: if ``nqubits`` is less than 2.
        """
        _validate_nqubits(nqubits, minimum=2, label="Heisenberg")
        if yy_coefficient is None:
            yy_coefficient = xx_coefficient
        if zz_coefficient is None:
            zz_coefficient = xx_coefficient
        logger.debug("[Hamiltonian] Building Heisenberg model over {} qubits", nqubits)
        pairs = list(combinations(range(nqubits), 2))
        elements: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = {
            (_get_pauli("X", first), _get_pauli("X", second)): xx_coefficient for first, second in pairs
        }
        elements.update({(_get_pauli("Y", first), _get_pauli("Y", second)): yy_coefficient for first, second in pairs})
        elements.update({(_get_pauli("Z", first), _get_pauli("Z", second)): zz_coefficient for first, second in pairs})
        elements.update({(_get_pauli("Z", qubit),): z_coefficient for qubit in range(nqubits)})
        return cls(elements)

    def randomize_coefficients(self, range: tuple[float, float] = (-1.0, 1.0), seed: int = 1) -> Hamiltonian:
        """
        Replace every coefficient with one drawn uniformly at random, in place.

        Example:
            .. code-block:: python

                from qilisdk.analog import Hamiltonian

                # A Sherrington-Kirkpatrick instance.
                H = Hamiltonian.ising(nqubits=4).randomize_coefficients()

                # A random-field Ising chain.
                H = Hamiltonian.ising_chain(nqubits=4, z_coefficient=1.0).randomize_coefficients(range=(-4.0, 4.0))

        Args:
            range (tuple[float, float], optional): the range from which each coefficient is drawn
                uniformly at random. Defaults to (-1.0, 1.0).
            seed (int, optional): the seed for the random number generator. Defaults to 1.

        Returns:
            Hamiltonian: this Hamiltonian, with randomized coefficients.

        Raises:
            ValueError: if ``range`` is not a well-ordered ``(low, high)`` pair.
        """
        _validate_coefficient_range(range, "range")
        logger.debug("[Hamiltonian] Randomizing {} coefficients from {}", len(self._elements), range)
        generator = np.random.default_rng(seed)
        for operators in self._elements:
            # Stored as complex, like every other coefficient: `elements` and `simplify` both rely on
            # numeric coefficients being complex instances, and a bare float is not one.
            self._elements[operators] = complex(_uniform(generator, range))
        # Every coefficient is now a plain number, so no symbolic parameters are left to track.
        self._parameters.clear()
        return self.simplify()

    @classmethod
    def from_qtensor(cls, tensor: QTensor, tol: float | None = None, prune: float | None = None) -> Hamiltonian:
        """
        Expand a qtensor (dense operator) on n qubits into a sum of Pauli strings,
        returning a qilisdk.analog.Hamiltonian.

        Args:
            tol (float): Hermiticity check tolerance. Defaults to global zero tolerance setting.
            prune (float): Drop coefficients whose absolute value satisfies ``abs(c) < prune`` to reduce numerical noise. Defaults to global zero tolerance setting.

        Returns:
            Hamiltonian: Sum_{P in {I,X,Y,Z}^{⊗ n}} c_P * P  with c_P = Tr(qt * P) / 2^n

        Raises:
            ValueError: If the input is not square, not a power-of-two dimension, or not Hermitian w.r.t. `tol`.
        """
        if tol is None:
            tol = get_settings().atol
        if prune is None:
            prune = get_settings().atol

        A = np.asarray(tensor.dense())

        dim = tensor.shape[0]
        n = round(np.log2(dim))
        logger.debug("[Hamiltonian] Expanding {}-qubit dense operator into Pauli basis", n)
        if not tensor.is_hermitian():
            raise ValueError("Matrix is not Hermitian within tolerance; cannot form a Hamiltonian.")

        # QiliSDK Pauli constructors indexed by qubit id
        pauli_for: dict[int, Callable[[int], Hamiltonian]] = {0: I, 1: X, 2: Y, 3: Z}

        # Normalization from orthonormality: Tr(P_a P_b) = 2^n δ_ab
        norm = 1.0 / (2**n)

        # Prebuild per-qubit operator “letters” so we can construct full strings quickly

        H = Hamiltonian()  # start additive Hamiltonian expression
        # Full Pauli basis (includes identity on any subset automatically)
        for word in product((0, 1, 2, 3), repeat=n):
            # Compose the word into a QiliSDK operator acting on the proper qubits
            # Example word=(3,1,0) for n=3 -> Z(0)*X(1)*I(2)
            op = Hamiltonian()
            for q, letter in enumerate(word):
                # multiply by the operator on qubit q
                op = op * pauli_for[letter](q) if op != 0 else pauli_for[letter](q)

            # Convert to dense once; no padding needed because it spans all n qubits
            P_dense = op.to_qtensor(n).dense()

            # Coefficient c_P = Tr(A P) / 2^n  (P is Hermitian)
            c = norm * np.trace(A @ P_dense)

            # Numerical safety: coefficients should be real for Hermitian A and P
            if abs(c.imag) < tol:
                c = c.real

            if abs(c) > prune:
                H += c * op

        # Optional: verify round-trip (use a slightly looser atol to tolerate pruning)
        if not np.allclose(H.to_qtensor(n).dense(), A, atol=max(10 * prune, 1e-9)):
            # If this triggers, consider lowering `prune` or raising `tol`.
            raise ValueError("Pauli expansion failed round-trip check; try adjusting tolerances.")

        return H

    @classmethod
    def parse(cls, hamiltonian_str: str) -> Hamiltonian:
        logger.debug("[Hamiltonian] Parsing Hamiltonian from string")
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
                # Make sure it's not empty
                if text:
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

    def commutator(self, h: Hamiltonian) -> Hamiltonian:
        """compute the commutator of the current hamiltonian with another hamiltonian (h)

        Args:
            h (Hamiltonian): the second hamiltonian.

        Returns:
            Hamiltonian: the commutator.
        """
        return self * h - h * self

    def anticommutator(self, h: Hamiltonian) -> Hamiltonian:
        """compute the anticommutator of the current hamiltonian with another hamiltonian (h)

        Args:
            h (Hamiltonian): the second hamiltonian.

        Returns:
            Hamiltonian: the anticommutator.
        """
        return self * h + h * self

    def commutes_with(self, h: Hamiltonian) -> bool:
        """Check whether this Hamiltonian commutes with another one (``[self, h] == 0``).

        This is faster than materialising the full commutator ``self * h - h * self``:
        every pair of Pauli strings either commutes or anticommutes, and which one
        holds can be decided with a cheap qubit-overlap parity check instead of a
        matrix/operator product. Commuting pairs contribute nothing to the commutator
        and are skipped entirely, so only the anticommuting pairs (for which
        ``[P, Q] = 2 P Q``) are ever multiplied out and accumulated.

        Args:
            h (Hamiltonian): the Hamiltonian to test commutation against.

        Returns:
            bool: ``True`` if the two Hamiltonians commute, ``False`` otherwise.
        """
        residual: dict[tuple[PauliOperator, ...], complex | Term | Parameter] = defaultdict(complex)
        for ops1, c1 in self._elements.items():
            for ops2, c2 in h._elements.items():
                if Hamiltonian._pauli_strings_commute(ops1, ops2):
                    continue
                # Anticommuting strings: [P, Q] = P Q - Q P = 2 P Q.
                phase, new_ops = self._multiply_sets(ops1, ops2)
                residual[new_ops] += 2 * phase * c1 * c2

        return Hamiltonian(residual) == Hamiltonian.ZERO

    def vector_norm(self) -> float:
        """
        Returns:
            float: the vector norm of the hamiltonian.
        """
        s = 0
        for coeff, _ in self:
            s += np.conj(coeff) * coeff
        return np.real(np.sqrt(s))

    def frobenius_norm(self) -> float:
        """
        Returns:
            float: the forbenius norm of the hamiltonian.
        """
        n = self.nqubits
        s = 0
        for coeff, _ in self:
            s += np.conj(coeff) * coeff
        return np.real(np.sqrt(s) * np.sqrt(2**n))

    def trace(self) -> Number:
        """
        Returns:
            float: the trace of the hamiltonian.
        """
        n = self.nqubits
        d = 2**n
        t = self._elements.get((PauliI(0),), 0)
        if isinstance(t, Parameter):
            return t.evaluate() * d
        if isinstance(t, Term):
            return t.evaluate({}) * d
        return t * d

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
    def _pauli_strings_commute(ops1: tuple[PauliOperator, ...], ops2: tuple[PauliOperator, ...]) -> bool:
        """Return whether two Pauli strings commute, using only a qubit-overlap parity check.

        Two Pauli strings always either commute or anticommute. They anticommute iff they
        disagree (both non-identity and a different Pauli) on an odd number of qubits, since
        single-qubit Paulis anticommute exactly when they are distinct and non-identity.
        """
        names2 = {op.qubit: op.name for op in ops2 if op.name != "I"}
        if not names2:
            return True
        anticommuting = 0
        for op in ops1:
            if op.name == "I":
                continue
            other = names2.get(op.qubit)
            if other is not None and other != op.name:
                anticommuting += 1
        return anticommuting % 2 == 0

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
            raise ValueError(_GENERIC_VARIABLE_IN_TERM_MESSAGE)
        out._add_inplace(other)
        return out.simplify()

    def __radd__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError(_GENERIC_VARIABLE_IN_TERM_MESSAGE)
        return self.__add__(other)

    def __sub__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError(_GENERIC_VARIABLE_IN_TERM_MESSAGE)
        out = copy.copy(self)
        out._sub_inplace(other)
        return out.simplify()

    def __rsub__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        # (other - self)
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError(_GENERIC_VARIABLE_IN_TERM_MESSAGE)
        out = copy.copy(other if isinstance(other, Hamiltonian) else Hamiltonian() + other)
        out._sub_inplace(self)
        return out.simplify()

    def __neg__(self) -> Hamiltonian:
        return -1 * self

    def __mul__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError(_GENERIC_VARIABLE_IN_TERM_MESSAGE)
        out = copy.copy(self)
        out._mul_inplace(other)
        return out.simplify()

    def __rmul__(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> Hamiltonian:
        if isinstance(other, Term) and not other.is_parameterized_term():
            raise ValueError(_GENERIC_VARIABLE_IN_TERM_MESSAGE)
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
        raise InvalidHamiltonianOperation(_DIVISION_BY_OPERATORS_MESSAGE)

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

            self._update_parameters(other._parameters)  # noqa: SLF001
        elif isinstance(other, PauliOperator):
            # Just add 1 to that single operator key
            self._elements[other,] += 1
        elif isinstance(other, (int, float, complex)):
            if abs(other) < get_settings().atol:
                return
            # Add the scalar to (I(0),)
            self._elements[PauliI(0),] += other
        elif isinstance(other, (Term, Parameter)):
            if isinstance(other, Term):
                if not other.is_parameterized_term():
                    raise ValueError(_GENERIC_VARIABLE_IN_HAMILTONIAN_MESSAGE)
                self._update_parameters({v.label: v for v in other if isinstance(v, Parameter)})
            else:
                self._add_parameter(other.label, other)
            self._elements[PauliI(0),] += other
        else:
            raise InvalidHamiltonianOperation(f"Invalid addition between Hamiltonian and {other.__class__.__name__}.")

    def _sub_inplace(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> None:
        if isinstance(other, Hamiltonian):
            for key, val in other._elements.items():  # noqa: SLF001
                self._elements[key] -= val
            self._update_parameters(other._parameters)  # noqa: SLF001
        elif isinstance(other, PauliOperator):
            self._elements[other,] -= 1
        elif isinstance(other, (int, float, complex)):
            if abs(other) < get_settings().atol:
                return
            self._elements[PauliI(0),] -= other
        elif isinstance(other, (Term, Parameter)):
            if isinstance(other, Term):
                if not other.is_parameterized_term():
                    raise ValueError(_GENERIC_VARIABLE_IN_HAMILTONIAN_MESSAGE)
                self._update_parameters({v.label: v for v in other if isinstance(v, Parameter)})
            else:
                self._add_parameter(other.label, other)
            self._elements[PauliI(0),] -= other
        else:
            raise InvalidHamiltonianOperation(
                f"Invalid subtraction between Hamiltonian and {other.__class__.__name__}."
            )

    def _mul_inplace(self, other: Number | PauliOperator | Hamiltonian | Term | Parameter) -> None:
        if isinstance(other, (int, float, complex)):
            # 0 short-circuit
            if abs(other) < get_settings().atol:
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
                    raise ValueError(_GENERIC_VARIABLE_IN_HAMILTONIAN_MESSAGE)
                self._update_parameters({v.label: v for v in other if isinstance(v, Parameter)})
            else:
                self._add_parameter(other.label, other)
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
            self._update_parameters(other._parameters)  # noqa: SLF001

        else:
            raise InvalidHamiltonianOperation(
                f"Invalid multiplication between Hamiltonian and {other.__class__.__name__}."
            )
        return None

    def _div_inplace(self, other: Number | PauliOperator | Hamiltonian) -> None:
        # Only valid for scalars
        if not isinstance(other, (int, float, complex)):
            raise InvalidHamiltonianOperation(_DIVISION_BY_OPERATORS_MESSAGE)
        if abs(other) < get_settings().atol:
            raise ZeroDivisionError("Cannot divide by zero.")
        self._mul_inplace(1 / other)
