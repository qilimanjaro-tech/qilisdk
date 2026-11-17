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

import math
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
from loguru import logger
from typing_extensions import Self

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core import Domain, Parameter, QTensor
from qilisdk.core.expression import Expression
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.settings import get_settings
from qilisdk.utils.hashing import hash as qili_hash
from qilisdk.utils.visualization import CircuitStyle
from qilisdk.yaml import yaml

from .exceptions import QubitOutOfRangeError
from .gates import BasicGate, Gate, _reject_analog_object

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qilisdk.analog.hamiltonian import PauliOperator
    from qilisdk.core.types import RealNumber

# Name used for this class in the error messages raised for analog objects.
_REJECT_TARGET = "a Circuit"

# Pauli rotation count above which double_bracket_iteration warns about circuit size.
_DOUBLE_BRACKET_ROTATION_WARNING = 100_000


def _complex_dtype() -> np.dtype:
    return get_settings().complex_precision.dtype


def _apply_gate_left(operator: np.ndarray, gate: Gate, nqubits: int) -> np.ndarray:
    gate_matrix = gate.matrix.dense()
    k = gate.nqubits
    if k == 0:
        return operator
    if k == nqubits and gate.qubits == tuple(range(nqubits)):
        return gate_matrix @ operator

    op_tensor = operator.reshape([2] * nqubits + [2] * nqubits)
    gate_tensor = gate_matrix.reshape([2] * k + [2] * k)

    output_axes = list(gate.qubits)
    contracted = np.tensordot(gate_tensor, op_tensor, axes=(list(range(k, 2 * k)), output_axes))

    output_set = set(output_axes)
    out_current = output_axes + [q for q in range(nqubits) if q not in output_set]
    out_positions = {q: i for i, q in enumerate(out_current)}
    perm_output = [out_positions[q] for q in range(nqubits)]
    perm = perm_output + list(range(nqubits, 2 * nqubits))

    return np.transpose(contracted, axes=perm).reshape(operator.shape)


@yaml.register_class
class Circuit(Parameterizable):
    def __init__(self, nqubits: int) -> None:
        """
        Initialize a Circuit instance with a specified number of qubits.

        Args:
            nqubits (int): The number of qubits in the circuit.
        """
        super(Circuit, self).__init__()
        self._nqubits: int = nqubits
        self._gates: list[Gate] = []
        self._init_state: np.ndarray = np.zeros(nqubits)
        self._parameters_link: dict[str, list[tuple[str, Gate]]] = {}
        logger.debug("[Circuit] Initialized Circuit with {} qubits", nqubits)

    @property
    def nqubits(self) -> int:
        """
        Retrieve the number of qubits in the circuit.

        Returns:
            int: The total number of qubits.
        """
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total number of parameters required by all parameterized gates in the circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """
        return len(self._parameters)

    @property
    def gates(self) -> list[Gate]:
        """
        Retrieve the list of gates in the circuit.

        Returns:
            list[Gate]: A list of gates that have been added to the circuit.
        """
        return self._gates

    def set_parameters(self, parameters: dict[str, RealNumber]) -> None:
        """Set the parameter values by their label. No need to provide the full list of parameters.

        Args:
            parameters (dict[str, float]): A dictionary with the labels of the parameters to be modified and their new value.

        Raises:
            ValueError: if the label provided doesn't correspond to a parameter defined in this circuit.
        """
        logger.trace("[Circuit] Setting parameters: {}", parameters)
        if not self.check_constraints(parameters):
            raise ValueError(
                f"New assignation of the parameters breaks the parameter constraints: \n{self.get_constraints()}"
            )
        for label, param in parameters.items():
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this circuit.")
            for link in self._parameters_link[label]:
                link[1].set_parameters({link[0]: param})

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        logger.trace("[Circuit] Setting parameter bounds: {}", ranges)
        for label, bound in ranges.items():
            if label not in self._parameters:
                raise ValueError(
                    f"The provided parameter label {label} is not defined in the list of parameters in this object."
                )
            for link in self._parameters_link[label]:
                link[1].set_parameter_bounds({link[0]: bound})

    def set_prefix(
        self,
        prefix: str,
        where: Callable[[Parameter], bool] | None = None,
    ) -> None:
        """Set a prefix on parameter labels.

        Args:
            prefix (str): Prefix to prepend to selected parameter labels.
            where (Callable[[Parameter], bool] | None): Optional predicate selecting local parameters.

        Notes:
            The ``where`` predicate is applied to local parameters only. Child parameterizable
            objects always receive the same prefix operation recursively.
        """
        logger.trace("[Circuit] Setting prefix '{}' for parameters with condition: {}", prefix, where)
        old_keys = list(self._filtered_parameter_map(where=where))
        for name in old_keys:
            if not name.startswith(prefix):
                _name = name.removeprefix(self._prefix) if self._prefix and name.startswith(self._prefix) else name
                self._parameters_link[prefix + _name] = self._parameters_link.pop(name)
        super().set_prefix(prefix=prefix, where=where)

    def _parse_params(self, gate: Gate) -> None:
        """Parse The parameters in the gate

        Args:
            gate (Gate): The gate to be parsed.
        """
        logger.trace("[Circuit] Parsing parameters for gate: {}", gate)
        if gate.is_parameterized:
            param_base_label = self.get_prefix() + f"{gate.name}({','.join(map(str, gate.qubits))})"
            for label in gate.get_parameter_names():
                _parameter_name = self._query_parameter_original_name(gate, label)
                if label == _parameter_name and label in gate.PARAMETER_NAMES:
                    parameter_label = param_base_label + f"_{label}_{len(self._parameters)}"
                else:
                    parameter_label = _parameter_name
                self._add_parameter_from(label, gate, new_label=parameter_label)
                if parameter_label not in self._parameters_link:
                    self._parameters_link[parameter_label] = [(label, gate)]
                else:
                    self._parameters_link[parameter_label].append((label, gate))

    def _add(self, gate: Gate) -> None:
        """
        Add a quantum gate to the circuit.

        Args:
            gate (Gate): The quantum gate or a list of quantum gates to be added to the circuit.

        Raises:
            NotAGateError: If the object to be added is an analog object rather than a gate.
            QubitOutOfRangeError: If any qubit index used by the gate is not within the circuit's qubit range.
        """
        logger.trace("[Circuit] Adding gate: {}", gate)
        _reject_analog_object(gate, _REJECT_TARGET)
        if any(qubit < 0 or qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError

        self._parse_params(gate)
        self._gates.append(gate)

    def add(self, gates: Gate | Iterable[Gate]) -> None:
        """
        Add a quantum gate to the circuit.

        Args:
            gates (Gate | list[Gate]): The quantum gate or a list of quantum gates to be added to the circuit.

        Raises:
            NotAGateError: If any of the objects to be added is an analog object rather than a gate.
        """
        logger.trace("[Circuit] Adding gates: {}", gates)
        _reject_analog_object(gates, _REJECT_TARGET)
        if isinstance(gates, Gate):
            self._add(gates)
            return
        for g in gates:
            self._add(g)

    def _insert(self, gate: Gate, index: int) -> None:
        """Insert a quantum gate to the circuit at a given index.

        Args:
            gate (Gate): The gate to be inserted.
            index (int): The index at which the gate is inserted.

        Raises:
            NotAGateError: If the object to be inserted is an analog object rather than a gate.
            QubitOutOfRangeError: If any qubit index used by the gate is not within the circuit's qubit range.
        """
        _reject_analog_object(gate, _REJECT_TARGET)
        if any(qubit < 0 or qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError

        logger.trace("[Circuit] Inserting gate: {} at index: {}", gate, index)
        self._parse_params(gate)
        self._gates.insert(index, gate)

    def insert(self, gates: Gate | Iterable[Gate], index: int | None = None) -> None:
        """Insert one or more quantum gates into the circuit at a given index.

        The gates are inserted consecutively, so their relative order is preserved.

        Args:
            gates (Gate | Iterable[Gate]): The gate or gates to be inserted.
            index (int | None, optional): The index at which the first gate is inserted, following the
                usual list indexing rules, so a negative value counts from the end of the circuit.
                Defaults to None, which appends the gates at the end of the circuit.

        Raises:
            QubitOutOfRangeError: If any qubit index used by a gate is not within the circuit's qubit range.
        """
        logger.trace("[Circuit] Inserting gates: {} at index: {}", gates, index)
        _reject_analog_object(gates, _REJECT_TARGET)
        if index is None:
            start = len(self._gates)
        elif index < 0:
            start = max(len(self._gates) + index, 0)
        else:
            start = index
        if isinstance(gates, Gate):
            self._insert(gates, start)
            return
        for offset, gate in enumerate(gates):
            self._insert(gate, start + offset)

    def append(self, circuit: Circuit) -> None:
        """Append circuit elements at the end of the current circuit.

        Args:
            circuit (Circuit): The circuit to be appended.

        Raises:
            QubitOutOfRangeError: If the appended circuit acts on more qubits than the current circuit.
        """
        logger.trace("[Circuit] Appending circuit: {}", circuit)
        if circuit.nqubits != self.nqubits:
            raise QubitOutOfRangeError(
                "the appended circuit contains different number of qubits than the current circuit."
            )

        for g in circuit.gates:
            self.add(g)

    def prepend(self, circuit: Circuit) -> None:
        """Prepend circuit elements to the beginning of the current circuit.

        Args:
            circuit (Circuit): The circuit to be prepended.

        Raises:
            QubitOutOfRangeError: If the circuit to be prepended acts on more qubits than the current circuit.
        """
        logger.trace("[Circuit] Prepending circuit: {}", circuit)
        if circuit.nqubits != self.nqubits:
            raise QubitOutOfRangeError(
                "the prepended circuit contains different number of qubits than the current circuit."
            )

        for i, g in enumerate(circuit.gates):
            self.insert(g, i)

    def copy(self) -> Self:
        """Return an independent deep copy of this circuit.

        Every gate and parameter is duplicated, so setting parameters, adding gates or otherwise
        modifying the returned circuit never affects this one. Parameters shared between several gates
        of this circuit stay shared between the corresponding gates of the copy.

        Returns:
            Circuit: A new circuit equal to this one but sharing no state with it.
        """
        logger.trace("[Circuit] Copying circuit with {} gates.", len(self._gates))
        return deepcopy(self)

    def to_matrix(self) -> np.ndarray:
        """Return the full unitary matrix representation of the circuit.

        Raises:
            GateHasNoMatrixError: if any gate does not define a matrix (e.g., measurement).
        """
        dim = 1 << self.nqubits
        logger.trace("[Circuit] Converting circuit to matrix with dimension: {}", dim)
        operator = np.eye(dim, dtype=_complex_dtype())
        for gate in self.gates:
            operator = _apply_gate_left(operator, gate, self.nqubits)
        return operator

    def to_qtensor(self) -> QTensor:
        logger.trace("[Circuit] Converting circuit to QTensor.")
        return QTensor(self.to_matrix())

    def __add__(self, other: Circuit | Gate) -> Circuit:
        """Return a new circuit with ``other`` composed at the end of this one.

        Args:
            other (Circuit | Gate): The circuit or gate to compose at the end of this circuit.

        Returns:
            Circuit: A new circuit holding this circuit's gates followed by ``other``'s.
        """
        logger.trace("[Circuit] Adding {} to circuit.", other)
        _reject_analog_object(other, _REJECT_TARGET)
        if not isinstance(other, (Circuit, Gate)):
            return NotImplemented
        new_circuit = Circuit(self.nqubits)
        new_circuit.append(self)
        if isinstance(other, Gate):
            new_circuit.add(other)
        else:
            new_circuit.append(other)
        return new_circuit

    def __radd__(self, other: Circuit | Gate) -> Circuit:
        """Return a new circuit with ``other`` composed at the beginning of this one.

        Args:
            other (Circuit | Gate): The circuit or gate to compose at the beginning of this circuit.

        Returns:
            Circuit: A new circuit holding ``other``'s gates followed by this circuit's.
        """
        logger.trace("[Circuit] Right-adding {} to circuit.", other)
        _reject_analog_object(other, _REJECT_TARGET)
        if not isinstance(other, (Circuit, Gate)):
            return NotImplemented
        new_circuit = Circuit(self.nqubits)
        if isinstance(other, Gate):
            new_circuit.add(other)
        else:
            new_circuit.append(other)
        new_circuit.append(self)
        return new_circuit

    def __eq__(self, other: object) -> bool:
        logger.trace("[Circuit] Checking equality of circuit with {}.", other)
        if not isinstance(other, Circuit):
            return NotImplemented
        if self.nqubits != other.nqubits:
            return False
        if len(self.gates) != len(other.gates):
            return False
        return all(g1 == g2 for g1, g2 in zip(self.gates, other.gates))

    def __hash__(self) -> int:
        h = qili_hash((self.nqubits, tuple(self.gates)))
        logger.trace("[Circuit] Hashing circuit: {}", h)
        return h

    def draw(self, style: CircuitStyle = CircuitStyle(), filepath: str | None = None) -> None:
        """
        Render this circuit with Matplotlib and optionally save it to a file.

        The circuit is rendered using the provided style configuration. If ``filepath`` is
        given, the resulting figure is saved to disk (the output format is inferred
        from the file extension, e.g. ``.png``, ``.pdf``, ``.svg``).

        Args:
            style (CircuitStyle): Visual style configuration applied to the plot.
                If not provided, the default :class:`CircuitStyle` is used.
            filepath (str | None): Destination file path for the rendered figure.
                If ``None``, the figure is not saved.
        """
        logger.debug("[Circuit] Drawing circuit with style: {} and filepath: {}", style, filepath)
        from qilisdk.utils.visualization.circuit_renderers import (  # ruff:ignore[import-outside-top-level]
            MatplotlibCircuitRenderer,
        )

        renderer = MatplotlibCircuitRenderer(self, style=style)
        renderer.plot()
        if filepath:
            renderer.save(filepath)

    @classmethod
    def random(
        cls,
        nqubits: int,
        single_qubit_gates: set[type[BasicGate]],
        two_qubit_gates: set[type[BasicGate]],
        ngates: int,
        seed: int | None = None,
    ) -> Self:
        """
        Generate a random quantum circuit from a given set of gates.

        Args:
            nqubits (int): The number of qubits in the circuit.
            single_qubit_gates (set[Gate]): A set of single-qubit gate classes to choose from.
            two_qubit_gates (set[Gate]): A set of two-qubit gate classes to choose from.
            ngates (int): The number of gates to include in the circuit.
            seed (int | None): Optional seed for the local random number generator, for reproducible circuits.
                The global :mod:`random` state is never used or modified.

        Returns:
            Circuit: A randomly generated quantum circuit.

        Raises:
            ValueError: If it is not possible to generate a full random circuit with the provided parameters

        """
        logger.debug(
            "[Circuit] Generating random circuit with {} qubits, {} gates, single_qubit_gates={}, two_qubit_gates={}",
            nqubits,
            ngates,
            single_qubit_gates,
            two_qubit_gates,
        )

        # If we only have one gate and one qubit, throw an error
        if nqubits == 1 and len(single_qubit_gates) == 1:
            raise ValueError("Cannot generate a full random circuit with only one qubit and one gate.")

        # Make sure we have given gates
        if len(single_qubit_gates) == 0 and len(two_qubit_gates) == 0:
            raise ValueError("At least one gate must be provided to generate a random circuit.")

        # Create a new rng with the seed
        rng = random.Random(seed)

        # Sort the gate lists for reproducibility
        gate_list: list[type[BasicGate]] = sorted(single_qubit_gates, key=lambda g: g.__name__)
        if nqubits > 1:
            gate_list.extend(sorted(two_qubit_gates, key=lambda g: g.__name__))

        # Start adding gates to the circuit
        new_circuit = cls(nqubits)
        prev_gate_type = None
        prev_qubits = None
        for gate_index in range(ngates):
            gate_class = rng.choice(gate_list)
            gate_nqubits = 1 if gate_class in single_qubit_gates else 2
            qubits = tuple(rng.sample(range(nqubits), gate_nqubits))
            logger.trace("[Circuit] Randomly selected gate {} on qubits {}", gate_class.__name__, qubits)

            # Avoid adding the same gate on the same qubits consecutively
            if gate_class == prev_gate_type and qubits == prev_qubits:
                # If we only have one qubit, pick a different gate
                if nqubits == 1:
                    possible_new_gates = sorted(
                        (g for g in single_qubit_gates if g != gate_class), key=lambda g: g.__name__
                    )
                    gate_class = rng.choice(possible_new_gates)

                # If the gate list does not include all qubits, change the first to be a different qubit
                if len(qubits) < nqubits:
                    possible_new_qubits: list[int] = [q for q in range(nqubits) if q not in qubits]
                    new_qubit = rng.choice(possible_new_qubits)
                    qubits = (new_qubit, *qubits[1:])

                # Otherwise, flip the order of the qubits
                else:
                    qubits = tuple(reversed(qubits))

            # Update previous gate info
            prev_gate_type = gate_class
            prev_qubits = qubits

            # Generate random parameters if needed
            params = {}
            if gate_class.PARAMETER_NAMES:
                for param_name in gate_class.PARAMETER_NAMES:
                    val = rng.uniform(-np.pi, np.pi)
                    params[param_name] = Parameter(label=f"{param_name}_{gate_index}", value=val, domain=Domain.REAL)

            # Add the gate to the circuit
            new_circuit.add(gate_class(*qubits, **params))  # ty:ignore[invalid-argument-type]

        return new_circuit

    @classmethod
    def double_bracket_iteration(
        cls,
        hamiltonian: Hamiltonian | QTensor,
        step_sizes: Sequence[RealNumber | Parameter],
        diagonal: Hamiltonian | None = None,
        trotter_steps: int = 1,
    ) -> Self:
        r"""
        Build a circuit that approximately diagonalizes a Hamiltonian using double-bracket iterations.

        A double-bracket iteration (Gluza, "Double-bracket quantum algorithms for diagonalization",
        `arXiv:2206.11772 <https://arxiv.org/abs/2206.11772>`_) rotates :math:`H_0` towards a diagonal matrix by
        repeating

        .. math::

            H_{k+1} = e^{s_k [D, H_k]} \, H_k \, e^{-s_k [D, H_k]},

        where :math:`D` is a fixed diagonal operator and :math:`s_k` is the size of step :math:`k`. Writing
        :math:`H_k = U_k H_0 U_k^\dagger`, each step is built from evolutions under :math:`H_0` and :math:`D` with
        the group commutator

        .. math::

            U_{k+1} = e^{-i\sqrt{s_k}D} \, U_k \, e^{i\sqrt{s_k}H_0} \, U_k^\dagger \, e^{i\sqrt{s_k}D} \, U_k \,
            e^{-i\sqrt{s_k}H_0}, \qquad U_0 = I,

        which equals :math:`e^{s_k [D, H_k]} U_k` up to an error of order :math:`s_k^{3/2}`. Evolutions under
        :math:`D` are exact, while evolutions under :math:`H_0` are Trotterized. Each step uses the previous circuit
        three times, so the number of gates grows like :math:`3^k`, and only a few steps are practical.

        The returned circuit is :math:`U_k^\dagger`, which maps computational basis states to approximate
        eigenstates of :math:`H_0`. For small enough step sizes the flow pairs low eigenvalues of :math:`H_0` with
        low diagonal entries of :math:`D`, so applying the circuit to the basis state that minimizes :math:`D`
        approximates the ground state:

        .. code-block:: python

            from qilisdk.analog import X, Z
            from qilisdk.digital import Circuit

            # The diagonal part -Z(0) - Z(1) is lowest on |00>, so the circuit acts on the default initial state.
            hamiltonian = -Z(0) - Z(1) + 0.5 * X(0) * X(1)
            circuit = Circuit.double_bracket_iteration(hamiltonian, step_sizes=[0.1, 0.1])

        Step sizes can be :class:`~qilisdk.core.variables.Parameter` objects, so they can be optimized with a
        :class:`~qilisdk.functionals.variational_program.VariationalProgram`.

        Args:
            hamiltonian (Hamiltonian | QTensor): The Hamiltonian :math:`H_0` to diagonalize. A
                :class:`~qilisdk.core.qtensor.QTensor` is converted with
                :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.from_qtensor`.
            step_sizes (Sequence[RealNumber | Parameter]): The non-negative step size :math:`s_k` of each iteration.
                The number of iterations is ``len(step_sizes)``.
            diagonal (Hamiltonian | None): The diagonal operator :math:`D`, made only of ``Z`` and identity terms.
                Defaults to the diagonal part of ``hamiltonian``, its terms made only of ``Z`` and identities.
            trotter_steps (int): The number of Trotter steps used for each evolution under :math:`H_0`.
                Defaults to 1.

        Returns:
            Circuit: The circuit :math:`U_k^\dagger` on enough qubits for both ``hamiltonian`` and ``diagonal``.

        Raises:
            ValueError: If a numeric step size is negative, if ``trotter_steps`` is less than 1, if ``diagonal``
                contains ``X`` or ``Y`` operators, or if :math:`D` has no terms other than the identity.
        """
        # Local import to avoid a circular import, as qilisdk.utils.trotterization imports qilisdk.digital.gates.
        from qilisdk.utils.trotterization.trotterization import (  # ruff:ignore[import-outside-top-level]
            _pauli_evolution,
        )

        logger.debug(
            "[Circuit] Building double-bracket iteration with {} steps and {} Trotter steps",
            len(step_sizes),
            trotter_steps,
        )
        if isinstance(hamiltonian, QTensor):
            hamiltonian = Hamiltonian.from_qtensor(hamiltonian)
        if trotter_steps < 1:
            raise ValueError(f"The number of Trotter steps must be at least 1, got {trotter_steps}.")
        if any(not isinstance(step, Expression) and step < 0 for step in step_sizes):
            raise ValueError(f"Step sizes must be non-negative, got {list(step_sizes)}.")
        if diagonal is None:
            diagonal = Hamiltonian(
                {
                    term: coeff
                    for term, coeff in hamiltonian.elements.items()
                    if all(op.name in {"I", "Z"} for op in term)
                }
            )
        elif any(op.name not in {"I", "Z"} for term in diagonal.elements for op in term):
            raise ValueError(f"The diagonal operator must contain only Z and identity terms, got {diagonal}.")

        # Each rotation (term, angle) is exp(-i * angle * term); the identity terms only add a global phase.
        hamiltonian_terms = [
            (term, float(np.real(coeff)) / trotter_steps)
            for part in hamiltonian.get_commuting_partitions()
            for term, coeff in part.items()
            if any(op.name != "I" for op in term)
        ] * trotter_steps
        diagonal_terms = [
            (term, float(np.real(coeff)))
            for term, coeff in diagonal.elements.items()
            if any(op.name != "I" for op in term)
        ]
        if not diagonal_terms:
            raise ValueError("The diagonal operator has no terms other than the identity, so the flow does nothing.")

        def inverse(
            rotations: list[tuple[tuple[PauliOperator, ...], float | Expression]],
        ) -> list[tuple[tuple[PauliOperator, ...], float | Expression]]:
            return [(term, -angle) for term, angle in reversed(rotations)]

        # The rotations of U_k in time order, so the first one acts first.
        rotations: list[tuple[tuple[PauliOperator, ...], float | Expression]] = []
        for step in step_sizes:
            root = step**0.5 if isinstance(step, Expression) else math.sqrt(step)
            hamiltonian_forward = [(term, coeff * root) for term, coeff in hamiltonian_terms]
            diagonal_forward = [(term, coeff * root) for term, coeff in diagonal_terms]
            rotations_inverse = inverse(rotations)
            rotations = (
                hamiltonian_forward
                + rotations
                + inverse(diagonal_forward)
                + rotations_inverse
                + inverse(hamiltonian_forward)
                + rotations
                + diagonal_forward
            )
        if len(rotations) > _DOUBLE_BRACKET_ROTATION_WARNING:
            logger.warning(
                "[Circuit] The double-bracket iteration has {} Pauli rotations, as the size grows like 3^steps",
                len(rotations),
            )

        circuit = cls(max(hamiltonian.nqubits, diagonal.nqubits))
        for term, angle in inverse(rotations):
            circuit.add(list(_pauli_evolution(term, angle, 1.0)))
        return circuit

    def __repr__(self) -> str:
        logger.trace(
            "[Circuit] Generating string representation of circuit with {} qubits and {} gates",
            self.nqubits,
            len(self.gates),
        )
        return f"Circuit(nqubits={self.nqubits}, gates={self.gates})"
