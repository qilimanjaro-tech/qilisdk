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

import random
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
from loguru import logger
from typing_extensions import Self

from qilisdk.core import Domain, Parameter, QTensor
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.settings import get_settings
from qilisdk.utils.hashing import hash as qili_hash
from qilisdk.utils.visualization import CircuitStyle
from qilisdk.yaml import yaml

from .exceptions import QubitOutOfRangeError
from .gates import CNOT, RY, RZ, BasicGate, Gate, _reject_analog_object

if TYPE_CHECKING:
    from qilisdk.core.types import RealNumber

# Name used for this class in the error messages raised for analog objects.
_REJECT_TARGET = "a Circuit"


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


def _uniformly_controlled_rotation(
    gate_class: type[RY | RZ], target: int, angles: np.ndarray, relevant: np.ndarray, atol: float
) -> list[Gate]:
    """Decompose a rotation on ``target`` uniformly controlled by qubits ``0..target-1`` into rotations and CNOTs.

    Uses the Gray-code construction of Möttönen et al. (quant-ph/0407010): the ``2**k`` rotation angles
    ``angles[j]``, one per computational basis state ``j`` of the ``k = target`` control qubits (qubit 0 being the
    most significant bit of ``j``), become ``2**k`` single-qubit rotations interleaved with ``2**k`` CNOTs. Rotations
    whose angle is below ``atol`` are omitted, together with any CNOTs that then cancel out. Angles that are not
    ``relevant`` (because they act on a zero-amplitude branch) are free, so they are set equal to a relevant one, which
    turns a rotation that only depends on the controls through irrelevant branches into a single uncontrolled one.

    Args:
        gate_class (type[RY | RZ]): The rotation gate, which must anticommute with X so a CNOT flips its angle.
        target (int): The target qubit, controlled by every qubit with a smaller index.
        angles (np.ndarray): The ``2**target`` rotation angles, indexed by the state of the control qubits.
        relevant (np.ndarray): Boolean mask of the angles that affect the result; the others may take any value.
        atol (float): Angles with an absolute value below this are treated as zero.

    Returns:
        list[Gate]: The gates implementing the uniformly controlled rotation.
    """
    if not relevant.any():
        return []
    angles = np.where(relevant, angles, angles[relevant][0])
    if np.all(np.abs(angles) < atol):
        return []
    if target == 0:
        return [gate_class(0, **{gate_class.PARAMETER_NAMES[0]: float(angles[0])})]
    size = len(angles)
    gray = np.arange(size) ^ (np.arange(size) >> 1)
    # thetas[i] = sum_j (-1)^popcount(gray[i] & j) angles[j] / size, i.e. a Gray-ordered fast Walsh-Hadamard transform.
    coeffs = angles.astype(float)
    for bit in range(target):
        pairs = coeffs.reshape(-1, 2, 1 << bit)
        coeffs = np.stack((pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]), axis=1).ravel()
    thetas = coeffs[gray] / size
    # CNOTs sharing a target commute and cancel in pairs, so only the net parity between rotations is emitted.
    gates: list[Gate] = []
    pending = 0
    for i, theta in enumerate(thetas):
        if abs(theta) >= atol:
            gates.extend(CNOT(target - 1 - bit, target) for bit in range(target) if pending >> bit & 1)
            gates.append(gate_class(target, **{gate_class.PARAMETER_NAMES[0]: float(theta)}))
            pending = 0
        pending ^= int(gray[i] ^ gray[(i + 1) % size])
    gates.extend(CNOT(target - 1 - bit, target) for bit in range(target) if pending >> bit & 1)
    return gates


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
    def from_state(cls, state: QTensor) -> Self:
        """Build a circuit that prepares ``state`` from the all-zero state ``|0...0>``.

        The circuit uses only :class:`RY`, :class:`RZ` and :class:`CNOT` gates, following the uniformly controlled
        rotation construction of Möttönen et al., "Transformation of quantum states using uniformly controlled
        rotations" (quant-ph/0407010). First a cascade of ``RY`` rotations sets the amplitude magnitudes, qubit by
        qubit from qubit 0, then a cascade of ``RZ`` rotations sets their relative phases. Preparing a generic
        ``n``-qubit state takes at most ``2**(n+1) - 4`` CNOTs; rotations with a zero angle are left out, along with
        the CNOTs they make redundant, so product states need fewer gates. A state whose amplitudes are real up to a
        global phase needs no ``RZ`` cascade at all, as the signs are set by the last qubit's ``RY`` rotations, which
        halves the CNOT count to at most ``2**n - 2``.

        The phase cascade runs from the last qubit to the first: ``diag(e^{i w0}, e^{i w1})`` on qubit ``k`` equals
        ``RZ(w1 - w0)`` times a phase ``(w0 + w1) / 2`` that is left on qubits ``0..k-1``. Phases of zero amplitudes are
        free, so a pair with a single nonzero amplitude of phase ``w`` can take any ``RZ`` angle ``d``, provided the
        phase left behind is ``w + d / 2`` (or ``w - d / 2``); ``d`` is chosen to match another pair, which lets
        rotations that would otherwise be controlled collapse into fewer gates.

        Example:
            .. code-block:: python

                from qilisdk.core import QTensor
                from qilisdk.digital import Circuit

                state = QTensor.random(3, seed=42)
                circuit = Circuit.from_state(state)

        Args:
            state (QTensor): The ket to prepare. An unnormalized ket is normalized, with a warning.

        Returns:
            Circuit: A circuit whose action on ``|0...0>`` is ``state`` up to a global phase.

        Raises:
            ValueError: If ``state`` is not a ket, or is the zero vector.
        """
        if not state.is_ket():
            raise ValueError(f"Circuit.from_state expects a ket, but got a QTensor of shape {state.shape}.")
        amplitudes = state.dense().ravel()
        norm = float(np.linalg.norm(amplitudes))
        atol = get_settings().atol
        if norm < atol:
            raise ValueError("Cannot prepare the zero vector, as it is not a valid quantum state.")
        if abs(norm - 1.0) > atol:
            logger.warning("[Circuit] The state given to Circuit.from_state has norm {}, normalizing it.", norm)
            amplitudes /= norm
        nqubits = state.nqubits
        logger.debug("[Circuit] Building a state preparation circuit for {} qubits", nqubits)

        circuit = cls(nqubits)
        # Removing the global phase of the largest amplitude lets states that are real up to a phase skip the RZ cascade.
        magnitudes = np.abs(amplitudes)
        amplitudes *= np.exp(-1j * np.angle(amplitudes[np.argmax(magnitudes)]))
        is_real = bool(np.all(np.abs(amplitudes.imag) < atol))
        # Qubit k is rotated by RY(2 atan2(|b=1|, |b=0|)), controlled by the state of qubits 0..k-1.
        for target in range(nqubits):
            branches = magnitudes.reshape(1 << target, 2, -1)
            branch_norms = np.linalg.norm(branches, axis=2)
            # A real state keeps its signs in the last qubit's rotation, so it needs no phase cascade.
            if is_real and target == nqubits - 1:
                branch_norms = amplitudes.real.reshape(-1, 2)
            angles = 2 * np.arctan2(branch_norms[:, 1], branch_norms[:, 0])
            relevant = np.abs(branch_norms).sum(axis=1) >= atol
            circuit.add(_uniformly_controlled_rotation(RY, target, angles, relevant, atol))
        if is_real:
            return circuit

        # Each qubit, last to first, absorbs the relative phase of its pairs and passes their mean phase upwards.
        phases = np.angle(amplitudes)
        weights = magnitudes >= atol
        phase_gates: list[Gate] = []
        for target in reversed(range(nqubits)):
            pair_phases = phases.reshape(-1, 2)
            pair_weights = weights.reshape(-1, 2)
            both = pair_weights.all(axis=1)
            # Wrapping into (-pi, pi] keeps differences that agree modulo 2 pi equal, so the RZ stays uncontrolled.
            diffs = np.angle(np.exp(1j * (pair_phases[:, 1] - pair_phases[:, 0])))
            diffs = np.where(both, diffs, diffs[both][0] if both.any() else 0.0)
            phase_gates = _uniformly_controlled_rotation(RZ, target, diffs, both, atol) + phase_gates
            phases = np.where(pair_weights[:, 0], pair_phases[:, 0] + diffs / 2, pair_phases[:, 1] - diffs / 2)
            weights = pair_weights.any(axis=1)
        circuit.add(phase_gates)
        return circuit

    def __repr__(self) -> str:
        logger.trace(
            "[Circuit] Generating string representation of circuit with {} qubits and {} gates",
            self.nqubits,
            len(self.gates),
        )
        return f"Circuit(nqubits={self.nqubits}, gates={self.gates})"
