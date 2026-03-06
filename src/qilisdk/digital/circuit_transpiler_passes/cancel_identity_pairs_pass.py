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

import cmath
import math
from copy import deepcopy
from typing import Any, ClassVar, TypeGuard

import numpy as np

from qilisdk.digital import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    U1,
    U2,
    U3,
    BasicGate,
    Circuit,
    Gate,
    H,
    I,
    X,
    Y,
    Z,
)
from qilisdk.digital.exceptions import GateHasNoMatrixError
from qilisdk.digital.gates import Adjoint, Controlled, M

from .circuit_transpiler_pass import CircuitTranspilerPass
from .numeric_helpers import _EPS, _round_float, _wrap_angle


def _is_controlled(gate: Gate) -> TypeGuard[Controlled[BasicGate]]:
    return isinstance(gate, Controlled)


def _is_adjoint(gate: Gate) -> TypeGuard[Adjoint[BasicGate]]:
    return isinstance(gate, Adjoint)


def _first_nonzero_phase(unitary: np.ndarray) -> float:
    """Return the phase of the first non-zero matrix entry.

    Args:
        unitary (np.ndarray): Matrix to inspect.

    Returns:
        float: Phase angle of the first entry whose modulus is greater than
            `_EPS`. Returns `0.0` when no such entry exists.
    """
    # phase of the first element with |.| > EPS
    for z in np.ravel(unitary, order="K"):
        val = complex(z)
        if abs(val) > _EPS:
            return cmath.phase(val)
    return 0.0


def _dephased_signature(unitary: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Return a rounded signature that is invariant to global phase.

    Args:
        unitary (np.ndarray): Matrix to convert into a hashable signature.

    Returns:
        tuple[tuple[float, float], ...]: Flattened sequence of rounded
            `(real, imag)` entries after removing global phase. Returns an empty
            tuple for empty input matrices.
    """
    if unitary.size == 0:
        return ()
    phi = _first_nonzero_phase(unitary)
    F = unitary * np.exp(-1j * phi)
    flat = F.reshape(-1)
    sig: list[tuple[float, float]] = []
    for z in flat:
        sig.append((_round_float(z.real), _round_float(z.imag)))
    return tuple(sig)


def _try_matrix(g: Gate) -> np.ndarray | None:
    """Safely fetch a gate matrix when available.

    Args:
        g (Gate): Gate whose matrix representation is requested.

    Returns:
        np.ndarray | None: Gate matrix, or `None` when the gate does not expose
            a matrix.
    """
    try:
        return g.matrix
    except GateHasNoMatrixError:
        return None


# ------------------------ Canonical keys for matching ------------------------


class CancelIdentityPairsPass(CircuitTranspilerPass):
    """
    Cancel pairs of gates whose product is identity (up to global phase), across
    disjoint-qubit operations. Runs to a fixed point.

    It handles:
      • Involutions: H, X, Y, Z, CNOT, CZ, SWAP (gate; same gate cancels).
      • Parameter inverses: RX(θ)/RY(θ)/RZ(φ)/U1(φ) with negative angles;
        U3(θ,φ,λ) with U3(-θ, -λ, -φ).
      • Adjoint pairing: G with Adjoint(G).
      • Controlled^k: Controlled^k(U) with Controlled^k(U†), same controls/target.
      • Fallback: any two gates with matrix product ≈ identity (on the same qubits).

    Blocking:
      • Any operation that touches a qubit clears any pending candidate on that qubit.
      • Operations on disjoint qubits do not block.

    Notes:
      • `I` is dropped immediately.
      • For symmetric 2Q gates (CZ, SWAP) we normalize the qubit key to an unordered pair
        so CZ(a,b) cancels with CZ(b,a).
      • Measurements (M) are barriers on their qubits.
    """

    # Self-inverse (involution) gate classes we recognize cheaply
    _INVOLUTION_TYPES: ClassVar[tuple[type[Gate], ...]] = (H, X, Y, Z, CNOT, CZ, SWAP)
    _SYMMETRIC_TWO_QUBIT_ARITY: ClassVar[int] = 2

    def run(self, circuit: Circuit) -> Circuit:
        """Cancel inverse gate pairs until no more cancellations are possible.

        Args:
            circuit (Circuit): Input circuit that is not mutated.

        Returns:
            Circuit: New circuit with cancellable identity pairs removed.
        """
        gates = list(circuit.gates)

        while True:
            stack: dict[tuple[Any, tuple[int, ...]], int] = {}
            to_delete: set[int] = set()

            for idx, g in enumerate(gates):
                # Drop identities immediately
                if isinstance(g, I):
                    to_delete.add(idx)
                    continue

                # Measurements are barriers; they just block
                if isinstance(g, M):
                    self._block_overlapping(stack, g.qubits)
                    continue

                # Compute keys
                qkey = self._qubits_key(g)
                fkey, invkey = self._forward_inverse_keys(g)

                if fkey is None:
                    # Unknown gate with no matrix; just block
                    self._block_overlapping(stack, g.qubits)
                    continue

                # If we previously saw an inverse on these qubits, cancel both
                inv_lookup = (invkey, qkey)
                if inv_lookup in stack:
                    prev_idx = stack.pop(inv_lookup)
                    to_delete.update((prev_idx, idx))
                    # Do not push current; pair is canceled
                    continue

                # Otherwise, this gate becomes the current candidate on these qubits
                self._block_overlapping(stack, g.qubits)
                stack[fkey, qkey] = idx

            if not to_delete:
                break

            gates = [gate for i, gate in enumerate(gates) if i not in to_delete]

        # Build new circuit (deepcopy to avoid Parameter sharing)
        out = Circuit(circuit.nqubits)
        for g in gates:
            out.add(deepcopy(g))

        self.append_circuit_to_context(out)
        return out

    # ----------------- key builders -----------------

    @staticmethod
    def _qubits_key(g: Gate) -> tuple[int, ...]:
        """Build the qubit key used for candidate matching.

        Args:
            g (Gate): Gate being considered for cancellation.

        Returns:
            tuple[int, ...]: Normalized qubit tuple. For symmetric 2-qubit
                gates (`CZ`, `SWAP`) the pair is sorted; otherwise gate qubit
                order is preserved.
        """
        qs = g.qubits
        if isinstance(g, (CZ, SWAP)) and len(qs) == CancelIdentityPairsPass._SYMMETRIC_TWO_QUBIT_ARITY:
            a, b = qs
            return (a, b) if a < b else (b, a)
        # For Controlled (non-CZ), direction matters, so keep order (controls then targets)
        return qs

    def _forward_inverse_keys(self, g: Gate) -> tuple[Any | None, Any | None]:
        """
        Compute forward and inverse matching keys for a gate.

        Args:
            g (Gate): Gate to encode.

        Returns:
            tuple[Any | None, Any | None]: Pair `(forward_key, inverse_key)`.
                Keys are hashable descriptors of the unitary up to global phase.
                Returns `(None, None)` when no key can be derived and the gate
                should be treated as a barrier.
        """
        # Self-inverse classes (no parameters)
        if isinstance(g, self._INVOLUTION_TYPES):
            # CZ/SWAP handled by unordered qubit key; CNOT is directional and self-inverse
            tag = type(g).__name__
            return (("INV", tag), ("INV", tag))

        # Parameterized rotations
        if isinstance(g, RX):
            a = _wrap_angle(g.theta)
            return (("RX", _round_float(a)), ("RX", _round_float(-a)))
        if isinstance(g, RY):
            a = _wrap_angle(g.theta)
            return (("RY", _round_float(a)), ("RY", _round_float(-a)))
        if isinstance(g, RZ):
            a = _wrap_angle(g.phi)
            return (("RZ", _round_float(a)), ("RZ", _round_float(-a)))
        if isinstance(g, U1):
            a = _wrap_angle(g.phi)
            return (("U1", _round_float(a)), ("U1", _round_float(-a)))
        if isinstance(g, U2):
            # Treat via U3 equivalence: U2(phi,gamma) == U3(pi/2, phi, gamma)
            theta, phi, gamma = (math.pi / 2.0, _wrap_angle(g.phi), _wrap_angle(g.gamma))
            return (
                ("U3", _round_float(theta), _round_float(phi), _round_float(gamma)),
                ("U3", _round_float(-theta), _round_float(-gamma), _round_float(-phi)),
            )
        if isinstance(g, U3):
            theta = _wrap_angle(g.theta)
            phi = _wrap_angle(g.phi)
            gamma = _wrap_angle(g.gamma)
            # U3(θ,φ,λ)† = U3(-θ, -λ, -φ)
            return (
                ("U3", _round_float(theta), _round_float(phi), _round_float(gamma)),
                ("U3", _round_float(-theta), _round_float(-gamma), _round_float(-phi)),
            )

        # Adjoint wrapper: swap forward/inverse of the base
        if _is_adjoint(g):
            f, inv = self._forward_inverse_keys(g.basic_gate)
            return (inv, f)

        # Controlled: propagate keys with control count; direction matters
        if _is_controlled(g):
            k = len(g.control_qubits)
            f_base, inv_base = self._forward_inverse_keys(g.basic_gate)
            if f_base is None:
                # Fallback to matrix signature
                return self._matrix_keys(g)
            return (("C", k, f_base), ("C", k, inv_base))

        # Generic w/ matrix fallback (includes Exponential, unknown 1Q BasicGate, etc.)
        f_inv = self._matrix_keys(g)
        return f_inv

    @staticmethod
    def _matrix_keys(g: Gate) -> tuple[Any | None, Any | None]:
        """Build matrix-derived fallback keys for generic gates.

        Args:
            g (Gate): Gate to encode from matrix data.

        Returns:
            tuple[Any | None, Any | None]: Pair of keys for `U` and `U^\u2020`,
                or `(None, None)` if no matrix is available.
        """
        U = _try_matrix(g)
        if U is None:
            return (None, None)
        sig = _dephased_signature(U)
        sig_inv = _dephased_signature(U.conj().T)
        return (("U", sig), ("U", sig_inv))

    # ----------------- blocking policy -----------------

    @staticmethod
    def _block_overlapping(
        stack: dict[tuple[Any, tuple[int, ...]], int],
        qubits: tuple[int, ...],
    ) -> None:
        """Drop pending candidates that overlap with blocked qubits.

        Args:
            stack (dict[tuple[Any, tuple[int, ...]], int]): Pending candidate
                map keyed by `(gate_key, qubit_key)` and valued by gate index.
            qubits (tuple[int, ...]): Qubits touched by the current operation.
        """
        if not qubits:
            stack.clear()
            return
        touched = set(qubits)
        for k in list(stack.keys()):
            _, k_qubits = k
            if touched.intersection(k_qubits):
                stack.pop(k, None)
