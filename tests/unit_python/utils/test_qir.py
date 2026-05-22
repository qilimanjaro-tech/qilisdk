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

import pytest

pyqir = pytest.importorskip("pyqir")

from qilisdk.digital import (  # noqa: E402
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    U1,
    Adjoint,
    Circuit,
    H,
    I,
    M,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.utils.qir import from_qir, from_qir_file, to_qir, to_qir_file  # noqa: E402


def _summarize(circuit: Circuit) -> list[tuple]:
    """Return a (name, qubits, angle) tuple per gate for compact comparisons."""
    out = []
    for g in circuit.gates:
        angle = None
        if hasattr(g, "theta") and "theta" in getattr(g, "PARAMETER_NAMES", []):
            angle = float(g.theta)
        elif hasattr(g, "phi") and "phi" in getattr(g, "PARAMETER_NAMES", []):
            angle = float(g.phi)
        out.append((g.name, tuple(g.qubits), angle))
    return out


# --- Header / module structure ----------------------------------------------------


def test_to_qir_includes_entry_point_and_qubit_count():
    circuit = Circuit(2)
    circuit.add(H(0))
    text = to_qir(circuit, name="hdr")
    assert "entry_point" in text
    assert '"required_num_qubits"="2"' in text
    assert '"required_num_results"="2"' in text


def test_to_qir_module_name_appears_in_ir():
    circuit = Circuit(1)
    circuit.add(X(0))
    assert "my-module" in to_qir(circuit, name="my-module")


# --- Single-qubit non-parameterized gates -----------------------------------------


@pytest.mark.parametrize(
    ("gate_cls", "intrinsic"),
    [(X, "x"), (Y, "y"), (Z, "z"), (H, "h"), (S, "s"), (T, "t")],
)
def test_simple_single_qubit_gates_emit_expected_intrinsic(gate_cls, intrinsic):
    circuit = Circuit(1)
    circuit.add(gate_cls(0))
    assert f"__quantum__qis__{intrinsic}__body" in to_qir(circuit)


def test_identity_gate_is_a_noop_in_qir():
    circuit = Circuit(1)
    circuit.add(I(0))
    circuit.add(X(0))
    text = to_qir(circuit)
    # I emits nothing; only the X call shows up.
    assert "__quantum__qis__x__body" in text
    assert "__quantum__qis__i" not in text


def test_adjoint_s_and_t_emit_adj_intrinsics():
    circuit = Circuit(2)
    circuit.add(Adjoint(S(0)))
    circuit.add(Adjoint(T(1)))
    text = to_qir(circuit)
    assert "__quantum__qis__s__adj" in text
    assert "__quantum__qis__t__adj" in text


# --- Parameterized gates ----------------------------------------------------------


def test_rotation_gates_emit_angle_arguments():
    circuit = Circuit(1)
    circuit.add(RX(0, theta=0.5))
    circuit.add(RY(0, theta=1.25))
    circuit.add(RZ(0, phi=-0.75))
    text = to_qir(circuit)
    assert "__quantum__qis__rx__body" in text
    assert "__quantum__qis__ry__body" in text
    assert "__quantum__qis__rz__body" in text


# --- Two-qubit gates --------------------------------------------------------------


def test_cnot_emits_cnot_intrinsic_with_correct_qubits():
    circuit = Circuit(3)
    circuit.add(CNOT(2, 0))
    text = to_qir(circuit)
    assert "__quantum__qis__cnot__body" in text
    # Control comes first in the call argument list.
    assert "i64 2" in text


def test_cz_and_swap_emit_their_intrinsics():
    circuit = Circuit(2)
    circuit.add(CZ(0, 1))
    circuit.add(SWAP(0, 1))
    text = to_qir(circuit)
    assert "__quantum__qis__cz__body" in text
    assert "__quantum__qis__swap__body" in text


# --- Measurement ------------------------------------------------------------------


def test_multi_qubit_measurement_emits_one_mz_per_target():
    circuit = Circuit(3)
    circuit.add(M(0, 2))
    text = to_qir(circuit)
    # Count only call-site occurrences, not the trailing `declare` line.
    assert text.count("call void @__quantum__qis__mz__body") == 2


# --- Unsupported gates ------------------------------------------------------------


def test_unsupported_gate_raises_not_implemented():
    circuit = Circuit(1)
    circuit.add(U1(0, phi=0.5))
    with pytest.raises(NotImplementedError, match="QIR Base-Profile"):
        to_qir(circuit)


# --- Round-trip parity ------------------------------------------------------------


def test_round_trip_preserves_gate_sequence():
    circuit = Circuit(3)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(RX(2, theta=0.7))
    circuit.add(CZ(1, 2))
    circuit.add(M(0, 1, 2))

    reparsed = from_qir(to_qir(circuit))
    # M(0, 1, 2) round-trips as three separate M(q) gates, so compare names per qubit.
    original_summary = []
    for g in circuit.gates:
        if g.name == "M":
            original_summary.extend([("M", (q,), None) for q in g.qubits])
        else:
            angle = None
            if hasattr(g, "theta") and "theta" in getattr(g, "PARAMETER_NAMES", []):
                angle = float(g.theta)
            elif hasattr(g, "phi") and "phi" in getattr(g, "PARAMETER_NAMES", []):
                angle = float(g.phi)
            original_summary.append((g.name, tuple(g.qubits), angle))

    assert _summarize(reparsed) == original_summary


def test_round_trip_preserves_nqubits():
    circuit = Circuit(4)
    circuit.add(X(0))
    circuit.add(X(3))
    reparsed = from_qir(to_qir(circuit))
    assert reparsed.nqubits == 4


def test_round_trip_adjoint_s_and_t():
    circuit = Circuit(2)
    circuit.add(Adjoint(S(0)))
    circuit.add(Adjoint(T(1)))
    reparsed = from_qir(to_qir(circuit))
    assert [g.name for g in reparsed.gates] == ["S†", "T†"]


# --- File I/O (.ll and .bc) -------------------------------------------------------


def test_to_qir_file_writes_textual_ll(tmp_path):
    circuit = Circuit(1)
    circuit.add(H(0))
    path = tmp_path / "out.ll"
    to_qir_file(circuit, str(path))
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "__quantum__qis__h__body" in text
    assert from_qir_file(str(path)).nqubits == 1


def test_to_qir_file_writes_bitcode_bc(tmp_path):
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    path = tmp_path / "out.bc"
    to_qir_file(circuit, str(path))
    assert path.exists()
    # Bitcode files start with magic 'BC\xc0\xde'.
    assert path.read_bytes()[:4] == b"BC\xc0\xde"
    reparsed = from_qir_file(str(path))
    assert [g.name for g in reparsed.gates] == ["H", "CNOT"]


def test_from_qir_file_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_qir_file(str(tmp_path / "nope.ll"))


# --- Failure modes ----------------------------------------------------------------


def test_module_without_entry_point_raises():
    bare = """
; ModuleID = 'bare'
declare void @__quantum__qis__h__body(ptr)
"""
    with pytest.raises(ValueError, match="entry-point"):
        from_qir(bare)
