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

import numpy as np
import pytest
from matplotlib.lines import Line2D

import qilisdk.utils.visualization.circuit_renderers
from qilisdk.backends.qilisim import QiliSim
from qilisdk.digital import CNOT, RX, Barrier, Circuit, H, M, X
from qilisdk.digital.circuit_transpiler import CircuitTranspiler
from qilisdk.digital.exceptions import GateHasNoMatrixError, QubitOutOfRangeError
from qilisdk.functionals import DigitalPropagation
from qilisdk.readout import Readout
from qilisdk.utils.openqasm import from_qasm2, from_qasm3, to_qasm2, to_qasm3
from qilisdk.utils.serialization import deserialize, serialize
from qilisdk.utils.visualization.circuit_renderers import MatplotlibCircuitRenderer
from qilisdk.utils.visualization.style import CircuitStyle


def bell_with_barriers() -> Circuit:
    circuit = Circuit(2)
    circuit.add([Barrier(), H(0), Barrier(0, 1), CNOT(0, 1), Barrier(1), M(0, 1), Barrier()])
    return circuit


def bell() -> Circuit:
    circuit = Circuit(2)
    circuit.add([H(0), CNOT(0, 1), M(0, 1)])
    return circuit


# --- Gate -------------------------------------------------------------------------------------------------------------


def test_barrier_gate():
    barrier = Barrier(2, 0)
    assert barrier.name == "Barrier"
    assert barrier.qubits == (2, 0)
    assert barrier.target_qubits == (2, 0)
    assert barrier.control_qubits == ()
    assert barrier.nqubits == 2
    assert not barrier.is_parameterized
    assert repr(barrier) == "Barrier(2, 0)"
    with pytest.raises(GateHasNoMatrixError):
        _ = barrier.matrix


def test_barrier_rejects_repeated_qubits():
    with pytest.raises(ValueError, match="same qubit"):
        Barrier(0, 0)


def test_barrier_equality():
    assert Barrier(0, 1) == Barrier(0, 1)
    assert Barrier(0, 1) != Barrier(0, 2)
    assert hash(Barrier(0, 1)) == hash(Barrier(0, 1))


def test_barrier_yaml_roundtrip():
    circuit = bell_with_barriers()
    assert deserialize(serialize(circuit), Circuit) == circuit


# --- Circuit ----------------------------------------------------------------------------------------------------------


def test_barrier_without_qubits_spans_the_circuit():
    circuit = Circuit(3)
    circuit.add(Barrier())
    circuit.insert(Barrier(), 0)
    assert all(gate.qubits == (0, 1, 2) for gate in circuit.gates)


def test_barrier_out_of_range():
    with pytest.raises(QubitOutOfRangeError):
        Circuit(2).add(Barrier(0, 2))


def test_barriers_do_not_change_the_unitary():
    with_barriers = Circuit(2)
    with_barriers.add([H(0), Barrier(), CNOT(0, 1), Barrier(1), RX(1, theta=0.3)])
    without = Circuit(2)
    without.add([H(0), CNOT(0, 1), RX(1, theta=0.3)])
    np.testing.assert_allclose(with_barriers.to_matrix(), without.to_matrix())


# --- Drawing ----------------------------------------------------------------------------------------------------------


def barrier_lines(renderer: MatplotlibCircuitRenderer) -> list[Line2D]:
    return [
        line
        for line in renderer.axes.get_lines()
        if line.get_zorder() == renderer._Z["barrier"] and line.get_xdata()[0] == line.get_xdata()[1]
    ]


def render(circuit: Circuit, monkeypatch: pytest.MonkeyPatch, **style) -> MatplotlibCircuitRenderer:
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt, "show", lambda: None)
    renderer = MatplotlibCircuitRenderer(circuit, style=CircuitStyle(**style))
    renderer.plot()
    return renderer


def test_barrier_is_drawn_as_a_vertical_line_across_its_qubits(monkeypatch):
    circuit = Circuit(3)
    circuit.add([H(0), Barrier(), X(0)])
    renderer = render(circuit, monkeypatch)

    (line,) = barrier_lines(renderer)
    y_bottom, y_top = sorted(line.get_ydata())
    assert y_bottom < renderer._ypos(2, n_qubits=3, sep=renderer.style.wire_sep)
    assert y_top > renderer._ypos(0, n_qubits=3, sep=renderer.style.wire_sep)
    # the line sits between the two gates around it
    x = line.get_xdata()[0]
    assert renderer._layer_x[0] < x < renderer._layer_x[2]


@pytest.mark.parametrize("layout", ["normal", "compact"])
def test_barrier_line_breaks_over_qubits_it_skips(monkeypatch, layout):
    circuit = Circuit(4)
    circuit.add(Barrier(0, 2, 3))
    renderer = render(circuit, monkeypatch, layout=layout)

    lines = barrier_lines(renderer)
    assert len(lines) == 2
    wire = renderer._ypos(1, n_qubits=4, sep=renderer.style.wire_sep)
    assert all(not min(line.get_ydata()) <= wire <= max(line.get_ydata()) for line in lines)


def test_barrier_separates_gates_that_would_share_a_column(monkeypatch):
    circuit = Circuit(2)
    circuit.add([H(0), H(1)])
    assert len(render(circuit, monkeypatch)._layer_gate_mapping) == 1

    circuit = Circuit(2)
    circuit.add([H(0), Barrier(), H(1)])
    renderer = render(circuit, monkeypatch)
    assert len(renderer._layer_gate_mapping) == 3
    assert list(renderer._layer_gate_mapping[2]) == [1]


def test_barrier_does_not_stop_final_measurements_sharing_a_column(monkeypatch):
    renderer = render(bell_with_barriers(), monkeypatch)
    assert renderer._deferred_qubits == set()
    assert renderer._inline_measures

    circuit = Circuit(2)
    circuit.add([H(0), Barrier(), M(0, 1)])
    assert render(circuit, monkeypatch)._deferred_qubits == {0, 1}


# --- Transpilation ----------------------------------------------------------------------------------------------------


def test_transpiler_removes_barriers():
    result = CircuitTranspiler.default(topology=[(0, 1)]).transpile(bell_with_barriers())
    assert not any(isinstance(gate, Barrier) for gate in result.circuit.gates)
    assert all(
        not any(isinstance(gate, Barrier) for gate in step.circuit.gates) for step in result.intermediate_results
    )


def test_transpiling_without_barriers_keeps_the_input_circuit():
    circuit = bell()
    assert CircuitTranspiler([]).transpile(circuit).circuit is circuit


# --- Serialization ----------------------------------------------------------------------------------------------------


def test_barrier_to_qasm2():
    circuit = Circuit(3)
    circuit.add([Barrier(), Barrier(0, 2)])
    lines = to_qasm2(circuit).splitlines()
    assert lines[-2:] == ["barrier q;", "barrier q[0], q[2];"]


def test_barrier_qasm2_roundtrip():
    circuit = Circuit(3)
    circuit.add([H(0), Barrier(), Barrier(2, 0), M(0)])
    assert from_qasm2(to_qasm2(circuit)) == circuit


def test_barrier_from_qasm2_with_several_registers():
    circuit = from_qasm2("OPENQASM 2.0;\nqreg a[2];\nqreg b[2];\nbarrier a, b[1];\nbarrier a[0], a;\n")
    assert [gate.qubits for gate in circuit.gates] == [(0, 1, 3), (0, 1)]


@pytest.mark.parametrize(
    ("line", "error"),
    [
        ("barrier;", "Invalid barrier instruction"),
        ("barrier q[0] q[1];", "Invalid barrier operand"),
        ("barrier r[0];", "Undeclared quantum register"),
        ("barrier q[2];", "out of range"),
    ],
)
def test_from_qasm2_rejects_invalid_barriers(line, error):
    with pytest.raises(ValueError, match=error):
        from_qasm2(f"OPENQASM 2.0;\nqreg q[2];\n{line}\n")


def test_barrier_qasm3_roundtrip():
    pytest.importorskip("openqasm3")
    circuit = Circuit(3)
    circuit.add([H(0), Barrier(), Barrier(2, 0), M(0)])
    qasm = to_qasm3(circuit)
    assert "barrier q;" in qasm
    assert "barrier q[2], q[0];" in qasm
    assert from_qasm3(qasm) == circuit


def test_barrier_from_qasm3_without_operands_spans_every_qubit():
    pytest.importorskip("openqasm3")
    circuit = from_qasm3('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nbarrier;\nbarrier q[1], q[2];\n')
    assert [gate.qubits for gate in circuit.gates] == [(0, 1, 2), (1, 2)]


def test_barrier_is_left_out_of_qir():
    pytest.importorskip("pyqir")
    from qilisdk.utils.qir import to_qir  # ruff: ignore[import-outside-top-level]

    assert to_qir(bell_with_barriers()) == to_qir(bell())


# --- Simulation -------------------------------------------------------------------------------------------------------


def test_qilisim_ignores_barriers():
    readout = Readout().with_state_tomography()
    with_barriers = Circuit(2)
    with_barriers.add([Barrier(), H(0), Barrier(0, 1), CNOT(0, 1), Barrier(1)])
    without = Circuit(2)
    without.add([H(0), CNOT(0, 1)])
    state = QiliSim().execute(DigitalPropagation(with_barriers), readout).get_state()
    expected = QiliSim().execute(DigitalPropagation(without), readout).get_state()
    np.testing.assert_allclose(state.dense(), expected.dense(), atol=1e-8)


def test_qilisim_samples_final_measurements_followed_by_a_barrier():
    samples = QiliSim().execute(DigitalPropagation(bell_with_barriers()), Readout().with_sampling(nshots=200))
    assert set(samples.get_samples()) <= {"00", "11"}


def test_qutip_ignores_barriers():
    pytest.importorskip("qutip_qip")
    from qilisdk.backends.qutip_backend import QutipBackend  # ruff: ignore[import-outside-top-level]

    samples = QutipBackend().execute(DigitalPropagation(bell_with_barriers()), Readout().with_sampling(nshots=200))
    assert set(samples.get_samples()) <= {"00", "11"}
