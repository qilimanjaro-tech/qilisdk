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

import builtins
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager as fm

import qilisdk.utils.visualization.circuit_renderers
import qilisdk.utils.visualization.schedule_renderers
from qilisdk.analog import Schedule, X, Z
from qilisdk.core import QTensor
from qilisdk.digital import CNOT, RX, SWAP, Circuit, Controlled, M
from qilisdk.digital import X as XGate
from qilisdk.digital import Y as YGate
from qilisdk.utils.visualization.circuit_renderers import MatplotlibCircuitRenderer
from qilisdk.utils.visualization.qtensor_renderers import MatplotlibQTensorRenderer
from qilisdk.utils.visualization.schedule_renderers import MatplotlibEigenvalueRenderer, MatplotlibScheduleRenderer
from qilisdk.utils.visualization.style import CircuitStyle, QTensorStyle, ScheduleStyle


def mock_show():
    return None


def mock_save(self, *args, **kwargs):
    return None


def test_schedule_style_init():
    style = ScheduleStyle()
    assert style.dpi == 150
    assert style.theme.background is not None
    assert style.fontsize == 10
    assert isinstance(style.font, fm.FontProperties)


def test_schedule_renderer_init():
    H0 = X(1) + X(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0}, coefficients={})
    style = ScheduleStyle()
    renderer = MatplotlibScheduleRenderer(schedule=schedule, style=style)
    assert renderer.schedule == schedule
    assert renderer.style == style
    assert renderer.ax is not None


def test_schedule_renderer_with_axes(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # Prevent actual rendering during tests
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)  # Prevent file saving during tests

    H0 = X(1) + X(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0}, coefficients={})
    style = ScheduleStyle()
    style.grid = True
    style.grid_style = {}
    ax = plt.gca()
    renderer = MatplotlibScheduleRenderer(schedule=schedule, style=style, ax=ax)
    assert renderer.schedule == schedule
    assert renderer.style == style
    assert renderer.ax is not None
    renderer.plot(ax=ax)


def test_schedule_draw(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    # Create a simple schedule for testing
    H0 = X(1) + X(0)
    H1 = Z(1) + Z(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={})
    schedule.draw()
    schedule.draw(filepath="test_schedule.png")


def test_circuit_style_init():
    style = CircuitStyle()
    assert np.isclose(style.padding, 0.3)


def test_circuit_renderer_init():
    circuit = Circuit(2)
    style = CircuitStyle()
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style)
    assert renderer.circuit == circuit
    assert renderer.style == style
    assert renderer._ax is not None


def test_circuit_renderer_with_axes(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # Prevent actual rendering during tests
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)  # Prevent file saving during tests

    circuit = Circuit(2)
    style = CircuitStyle()
    ax = plt.gca()
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style, ax=ax)
    assert renderer.circuit == circuit
    assert renderer.style == style
    assert renderer._ax is not None
    renderer.plot()


def test_circuit_draw(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt.Figure, "savefig", mock_save)

    # Create a simple circuit for testing
    circuit = Circuit(2)
    circuit.add(XGate(0))
    circuit.add(SWAP(0, 1))
    circuit.add(CNOT(0, 1))
    circuit.add(M(0))
    circuit.add(Controlled(0, basic_gate=XGate(1)))
    circuit.add(M(0))
    circuit.draw()
    circuit.draw(filepath="test_circuit.png")


def test_compact_layout(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt.Figure, "savefig", mock_save)

    circuit = Circuit(3)
    circuit.add(XGate(0))
    circuit.add(SWAP(0, 1))
    circuit.add(M(0))
    circuit.add(RX(0, theta=np.pi / 2))
    circuit.add(Controlled(0, basic_gate=XGate(1)))
    circuit.add(Controlled(0, basic_gate=YGate(1)))
    circuit.add(Controlled(2, basic_gate=SWAP(0, 1)))
    circuit.add(M(0))

    style = CircuitStyle()
    style.layout = "compact"
    style.title = "Compact Layout Test"
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style)
    assert renderer.style.layout == "compact"
    renderer.plot()


def test_ipython(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt.Figure, "savefig", mock_save)

    monkeypatch.setattr(
        builtins,
        "get_ipython",
        MagicMock(return_value=True),
        raising=False,
    )

    circuit = Circuit(2)
    style = CircuitStyle()
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style)
    renderer.plot()


def test_pi_fraction():
    circuit = Circuit(1)
    style = CircuitStyle()
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style)

    # backslash blackslash pi
    assert renderer._pi_fraction(np.pi / 2) == "\\pi/2"
    assert renderer._pi_fraction(np.pi) == "\\pi"
    assert renderer._pi_fraction(3 * np.pi / 4) == "3\\pi/4"
    assert renderer._pi_fraction(np.pi / 3) == "\\pi/3"
    assert renderer._pi_fraction(2 * np.pi / 3) == "2\\pi/3"
    assert renderer._pi_fraction(np.pi / 6) == "\\pi/6"
    assert renderer._pi_fraction(0) == "0"
    assert renderer._pi_fraction(np.sqrt(2), tol=1e-7) == "1.41"


def test_superscript_dagger():
    circuit = Circuit(1)
    style = CircuitStyle()
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style)

    assert renderer._with_superscript_dagger("RX†") == "$\\mathrm{RX}^{\\dagger}$"


def test_multi_target_gates(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt.Figure, "savefig", mock_save)

    # note: I'm not sure if we actually support these in qilisdk, so I'm mocking it
    three_target_gate = YGate(0)
    three_target_gate._target_qubits = (0, 1, 2)
    circuit = Circuit(3)
    style = CircuitStyle()
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style)
    circuit.add(three_target_gate)
    renderer.plot()


def test_layer_stacking(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers.plt.Figure, "savefig", mock_save)

    circuit = Circuit(3)
    style = CircuitStyle()
    style.layout = "compact"
    renderer = MatplotlibCircuitRenderer(circuit=circuit, style=style)
    circuit.add(XGate(1))
    circuit.add(CNOT(0, 2))
    renderer.plot()


def test_qtensor_draw_runs(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.qtensor_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.qtensor_renderers.plt.Figure, "savefig", mock_save)
    qobj = QTensor.ket(0)
    qobj.draw()


def test_qtensor_draw_with_filepath_runs(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.qtensor_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.qtensor_renderers.plt.Figure, "savefig", mock_save)
    qobj = QTensor.ket(0)
    qobj.draw(filepath="test_output.png")


def test_qtensor_draw_with_style_runs(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.qtensor_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.qtensor_renderers.plt.Figure, "savefig", mock_save)
    qobj = QTensor.ket(0)
    style = QTensorStyle(title="Custom Title")
    qobj.draw(filepath="test_output.png", style=style)


def test_qtensor_draw_many_qubits_raises():
    qobj = QTensor.ket(0, 0, 0, 0)  # 4 qubits
    with pytest.raises(ValueError, match="Drawing is only supported for single-qubit states"):
        qobj.draw()


def test_qtensor_draw_non_ket_raises():
    qobj = QTensor(np.eye(2))  # Not a ket
    with pytest.raises(ValueError, match="Drawing is only supported for state vectors"):
        qobj.draw()


def test_qtensor_make_axes_bad_type(monkeypatch):
    mock_axes = MagicMock()
    monkeypatch.setattr(
        qilisdk.utils.visualization.qtensor_renderers.plt, "subplots", lambda *args, **kwargs: (MagicMock(), mock_axes)
    )
    with pytest.raises(TypeError, match="Expected axes of type"):
        MatplotlibQTensorRenderer._make_axes(dpi=100)


def test_schedule_draw_eigenvalues(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    # Create a simple schedule for testing
    H0 = X(1) + X(0)
    H1 = Z(1) + Z(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={}, dt=1.0)
    states = [QTensor.ket(0, 0) for _ in range(11)]
    schedule.draw_eigenvalues(intermediate_states=states, show_overlaps=True)
    schedule.draw_eigenvalues(filepath="test_schedule.png")


def test_schedule_draw_eigenvalues_with_no_state_but_overlaps_raises(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    # Create a simple schedule for testing
    H0 = X(1) + X(0)
    H1 = Z(1) + Z(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={})
    with pytest.raises(ValueError, match="without intermediate states"):
        schedule.draw_eigenvalues(show_overlaps=True)


def test_schedule_draw_eigenvalues_too_many_qubits_raises(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    # Create a schedule with many qubits
    H0 = sum(X(i) for i in range(10))
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0}, coefficients={})
    states = [QTensor.ket(0, 0, 0) for _ in range(11)]
    with pytest.raises(ValueError, match="with more than"):
        schedule.draw_eigenvalues(intermediate_states=states)


def test_schedule_draw_eigenvalues_not_hamiltonian_raises(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    fake_h = MagicMock()
    fake_h.nqubits = 2

    # Create a schedule with a non-Hamiltonian functional
    schedule = Schedule(total_time=10, hamiltonians={"H0": fake_h}, coefficients={})
    states = [QTensor.ket(0) for _ in range(11)]
    with pytest.raises(ValueError, match="to be a Hamiltonian"):
        schedule.draw_eigenvalues(intermediate_states=states)


# icmethod
#     def _calculate_overlaps(
#         state: QTensor,
#         eigenstates: list[QTensor],
#         eigenvalues: list[list[float]],
#         time_index: int,
#         eigen_range: float,
#         sig_figs: int,
#     ) -> list[tuple[float, float]]:
#         overlaps = []
#         for j, eig in enumerate(eigenstates):
#             overlap = 100.0 * state.fidelity(eig)
#             y_loc = eigenvalues[j][time_index]
#             if overlap > 10 ** (-sig_figs):
#                 overlaps.append((y_loc, overlap))

#         # Group nearby overlaps together to avoid clutter
#         grouped_overlaps: list[tuple[float, float]] = []
#         for overlap in overlaps:
#             found_group = False
#             for idx, grouped_overlap in enumerate(grouped_overlaps):
#                 if abs(grouped_overlap[0] - overlap[0]) < 0.05 * eigen_range:
#                     # If within 5% of the eigenvalue range, group them together by averaging the y location and summing the overlap percentage
#                     new_y_loc = (grouped_overlap[0] + overlap[0]) / 2
#                     new_overlap = grouped_overlap[1] + overlap[1]
#                     grouped_overlaps[idx] = (new_y_loc, new_overlap)
#                     found_group = True
#                     break
#             if not found_group:
#                 grouped_overlaps.append(overlap)

#         return grouped_overlaps


def test_schedule_draw_eigenvalues_calculate_overlaps(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    # Create a simple schedule for testing
    H0 = X(1) + X(0)
    H1 = Z(1) + Z(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={}, dt=1.0)
    states = [QTensor.ket(0, 0) for _ in range(11)]
    renderer = MatplotlibEigenvalueRenderer(schedule=schedule, style=ScheduleStyle())
    overlaps = renderer._calculate_overlaps(
        state=states[0],
        eigenstates=[QTensor.ket(0, 0), QTensor.ket(0, 1)],
        eigenvalues=[[0.5], [0.5]],
        time_index=0,
        eigen_range=4.0,
        sig_figs=2,
    )
    assert overlaps == [(0.5, 100.0)]
