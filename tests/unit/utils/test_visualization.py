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
from matplotlib import font_manager as fm

import qilisdk.utils.visualization.circuit_renderers
import qilisdk.utils.visualization.schedule_renderers
from qilisdk.analog import Schedule, X, Z
from qilisdk.digital import CNOT, RX, SWAP, Circuit, Controlled, M
from qilisdk.digital import X as XGate
from qilisdk.digital import Y as YGate
from qilisdk.utils.visualization.circuit_renderers import MatplotlibCircuitRenderer
from qilisdk.utils.visualization.schedule_renderers import MatplotlibScheduleRenderer
from qilisdk.utils.visualization.style import CircuitStyle, ScheduleStyle


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
