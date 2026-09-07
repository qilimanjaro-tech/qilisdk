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
import itertools
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager as fm
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import qilisdk.utils.visualization.circuit_renderers
import qilisdk.utils.visualization.hamiltonian_renderers
import qilisdk.utils.visualization.schedule_renderers
from qilisdk.analog import Hamiltonian, I, Schedule, X, Y, Z
from qilisdk.core import QTensor
from qilisdk.digital import CNOT, RX, SWAP, Circuit, Controlled, M
from qilisdk.digital import X as XGate
from qilisdk.digital import Y as YGate
from qilisdk.utils.visualization.circuit_renderers import MatplotlibCircuitRenderer
from qilisdk.utils.visualization.hamiltonian_renderers import MatplotlibHamiltonianRenderer
from qilisdk.utils.visualization.qtensor_renderers import MatplotlibQTensorRenderer
from qilisdk.utils.visualization.schedule_renderers import MatplotlibEigenvalueRenderer, MatplotlibScheduleRenderer
from qilisdk.utils.visualization.style import CircuitStyle, HamiltonianStyle, QTensorStyle, ScheduleStyle
from qilisdk.utils.visualization.themes import dark


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


def test_schedule_draw_eigenvalues_with_no_state_but_overlaps_runs_with_warning(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    # Create a simple schedule for testing
    H0 = X(1) + X(0)
    H1 = Z(1) + Z(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={})
    schedule.draw_eigenvalues(show_overlaps=True)  # Should warn but not fail


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


def test_schedule_draw_eigenvalues_calculate_overlaps(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    # Create a simple schedule for testing
    H0 = X(1) + X(0)
    H1 = Z(1) + Z(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={}, dt=1.0)
    states = [QTensor.uniform(2) for _ in range(11)]
    renderer = MatplotlibEigenvalueRenderer(schedule=schedule, style=ScheduleStyle())
    overlaps = renderer._calculate_overlaps(
        state=states[0],
        eigenstates=[QTensor.ket(0, 0), QTensor.ket(0, 1), QTensor.ket(1, 0), QTensor.ket(1, 1)],
        eigenvalues=[[0.5], [0.5], [0.5], [0.5]],
        time_index=0,
        eigen_range=4.0,
        sig_figs=2,
    )
    assert overlaps == [(0.5, 100.0)]


def test_calculate_expectation_values_too_few_states_raises():
    H0 = X(1) + X(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0}, coefficients={}, dt=1.0)
    # tlist has 11 points (0..10), but only 5 states provided
    states = [QTensor.ket(0, 0) for _ in range(5)]
    renderer = MatplotlibEigenvalueRenderer(schedule=schedule, style=ScheduleStyle(), intermediate_states=states)
    with pytest.raises(ValueError, match="Length of intermediate_states must match"):
        renderer._calculate_expectation_values()


def test_calculate_expectation_values_non_hamiltonian_raises():
    fake_h = MagicMock()
    fake_h.nqubits = 1
    schedule = Schedule(total_time=10, hamiltonians={"H0": fake_h}, coefficients={}, dt=1.0)
    states = [QTensor.ket(0) for _ in range(11)]
    renderer = MatplotlibEigenvalueRenderer(schedule=schedule, style=ScheduleStyle(), intermediate_states=states)
    with pytest.raises(ValueError, match="Expected full_hamiltonian to be a Hamiltonian"):
        renderer._calculate_expectation_values()


# ---------------------------------------------------------------------------
# Schedule plot titles
# ---------------------------------------------------------------------------


@pytest.fixture
def plot_titles(monkeypatch):
    """Collect the axes title of every schedule plot rendered during the test.

    Both schedule renderers set their title in ``MatplotlibScheduleRenderer.setup_axes``,
    so spying there captures what ``Schedule.draw`` and ``Schedule.draw_eigenvalues``
    ended up titling their plot without reaching into the global pyplot state.
    """
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt, "draw", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.schedule_renderers.plt.Figure, "savefig", mock_save)

    titles = []
    setup_axes = MatplotlibScheduleRenderer.setup_axes

    def spy(self):
        setup_axes(self)
        titles.append(self.ax.get_title())

    monkeypatch.setattr(MatplotlibScheduleRenderer, "setup_axes", spy)
    return titles


def make_schedule():
    return Schedule(total_time=10, hamiltonians={"H0": X(1) + X(0), "H1": Z(1) + Z(0)}, coefficients={}, dt=1.0)


def test_schedule_draw_is_not_titled_as_eigenvalues(plot_titles):
    # draw() used to fall through to the renderer default, which titled the coefficient
    # plot "Schedule Eigenvalues".
    make_schedule().draw()
    (title,) = plot_titles
    assert "Coefficient" in title
    assert "Eigenvalue" not in title


def test_schedule_draw_eigenvalues_is_titled_as_eigenvalues(plot_titles):
    make_schedule().draw_eigenvalues()
    (title,) = plot_titles
    assert "Eigenvalue" in title


def test_schedule_draw_and_draw_eigenvalues_have_distinct_titles(plot_titles):
    # The two plots showing the same title was the bug, independently of the wording chosen.
    schedule = make_schedule()
    schedule.draw()
    schedule.draw_eigenvalues()
    coefficient_title, eigenvalue_title = plot_titles
    assert coefficient_title != eigenvalue_title


@pytest.mark.parametrize("method", ["draw", "draw_eigenvalues"])
def test_schedule_draw_keeps_an_explicit_style_title(plot_titles, method):
    getattr(make_schedule(), method)(style=ScheduleStyle(title="My Own Title"))
    assert plot_titles == ["My Own Title"]


def test_schedule_renderer_falls_back_to_a_generic_title(plot_titles):
    # A renderer used directly, with a style carrying no title, gets the neutral default.
    MatplotlibScheduleRenderer(schedule=make_schedule(), style=ScheduleStyle()).plot()
    assert plot_titles == ["Schedule"]


# ---------------------------------------------------------------------------
# Hamiltonian renderer
# ---------------------------------------------------------------------------


@pytest.fixture
def no_plot(monkeypatch):
    monkeypatch.setattr(qilisdk.utils.visualization.hamiltonian_renderers.plt, "show", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.hamiltonian_renderers.plt, "draw", mock_show)
    monkeypatch.setattr(qilisdk.utils.visualization.hamiltonian_renderers.plt.Figure, "savefig", mock_save)


def test_hamiltonian_style_defaults():
    style = HamiltonianStyle()
    assert style.layout == "spring"
    assert style.show_colorbar is True
    assert style.coupling_line_styles["ZZ"] == "-"
    assert isinstance(style.font, fm.FontProperties)


def test_hamiltonian_renderer_collects_terms(no_plot):
    H = 2 * X(0) - 1.5 * Z(0) + Y(1) + 0.5 * Z(0) * Z(1) + 0.25 * X(0) * X(1) + 3.0 * I(0)
    renderer = MatplotlibHamiltonianRenderer(H)
    renderer.plot()

    assert renderer._fields[0] == {"X": 2.0, "Z": -1.5}
    assert renderer._fields[1] == {"Y": 1.0}
    assert renderer._couplings[0, 1] == {"ZZ": 0.5, "XX": 0.25}
    assert renderer._multi_body == []
    assert renderer._offset == 3.0
    assert renderer._use_magnitude is False
    assert renderer.axes is renderer.ax


def test_hamiltonian_renderer_build_graph_matches_qubits():
    H = X(0) + Z(0) * Z(1) + Y(2)
    graph = MatplotlibHamiltonianRenderer(H).build_graph()
    assert graph.num_nodes() == 3
    assert sorted(graph.edge_list()) == [(0, 1)]
    assert graph[0] == {"X": 1.0}
    assert graph[2] == {"Y": 1.0}


def test_hamiltonian_renderer_isolated_qubit_and_no_field_node(no_plot):
    # q1 carries no local field, so its node is drawn as a plain disc.
    H = X(0) + Z(0) * Z(1)
    renderer = MatplotlibHamiltonianRenderer(H, style=HamiltonianStyle(layout="circular"))
    renderer.plot()
    assert renderer._fields.get(1) is None


def test_hamiltonian_renderer_multi_body_is_drawn(no_plot):
    H = Z(0) * Z(1) * Z(2) + X(0)
    renderer = MatplotlibHamiltonianRenderer(H)
    renderer.plot()
    assert renderer._multi_body == [((0, 1, 2), "ZZZ", 1.0)]
    assert renderer._line_style("ZZZ") in HamiltonianStyle().default_coupling_line_styles

    # A three-body term never reuses the line style of a two-body term drawn alongside it.
    renderer = MatplotlibHamiltonianRenderer(H + Z(0) * Z(1))
    renderer.plot()
    assert renderer._line_style("ZZZ") != renderer._line_style("ZZ")


def test_hamiltonian_renderer_multi_body_can_be_hidden(no_plot):
    H = Z(0) * Z(1) * Z(2) + X(0)
    renderer = MatplotlibHamiltonianRenderer(H, style=HamiltonianStyle(show_multi_body=False))
    renderer.plot()
    assert renderer._coupling_values() == []


def test_hamiltonian_renderer_complex_coefficients_use_magnitude(no_plot):
    H = 3j * X(0) + Z(0) * Z(1)
    renderer = MatplotlibHamiltonianRenderer(H)
    renderer.plot()
    assert renderer._use_magnitude is True
    assert renderer._fields[0] == {"X": 3.0}


def test_hamiltonian_renderer_separate_color_scales(no_plot):
    H = 5 * X(0) + 0.1 * Z(0) * Z(1)
    renderer = MatplotlibHamiltonianRenderer(H, style=HamiltonianStyle(separate_color_scales=True))
    renderer.plot()
    field_norm, coupling_norm = renderer._make_norms()
    assert field_norm is not coupling_norm
    assert coupling_norm.vmin == pytest.approx(-0.4)


def test_hamiltonian_renderer_shared_color_scale_is_symmetric_around_zero(no_plot):
    H = 2 * X(0) - 0.5 * Z(1) + Z(0) * Z(1)
    renderer = MatplotlibHamiltonianRenderer(H)
    renderer.plot()
    field_norm, coupling_norm = renderer._make_norms()
    assert field_norm is coupling_norm
    assert field_norm.vmin == pytest.approx(-2.0)
    assert field_norm.vmax == pytest.approx(2.0)


def test_hamiltonian_renderer_norm_edge_cases():
    norm_for = MatplotlibHamiltonianRenderer._norm_for
    empty = norm_for([])
    assert (empty.vmin, empty.vmax) == (-1.0, 1.0)
    flat = norm_for([2.0, 2.0])
    assert (flat.vmin, flat.vmax) == (1.5, 2.5)
    positive = norm_for([1.0, 3.0])
    assert (positive.vmin, positive.vmax) == (1.0, 3.0)


def test_hamiltonian_renderer_custom_colormap(no_plot):
    H = X(0) + Z(0) * Z(1)
    renderer = MatplotlibHamiltonianRenderer(H, style=HamiltonianStyle(colormap="viridis"))
    renderer.plot()
    assert renderer._make_colormap().name == "viridis"


def test_hamiltonian_renderer_contrast_color():
    assert MatplotlibHamiltonianRenderer._contrast_color((1.0, 1.0, 1.0, 1.0)) == "black"
    assert MatplotlibHamiltonianRenderer._contrast_color((0.0, 0.0, 0.0, 1.0)) == "white"


@pytest.mark.parametrize("layout", ["spring", "circular", "shell", "spiral", "random"])
def test_hamiltonian_renderer_layouts(no_plot, layout):
    H = sum(Z(i) * Z(i + 1) for i in range(3)) + X(0)
    renderer = MatplotlibHamiltonianRenderer(H, style=HamiltonianStyle(layout=layout))
    renderer.plot()
    positions = renderer._compute_positions(renderer.build_graph())
    assert set(positions) == {0, 1, 2, 3}
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in positions.values())


def test_hamiltonian_renderer_explicit_positions(no_plot):
    H = X(0) + Z(0) * Z(1)
    style = HamiltonianStyle(positions={0: (0.0, 0.0), 1: (2.0, 0.0)})
    renderer = MatplotlibHamiltonianRenderer(H, style=style)
    renderer.plot()
    assert renderer._compute_positions(renderer.build_graph()) == {0: (0.0, 0.5), 1: (1.0, 0.5)}


def test_hamiltonian_renderer_incomplete_positions_raise():
    H = X(0) + Z(0) * Z(1)
    renderer = MatplotlibHamiltonianRenderer(H, style=HamiltonianStyle(positions={0: (0.0, 0.0)}))
    with pytest.raises(ValueError, match="missing entries for qubits"):
        renderer.plot()


def test_hamiltonian_renderer_coincident_positions_are_centred():
    normalized = MatplotlibHamiltonianRenderer._normalize_positions({0: (1.0, 1.0), 1: (1.0, 1.0)})
    assert normalized == {0: (0.5, 0.5), 1: (0.5, 0.5)}


def test_hamiltonian_renderer_single_qubit_radius(no_plot):
    H = X(0) + Z(0)
    renderer = MatplotlibHamiltonianRenderer(H)
    renderer.plot()
    assert renderer._compute_radius({0: (0.5, 0.5)}) == pytest.approx(HamiltonianStyle().node_radius)


def test_hamiltonian_renderer_radius_respects_lower_bound():
    style = HamiltonianStyle(node_radius=0.3, min_node_radius=0.2)
    renderer = MatplotlibHamiltonianRenderer(X(0) + X(1), style=style)
    assert renderer._compute_radius({0: (0.0, 0.0), 1: (0.1, 0.0)}) == pytest.approx(0.2)


def test_hamiltonian_renderer_label_position_and_angles():
    renderer = MatplotlibHamiltonianRenderer(X(0))
    straight = renderer._label_position((0.0, 0.0), (2.0, 0.0), 0.0)
    assert straight == (1.0, 0.0)
    curved = renderer._label_position((0.0, 0.0), (2.0, 0.0), 0.5)
    assert curved == (1.0, -1.0)

    trimmed_start, trimmed_end = renderer._trim((0.0, 0.0), (1.0, 0.0), 0.25)
    assert trimmed_start == pytest.approx((0.25, 0.0))
    assert trimmed_end == pytest.approx((0.75, 0.0))
    # Coincident nodes cannot be trimmed.
    assert renderer._trim((0.0, 0.0), (0.0, 0.0), 0.25) == ((0.0, 0.0), (0.0, 0.0))


def test_hamiltonian_renderer_qubit_label_avoids_edges():
    H = Z(0) * Z(1) + Z(0) * Z(2) + Z(0) * Z(1) * Z(2)
    renderer = MatplotlibHamiltonianRenderer(H)
    renderer._collect_terms()
    positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}
    # q0's edges leave to the right and upwards, so its label goes down-left.
    angle = renderer._label_angle(0, positions)
    assert -np.pi < angle < 0
    assert np.cos(angle) < 0
    # An isolated qubit keeps the default placement below the node.
    isolated = MatplotlibHamiltonianRenderer(X(0))
    isolated._collect_terms()
    assert isolated._label_angle(0, {0: (0.0, 0.0)}) == pytest.approx(-np.pi / 2)


def test_hamiltonian_renderer_alignment_helpers():
    renderer = MatplotlibHamiltonianRenderer
    assert renderer._horizontal_alignment(1.0) == "left"
    assert renderer._horizontal_alignment(-1.0) == "right"
    assert renderer._horizontal_alignment(0.0) == "center"
    assert renderer._vertical_alignment(1.0) == "bottom"
    assert renderer._vertical_alignment(-1.0) == "top"
    assert renderer._vertical_alignment(0.0) == "center"


def test_hamiltonian_renderer_unknown_coupling_type_falls_back():
    renderer = MatplotlibHamiltonianRenderer(X(0))
    assert renderer._line_style("ZZ") == HamiltonianStyle().default_coupling_line_styles[0]


def test_hamiltonian_renderer_configured_style_wins_over_cycle(no_plot):
    style = HamiltonianStyle(coupling_line_styles={"XY": (0, (1, 1))})
    renderer = MatplotlibHamiltonianRenderer(X(0) * Y(1) + Z(0) * Z(1), style=style)
    renderer.plot()
    assert renderer._line_style("XY") == (0, (1, 1))
    assert renderer._line_style("ZZ") != (0, (1, 1))


def test_hamiltonian_renderer_without_decorations(no_plot):
    style = HamiltonianStyle(
        show_colorbar=False,
        show_legend=False,
        show_field_labels=False,
        show_qubit_labels=False,
        show_coupling_labels=False,
        show_identity_offset=False,
        tight_layout=False,
        theme=dark,
    )
    H = X(0) + Z(0) * Z(1) + Z(0) * Z(1) * Z(2) + 2.0 * I(0)
    renderer = MatplotlibHamiltonianRenderer(H, style=style)
    renderer.plot()
    assert renderer.ax.get_legend() is None


def test_hamiltonian_renderer_legend_skipped_without_couplings(no_plot):
    renderer = MatplotlibHamiltonianRenderer(X(0) + Z(0))
    renderer.plot()
    assert renderer.ax.get_legend() is None


def test_hamiltonian_renderer_uses_given_axes(no_plot):
    _, ax = plt.subplots()
    renderer = MatplotlibHamiltonianRenderer(X(0) + Z(0) * Z(1), ax=ax)
    renderer.plot()
    assert renderer.axes is ax
    assert ax.get_title() == "Hamiltonian"


def test_hamiltonian_renderer_title_and_save(no_plot, tmp_path):
    renderer = MatplotlibHamiltonianRenderer(X(0) + Z(0) * Z(1), style=HamiltonianStyle(title="My H"))
    renderer.plot()
    assert renderer.ax.get_title() == "My H"
    renderer.save(str(tmp_path / "h.png"))
    renderer.show()


def test_hamiltonian_renderer_empty_hamiltonian_raises():
    empty_H = Hamiltonian()
    with pytest.raises(ValueError, match="does not act on any qubit"):
        empty_H.draw()


def test_hamiltonian_renderer_optional_edge_labels(no_plot):
    # Edge labels are off by default because the legend already maps styles to coupling types.
    assert HamiltonianStyle().show_coupling_labels is False
    H = Z(0) * Z(1) + X(0) * X(1) + Z(0) * Z(1) * Z(2)
    renderer = MatplotlibHamiltonianRenderer(H, style=HamiltonianStyle(show_coupling_labels=True))
    renderer.plot()
    assert {"ZZ", "XX", "ZZZ"} <= {text.get_text() for text in renderer.ax.texts}


def test_hamiltonian_renderer_separate_colorbar_texts_do_not_overlap(no_plot):
    H = 5 * X(0) - 3 * Z(0) + X(1) + 0.1 * Z(0) * Z(1) + 0.05 * X(0) * X(1)
    renderer = MatplotlibHamiltonianRenderer(
        H, style=HamiltonianStyle(separate_color_scales=True, title="My Hamiltonian")
    )
    renderer.plot()

    figure = renderer.ax.figure
    figure.canvas.draw()
    bars = [ax for ax in figure.axes if ax is not renderer.ax]
    assert len(bars) == 2
    # The inner bar keeps its label on the side that the neighbouring bar does not occupy.
    assert bars[0].yaxis.get_label_position() == "right"
    assert bars[1].yaxis.get_label_position() == "left"

    boxes = [
        text.get_window_extent() for bar in bars for text in bar.findobj(Text) if text.get_text() and text.get_visible()
    ]
    for first, second in itertools.combinations(boxes, 2):
        overlap = Bbox.intersection(first, second)
        # A one-pixel touch is fine, real overlap is not.
        assert overlap is None or overlap.width * overlap.height < 1.0
