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

import math
import random
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest

import qilisdk.utils.visualization.circuit_renderers
from qilisdk.analog import X as analog_X
from qilisdk.analog import Y as analog_Y
from qilisdk.analog.hamiltonian import PauliX
from qilisdk.core import Parameter, QTensor
from qilisdk.core.comparison import LEQ
from qilisdk.digital import CNOT, CZ, RX, RY, RZ, SWAP, U1, U2, U3, Circuit, H, I, M, S, T, X, Y, Z
from qilisdk.digital.circuit import _apply_gate_left
from qilisdk.digital.exceptions import (
    GateHasNoMatrixError,
    NotAGateError,
    QubitOutOfRangeError,
    UnsupportedGateError,
)
from qilisdk.digital.gates import Adjoint, BasicGate, Controlled, Exponential, Gate
from qilisdk.utils.serialization import deserialize, serialize


def test_circuit_initialization():
    """
    Test basic initialization of the Circuit class.
    """
    c = Circuit(nqubits=3)
    assert c.nqubits == 3
    assert c.nparameters == 0  # No gates yet
    assert c.get_parameter_values() == []
    assert c.gates == []
    assert c.get_parameter_names() == []


def test_apply_gate_left_noop_for_zero_qubit_gate():
    class ZeroQubitGate(BasicGate):
        nqubits = 0
        qubits = ()
        matrix = QTensor(np.array([[1.0]], dtype=np.complex128))
        name = "ZeroQubitGate"

        def _generate_matrix(self):
            return self.matrix

    operator = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    out = _apply_gate_left(operator, ZeroQubitGate([0]), nqubits=1)
    np.testing.assert_array_equal(out, operator)


def test_circuit_set_parameters_bad():
    c = Circuit(nqubits=1)
    assert c.nparameters == 0
    with pytest.raises(ValueError, match="not defined for this object"):
        c.set_parameters({"param1": 0.5})
    with pytest.raises(ValueError, match="not defined in the list of"):
        c.set_parameter_bounds({"param1": (0.0, np.pi)})


def test_circuit_set_parameters():
    c = Circuit(nqubits=1)
    param = Parameter("param1", 0.0)
    c.add(RX(qubit=0, theta=param))
    assert c.nparameters == 1
    c.set_parameters({"param1": 0.5})
    assert c.get_parameter_values() == [0.5]


def test_circuit_set_parameters_values():
    c = Circuit(nqubits=1)
    param = Parameter("param1", 0.0)
    c.add(RX(qubit=0, theta=param))
    assert c.nparameters == 1
    c.set_parameter_values([0.5])
    assert c.get_parameter_values() == [0.5]


def test_circuit_set_prefix_keeps_parameter_links():
    c = Circuit(nqubits=1)
    c.add(RX(qubit=0, theta=np.pi / 4))

    c.set_prefix("p_")
    assert c.get_parameter_names() == ["p_RX(0)_theta_0"]

    c.set_parameters({"p_RX(0)_theta_0": np.pi / 3})
    assert np.isclose(c.get_parameter_values()[0], np.pi / 3)


def test_circuit_constraints_are_enforced_on_set_parameters():
    c = Circuit(nqubits=1)
    left = Parameter("left", 0.1)
    right = Parameter("right", 0.2)
    c.add(RX(qubit=0, theta=left))
    c.add(RY(qubit=0, theta=right))
    c.add_parameter_constraint(LEQ(left, right))

    with pytest.raises(
        ValueError,
        match=r"New assignation of the parameters breaks the parameter constraints:",
    ):
        c.set_parameters({"left": 0.9, "right": 0.1})

    c.set_parameters({"left": 0.2, "right": 0.9})
    assert c.get_parameters() == {"left": 0.2, "right": 0.9}


def test_parameter_bounds():
    c = Circuit(nqubits=1)
    param = Parameter("param1", 0.0, bounds=(0.0, np.pi))
    c.add(RX(qubit=0, theta=param))
    bounds = c.get_parameter_bounds()
    assert bounds == {"param1": (0.0, np.pi)}
    c.set_parameter_bounds({"param1": (0.0, np.pi - 0.1)})
    bounds = c.get_parameter_bounds()
    assert bounds == {"param1": (0.0, np.pi - 0.1)}


def test_add_non_parameterized_gate():
    """
    Test adding a single-qubit, non-parameterized gate (e.g., X) within range.
    """
    c = Circuit(nqubits=2)
    x_gate = X(qubit=0)
    c.add(x_gate)

    # Verify internal state
    # The gate should be in _gates, but not in _parameterized_gates
    assert x_gate in c._gates
    assert len(c._parameters) == 0

    # Property should also work
    assert x_gate in c.gates

    # No parameters
    assert c.nparameters == 0
    assert c.get_parameter_values() == []


def test_add_parameterized_gate():
    """
    Test adding a single-qubit, parameterized gate (e.g., RX) within range.
    """
    c = Circuit(nqubits=2)
    rx_gate = RX(qubit=1, theta=np.pi / 2)  # parameterized gate
    c.add(rx_gate)

    # rx_gate should be in both _gates and _parameterized_gates
    assert rx_gate in c._gates
    assert len(c._parameters) == 1
    assert any(rx_gate.name in label for label in c._parameters)

    # Property should also work
    assert rx_gate in c.gates

    # Should have exactly 1 parameter
    assert c.nparameters == 1
    # The parameter value is [np.pi/2]
    assert c.get_parameter_values() == [np.pi / 2]


def test_add_gate_out_of_range():
    """
    Check that adding a gate whose target qubit is >= nqubits raises QubitOutOfRangeError.
    """
    c = Circuit(nqubits=1)
    # For instance, qubit index 1 is invalid in a 1-qubit circuit
    with pytest.raises(QubitOutOfRangeError):
        c.add(X(qubit=1))


def test_multiple_parameterized_gates():
    """
    Test adding multiple parameterized gates and ensure nparameters and get_parameter_values()
    reflect the sum of all gate parameters in the correct order.
    """
    c = Circuit(nqubits=2)

    # RX has 1 parameter, RZ has 1 parameter
    rx_gate = RX(qubit=0, theta=0.1)
    rz_gate = RZ(qubit=1, phi=0.2)
    c.add(rx_gate)
    c.add(rz_gate)

    # Now the circuit has 2 parameterized gates, each with 1 parameter => total 2
    assert c.nparameters == 2
    # Check get_parameter_values => [0.1, 0.2]
    assert c.get_parameter_values() == [0.1, 0.2]


def test_set_parameter_values_correct():
    """
    Test setting parameter values correctly for multiple gates.
    """
    c = Circuit(nqubits=2)
    rx_gate = RX(qubit=0, theta=0.0)
    rz_gate = RZ(qubit=1, phi=0.0)
    c.add(rx_gate)
    c.add(rz_gate)

    # Circuit should have 2 parameters total (theta for RX, theta for RZ)
    assert c.nparameters == 2

    # We'll set them to [0.5, 1.0]
    new_values = [0.5, 1.0]
    c.set_parameter_values(new_values)

    # Verify that each gate's parameters are updated
    assert rx_gate.get_parameter_values() == [0.5]
    assert rz_gate.get_parameter_values() == [1.0]

    # get_parameter_values should reflect the new updates
    assert c.get_parameter_values() == new_values


def test_set_parameter_values_incorrect():
    """
    Test that setting parameter values with a mismatched list length raises ParametersNotEqualError.
    """
    c = Circuit(nqubits=2)
    rx_gate = RX(qubit=0, theta=0.0)
    rz_gate = RZ(qubit=1, phi=0.0)
    c.add(rx_gate)
    c.add(rz_gate)

    # circuit has 2 parameters total. Let's provide a list of length 3 instead.
    with pytest.raises(ValueError, match=r"Provided 3 but this object has 2 parameters."):
        c.set_parameter_values([0.1, 0.2, 0.3])


def test_empty_circuit_set_parameter_values():
    """
    Test that setting parameter values on an empty circuit
    doesn't cause an error if the list is empty.
    """
    c = Circuit(nqubits=2)
    assert c.nparameters == 0

    # Setting an empty list on a circuit with no parameters is valid
    c.set_parameter_values([])  # Should not raise any error
    # Still no parameters
    assert c.get_parameter_values() == []


def test_user_provides_custom_parameter():
    """
    Test that providing a custom Parameter to one or more gates,
    overrides default naming scheme
    and allows setting multiple gates' parameters at once.
    """
    angle = Parameter("angle", 0.0)
    c = Circuit(nqubits=2)
    c.add(RX(0, theta=angle))
    c.add(RZ(1, phi=angle))

    # Circuit has 1 parameter.
    assert c.nparameters == 1

    # Even though the total parameters of gates are 2.
    assert sum(gate.nparameters for gate in c.gates) == 2

    # Check that circuit's parameter has the correct label and value
    assert c.get_parameters() == {angle.label: angle.value}

    # Check that gates' parameters have the correct label and value
    assert all(
        label in gate.PARAMETER_NAMES and parameter.label == angle.label and parameter.value == angle.value
        for gate in c.gates
        for label, parameter in gate._parameters.items()
    )


def test_add_list_of_gates():
    c = Circuit(nqubits=2)
    x_gate = X(0)
    rx_gate = RX(1, theta=0.1)

    c.add([x_gate, rx_gate])

    assert c.gates == [x_gate, rx_gate]
    assert c.nparameters == 1
    assert c.get_parameter_values() == [0.1]


def test_insert_list_of_gates_keeps_order():
    c = Circuit(nqubits=2)
    x_gate = X(0)
    c.add(x_gate)

    rz_gate = RZ(0, phi=0.2)
    x2_gate = X(1)
    c.insert([rz_gate, x2_gate], index=1)

    assert c.gates == [x_gate, rz_gate, x2_gate]
    assert c.nparameters == 1


def test_append_and_prepend_circuits():
    base = Circuit(nqubits=2)
    x_gate = X(0)
    base.add(x_gate)

    appended = Circuit(nqubits=2)
    rx_gate = RX(1, theta=0.4)
    appended.add(rx_gate)
    base.append(appended)

    prepended = Circuit(nqubits=2)
    rz_gate = RZ(0, phi=0.2)
    x2_gate = X(1)
    prepended.add([rz_gate, x2_gate])
    base.prepend(prepended)

    assert base.gates == [rz_gate, x2_gate, x_gate, rx_gate]


def test_add_operator_supports_gate_and_circuit():
    c = Circuit(nqubits=2)
    x_gate = X(0)
    c = c + x_gate
    assert c.gates == [x_gate]

    other = Circuit(nqubits=2)
    angle = Parameter("RZ_phi", 0.3)
    rz_gate = RZ(1, phi=angle)
    other.add(rz_gate)
    c = c + other
    assert c.gates == [x_gate, rz_gate]

    prepend_gate = RX(0, theta=angle)
    c = prepend_gate + c
    assert c.gates == [prepend_gate, x_gate, rz_gate]

    # Change circuit's parameter value
    c.set_parameter_values([1.0])

    # Check that Parameter object has changed its value
    assert np.isclose(angle.value, 1.0)

    # Check that circuit's parameter has the correct label and value
    assert c.get_parameters() == {angle.label: angle.value}

    # Check that gates' parameters have the correct label and value
    assert all(
        label in gate.PARAMETER_NAMES and parameter.label == angle.label and parameter.value == angle.value
        for gate in c.gates
        for label, parameter in gate._parameters.items()
    )


def test_random_circuit():
    single_qubit_gates = {X, RX, S, RY, U1, U2, U3}
    two_qubit_gates = {CNOT}
    nqubits = 3
    ngates = 30

    c = Circuit.random(
        nqubits=nqubits,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        ngates=ngates,
        seed=42,
    )

    # Check that the circuit has the correct number of gates
    assert len(c.gates) == ngates

    # Check that all gates are from the provided sets
    for gate in c.gates:
        assert type(gate) in single_qubit_gates or type(gate) in two_qubit_gates

    # Make sure there are no duplicate gates next to each other
    for i in range(1, len(c.gates)):
        assert not (c.gates[i] is c.gates[i - 1] and c.gates[i].qubits == c.gates[i - 1].qubits)

    # Check that all target qubits are within range
    for gate in c.gates:
        for qubit in gate.qubits:
            assert 0 <= qubit < nqubits


def test_random_single_qubit_circuit():
    single_qubit_gates = {X, RX, RZ}
    two_qubit_gates = {CNOT}
    nqubits = 1
    ngates = 10

    c = Circuit.random(
        nqubits=nqubits,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        ngates=ngates,
        seed=42,
    )

    # Check that the circuit has the correct number of gates
    assert len(c.gates) == ngates

    # Check that all gates are from the provided set
    for gate in c.gates:
        assert type(gate) in single_qubit_gates
        for qubit in gate.qubits:
            assert 0 <= qubit < nqubits

    # Make sure there are no duplicate gates next to each other
    for i in range(1, len(c.gates)):
        assert not (c.gates[i] is c.gates[i - 1] and c.gates[i].qubits == c.gates[i - 1].qubits)


def test_random_circuit_no_gates():
    with pytest.raises(ValueError, match=r"At least one gate must be provided to generate a random circuit."):
        Circuit.random(nqubits=0, single_qubit_gates=set(), two_qubit_gates=set(), ngates=1000)


def test_random_circuit_single_qubit_one_gate():
    with pytest.raises(ValueError, match=r"Cannot generate a full random circuit with only one qubit and one gate."):
        Circuit.random(nqubits=1, single_qubit_gates={X}, two_qubit_gates=set(), ngates=1000)


def test_circuit_bad_insert_append():
    c = Circuit(nqubits=2)
    x_gate = X(10)
    with pytest.raises(QubitOutOfRangeError):
        c.insert(x_gate, index=0)
    with pytest.raises(QubitOutOfRangeError):
        c.append(x_gate)


def test_circuit_bad_prepend():
    c = Circuit(nqubits=2)
    c_2 = Circuit(nqubits=3)
    with pytest.raises(QubitOutOfRangeError, match="the prepended circuit contains"):
        c.prepend(c_2)


def test_circuit_add_not_supported():
    c = Circuit(nqubits=2)
    with pytest.raises(TypeError):
        c + 5
    with pytest.raises(TypeError):
        5 + c


def test_add_operators_do_not_mutate_operands():
    left = Circuit(nqubits=2)
    x_gate = X(0)
    left.add(x_gate)

    right = Circuit(nqubits=2)
    rz_gate = RZ(1, phi=0.3)
    right.add(rz_gate)

    combined = left + right
    assert combined is not left
    assert combined.gates == [x_gate, rz_gate]
    assert left.gates == [x_gate]
    assert right.gates == [rz_gate]

    prepend_gate = RX(0, theta=0.4)
    prepended = prepend_gate + left
    assert prepended is not left
    assert prepended.gates == [prepend_gate, x_gate]
    assert left.gates == [x_gate]

    appended_gate = X(1)
    assert (left + appended_gate).gates == [x_gate, appended_gate]
    assert left.gates == [x_gate]

    # Adding gates to the result leaves the operands alone.
    combined.add(X(1))
    assert left.gates == [x_gate]
    assert right.gates == [rz_gate]


def test_copy_is_independent():
    original = Circuit(nqubits=2)
    angle = Parameter("angle", 0.5)
    original.add([X(0), RX(0, theta=angle), RX(1, theta=angle)])

    clone = original.copy()
    assert clone == original
    assert clone.get_parameter_names() == original.get_parameter_names()
    assert all(g1 is not g2 for g1, g2 in zip(clone.gates, original.gates))

    # The parameter shared by both RX gates is still shared inside the copy.
    assert clone.nparameters == 1
    clone.set_parameter_values([1.25])
    assert clone.get_parameter_values() == [1.25]
    assert original.get_parameter_values() == [0.5]
    assert angle.value == 0.5
    assert [g.get_parameter_values() for g in clone.gates] == [[], [1.25], [1.25]]

    clone.add(X(1))
    assert len(original.gates) == 3


def test_append_and_prepend_copies_keep_circuits_independent():
    source = Circuit(nqubits=1)
    source.add(RX(0, theta=0.5))

    target = Circuit(nqubits=1)
    target.append(source.copy())
    target.set_parameter_values([1.234])

    assert target.gates[0] is not source.gates[0]
    assert source.get_parameter_values() == [0.5]
    assert target.get_parameter_values() == [1.234]


def test_insert_default_index_appends_in_order():
    c = Circuit(nqubits=2)
    x_gate, x2_gate = X(0), X(1)
    c.add([x_gate, x2_gate])

    rz_gate, rx_gate = RZ(0, phi=0.2), RX(1, theta=0.3)
    c.insert([rz_gate, rx_gate])
    assert c.gates == [x_gate, x2_gate, rz_gate, rx_gate]

    single_gate = X(0)
    c.insert(single_gate)
    assert c.gates == [x_gate, x2_gate, rz_gate, rx_gate, single_gate]


def test_insert_negative_index_keeps_order():
    c = Circuit(nqubits=2)
    x_gate, x2_gate = X(0), X(1)
    c.add([x_gate, x2_gate])

    rz_gate, rx_gate = RZ(0, phi=0.2), RX(1, theta=0.3)
    c.insert([rz_gate, rx_gate], index=-1)
    assert c.gates == [x_gate, rz_gate, rx_gate, x2_gate]

    # Negative indices below the start of the circuit clamp to the beginning.
    first_gate = X(0)
    c.insert(first_gate, index=-100)
    assert c.gates == [first_gate, x_gate, rz_gate, rx_gate, x2_gate]


def test_circuit_draw_runs(monkeypatch):
    c = Circuit(nqubits=2)
    x_gate = X(0)
    c.add(x_gate)

    renderer = MagicMock()
    monkeypatch.setattr(qilisdk.utils.visualization.circuit_renderers, "MatplotlibCircuitRenderer", renderer)
    c.draw(filepath="some/path.png")
    assert renderer.called


def test_to_matrix_single_qubit_sequence():
    c = Circuit(nqubits=1)
    g1 = X(0)
    g2 = RZ(0, phi=np.pi / 3)
    c.add([g1, g2])

    expected = (g2.matrix @ g1.matrix).dense()
    assert np.allclose(c.to_matrix(), expected)
    assert np.allclose(c.to_qtensor().dense(), expected)


def test_to_matrix_multi_qubit_noncontiguous():
    c = Circuit(nqubits=3)
    g1 = CNOT(2, 0)
    g2 = X(1)
    c.add([g1, g2])

    # Expected operator in computational basis ordering |q0 q1 q2>.
    expected = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.complex128,
    )
    np.testing.assert_array_equal(c.to_matrix(), expected)
    np.testing.assert_array_equal(c.to_qtensor().dense(), expected)


def test_to_matrix_measurement_raises():
    c = Circuit(nqubits=1)
    c.add(M(0))
    with pytest.raises(GateHasNoMatrixError):
        c.to_matrix()
    with pytest.raises(GateHasNoMatrixError):
        c.to_qtensor()


def test_prepend_circuit():
    base = Circuit(nqubits=2)
    x_gate = X(0)
    base.add(x_gate)

    prepended = Circuit(nqubits=2)
    rz_gate = RZ(0, phi=0.2)
    x2_gate = X(1)
    prepended.add([rz_gate, x2_gate])
    new = base.__radd__(prepended)  # ruff: ignore[unnecessary-dunder-call]

    assert new.gates == [rz_gate, x2_gate, x_gate]


def test_random_circuit_seedable():
    single_qubit_gates = {X}
    two_qubit_gates = {CNOT}
    nqubits = 3
    ngates = 200

    state_before = random.getstate()
    c1 = Circuit.random(
        nqubits=nqubits,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        ngates=ngates,
        seed=42,
    )

    c2 = Circuit.random(
        nqubits=nqubits,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        ngates=ngates,
        seed=42,
    )

    # The global random state must be left untouched
    assert random.getstate() == state_before

    # Check that both circuits have the same gates in the same order
    assert len(c1.gates) == len(c2.gates)
    for gate1, gate2 in zip(c1.gates, c2.gates):
        assert isinstance(gate1, Gate)
        assert isinstance(gate2, Gate)
        assert gate1.name == gate2.name
        assert gate1.qubits == gate2.qubits
        assert gate1.get_parameter_values() == gate2.get_parameter_values()


def test_random_circuit_different_seeds_differ():
    kwargs = {"nqubits": 3, "single_qubit_gates": {RX}, "two_qubit_gates": {CNOT}, "ngates": 20}
    c1 = Circuit.random(**kwargs, seed=1)
    c2 = Circuit.random(**kwargs, seed=2)
    assert c1.get_parameter_values() != c2.get_parameter_values()


def test_random_circuit_parameters_are_assignable():
    c = Circuit.random(nqubits=2, single_qubit_gates={RX, U3}, two_qubit_gates={CNOT}, ngates=10, seed=7)
    assert c.nparameters > 0

    # Labels must not embed the random values, and must be unique
    names = c.get_parameter_names()
    assert len(set(names)) == len(names)
    for name in names:
        assert "." not in name

    # Parameters are unbounded, so any real value can be assigned
    values = [0.1 * i for i in range(c.nparameters)]
    c.set_parameter_values(values)
    assert c.get_parameter_values() == values


def test_repr():
    c = Circuit(nqubits=2)
    x_gate = X(0)
    c.add(x_gate)
    repr_str = str(c)
    assert "Circuit" in repr_str
    assert "nqubits=2" in repr_str
    assert "gates=[X(0)]" in repr_str


def test_circuit_set_bad_param_raises(monkeypatch):
    c = Circuit(nqubits=1)
    monkeypatch.setattr(c, "check_constraints", lambda params: True)  # Bypass constraint checking for this test
    with pytest.raises(ValueError, match="is not defined in this circuit"):
        c.set_parameters({"bad_param": 0.5})


def test_circuit_no_equality_different_qubits():
    c1 = Circuit(nqubits=2)
    c2 = Circuit(nqubits=3)
    assert c1 != c2


def test_circuit_no_equality_different_gate_count():
    c1 = Circuit(nqubits=2)
    c2 = Circuit(nqubits=2)
    c1.add(X(0))
    assert c1 != c2


# --- Parameter Cache Invalidation Tests ---


def test_circuit_set_parameters_clears_gate_matrix_cache():
    """Circuit.set_parameters must invalidate each gate's cached_property matrix.

    Gate.matrix is a @cached_property. If set_parameters only updates the
    Parameter value without calling _invalidate_matrix, the stale matrix is
    returned on the next access.
    """
    # Use a name that differs from the gate's slot name ("theta") so the
    # circuit registers the parameter under the user-supplied label "angle".
    gate = RX(0, theta=Parameter("angle", np.pi / 4))
    c = Circuit(nqubits=1)
    c.add(gate)

    # Access matrix to populate the cached_property.
    matrix_before = gate.matrix.dense()

    c.set_parameters({"angle": np.pi / 2})

    # Matrix must reflect the new angle, not the cached one.
    assert not np.allclose(matrix_before, gate.matrix.dense())


def test_circuit_set_parameter_values_clears_gate_matrix_cache():
    """Circuit.set_parameter_values must invalidate each gate's cached matrix."""
    gate = RX(0, theta=Parameter("angle", np.pi / 4))
    c = Circuit(nqubits=1)
    c.add(gate)

    matrix_before = gate.matrix.dense()

    c.set_parameter_values([np.pi / 2])

    assert not np.allclose(matrix_before, gate.matrix.dense())


# --- Shared Parameter Tests ---
# Regression coverage for a bug where _parameters_link stored a single
# (label, gate) tuple per parameter, so when the same Parameter object was
# shared across multiple gates only the last-added gate received updates from
# Circuit.set_parameters / set_parameter_bounds. Earlier gates kept stale
# matrices because their _invalidate_matrix was never called.


def test_shared_parameter_links_all_gates():
    """All gates sharing a Parameter must be registered in _parameters_link."""
    shared = Parameter("shared", math.pi)
    c = Circuit(nqubits=2)
    g0 = U1(0, phi=shared)
    g1 = U2(1, phi=shared, gamma=0.1)
    c.add(g0)
    c.add(g1)

    link = c._parameters_link["shared"]
    assert isinstance(link, list)
    linked_gates = [gate for _, gate in link]
    assert g0 in linked_gates
    assert g1 in linked_gates
    assert all(label == "phi" for label, _ in link)


def test_circuit_set_parameters_updates_all_shared_gates():
    """set_parameters must update every gate sharing a Parameter object."""
    shared = Parameter("shared", math.pi)
    c = Circuit(nqubits=2)
    g0 = U1(0, phi=shared)
    g1 = U2(1, phi=shared, gamma=0.1)
    c.add(g0)
    c.add(g1)

    # Prime the cached_property matrix on both gates.
    _ = g0.matrix
    _ = g1.matrix

    c.set_parameters({"shared": 0.0})

    # U1(phi=0) is the identity on its slot.
    np.testing.assert_allclose(g0.matrix.dense(), np.eye(2))
    # U2(phi=0, gamma=0.1) has [1, 0] = 1/sqrt(2) (independent of gamma).
    assert np.isclose(g1.matrix[1, 0], 1 / np.sqrt(2))


def test_circuit_set_parameter_bounds_updates_all_shared_gates():
    """set_parameter_bounds must propagate to every gate sharing a Parameter."""
    shared = Parameter("shared", 0.5, bounds=(0.0, 1.0))
    c = Circuit(nqubits=2)
    g0 = U1(0, phi=shared)
    g1 = U2(1, phi=shared, gamma=0.1)
    c.add(g0)
    c.add(g1)

    c.set_parameter_bounds({"shared": (0.1, 0.9)})

    assert g0.get_parameter_bounds()["phi"] == (0.1, 0.9)
    assert g1.get_parameter_bounds()["phi"] == (0.1, 0.9)


def test_circuit_set_prefix_then_set_parameters_updates_all_shared_gates():
    """After set_prefix the shared link must still drive every gate.

    Previously set_prefix renamed the link key but the value remained a single
    (label, gate) tuple, so the prefix path lost track of all but one gate.
    """
    shared = Parameter("shared", math.pi)
    c = Circuit(nqubits=2)
    g0 = U1(0, phi=shared)
    g1 = U2(1, phi=shared, gamma=0.1)
    c.add(g0)
    c.add(g1)

    c.set_prefix("prefix_")
    assert "prefix_shared" in c._parameters_link
    assert isinstance(c._parameters_link["prefix_shared"], list)
    assert len(c._parameters_link["prefix_shared"]) == 2

    _ = g0.matrix
    _ = g1.matrix
    c.set_parameters({"prefix_shared": 5.0})

    expected = np.exp(1j * 5.0)
    assert np.isclose(g0.matrix[1, 1], expected)
    assert np.isclose(g1.matrix[1, 0], expected / np.sqrt(2))


@pytest.mark.parametrize("bad_qubit", [-1, 2, 5])
def test_circuit_rejects_out_of_range_qubit(bad_qubit):
    """QSDK-05: the Python layer must reject both negative and too-large qubit
    indices (the negative case was previously unguarded)."""
    circuit = Circuit(2)
    gate = X(bad_qubit)
    with pytest.raises(QubitOutOfRangeError):
        circuit.add(gate)


def test_circuit_add_analog_hamiltonian_raises_helpful_error():
    """Adding an analog Hamiltonian to a circuit points at the analog/digital import mixup."""
    circuit = Circuit(1)
    hamiltonian = analog_X(0)
    with pytest.raises(NotAGateError, match="did you mean to import X, Y, Z or I from qilisdk"):
        circuit.add(hamiltonian)


def test_circuit_add_analog_pauli_operator_raises_helpful_error():
    """Adding a bare analog Pauli operator to a circuit gives the same guidance."""
    circuit = Circuit(1)
    pauli = PauliX(0)
    with pytest.raises(NotAGateError, match="did you mean to import X, Y, Z or I from qilisdk"):
        circuit.add(pauli)


def test_circuit_insert_analog_hamiltonian_raises_helpful_error():
    """Inserting an analog Hamiltonian into a circuit points at the analog/digital import mixup."""
    circuit = Circuit(1)
    hamiltonian = analog_X(0)
    with pytest.raises(NotAGateError, match="did you mean to import X, Y, Z or I from qilisdk"):
        circuit.insert(hamiltonian, 0)


def test_circuit_add_operator_with_analog_hamiltonian_raises_helpful_error():
    """The ``+`` operator rejects analog Hamiltonians with the same guidance."""
    circuit = Circuit(1)
    hamiltonian = analog_X(0)
    with pytest.raises(NotAGateError, match="did you mean to import X, Y, Z or I from qilisdk"):
        circuit += hamiltonian


def test_gate_plus_gate_is_not_supported():
    """Gates do not support arithmetic, so adding two of them raises the usual TypeError."""
    first, second = X(0), Y(0)
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = first + second


def test_gate_plus_analog_hamiltonian_raises_helpful_error():
    """Adding an analog Hamiltonian to a digital gate points at the import mixup."""
    gate = X(0)
    hamiltonian = analog_Y(0)
    with pytest.raises(NotAGateError, match="did you mean to import X, Y, Z or I from qilisdk"):
        _ = gate + hamiltonian


def test_gate_still_supports_addition_with_a_circuit():
    """Rejecting gate arithmetic does not break prepending a gate to a circuit."""
    circuit = Circuit(1)
    circuit.add(X(0))
    result = Y(0) + circuit
    assert result.gates == [Y(0), X(0)]


def test_gate_minus_gate_is_not_supported():
    """Subtracting one gate from another raises the usual TypeError."""
    first, second = X(0), Y(0)
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = first - second


def test_gate_minus_analog_hamiltonian_raises_helpful_error():
    """Subtracting an analog Hamiltonian from a digital gate points at the import mixup."""
    gate = X(0)
    hamiltonian = analog_Y(0)
    with pytest.raises(NotAGateError, match="did you mean to import X, Y, Z or I from qilisdk"):
        _ = gate - hamiltonian


def _adjoint_test_circuit() -> Circuit:
    circuit = Circuit(3)
    circuit.add(
        [
            H(0),
            RX(1, theta=Parameter("angle", 0.3)),
            CNOT(0, 1),
            T(2),
            U3(2, theta=0.1, phi=0.2, gamma=0.3),
            Controlled(0, basic_gate=RY(2, theta=0.4)),
            Adjoint(S(1)),
            Controlled(0, 1, basic_gate=X(2)),
            CZ(1, 2),
            RZ(0, phi=0.5),
            SWAP(1, 2),
        ]
    )
    return circuit


def test_circuit_adjoint_is_inverse():
    circuit = _adjoint_test_circuit()
    adjoint = circuit.adjoint()
    assert adjoint.nqubits == circuit.nqubits
    np.testing.assert_allclose(adjoint.to_matrix(), circuit.to_matrix().conj().T, atol=1e-12)
    np.testing.assert_allclose(adjoint.to_matrix() @ circuit.to_matrix(), np.eye(8), atol=1e-12)


def test_circuit_adjoint_gate_structure():
    adjoint = _adjoint_test_circuit().adjoint()
    assert [gate.name for gate in adjoint.gates] == [
        "SWAP",
        "RZ†",
        "CZ",
        "CCX",
        "S",
        "CRY†",
        "U3†",
        "T†",
        "CNOT",
        "RX†",
        "H",
    ]
    assert [gate.qubits for gate in adjoint.gates][:4] == [(1, 2), (0,), (1, 2), (0, 1, 2)]


def test_circuit_adjoint_of_adjoint_is_original():
    circuit = _adjoint_test_circuit()
    assert circuit.adjoint().adjoint() == circuit


def test_circuit_adjoint_of_empty_circuit():
    adjoint = Circuit(2).adjoint()
    assert adjoint.nqubits == 2
    assert adjoint.gates == []


def test_circuit_adjoint_of_exponential_gate_is_conjugate_transpose():
    circuit = Circuit(2)
    circuit.add([Exponential(RX(0, theta=0.7)), CNOT(0, 1)])
    np.testing.assert_allclose(circuit.adjoint().to_matrix(), circuit.to_matrix().conj().T, atol=1e-12)


def test_circuit_adjoint_is_independent_of_original():
    circuit = _adjoint_test_circuit()
    adjoint = circuit.adjoint()
    adjoint.set_parameters({"angle": 1.2})
    assert circuit.get_parameters()["angle"] == pytest.approx(0.3)
    assert adjoint.get_parameters()["angle"] == pytest.approx(1.2)


def test_circuit_adjoint_keeps_parameter_values_and_bounds():
    circuit = Circuit(1)
    circuit.add(RX(0, theta=Parameter("angle", 0.3, bounds=(0.0, 1.0))))
    adjoint = circuit.adjoint()
    assert adjoint.get_parameters() == {"angle": pytest.approx(0.3)}
    assert adjoint.get_parameter_bounds() == {"angle": (0.0, 1.0)}


def test_circuit_plus_adjoint_shares_named_parameters():
    circuit = _adjoint_test_circuit()
    combined = circuit + circuit.adjoint()
    combined.set_parameters({"angle": 1.1})
    np.testing.assert_allclose(combined.to_matrix(), np.eye(8), atol=1e-12)


def test_circuit_adjoint_with_measurement_raises():
    circuit = Circuit(1)
    circuit.add([H(0), M(0)])
    with pytest.raises(UnsupportedGateError, match="measurements"):
        circuit.adjoint()


def _assert_is_adjoint(adjoint: Circuit, circuit: Circuit) -> None:
    np.testing.assert_allclose(adjoint.to_matrix(), circuit.to_matrix().conj().T, atol=1e-12)


@pytest.mark.parametrize(
    "gate",
    [
        I(0),
        X(0),
        Y(0),
        Z(0),
        H(0),
        S(0),
        T(0),
        RX(0, theta=0.7),
        RY(0, theta=-1.3),
        RZ(0, phi=2.1),
        U1(0, phi=0.4),
        U2(0, phi=0.5, gamma=-0.6),
        U3(0, theta=0.1, phi=0.2, gamma=0.3),
        CNOT(0, 1),
        CNOT(1, 0),
        CZ(0, 1),
        SWAP(0, 1),
    ],
    ids=lambda gate: gate.name,
)
def test_circuit_adjoint_of_single_gate(gate):
    circuit = Circuit(2)
    circuit.add(gate)
    adjoint = circuit.adjoint()
    assert len(adjoint.gates) == 1
    assert adjoint.gates[0].qubits == gate.qubits
    _assert_is_adjoint(adjoint, circuit)


@pytest.mark.parametrize("gate", [I(0), X(0), Y(0), Z(0), H(0), SWAP(0, 1), CNOT(0, 1), CZ(1, 0)], ids=str)
def test_circuit_adjoint_keeps_self_adjoint_gates_unwrapped(gate):
    circuit = Circuit(2)
    circuit.add(gate)
    adjoint_gate = circuit.adjoint().gates[0]
    assert type(adjoint_gate) is type(gate)
    assert adjoint_gate == gate


def test_circuit_adjoint_keeps_multi_controlled_self_adjoint_gates_unwrapped():
    circuit = Circuit(4)
    circuit.add(
        [Controlled(0, 1, 2, basic_gate=X(3)), Controlled(0, basic_gate=SWAP(1, 2)), Controlled(3, basic_gate=H(0))]
    )
    adjoint = circuit.adjoint()
    assert all(isinstance(gate, Controlled) for gate in adjoint.gates)
    assert [gate.control_qubits for gate in adjoint.gates] == [(3,), (0,), (0, 1, 2)]
    _assert_is_adjoint(adjoint, circuit)


@pytest.mark.parametrize("gate", [S(1), T(1), RX(1, theta=0.3), U3(1, theta=0.1, phi=0.2, gamma=0.3)], ids=str)
def test_circuit_adjoint_wraps_controlled_non_self_adjoint_gates(gate):
    circuit = Circuit(3)
    circuit.add([Controlled(0, basic_gate=gate), Controlled(2, 0, basic_gate=deepcopy(gate))])
    adjoint = circuit.adjoint()
    assert all(isinstance(g, Adjoint) and isinstance(g.basic_gate, Controlled) for g in adjoint.gates)
    assert [g.control_qubits for g in adjoint.gates] == [(2, 0), (0,)]
    _assert_is_adjoint(adjoint, circuit)


def test_circuit_adjoint_unwraps_adjoint_gates():
    circuit = Circuit(3)
    circuit.add(
        [
            Adjoint(S(0)),
            Adjoint(X(1)),
            Adjoint(CNOT(0, 2)),  # ty:ignore[invalid-argument-type]
            Adjoint(Controlled(0, basic_gate=T(1))),  # ty:ignore[invalid-argument-type]
        ]
    )
    adjoint = circuit.adjoint()
    assert [type(gate) for gate in adjoint.gates] == [Controlled, CNOT, X, S]
    _assert_is_adjoint(adjoint, circuit)


def test_circuit_adjoint_wraps_exponential_of_self_adjoint_gate():
    circuit = Circuit(1)
    circuit.add(Exponential(Y(0)))
    adjoint = circuit.adjoint()
    assert isinstance(adjoint.gates[0], Adjoint)
    assert isinstance(adjoint.gates[0].basic_gate, Exponential)
    _assert_is_adjoint(adjoint, circuit)


@pytest.mark.parametrize("seed", range(5))
def test_circuit_adjoint_of_random_circuit_is_inverse(seed):
    circuit = Circuit.random(
        nqubits=3,
        single_qubit_gates={I, X, Y, Z, H, S, T, RX, RY, RZ, U1, U2, U3},
        two_qubit_gates={CNOT, CZ, SWAP},  # ty:ignore[invalid-argument-type]
        ngates=40,
        seed=seed,
    )
    adjoint = circuit.adjoint()
    assert len(adjoint.gates) == len(circuit.gates)
    assert [gate.qubits for gate in adjoint.gates] == [gate.qubits for gate in reversed(circuit.gates)]
    np.testing.assert_allclose(adjoint.to_matrix() @ circuit.to_matrix(), np.eye(8), atol=1e-10)
    np.testing.assert_allclose(circuit.to_matrix() @ adjoint.to_matrix(), np.eye(8), atol=1e-10)


def test_circuit_adjoint_keeps_qubit_count_with_idle_qubits():
    circuit = Circuit(5)
    circuit.add([S(4), CNOT(4, 1)])
    adjoint = circuit.adjoint()
    assert adjoint.nqubits == 5
    assert [gate.qubits for gate in adjoint.gates] == [(4, 1), (4,)]
    _assert_is_adjoint(adjoint, circuit)


def test_circuit_adjoint_of_empty_circuit_with_zero_qubits():
    adjoint = Circuit(0).adjoint()
    assert adjoint.nqubits == 0
    assert adjoint.gates == []


def test_circuit_adjoint_does_not_modify_original():
    circuit = _adjoint_test_circuit()
    reference = circuit.copy()
    original_gates = list(circuit.gates)
    circuit.adjoint()
    assert circuit == reference
    assert all(a is b for a, b in zip(circuit.gates, original_gates, strict=True))
    assert circuit.get_parameters() == reference.get_parameters()


def test_circuit_adjoint_shares_no_gate_objects_with_original():
    circuit = _adjoint_test_circuit()
    adjoint = circuit.adjoint()
    original_ids = {id(gate) for gate in circuit.gates}
    for gate in adjoint.gates:
        assert id(gate) not in original_ids
        if isinstance(gate, Adjoint):
            assert id(gate.basic_gate) not in original_ids


def test_circuit_adjoint_is_unaffected_by_later_changes_to_original():
    circuit = _adjoint_test_circuit()
    adjoint = circuit.adjoint()
    expected = adjoint.to_matrix()
    circuit.add(T(0))
    circuit.set_parameters({"angle": 2.0})
    assert len(adjoint.gates) == 11
    np.testing.assert_allclose(adjoint.to_matrix(), expected, atol=1e-12)


def test_circuit_adjoint_keeps_shared_parameters_shared():
    angle = Parameter("angle", 0.2)
    circuit = Circuit(2)
    circuit.add([RX(0, theta=angle), RY(1, theta=angle)])
    adjoint = circuit.adjoint()
    assert adjoint.get_parameters() == {"angle": pytest.approx(0.2)}
    adjoint.set_parameters({"angle": 0.9})
    assert [gate.get_parameter_values() for gate in adjoint.gates] == [[0.9], [0.9]]
    circuit.set_parameters({"angle": 0.9})
    _assert_is_adjoint(adjoint, circuit)


def test_circuit_adjoint_matrix_follows_parameter_updates():
    circuit = Circuit(2)
    circuit.add([RX(0, theta=Parameter("a", 0.1)), Controlled(0, basic_gate=RZ(1, phi=Parameter("b", 0.2)))])
    adjoint = circuit.adjoint()
    adjoint.to_matrix()
    adjoint.set_parameters({"a": 1.4, "b": -0.8})
    circuit.set_parameters({"a": 1.4, "b": -0.8})
    _assert_is_adjoint(adjoint, circuit)


def test_circuit_adjoint_keeps_constraints():
    left = Parameter("left", 0.1)
    right = Parameter("right", 0.5)
    circuit = Circuit(1)
    circuit.add([RX(0, theta=left), RY(0, theta=right)])
    circuit.add_parameter_constraint(LEQ(left, right))
    adjoint = circuit.adjoint()
    assert len(adjoint.get_constraints()) == 1
    with pytest.raises(ValueError, match="breaks the parameter constraints"):
        adjoint.set_parameters({"left": 0.9})
    adjoint.set_parameters({"left": 0.4})
    assert adjoint.get_parameters()["left"] == pytest.approx(0.4)
    assert circuit.get_parameters()["left"] == pytest.approx(0.1)


def test_circuit_adjoint_keeps_prefix():
    circuit = Circuit(1)
    circuit.add([RX(0, theta=Parameter("a", 0.1)), RY(0, theta=Parameter("b", 0.2))])
    circuit.set_prefix("p_")
    adjoint = circuit.adjoint()
    assert adjoint.get_prefix() == "p_"
    assert adjoint.get_parameters() == {"p_b": pytest.approx(0.2), "p_a": pytest.approx(0.1)}
    adjoint.set_parameters({"p_a": 0.7})
    assert adjoint.gates[1].get_parameter_values() == [0.7]


def test_circuit_adjoint_with_measurement_in_middle_raises_and_leaves_original_intact():
    circuit = Circuit(2)
    circuit.add([H(0), M(1), S(0)])
    reference = circuit.copy()
    with pytest.raises(UnsupportedGateError):
        circuit.adjoint()
    assert circuit == reference


def test_circuit_adjoint_of_adjoint_has_same_hash():
    circuit = _adjoint_test_circuit()
    assert hash(circuit.adjoint().adjoint()) == hash(circuit)


def test_circuit_adjoint_serialization_roundtrip():
    adjoint = _adjoint_test_circuit().adjoint()
    restored = deserialize(serialize(adjoint), Circuit)
    assert [gate.name for gate in restored.gates] == [gate.name for gate in adjoint.gates]
    assert restored.get_parameters() == adjoint.get_parameters()
    np.testing.assert_allclose(restored.to_matrix(), adjoint.to_matrix(), atol=1e-12)
