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
from unittest.mock import MagicMock

import numpy as np
import pytest
from loguru_caplog import loguru_caplog as caplog  # ruff: ignore[unused-import]

import qilisdk.utils.visualization.circuit_renderers
from qilisdk.analog import X as analog_X
from qilisdk.analog import Y as analog_Y
from qilisdk.analog.hamiltonian import PauliX
from qilisdk.core import Parameter, QTensor
from qilisdk.core.comparison import LEQ
from qilisdk.digital import CNOT, RX, RY, RZ, U1, U2, U3, Circuit, M, S, X, Y
from qilisdk.digital.circuit import _apply_gate_left, _uniformly_controlled_rotation
from qilisdk.digital.exceptions import GateHasNoMatrixError, NotAGateError, QubitOutOfRangeError
from qilisdk.digital.gates import Adjoint, BasicGate, Exponential, Gate


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


def test_adjoint_gates_get_distinct_parameter_labels():
    a = U3(0, theta=0.1, phi=0.2, gamma=0.3)
    b = U3(1, theta=1.1, phi=1.2, gamma=1.3)
    c = Circuit(2)
    c.add([a.adjoint(), b.adjoint()])

    assert c.get_parameters() == {
        "U3†(0)_theta_0": 0.1,
        "U3†(0)_phi_1": 0.2,
        "U3†(0)_gamma_2": 0.3,
        "U3†(1)_theta_3": 1.1,
        "U3†(1)_phi_4": 1.2,
        "U3†(1)_gamma_5": 1.3,
    }

    before = c.to_matrix()
    c.set_parameter_values(c.get_parameter_values())
    assert a.theta == 0.1
    assert np.allclose(c.to_matrix(), before)


def test_controlled_gates_get_distinct_parameter_labels():
    c = Circuit(3)
    c.add([RX(0, theta=0.1).controlled(2), RX(1, theta=0.9).controlled(2)])
    assert c.get_parameters() == {"CRX(2,0)_theta_0": 0.1, "CRX(2,1)_theta_1": 0.9}


def test_nested_modified_gates_get_distinct_parameter_labels():
    c = Circuit(3)
    c.add([Adjoint(RX(0, theta=0.1).controlled(2)), Adjoint(RX(1, theta=0.9).controlled(2))])
    assert c.get_parameters() == {"CRX†(2,0)_theta_0": 0.1, "CRX†(2,1)_theta_1": 0.9}


def test_exponential_gates_get_distinct_parameter_labels():
    c = Circuit(2)
    c.add([Exponential(RX(0, theta=0.1)), Exponential(RX(1, theta=0.2))])
    assert c.get_parameters() == {"e^RX(0)_theta_0": 0.1, "e^RX(1)_theta_1": 0.2}


def test_inserted_modified_gates_get_distinct_parameter_labels():
    c = Circuit(2)
    c.add(RX(0, theta=0.1).adjoint())
    c.insert(RX(1, theta=0.2).adjoint(), 0)
    assert c.get_parameters() == {"RX†(0)_theta_0": 0.1, "RX†(1)_theta_1": 0.2}


def test_set_parameters_on_modified_gate_label_only_updates_that_gate():
    a = RX(0, theta=0.1)
    b = RX(1, theta=0.9)
    c = Circuit(3)
    c.add([a.controlled(2), b.controlled(2)])

    c.set_parameters({"CRX(2,0)_theta_0": 0.5})

    assert a.theta == 0.5
    assert b.theta == 0.9
    assert c.get_parameters() == {"CRX(2,0)_theta_0": 0.5, "CRX(2,1)_theta_1": 0.9}


def test_user_named_parameter_shared_across_modified_gates():
    shared = Parameter("shared", 0.3)
    a = RX(0, theta=shared)
    b = RX(1, theta=shared)
    c = Circuit(2)
    c.add([a.adjoint(), b.adjoint()])

    assert c.get_parameters() == {"shared": 0.3}

    c.set_parameters({"shared": 0.7})
    assert a.theta == 0.7
    assert b.theta == 0.7


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


def _assert_prepares(circuit: Circuit, target: np.ndarray) -> None:
    """The circuit maps |0...0> to ``target`` up to a global phase."""
    target = target / np.linalg.norm(target)
    assert np.isclose(abs(np.vdot(target, circuit.to_matrix()[:, 0])), 1.0)


@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5])
def test_from_state_prepares_random_state(nqubits):
    state = QTensor.random(nqubits, seed=nqubits)
    circuit = Circuit.from_state(state)
    assert circuit.nqubits == nqubits
    _assert_prepares(circuit, state.dense().ravel())


@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5])
def test_from_state_uses_only_rotations_and_at_most_the_mottonen_cnot_count(nqubits):
    circuit = Circuit.from_state(QTensor.random(nqubits, seed=7))
    assert all(isinstance(gate, (RY, RZ, CNOT)) for gate in circuit.gates)
    assert sum(isinstance(gate, CNOT) for gate in circuit.gates) <= max(0, 2 ** (nqubits + 1) - 4)


def test_from_state_prepares_sparse_complex_states():
    rng = np.random.default_rng(1)
    for _ in range(200):
        nqubits = int(rng.integers(1, 5))
        dim = 1 << nqubits
        amplitudes = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        amplitudes[rng.random(dim) < rng.random()] = 0
        amplitudes[rng.integers(dim)] = 1j
        _assert_prepares(Circuit.from_state(QTensor(amplitudes.reshape(-1, 1))), amplitudes)


@pytest.mark.parametrize("bits", [(0,), (1,), (0, 0), (1, 0, 1), (1, 1, 0, 1)])
def test_from_state_prepares_basis_state_without_entangling_gates(bits):
    circuit = Circuit.from_state(QTensor.ket(*bits))
    assert [gate.qubits for gate in circuit.gates] == [(q,) for q, bit in enumerate(bits) if bit]
    assert all(isinstance(gate, RY) for gate in circuit.gates)
    _assert_prepares(circuit, QTensor.ket(*bits).dense().ravel())


def test_from_state_prepares_uniform_state_without_entangling_gates():
    circuit = Circuit.from_state(QTensor.uniform(3))
    assert len(circuit.gates) == 3
    assert all(isinstance(gate, RY) for gate in circuit.gates)
    _assert_prepares(circuit, QTensor.uniform(3).dense().ravel())


@pytest.mark.parametrize("state", [QTensor.ghz(3), QTensor.w(3)])
def test_from_state_real_amplitudes_need_no_phase_rotations(state):
    circuit = Circuit.from_state(state)
    assert not any(isinstance(gate, RZ) for gate in circuit.gates)
    _assert_prepares(circuit, state.dense().ravel())


@pytest.mark.parametrize("global_phase", [1.0, 1j, np.exp(0.3j)])
def test_from_state_signed_real_states_need_no_phase_rotations(global_phase):
    rng = np.random.default_rng(3)
    for _ in range(100):
        nqubits = int(rng.integers(1, 7))
        dim = 1 << nqubits
        amplitudes = rng.normal(size=dim)
        amplitudes[rng.random(dim) < rng.random()] = 0
        amplitudes[rng.integers(dim)] = -1
        amplitudes = global_phase * amplitudes
        circuit = Circuit.from_state(QTensor(amplitudes.reshape(-1, 1)))
        assert all(isinstance(gate, (RY, CNOT)) for gate in circuit.gates)
        assert sum(isinstance(gate, CNOT) for gate in circuit.gates) <= max(0, 2**nqubits - 2)
        _assert_prepares(circuit, amplitudes)


def test_from_state_prepares_relative_phase():
    amplitudes = np.array([1, 1j, -1, -1j]) / 2
    circuit = Circuit.from_state(QTensor(amplitudes.reshape(-1, 1)))
    _assert_prepares(circuit, amplitudes)


def test_from_state_normalizes_with_warning(caplog):  # ruff: ignore[redefined-while-unused]
    amplitudes = np.array([3.0, 4.0j])
    circuit = Circuit.from_state(QTensor(amplitudes.reshape(-1, 1)))
    assert "normalizing it" in caplog.text
    _assert_prepares(circuit, amplitudes)


def test_from_state_normalized_state_does_not_warn(caplog):  # ruff: ignore[redefined-while-unused]
    Circuit.from_state(QTensor.ket(0, 1))
    assert "normalizing it" not in caplog.text


@pytest.mark.parametrize("state", [QTensor.bra(0, 1), QTensor.identity(1)])
def test_from_state_rejects_non_kets(state):
    with pytest.raises(ValueError, match="expects a ket"):
        Circuit.from_state(state)


def test_from_state_rejects_zero_vector():
    zero = QTensor(np.zeros((4, 1)))
    with pytest.raises(ValueError, match="zero vector"):
        Circuit.from_state(zero)


def _uniformly_controlled_matrix(gate_class, angles):
    """Block-diagonal matrix applying ``gate_class(angles[j])`` to the last qubit when the others are in state ``j``."""
    blocks = [gate_class(0, **{gate_class.PARAMETER_NAMES[0]: float(angle)}).matrix.dense() for angle in angles]
    size = 2 * len(blocks)
    matrix = np.zeros((size, size), dtype=complex)
    for j, block in enumerate(blocks):
        matrix[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = block
    return matrix


@pytest.mark.parametrize("gate_class", [RY, RZ])
@pytest.mark.parametrize("target", [1, 2, 3, 4])
def test_uniformly_controlled_rotation_matches_block_diagonal(gate_class, target):
    rng = np.random.default_rng(target)
    angles = rng.uniform(-np.pi, np.pi, size=1 << target)
    circuit = Circuit(target + 1)
    circuit.add(_uniformly_controlled_rotation(gate_class, target, angles, np.ones(1 << target, dtype=bool), 1e-10))
    assert np.allclose(circuit.to_matrix(), _uniformly_controlled_matrix(gate_class, angles))
    assert sum(isinstance(gate, CNOT) for gate in circuit.gates) == 1 << target


@pytest.mark.parametrize("target", [1, 2, 3])
def test_uniformly_controlled_rotation_ignores_irrelevant_angles(target):
    rng = np.random.default_rng(target)
    angles = rng.uniform(-np.pi, np.pi, size=1 << target)
    relevant = rng.random(1 << target) < 0.5
    relevant[0] = True
    circuit = Circuit(target + 1)
    circuit.add(_uniformly_controlled_rotation(RY, target, angles, relevant, 1e-10))
    blocks = np.repeat(relevant, 2)
    expected = _uniformly_controlled_matrix(RY, angles)
    assert np.allclose(circuit.to_matrix()[np.ix_(blocks, blocks)], expected[np.ix_(blocks, blocks)])


def test_uniformly_controlled_rotation_with_equal_angles_is_a_single_rotation():
    gates = _uniformly_controlled_rotation(RY, 3, np.full(8, 0.7), np.ones(8, dtype=bool), 1e-10)
    assert len(gates) == 1
    assert isinstance(gates[0], RY)
    assert gates[0].qubits == (3,)
    assert np.isclose(gates[0].theta, 0.7)


def test_from_state_prepares_larger_random_state():
    state = QTensor.random(8, seed=3)
    _assert_prepares(Circuit.from_state(state), state.dense().ravel())


def test_from_state_product_state_needs_no_entangling_gates():
    rng = np.random.default_rng(5)
    amplitudes = np.ones(1)
    for _ in range(4):
        single = rng.normal(size=2) + 1j * rng.normal(size=2)
        amplitudes = np.kron(amplitudes, single / np.linalg.norm(single))
    circuit = Circuit.from_state(QTensor(amplitudes.reshape(-1, 1)))
    assert not any(isinstance(gate, CNOT) for gate in circuit.gates)
    _assert_prepares(circuit, amplitudes)


def test_from_state_ignores_global_phase():
    state = QTensor.random(3, seed=11)
    shifted = QTensor(np.exp(0.9j) * state.dense())
    original_gates = Circuit.from_state(state).gates
    shifted_gates = Circuit.from_state(shifted).gates
    assert [(type(gate), gate.qubits) for gate in original_gates] == [
        (type(gate), gate.qubits) for gate in shifted_gates
    ]
    assert np.allclose(
        [value for gate in original_gates for value in gate.get_parameter_values()],
        [value for gate in shifted_gates for value in gate.get_parameter_values()],
    )


def test_from_state_basis_state_with_phase_needs_no_phase_rotations():
    amplitudes = np.zeros(8, dtype=complex)
    amplitudes[5] = np.exp(1.3j)
    circuit = Circuit.from_state(QTensor(amplitudes.reshape(-1, 1)))
    assert all(isinstance(gate, RY) for gate in circuit.gates)
    _assert_prepares(circuit, amplitudes)


def test_from_state_drops_amplitudes_below_tolerance():
    amplitudes = np.array([1, 0, 0, 1e-13j]) / np.sqrt(1 + 1e-26)
    circuit = Circuit.from_state(QTensor(amplitudes.reshape(-1, 1)))
    assert circuit.gates == []
    _assert_prepares(circuit, amplitudes)


def test_from_state_does_not_modify_input(caplog):  # ruff: ignore[redefined-while-unused]
    state = QTensor(np.array([[3.0], [4.0]]))
    Circuit.from_state(state)
    assert np.allclose(state.dense().ravel(), [3.0, 4.0])


def test_from_state_omits_negligible_rotations():
    circuit = Circuit.from_state(QTensor.random(5, seed=2))
    assert all(abs(gate.get_parameter_values()[0]) >= 1e-10 for gate in circuit.gates if not isinstance(gate, CNOT))
