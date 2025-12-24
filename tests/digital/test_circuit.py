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

import random

import numpy as np
import pytest

from qilisdk.core import Parameter
from qilisdk.digital import CNOT, RX, RZ, Circuit, X
from qilisdk.digital.exceptions import ParametersNotEqualError, QubitOutOfRangeError


def test_circuit_initialization():
    """
    Test basic initialization of the Circuit class.
    """
    c = Circuit(nqubits=3)
    assert c.nqubits == 3
    assert c.nparameters == 0  # No gates yet
    assert c.get_parameter_values() == []
    assert c.gates == []


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
    with pytest.raises(ParametersNotEqualError):
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
        for label, parameter in gate.parameters.items()
    )

    # Change circuit's parameter value
    c.set_parameter_values([1.0])

    # Check that Parameter object has changed its value
    assert angle.value == 1.0

    # Check that circuit's parameter has the correct label and value
    assert c.get_parameters() == {angle.label: angle.value}

    # Check that gates' parameters have the correct label and value
    assert all(
        label in gate.PARAMETER_NAMES and parameter.label == angle.label and parameter.value == angle.value
        for gate in c.gates
        for label, parameter in gate.parameters.items()
    )


def test_randomize_circuit():
    """
    Test the randomize method to ensure it adds the correct number of gates
    and only uses the provided gate sets.
    """
    single_qubit_gates = {X, RX}
    two_qubit_gates = {CNOT}
    nqubits = 3
    ngates = 10

    c = Circuit(nqubits=nqubits)
    random.seed(42)  # Set seed for reproducibility
    c.randomize(single_qubit_gates=single_qubit_gates, two_qubit_gates=two_qubit_gates, ngates=ngates)

    # Check that the circuit has the correct number of gates
    assert len(c.gates) == ngates

    # Check that all gates are from the provided sets
    for gate in c.gates:
        if gate.nqubits == 1:
            assert type(gate) in single_qubit_gates
        elif gate.nqubits == 2:
            assert type(gate) in two_qubit_gates
        else:
            pytest.fail("Gate with invalid number of qubits added to circuit.")

    # Check that all target qubits are within range
    for gate in c.gates:
        for qubit in gate.qubits:
            assert 0 <= qubit < nqubits
