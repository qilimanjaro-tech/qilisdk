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

from qilisdk.digital import RX, RZ, Circuit, X
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
    assert x_gate not in c._parameterized_gates

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
    assert rx_gate in c._parameterized_gates

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
    assert rx_gate.parameter_values == [0.5]
    assert rz_gate.parameter_values == [1.0]

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
