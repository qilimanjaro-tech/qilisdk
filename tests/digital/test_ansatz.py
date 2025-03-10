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

import pytest

from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.digital.gates import U1, M

# --- Helper functions for tests ---


def create_ansatz(
    n_qubits: int, layers: int = 1, connectivity="Linear", structure="grouped", one_qubit_gate="U1", two_qubit_gate="CZ"
) -> HardwareEfficientAnsatz:
    return HardwareEfficientAnsatz(
        n_qubits=n_qubits,
        layers=layers,
        connectivity=connectivity,
        structure=structure,
        one_qubit_gate=one_qubit_gate,
        two_qubit_gate=two_qubit_gate,
    )


# --- Test cases ---


def test_nparameters_property():
    """
    Test that the nparameters property returns the expected number.
    Expected: n_qubits * (layers + 1) * (number of parameters in the one-qubit gate)
    For U1 (assumed to have 1 parameter), with 3 qubits and 2 layers:
       expected = 3 * (2 + 1) * 1 = 9.
    """
    n_qubits = 3
    layers = 2
    ansatz = create_ansatz(n_qubits=n_qubits, layers=layers, connectivity="Linear", structure="grouped")
    expected = n_qubits * (layers + 1) * len(U1.PARAMETER_NAMES)
    assert ansatz.nparameters == expected


def test_construct_circuit_grouped_structure():
    """
    Test circuit construction with 'grouped' structure.
    For n_qubits=3, layers=1 and Linear connectivity (which gives 2 two-qubit gates),
    the expected gate count is:
      - Initial layer: 3 one-qubit gates.
      - For each layer (1 layer): grouped: 3 one-qubit gates + 2 two-qubit gates.
      - Final measurement: 1 gate.
    Total = 3 + (3+2) + 1 = 9.
    """
    n_qubits = 3
    layers = 1
    ansatz = create_ansatz(n_qubits=n_qubits, layers=layers, connectivity="Linear", structure="grouped")
    params = [0.5] * ansatz.nparameters
    circuit = ansatz.get_circuit(params)
    # Check the measurement gate is added at the end.
    assert isinstance(circuit.gates[-1], M)
    # Check total number of gates.
    assert len(circuit.gates) == 9


def test_construct_circuit_interposed_structure():
    """
    Test circuit construction with 'interposed' structure.
    For n_qubits=3, layers=1 and Linear connectivity (2 connectivity pairs),
    the expected gate count is:
      - Initial layer: 3 one-qubit gates.
      - For each layer:
          For each qubit: one one-qubit gate (3 total)
          and for each connectivity pair (2 pairs) added for every qubit: 3*2 = 6 two-qubit gates.
      - Final measurement: 1 gate.
    Total = 3 + (3+6) + 1 = 13.
    """
    n_qubits = 3
    layers = 1
    ansatz = create_ansatz(n_qubits=n_qubits, layers=layers, connectivity="Linear", structure="interposed")
    params = [0.5] * ansatz.nparameters
    circuit = ansatz.get_circuit(params)
    assert len(circuit.gates) == 13


def test_insufficient_parameters():
    """
    Test that constructing the circuit with too few parameters raises a ValueError.
    """
    n_qubits = 4
    layers = 1
    ansatz = create_ansatz(
        n_qubits=n_qubits, layers=layers, connectivity="Linear", structure="grouped", two_qubit_gate="CNOT"
    )
    params = [0.1] * (ansatz.nparameters - 1)
    with pytest.raises(ValueError, match="Expecting"):
        ansatz.get_circuit(params)


def test_invalid_connectivity():
    """
    Test that an invalid connectivity string raises a ValueError.
    """
    n_qubits = 3
    with pytest.raises(ValueError, match="Unrecognized connectivity type"):
        create_ansatz(n_qubits=n_qubits, connectivity="XYZ", structure="grouped")


def test_invalid_structure():
    """
    Test that an invalid structure string raises a ValueError.
    """
    n_qubits = 3
    with pytest.raises(ValueError, match="provided structure random is not supported"):
        create_ansatz(n_qubits=n_qubits, connectivity="Linear", structure="random")


def test_custom_connectivity():
    """
    Test that providing a custom connectivity list is preserved.
    """
    n_qubits = 4
    custom_conn = [(0, 2), (1, 3)]
    ansatz = create_ansatz(n_qubits=n_qubits, connectivity=custom_conn, structure="grouped")
    assert ansatz.connectivity == custom_conn


def test_connectivity_full_option():
    """
    Test that the "Full" connectivity option produces the correct pairs.
    For n_qubits=3, Full connectivity should produce: [(0,1), (0,2), (1,2)].
    """
    n_qubits = 3
    ansatz = create_ansatz(n_qubits=n_qubits, connectivity="Full", structure="grouped")
    expected = [(0, 1), (0, 2), (1, 2)]
    # Sorting for order-independence.
    assert sorted(ansatz.connectivity) == sorted(expected)


def test_connectivity_circular_option():
    """
    Test that the "Circular" connectivity option produces the correct pairs.
    For n_qubits=3, Circular connectivity should produce: [(0,1), (1,2), (2,0)].
    """
    n_qubits = 3
    ansatz = create_ansatz(n_qubits=n_qubits, connectivity="Circular", structure="grouped")
    expected = [(0, 1), (1, 2), (2, 0)]
    assert sorted(ansatz.connectivity) == sorted(expected)


def test_connectivity_linear_option():
    """
    Test that the "Linear" connectivity option produces the correct pairs.
    For n_qubits=4, Linear connectivity should produce: [(0,1), (1,2), (2,3)].
    """
    n_qubits = 4
    ansatz = create_ansatz(n_qubits=n_qubits, connectivity="Linear", structure="grouped")
    expected = [(0, 1), (1, 2), (2, 3)]
    assert ansatz.connectivity == expected


def test_circuit_reset_on_multiple_constructs():
    """
    Test that calling construct_circuit multiple times resets the circuit.
    """
    n_qubits = 3
    layers = 1
    ansatz = create_ansatz(n_qubits=n_qubits, layers=layers, connectivity="Linear", structure="grouped")
    params1 = [0.5] * ansatz.nparameters
    circuit1 = ansatz.get_circuit(params1)
    params2 = [0.5] * ansatz.nparameters
    circuit2 = ansatz.get_circuit(params2)
    # Verify that a new Circuit instance is created on each call.
    assert circuit1 is not circuit2
    # Also check that the new circuit's gate list is not a concatenation of previous gates.
    assert len(circuit2.gates) < len(circuit1.gates) * 2


def test_measurement_gate_added():
    """
    Test that the final gate in the constructed circuit is a measurement gate
    covering all qubits.
    """
    n_qubits = 3
    ansatz = create_ansatz(n_qubits=n_qubits, connectivity="Linear", structure="grouped")
    params = [0.5] * ansatz.nparameters
    circuit = ansatz.get_circuit(params)
    # Check that the last gate is a measurement gate (instance of M)
    meas_gate = circuit.gates[-1]
    assert isinstance(meas_gate, M)
    # Verify that the measurement is applied to all qubits.
    assert meas_gate.target_qubits == tuple(range(n_qubits))
