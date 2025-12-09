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

from qilisdk.analog.hamiltonian import Hamiltonian, PauliX, PauliZ
from qilisdk.digital.ansatz import QAOA, HardwareEfficientAnsatz
from qilisdk.digital.gates import CNOT, CZ, U1

# ------------------------------ Helpers ------------------------------


def _gate_counts(ansatz: HardwareEfficientAnsatz) -> tuple[int, int]:
    """(one_qubit_count, two_qubit_count) by simple isinstance checks."""
    one = 0
    two = 0
    for gate in ansatz.gates:
        if gate.nqubits == 1:
            one += 1
        elif gate.nqubits == 2:
            two += 1
    return one, two


# ------------------------------ Tests ------------------------------


def test_nparameters_property():
    """
    nparameters should equal: n_qubits * (layers + 1) * len(one_qubit_gate.PARAMETER_NAMES)
    Example with U1 (1 param): 3 qubits, 2 layers => 3 * (2+1) * 1 = 9.
    """
    n_qubits = 3
    layers = 2
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, layers=layers, connectivity="Linear", structure="grouped")
    expected = n_qubits * (layers) * len(U1.PARAMETER_NAMES)
    assert ansatz.nparameters == expected


def test_construct_circuit_grouped_structure_gate_count():
    """
    grouped schedule:
      - one-qubit gates: (layers + 1) * n
      - two-qubit gates: layers * |E|
    """
    n_qubits = 3
    layers = 1
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, layers=layers, connectivity="Linear", structure="grouped")
    E = len(list(ansatz.connectivity))
    one_q, two_q = _gate_counts(ansatz)

    assert one_q == (layers) * n_qubits
    assert two_q == layers * E
    assert len(ansatz.gates) == one_q + two_q  # total gates


def test_construct_circuit_interposed_structure_gate_count():
    """
    interposed schedule:
      - one-qubit gates: (layers + 1) * n
      - two-qubit gates: layers * n * |E|
    """
    n_qubits = 3
    layers = 1
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, layers=layers, connectivity="Linear", structure="interposed")
    E = len(list(ansatz.connectivity))
    one_q, two_q = _gate_counts(ansatz)

    assert one_q == (layers) * n_qubits
    assert two_q == layers * n_qubits * E
    assert len(ansatz.gates) == one_q + two_q


def test_layers_zero_only_initial_block():
    """layers=0 → only U(0) block is applied; no entanglers."""
    n_qubits = 5
    layers = 0
    for structure in ("grouped", "interposed"):
        ans = HardwareEfficientAnsatz(nqubits=n_qubits, layers=layers, connectivity="Linear", structure=structure)
        one_q, two_q = _gate_counts(ans)
        assert one_q == (layers) * n_qubits
        assert two_q == 0
        assert len(ans.gates) == 0


def test_invalid_connectivity_string_raises():
    with pytest.raises(ValueError, match="Unrecognized connectivity"):
        HardwareEfficientAnsatz(nqubits=3, connectivity="XYZ", structure="grouped")


def test_structure_normalization_and_defaulting():
    """Structure is normalized: 'GROUPED' -> 'grouped'; unknown -> 'interposed' (current behavior)."""
    a = HardwareEfficientAnsatz(nqubits=2, structure="GROUPED")
    assert a.structure == "grouped"
    b = HardwareEfficientAnsatz(nqubits=2, structure="random")
    assert b.structure == "interposed"  # current constructor behavior


def test_custom_connectivity_preserved_as_pairs():
    n_qubits = 4
    custom_conn = [(0, 2), (1, 3)]
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, connectivity=custom_conn, structure="grouped")
    # Accept tuple/list forms; compare as sets of pairs.
    assert set(map(tuple, ansatz.connectivity)) == set(custom_conn)


def test_connectivity_full_option_pairs():
    n_qubits = 3
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, connectivity="Full", structure="grouped")
    expected = {(0, 1), (0, 2), (1, 2)}
    assert set(map(tuple, ansatz.connectivity)) == expected


def test_connectivity_circular_option_pairs():
    n_qubits = 3
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, connectivity="Circular", structure="grouped")
    expected = {(0, 1), (1, 2), (2, 0)}
    assert set(map(tuple, ansatz.connectivity)) == expected


def test_connectivity_circular_with_one_qubit_yields_no_edges():
    n_qubits = 1
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, connectivity="Circular", structure="grouped")
    assert list(ansatz.connectivity) == []


def test_connectivity_linear_option_pairs():
    n_qubits = 4
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, connectivity="Linear", structure="grouped")
    expected = [(0, 1), (1, 2), (2, 3)]
    assert list(ansatz.connectivity) == expected


def test_out_of_range_or_self_edge_in_custom_connectivity_raises():
    # out of range
    with pytest.raises(ValueError, match="out of range"):
        HardwareEfficientAnsatz(nqubits=3, connectivity=[(0, 3)], structure="grouped")
    # self edge
    with pytest.raises(ValueError, match="Self-edge"):
        HardwareEfficientAnsatz(nqubits=3, connectivity=[(1, 1)], structure="grouped")


def test_two_qubit_gate_choice_reflected_in_circuit():
    """When two_qubit_gate=CNOT, all two-qubit gates should be CNOT."""
    n_qubits = 3
    ansatz = HardwareEfficientAnsatz(nqubits=n_qubits, layers=1, connectivity="Linear", two_qubit_gate=CNOT)
    two_gates = [g for g in ansatz.gates if isinstance(g, (CNOT, CZ))]
    assert all(isinstance(g, CNOT) for g in two_gates)


def test_independent_instances_do_not_share_state():
    """Building multiple instances should not share gate lists."""
    a1 = HardwareEfficientAnsatz(nqubits=3, layers=1, connectivity="Linear", structure="grouped")
    a2 = HardwareEfficientAnsatz(nqubits=3, layers=1, connectivity="Linear", structure="grouped")
    assert a1 is not a2
    assert a1.gates is not a2.gates
    assert len(a1.gates) == len(a2.gates)


def test_case_insensitivity_of_connectivity():
    for word in ("linear", "Linear", "LINEAR"):
        ans = HardwareEfficientAnsatz(nqubits=3, connectivity=word)
        assert list(ans.connectivity) == [(0, 1), (1, 2)]


def test_nparameters_property_qaoa():
    """
    nparameters should equal: layers * 2 for QAOA (gamma and alpha per layer).
    Example: 2 layers => 3 * 2 * 2 = 4.
    """
    n_qubits = 3
    layers = 2
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    ansatz = QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, layers=layers)
    expected = layers * 2
    assert ansatz.nparameters == expected


def test_numqubits_qaoa():
    """numqubits property should reflect nqubits argument."""
    n_qubits = 3
    layers = 2
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    ansatz = QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, layers=layers)
    assert ansatz.nqubits == n_qubits


def test_qaoa_invalid_layers_raises():
    """Providing non-positive layers should raise ValueError."""
    n_qubits = 3
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    with pytest.raises(ValueError, match="layers must be >= 1"):
        QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, layers=0)
    with pytest.raises(ValueError, match="layers must be >= 1"):
        QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, layers=-1)


def test_qaoa_nqubits_must_match_problem_hamiltonian():
    """nqubits must match the number of qubits in problem_hamiltonian."""
    n_qubits = 3
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    with pytest.raises(ValueError, match="nqubits must match the number of qubits in problem_hamiltonian"):
        QAOA(nqubits=2, problem_hamiltonian=test_hamiltonian, layers=1)


def test_qaoa_nqubits_must_match_mixer_hamiltonian():
    """nqubits must match the number of qubits in mixer_hamiltonian if provided."""
    n_qubits = 3
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    mixer_hamiltonian = Hamiltonian({(PauliX(q),): 1.0 for q in range(n_qubits + 1)})
    with pytest.raises(ValueError, match="nqubits must match the number of qubits in mixer_hamiltonian"):
        QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, mixer_hamiltonian=mixer_hamiltonian, layers=1)


def test_qaoa_trotter_steps_validation():
    """trotter_steps must be a positive integer."""
    n_qubits = 3
    layers = 1
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    with pytest.raises(ValueError, match="trotter_steps must be >= 1"):
        QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, layers=layers, trotter_steps=0)
    with pytest.raises(ValueError, match="trotter_steps must be >= 1"):
        QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, layers=layers, trotter_steps=-2)


def test_qaoa_default_mixer_hamiltonian():
    """If no mixer_hamiltonian is provided, it defaults to X-mixer."""
    n_qubits = 2
    layers = 1
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    ansatz = QAOA(nqubits=n_qubits, problem_hamiltonian=test_hamiltonian, layers=layers)
    # Construct expected X-mixer
    expected_mixer = Hamiltonian({(PauliX(q),): 1.0 for q in range(n_qubits)})
    assert ansatz.mixer_hamiltonian == expected_mixer


def test_qaoa_gate_count():
    """Verify the number of gates in the QAOA ansatz."""
    n_qubits = 2
    layers = 2
    test_hamiltonian = Hamiltonian({(PauliZ(q),): 1.0 for q in range(n_qubits)})
    test_mixer = Hamiltonian({(PauliX(q),): 1.0 for q in range(n_qubits)})
    ansatz = QAOA(
        nqubits=n_qubits,
        problem_hamiltonian=test_hamiltonian,
        layers=layers,
        trotter_steps=1,
        mixer_hamiltonian=test_mixer,
    )

    # Each layer has (assuming Z problem Hamiltonian and X mixer):
    # - Problem unitary: 2m-1 gates per term of size m in problem_hamiltonian (CNOT, RZ, CNOT)
    # - Mixer unitary: 3 gates per qubit (H RZ H)
    expected_gates_per_layer = (2 * len(test_hamiltonian.elements) - 1) + 3 * n_qubits
    expected_total_gates = expected_gates_per_layer * layers
    assert len(ansatz.gates) == expected_total_gates
