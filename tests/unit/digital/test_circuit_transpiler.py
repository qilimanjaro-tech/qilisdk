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


from qilisdk.digital import CNOT, Circuit, X
from qilisdk.digital.circuit_transpiler import CircuitTranspiler
from qilisdk.digital.circuit_transpiler_passes import CancelIdentityPairsPass
from qilisdk.digital.gates import Controlled, Gate

from .circuit_transpiler_passes.utils import _sequences_equivalent


def _describe_gate(gate: Gate) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    return (type(gate).__name__, gate.qubits, tuple(gate.get_parameter_values()))


def _describe_circuit(circuit: Circuit) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
    return [_describe_gate(gate) for gate in circuit.gates]


def test_circuit_transpiler_preserves_semantics_for_simple_circuit() -> None:
    circuit = Circuit(2)
    circuit.add(CNOT(0, 1))

    transpiler = CircuitTranspiler.default()
    transpiled_circuit = transpiler.transpile(circuit)

    assert transpiled_circuit is not circuit
    assert transpiled_circuit.nqubits == circuit.nqubits
    assert _sequences_equivalent(circuit.gates, transpiled_circuit.gates, circuit.nqubits)


def test_circuit_transpiler_does_not_mutate_input_circuit() -> None:
    circuit = Circuit(1)
    circuit.add(X(0))
    circuit.add(X(0))
    original_snapshot = _describe_circuit(circuit)

    transpiled_circuit = CircuitTranspiler.default().transpile(circuit)

    assert _describe_circuit(circuit) == original_snapshot
    assert transpiled_circuit.gates == []


def test_circuit_transpiler_default_pipeline_decomposes_multi_controlled_gates() -> None:
    circuit = Circuit(3)
    circuit.add(Controlled(0, 1, basic_gate=X(2)))

    transpiled_circuit = CircuitTranspiler.default().transpile(circuit)

    for gate in transpiled_circuit.gates:
        if isinstance(gate, Controlled):
            assert len(gate.control_qubits) <= 1
    assert _sequences_equivalent(circuit.gates, transpiled_circuit.gates, circuit.nqubits)


def test_circuit_transpiler_accepts_custom_pipeline() -> None:
    circuit = Circuit(3)
    circuit.add(Controlled(0, 1, basic_gate=X(2)))

    transpiled_circuit = CircuitTranspiler(pipeline=[CancelIdentityPairsPass()]).transpile(circuit)

    assert len(transpiled_circuit.gates) == 1
    assert isinstance(transpiled_circuit.gates[0], Controlled)
    assert len(transpiled_circuit.gates[0].control_qubits) == 2


def test_circuit_transpiler_run_returns_per_pass_results() -> None:
    circuit = Circuit(1)
    circuit.add(X(0))
    circuit.add(X(0))

    run_result = CircuitTranspiler(pipeline=[CancelIdentityPairsPass()]).run(circuit)

    assert run_result.circuit is run_result.final_circuit
    assert run_result.final_circuit.gates == []
    assert len(run_result.pass_results) == 1

    pass_result = run_result.pass_results[0]
    assert pass_result.pass_name == "CancelIdentityPairsPass"
    assert pass_result.transpiled_circuit is run_result.final_circuit
    assert pass_result.layout.initial_layout is None
    assert pass_result.layout.final_layout is None


def test_circuit_transpiler_run_reports_layout_result() -> None:
    circuit = Circuit(2)
    circuit.add(CNOT(0, 1))
    mapping = {0: 0, 1: 1}

    run_result = CircuitTranspiler.default(topology=[(0, 1)], qubit_mapping=mapping).run(circuit)

    assert run_result.layout.initial_layout == mapping
    assert run_result.layout.final_layout == mapping
