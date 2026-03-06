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
from qilisdk.digital.circuit_transpiler_passes import (
    CancelIdentityPairsPass,
    CircuitTranspilerPass,
    SingleQubitGateBasis,
)
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
    transpilation_result = transpiler.transpile(circuit)
    transpiled_circuit = transpilation_result.circuit

    assert transpiled_circuit is not circuit
    assert transpiled_circuit.nqubits == circuit.nqubits
    assert _sequences_equivalent(circuit.gates, transpiled_circuit.gates, circuit.nqubits)


def test_circuit_transpiler_does_not_mutate_input_circuit() -> None:
    circuit = Circuit(1)
    circuit.add(X(0))
    circuit.add(X(0))
    original_snapshot = _describe_circuit(circuit)

    transpilation_result = CircuitTranspiler.default().transpile(circuit)

    assert _describe_circuit(circuit) == original_snapshot
    assert transpilation_result.circuit.gates == []


def test_circuit_transpiler_default_pipeline_decomposes_multi_controlled_gates() -> None:
    circuit = Circuit(3)
    circuit.add(Controlled(0, 1, basic_gate=X(2)))

    transpilation_result = CircuitTranspiler.default().transpile(circuit)
    transpiled_circuit = transpilation_result.circuit

    for gate in transpiled_circuit.gates:
        if isinstance(gate, Controlled):
            assert len(gate.control_qubits) <= 1
    assert _sequences_equivalent(circuit.gates, transpiled_circuit.gates, circuit.nqubits)


def test_circuit_transpiler_default_pipeline_passes_single_qubit_basis_to_fuse_passes() -> None:
    transpiler = CircuitTranspiler.default(single_qubit_basis=SingleQubitGateBasis.RxRyRz)

    fuse_passes = [
        transpiler_pass
        for transpiler_pass in transpiler._pipeline
        if transpiler_pass.__class__.__name__ == "FuseSingleQubitGatesPass"
    ]

    assert fuse_passes
    assert all(pass_instance.single_qubit_basis == SingleQubitGateBasis.RxRyRz for pass_instance in fuse_passes)


def test_circuit_transpiler_accepts_custom_pipeline() -> None:
    circuit = Circuit(3)
    circuit.add(Controlled(0, 1, basic_gate=X(2)))

    transpilation_result = CircuitTranspiler(pipeline=[CancelIdentityPairsPass()]).transpile(circuit)
    transpiled_circuit = transpilation_result.circuit

    assert len(transpiled_circuit.gates) == 1
    assert isinstance(transpiled_circuit.gates[0], Controlled)
    assert len(transpiled_circuit.gates[0].control_qubits) == 2


def test_circuit_transpiler_transpile_returns_intermediate_results() -> None:
    circuit = Circuit(1)
    circuit.add(X(0))
    circuit.add(X(0))

    transpilation_result = CircuitTranspiler(pipeline=[CancelIdentityPairsPass()]).transpile(circuit)

    assert transpilation_result.circuit.gates == []
    assert len(transpilation_result.intermediate_results) == 1

    intermediate_result = transpilation_result.intermediate_results[0]
    assert intermediate_result.name == "CancelIdentityPairsPass"
    assert intermediate_result.circuit is transpilation_result.circuit


def test_circuit_transpiler_transpile_reports_layout_result() -> None:
    circuit = Circuit(2)
    circuit.add(CNOT(0, 1))
    mapping = {0: 0, 1: 1}

    transpilation_result = CircuitTranspiler.default(topology=[(0, 1)], qubit_mapping=mapping).transpile(circuit)

    assert transpilation_result.layout == mapping


class _RecordLayoutPass(CircuitTranspilerPass):
    def run(self, circuit: Circuit) -> Circuit:
        if self.context is not None:
            self.context.initial_layout = [1, 0]
            self.context.final_layout = {0: 0, 1: 1}
        return circuit


def test_circuit_transpiler_transpile_reports_only_final_layout() -> None:
    circuit = Circuit(2)

    transpilation_result = CircuitTranspiler(pipeline=[_RecordLayoutPass()]).transpile(circuit)

    assert transpilation_result.layout == {0: 0, 1: 1}


class _RecordInitialLayoutOnlyPass(CircuitTranspilerPass):
    def run(self, circuit: Circuit) -> Circuit:
        if self.context is not None:
            self.context.initial_layout = [1, 0]
        return circuit


def test_circuit_transpiler_transpile_does_not_expose_initial_layout() -> None:
    circuit = Circuit(2)

    transpilation_result = CircuitTranspiler(pipeline=[_RecordInitialLayoutOnlyPass()]).transpile(circuit)

    assert transpilation_result.layout == {}
