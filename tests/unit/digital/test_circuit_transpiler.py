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


from qilisdk.digital import CNOT, Circuit
from qilisdk.digital.circuit_transpiler import CircuitTranspiler


def test_circuit_transpiler_qaoa():
    circuit = Circuit(2)
    circuit.add(CNOT(0, 1))
    transpiler = CircuitTranspiler()
    transpiled_circuit = transpiler.transpile(circuit)
    assert transpiled_circuit.nqubits == circuit.nqubits
    assert len(transpiled_circuit.gates) == len(circuit.gates)
    assert transpiled_circuit.gates[0] == circuit.gates[0]
