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
from __future__ import annotations

from enum import Enum

import cudaq

from qilisdk.digital import (
    CNOT,
    RX,
    RY,
    RZ,
    Circuit,
    H,
    M,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.exceptions import UnsupportedGateError


class SimulationMethod(str, Enum):
    STATE_VECTOR = "state_vector"
    TENSORNET = "tensornet"
    MATRIX_PRODUCT_STATE = "matrix_product_state"


class CudaBackend:
    def __init__(self, simulation_method: SimulationMethod = SimulationMethod.STATE_VECTOR) -> None:
        self.simulation_method = simulation_method

    def _apply_simulation_method(self) -> None:
        if self.simulation_method == SimulationMethod.STATE_VECTOR:
            if cudaq.num_available_gpus() == 0:
                cudaq.set_target("qpp-cpu")
            else:
                cudaq.set_target("nvidia")
        elif self.simulation_method == SimulationMethod.TENSORNET:
            cudaq.set_target("tensornet")
        else:
            cudaq.set_target("tensornet-mps")

    def execute(self, circuit: Circuit, shots: int = 1000):  # noqa: ANN201
        self._apply_simulation_method()
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(circuit.nqubits)
        for gate in circuit.gates:
            if isinstance(gate, X):
                kernel.x(qubits[gate.target_qubits[0]])
            elif isinstance(gate, Y):
                kernel.y(qubits[gate.target_qubits[0]])
            elif isinstance(gate, Z):
                kernel.z(qubits[gate.target_qubits[0]])
            elif isinstance(gate, H):
                kernel.h(qubits[gate.target_qubits[0]])
            elif isinstance(gate, S):
                kernel.s(qubits[gate.target_qubits[0]])
            elif isinstance(gate, T):
                kernel.t(qubits[gate.target_qubits[0]])
            elif isinstance(gate, RX):
                kernel.rx(*gate.parameter_values, qubits[gate.target_qubits[0]])
            elif isinstance(gate, RY):
                kernel.ry(*gate.parameter_values, qubits[gate.target_qubits[0]])
            elif isinstance(gate, RZ):
                kernel.rz(*gate.parameter_values, qubits[gate.target_qubits[0]])
            elif isinstance(gate, CNOT):
                target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
                target_kernel.x(qubit)
                kernel.control(target_kernel, qubits[gate.control_qubits[0]], qubits[gate.target_qubits[0]])
            elif isinstance(gate, M):
                if len(gate.target_qubits) == circuit.nqubits:
                    kernel.mz(qubits)
                else:
                    kernel.mz([qubits[idx] for idx in gate.target_qubits])
            else:
                raise UnsupportedGateError
        results = cudaq.sample(kernel, shots_count=shots)
        return results
