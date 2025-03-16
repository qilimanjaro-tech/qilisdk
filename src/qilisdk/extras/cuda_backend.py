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
from typing import Callable, Type, TypeVar

import cudaq
import numpy as np

from qilisdk.digital import (
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    Adjoint,
    BasicGate,
    Circuit,
    Controlled,
    DigitalResult,
    H,
    M,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.exceptions import UnsupportedGateError

TBasicGate = TypeVar("TBasicGate", bound=BasicGate)
BasicGateHandlersMapping = dict[Type[TBasicGate], Callable[[cudaq.Kernel, TBasicGate, cudaq.QuakeValue], None]]


class SimulationMethod(str, Enum):
    STATE_VECTOR = "state_vector"
    TENSOR_NETWORK = "tensor_network"
    MATRIX_PRODUCT_STATE = "matrix_product_state"


class CudaBackend:
    def __init__(self, simulation_method: SimulationMethod = SimulationMethod.STATE_VECTOR) -> None:
        self.simulation_method = simulation_method
        self._basic_gate_handlers: BasicGateHandlersMapping = {
            X: CudaBackend._handle_X,
            Y: CudaBackend._handle_Y,
            Z: CudaBackend._handle_Z,
            H: CudaBackend._handle_H,
            S: CudaBackend._handle_S,
            T: CudaBackend._handle_T,
            RX: CudaBackend._handle_RX,
            RY: CudaBackend._handle_RY,
            RZ: CudaBackend._handle_RZ,
            U1: CudaBackend._handle_U1,
            U2: CudaBackend._handle_U2,
            U3: CudaBackend._handle_U3,
        }

    def _apply_simulation_method(self) -> None:
        if self.simulation_method == SimulationMethod.STATE_VECTOR:
            if cudaq.num_available_gpus() == 0:
                cudaq.set_target("qpp-cpu")
            else:
                cudaq.set_target("nvidia")
        elif self.simulation_method == SimulationMethod.TENSOR_NETWORK:
            cudaq.set_target("tensornet")
        else:
            cudaq.set_target("tensornet-mps")

    def execute(self, circuit: Circuit, nshots: int = 1000) -> DigitalResult:
        self._apply_simulation_method()
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(circuit.nqubits)

        for gate in circuit.gates:
            if isinstance(gate, Controlled):
                self._handle_controlled(kernel, gate, qubits[gate.control_qubits[0]], qubits[gate.target_qubits[0]])
            elif isinstance(gate, Adjoint):
                self._handle_adjoint(kernel, gate, qubits[gate.target_qubits[0]])
            elif isinstance(gate, M):
                self._handle_M(kernel, gate, circuit, qubits)
            else:
                handler = self._basic_gate_handlers.get(type(gate), None)
                if handler is None:
                    raise UnsupportedGateError
                handler(kernel, gate, qubits[gate.target_qubits[0]])

        cudaq_result = cudaq.sample(kernel, shots_count=nshots)
        return DigitalResult(nshots=nshots, samples=dict(cudaq_result.items()))

    def _handle_controlled(
        self, kernel: cudaq.Kernel, gate: Controlled, control_qubit: cudaq.QuakeValue, target_qubit: cudaq.QuakeValue
    ) -> None:
        if len(gate.control_qubits) != 1:
            raise UnsupportedGateError
        target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
        handler = self._basic_gate_handlers.get(type(gate.basic_gate), None)
        if handler is None:
            raise UnsupportedGateError
        handler(target_kernel, gate.basic_gate, qubit)
        kernel.control(target_kernel, control_qubit, target_qubit)

    def _handle_adjoint(self, kernel: cudaq.Kernel, gate: Adjoint, target_qubit: cudaq.QuakeValue) -> None:
        target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
        handler = self._basic_gate_handlers.get(type(gate.basic_gate), None)
        if handler is None:
            raise UnsupportedGateError
        handler(target_kernel, gate.basic_gate, qubit)
        kernel.adjoint(target_kernel, target_qubit)

    @staticmethod
    def _handle_M(kernel: cudaq.Kernel, gate: M, circuit: Circuit, qubits: cudaq.QuakeValue) -> None:
        if gate.nqubits == circuit.nqubits:
            kernel.mz(qubits)
        else:
            for idx in gate.target_qubits:
                kernel.mz(qubits[idx])

    @staticmethod
    def _handle_X(kernel: cudaq.Kernel, gate: X, qubit: cudaq.QuakeValue) -> None:
        kernel.x(qubit)

    @staticmethod
    def _handle_Y(kernel: cudaq.Kernel, gate: Y, qubit: cudaq.QuakeValue) -> None:
        kernel.y(qubit)

    @staticmethod
    def _handle_Z(kernel: cudaq.Kernel, gate: Z, qubit: cudaq.QuakeValue) -> None:
        kernel.z(qubit)

    @staticmethod
    def _handle_H(kernel: cudaq.Kernel, gate: H, qubit: cudaq.QuakeValue) -> None:
        kernel.h(qubit)

    @staticmethod
    def _handle_S(kernel: cudaq.Kernel, gate: S, qubit: cudaq.QuakeValue) -> None:
        kernel.s(qubit)

    @staticmethod
    def _handle_T(kernel: cudaq.Kernel, gate: T, qubit: cudaq.QuakeValue) -> None:
        kernel.t(qubit)

    @staticmethod
    def _handle_RX(kernel: cudaq.Kernel, gate: RX, qubit: cudaq.QuakeValue) -> None:
        kernel.rx(*gate.parameter_values, qubit)

    @staticmethod
    def _handle_RY(kernel: cudaq.Kernel, gate: RY, qubit: cudaq.QuakeValue) -> None:
        kernel.ry(*gate.parameter_values, qubit)

    @staticmethod
    def _handle_RZ(kernel: cudaq.Kernel, gate: RZ, qubit: cudaq.QuakeValue) -> None:
        kernel.rz(*gate.parameter_values, qubit)

    @staticmethod
    def _handle_U1(kernel: cudaq.Kernel, gate: U1, qubit: cudaq.QuakeValue) -> None:
        kernel.u3(theta=0.0, phi=gate.phi, delta=0.0, target=qubit)

    @staticmethod
    def _handle_U2(kernel: cudaq.Kernel, gate: U2, qubit: cudaq.QuakeValue) -> None:
        kernel.u3(theta=np.pi / 2, phi=gate.phi, delta=gate.gamma, target=qubit)

    @staticmethod
    def _handle_U3(kernel: cudaq.Kernel, gate: U3, qubit: cudaq.QuakeValue) -> None:
        kernel.u3(theta=gate.theta, phi=gate.phi, delta=gate.gamma, target=qubit)
