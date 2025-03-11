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

from typing import Any, Callable

from qibo.gates import gates as QiboGates
from qibo.models.circuit import Circuit as QiboCircuit

from qilisdk.digital import (
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    Circuit,
    Controlled,
    Gate,
    H,
    M,
    S,
    SimulationDigitalResults,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.exceptions import UnsupportedGateError


class QiboBackend:
    def __init__(self) -> None: ...

    def execute(self, circuit: Circuit, nshots: int = 1000) -> SimulationDigitalResults:  # noqa: PLR6301
        qibo_circuit = QiboBackend.to_qibo(circuit=circuit)
        qibo_results = qibo_circuit.execute(nshots=nshots)
        return SimulationDigitalResults(
            state=qibo_results.state(),
            probabilities=qibo_results.probabilities(),
            samples=qibo_results.samples(),
            frequencies=qibo_results.frequencies(),
            nshots=nshots,
        )

    @staticmethod
    def to_qibo(circuit: Circuit) -> QiboCircuit:
        to_qibo_converters: dict[type[Gate], Callable[[Any], QiboGates.Gate]] = {
            X: QiboBackend._to_qibo_X,
            Y: QiboBackend._to_qibo_Y,
            Z: QiboBackend._to_qibo_Z,
            H: QiboBackend._to_qibo_H,
            S: QiboBackend._to_qibo_S,
            T: QiboBackend._to_qibo_T,
            RX: QiboBackend._to_qibo_RX,
            RY: QiboBackend._to_qibo_RY,
            RZ: QiboBackend._to_qibo_RZ,
            U1: QiboBackend._to_qibo_U1,
            U2: QiboBackend._to_qibo_U2,
            U3: QiboBackend._to_qibo_U3,
            Controlled: QiboBackend._to_qibo_Controlled,
            M: QiboBackend._to_qibo_M,
        }
        qibo_circuit = QiboCircuit(nqubits=circuit.nqubits)
        for gate in circuit.gates:
            converter = to_qibo_converters.get(type(gate))
            if converter is None:
                raise UnsupportedGateError(f"Unsupported gate type: {type(gate).__name__}")
            qibo_circuit.add(converter(gate))
        return qibo_circuit

    @staticmethod
    def _to_qibo_X(gate: X) -> QiboGates.X:
        return QiboGates.X(gate.target_qubits[0])

    @staticmethod
    def _to_qibo_Y(gate: Y) -> QiboGates.Y:
        return QiboGates.Y(gate.target_qubits[0])

    @staticmethod
    def _to_qibo_Z(gate: Z) -> QiboGates.Z:
        return QiboGates.Z(gate.target_qubits[0])

    @staticmethod
    def _to_qibo_H(gate: H) -> QiboGates.H:
        return QiboGates.H(gate.target_qubits[0])

    @staticmethod
    def _to_qibo_S(gate: S) -> QiboGates.S:
        return QiboGates.S(gate.target_qubits[0])

    @staticmethod
    def _to_qibo_T(gate: T) -> QiboGates.T:
        return QiboGates.T(gate.target_qubits[0])

    @staticmethod
    def _to_qibo_RX(gate: RX) -> QiboGates.RX:
        return QiboGates.RX(gate.target_qubits[0], theta=gate.parameters["theta"])

    @staticmethod
    def _to_qibo_RY(gate: RY) -> QiboGates.RY:
        return QiboGates.RY(gate.target_qubits[0], theta=gate.parameters["theta"])

    @staticmethod
    def _to_qibo_RZ(gate: RZ) -> QiboGates.RZ:
        return QiboGates.RZ(gate.target_qubits[0], theta=gate.parameters["phi"])

    @staticmethod
    def _to_qibo_U1(gate: U1) -> QiboGates.U1:
        return QiboGates.U1(gate.target_qubits[0], theta=gate.parameters["phi"])

    @staticmethod
    def _to_qibo_U2(gate: U2) -> QiboGates.U2:
        return QiboGates.U2(gate.target_qubits[0], phi=gate.parameters["phi"], lam=gate.parameters["gamma"])

    @staticmethod
    def _to_qibo_U3(gate: U3) -> QiboGates.U3:
        return QiboGates.U3(
            gate.target_qubits[0],
            theta=gate.parameters["theta"],
            phi=gate.parameters["phi"],
            lam=gate.parameters["gamma"],
        )

    @staticmethod
    def _to_qibo_Controlled(gate: Controlled) -> QiboGates.CNOT | QiboGates.CZ:
        if gate.is_modified_from(X) and gate.nqubits == 2:  # noqa: PLR2004
            return QiboGates.CNOT(**gate.qubits)
        if gate.is_modified_from(Z) and gate.nqubits == 2:  # noqa: PLR2004
            return QiboGates.CZ(**gate.qubits)
        if gate.is_modified_from(Z) and len(gate.control_qubits) == 3:  # noqa: PLR2004
            return QiboGates.CCZ(**gate.qubits)
        raise UnsupportedGateError("Unsupported gate", gate)

    @staticmethod
    def _to_qibo_M(gate: M) -> QiboGates.M:
        return QiboGates.M(*gate.target_qubits)
