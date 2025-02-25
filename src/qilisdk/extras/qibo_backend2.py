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

from typing import ClassVar

from qibo.gates import gates as QiboGates
from qibo.models.circuit import Circuit as QiboCircuit

from qilisdk.digital import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    Circuit,
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
        qibo_circuit = QiboCircuit(nqubits=circuit.nqubits)

        for gate in circuit.gates:
            qibo_circuit.add(QiboBackend.to_qibo_gate(gate))

        return qibo_circuit

    @staticmethod
    def to_qibo_gate(gate: Gate) -> QiboGates.Gate:
        # Map gate type and args:
        qibo_gate_class: type[QiboGates.Gate] = QiboBackend.to_qibo_gate_map(gate.__class__)
        qibo_gate_args: dict = {QiboBackend.to_qibo_arg_map(k, gate.__class__): v for k, v in gate.parameters.items()}

        # Handle measurements
        if qibo_gate_class == QiboGates.M:
            return qibo_gate_class(*[qibo_gate_args["q"]])

        return qibo_gate_class(**qibo_gate_args)

    @staticmethod
    def to_qibo_arg_map(name: str, gate_type: type[Gate]) -> str:
        arg_map = QiboBackend._equiv_qibo_arg

        # Specific exceptions mapping:
        if gate_type in {RZ, U1}:
            arg_map["phi"] = "theta"

        return arg_map[name] if name in map else name

    @staticmethod
    def to_qibo_gate_map(type_gate: type[Gate]) -> type[QiboGates.Gate]:
        if type_gate not in QiboBackend._equiv_qibo_gate_type:
            raise UnsupportedGateError(f"Unsupported gate type: {type_gate.__name__}")

        return QiboBackend._equiv_qibo_gate_type[type_gate]

    # General mapping qilisdk -> qibo args:
    _equiv_qibo_arg: ClassVar[dict[str, str]] = {
        "qubit": "q",
        "control": "q0",
        "target": "q1",
    }

    # General mapping qilisdk -> qibo gate type:
    _equiv_qibo_gate_type: ClassVar[dict[type[Gate], type[QiboGates.Gate]]] = {
        X: QiboGates.X,
        Y: QiboGates.Y,
        Z: QiboGates.Z,
        H: QiboGates.H,
        S: QiboGates.S,
        T: QiboGates.T,
        RX: QiboGates.RX,
        RY: QiboGates.RY,
        RZ: QiboGates.RZ,
        U1: QiboGates.U1,
        U2: QiboGates.U2,
        U3: QiboGates.U3,
        CNOT: QiboGates.CNOT,
        CZ: QiboGates.CZ,
        M: QiboGates.M,
    }
