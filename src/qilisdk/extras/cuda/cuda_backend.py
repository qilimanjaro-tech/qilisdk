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

from typing import TYPE_CHECKING, Callable, Type, TypeVar

import cudaq
import numpy as np
from cudaq import State
from cudaq.operator import ElementaryOperator, OperatorSum, ScalarOperator, evolve, spin
from cudaq.operator import Schedule as cuda_schedule

from qilisdk.analog.analog_backend import AnalogBackend
from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliOperator, PauliX, PauliY, PauliZ
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.digital import (
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    Circuit,
    H,
    M,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.digital_backend import DigitalBackend, DigitalSimulationMethod
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import Adjoint, BasicGate, Controlled

from .cuda_analog_result import CudaAnalogResult
from .cuda_digital_result import CudaDigitalResult

if TYPE_CHECKING:
    from qilisdk.analog.schedule import Schedule


TBasicGate = TypeVar("TBasicGate", bound=BasicGate)
BasicGateHandlersMapping = dict[Type[TBasicGate], Callable[[cudaq.Kernel, TBasicGate, cudaq.QuakeValue], None]]

TPauliOperator = TypeVar("TPauliOperator", bound=PauliOperator)
PauliOperatorHandlersMapping = dict[Type[TPauliOperator], Callable[[TPauliOperator], ElementaryOperator]]


class CudaBackend(DigitalBackend, AnalogBackend):
    """
    Digital backend implementation using CUDA-based simulation.

    This backend translates a quantum circuit into a CUDA-compatible kernel and executes it
    using the cudaq library. It supports different simulation methods including state vector,
    tensor network, and matrix product state simulations. Gate operations in the circuit are
    mapped to CUDA operations via dedicated handler functions.
    """

    def __init__(
        self, digital_simulation_method: DigitalSimulationMethod = DigitalSimulationMethod.STATE_VECTOR
    ) -> None:
        """
        Initialize the CudaBackend.

        Args:
            digital_simulation_method (SimulationMethod, optional): The simulation method to use for executing circuits.
                Options include STATE_VECTOR, TENSOR_NETWORK, or MATRIX_PRODUCT_STATE.
                Defaults to STATE_VECTOR.
        """
        super().__init__(digital_simulation_method=digital_simulation_method)
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
        self._pauli_operator_handlers: PauliOperatorHandlersMapping = {
            PauliX: CudaBackend._handle_PauliX,
            PauliY: CudaBackend._handle_PauliY,
            PauliZ: CudaBackend._handle_PauliZ,
            PauliI: CudaBackend._handle_PauliI,
        }

    def _apply_digital_simulation_method(self) -> None:
        """
        Configure the cudaq simulation target based on the selected simulation method.

        For the STATE_VECTOR method, it checks for GPU availability and selects an appropriate target.
        For TENSOR_NETWORK and MATRIX_PRODUCT_STATE methods, it explicitly sets the target to use tensor network-based simulations.
        """
        if self.digital_simulation_method == DigitalSimulationMethod.STATE_VECTOR:
            if cudaq.num_available_gpus() == 0:
                cudaq.set_target("qpp-cpu")
            else:
                cudaq.set_target("nvidia")
        elif self.digital_simulation_method == DigitalSimulationMethod.TENSOR_NETWORK:
            cudaq.set_target("tensornet")
        else:
            cudaq.set_target("tensornet-mps")

    def execute(self, circuit: Circuit, nshots: int = 1000) -> CudaDigitalResult:
        """
        Execute a quantum circuit and return the measurement results.

        This method applies the selected simulation method, translates the circuit's gates into
        CUDA operations via their respective handlers, runs the simulation, and returns the result
        as a CudaDigitalResult.

        Args:
            circuit (Circuit): The quantum circuit to be executed.
            nshots (int, optional): The number of measurement shots to perform. Defaults to 1000.

        Returns:
            DigitalResult: A result object containing the measurement samples and computed probabilities.

        Raises:
            UnsupportedGateError: If the circuit contains a gate for which no handler is registered.
        """
        self._apply_digital_simulation_method()
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
        return CudaDigitalResult(nshots=nshots, samples=dict(cudaq_result.items()))

    def _handle_controlled(
        self, kernel: cudaq.Kernel, gate: Controlled, control_qubit: cudaq.QuakeValue, target_qubit: cudaq.QuakeValue
    ) -> None:
        """
        Handle a controlled gate operation.

        This method processes a controlled gate by creating a temporary kernel for the basic gate,
        applying its handler, and then integrating it into the main kernel as a controlled operation.

        Args:
            kernel (cudaq.Kernel): The main CUDA kernel being constructed.
            gate (Controlled): The controlled gate to be handled.
            control_qubit (cudaq.QuakeValue): The control qubit for the gate.
            target_qubit (cudaq.QuakeValue): The target qubit for the gate.

        Raises:
            UnsupportedGateError: If the number of control qubits is not equal to one or if the basic gate is unsupported.
        """
        if len(gate.control_qubits) != 1:
            raise UnsupportedGateError
        target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
        handler = self._basic_gate_handlers.get(type(gate.basic_gate), None)
        if handler is None:
            raise UnsupportedGateError
        handler(target_kernel, gate.basic_gate, qubit)
        kernel.control(target_kernel, control_qubit, target_qubit)

    def _handle_adjoint(self, kernel: cudaq.Kernel, gate: Adjoint, target_qubit: cudaq.QuakeValue) -> None:
        """
        Handle an adjoint (inverse) gate operation.

        This method creates a temporary kernel for the basic gate wrapped by the adjoint,
        applies the corresponding handler, and then integrates it into the main kernel as an adjoint operation.

        Args:
            kernel (cudaq.Kernel): The main CUDA kernel being constructed.
            gate (Adjoint): The adjoint gate to be handled.
            target_qubit (cudaq.QuakeValue): The target qubit for the gate.

        Raises:
            UnsupportedGateError: If the basic gate inside the adjoint is unsupported.
        """
        target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
        handler = self._basic_gate_handlers.get(type(gate.basic_gate), None)
        if handler is None:
            raise UnsupportedGateError
        handler(target_kernel, gate.basic_gate, qubit)
        kernel.adjoint(target_kernel, target_qubit)

    def _hamiltonian_to_cuda(self, hamiltonian: Hamiltonian) -> OperatorSum:
        out = None
        for offset, terms in hamiltonian:
            if out is None:
                out = offset * np.prod([self._pauli_operator_handlers[type(pauli)](pauli) for pauli in terms])
            else:
                out += offset * np.prod([self._pauli_operator_handlers[type(pauli)](pauli) for pauli in terms])
        return out

    def evolve(
        self,
        schedule: Schedule,
        initial_state: QuantumObject,
        observables: list[PauliOperator | Hamiltonian],
        store_intermediate_results: bool = False,
    ) -> CudaAnalogResult:
        """computes the time evolution under of an initial state under the given schedule.

        Args:
            schedule (Schedule): The evolution schedule of the system.
            initial_state (QuantumObject): the initial state of the evolution.
            observables (list[PauliOperator  |  Hamiltonian]): the list of observables to be measured at the end of the evolution.
            store_intermediate_results (bool): A flag to store the intermediate results along the evolution.

        Raises:
            ValueError: if the observables provided are not an instance of the PauliOperator or a Hamiltonian class.

        Returns:
            AnalogResult: The results of the evolution.
        """
        cudaq.set_target("dynamics")

        cuda_hamiltonian = None
        steps = np.linspace(0, schedule.T, int(schedule.T / schedule.dt))

        def parameter_values(time_steps: np.ndarray) -> cuda_schedule:
            def compute_value(param_name: str, step_idx: int) -> float:
                return schedule.get_coefficient(time_steps[int(step_idx)], param_name)

            return cuda_schedule(list(range(len(time_steps))), list(schedule.hamiltonians), compute_value)

        _cuda_schedule = parameter_values(steps)

        def get_schedule(key: str) -> Callable:
            return lambda **args: args[key]

        cuda_hamiltonian = sum(
            ScalarOperator(get_schedule(key)) * self._hamiltonian_to_cuda(ham)
            for key, ham in schedule.hamiltonians.items()
        )

        cuda_observables = []
        for observable in observables:
            if isinstance(observable, PauliOperator):
                cuda_observables.append(self._pauli_operator_handlers[type(observable)](observable))
            elif isinstance(observable, Hamiltonian):
                cuda_observables.append(self._hamiltonian_to_cuda(observable))
            else:
                raise ValueError(f"unsupported observable type of {observable.__class__}")

        evolution_result = evolve(
            hamiltonian=cuda_hamiltonian,
            dimensions=dict.fromkeys(range(schedule.nqubits), 2),
            schedule=_cuda_schedule,
            initial_state=State.from_data(np.array(initial_state.to_density_matrix().dense, dtype=np.complex128)),
            observables=cuda_observables,
            collapse_operators=[],
            store_intermediate_results=store_intermediate_results,
        )

        return CudaAnalogResult(
            final_expected_values=np.array(
                [exp_val.expectation() for exp_val in evolution_result.final_expectation_values()[0]]
            ),
            expected_values=(
                np.array(
                    [[val.expectation() for val in exp_vals] for exp_vals in evolution_result.expectation_values()]
                )
                if evolution_result.expectation_values() is not None
                else None
            ),
            final_state=(
                QuantumObject(np.array(evolution_result.final_state())).adjoint()
                if evolution_result.final_state() is not None
                else None
            ),
            intermediate_states=(
                [QuantumObject(np.array(state)).adjoint() for state in evolution_result.intermediate_states()]
                if evolution_result.intermediate_states() is not None
                else None
            ),
        )

    @staticmethod
    def _handle_M(kernel: cudaq.Kernel, gate: M, circuit: Circuit, qubits: cudaq.QuakeValue) -> None:
        """
        Handle a measurement gate.

        Depending on whether the measurement targets all qubits or a subset,
        this method applies measurement operations accordingly.

        Args:
            kernel (cudaq.Kernel): The CUDA kernel being constructed.
            gate (M): The measurement gate.
            circuit (Circuit): The circuit containing the measurement gate.
            qubits (cudaq.QuakeValue): The allocated qubits for the circuit.
        """
        if gate.nqubits == circuit.nqubits:
            kernel.mz(qubits)
        else:
            for idx in gate.target_qubits:
                kernel.mz(qubits[idx])

    @staticmethod
    def _handle_X(kernel: cudaq.Kernel, gate: X, qubit: cudaq.QuakeValue) -> None:
        """Handle an X gate operation."""
        kernel.x(qubit)

    @staticmethod
    def _handle_Y(kernel: cudaq.Kernel, gate: Y, qubit: cudaq.QuakeValue) -> None:
        """Handle an Y gate operation."""
        kernel.y(qubit)

    @staticmethod
    def _handle_Z(kernel: cudaq.Kernel, gate: Z, qubit: cudaq.QuakeValue) -> None:
        """Handle an Z gate operation."""
        kernel.z(qubit)

    @staticmethod
    def _handle_H(kernel: cudaq.Kernel, gate: H, qubit: cudaq.QuakeValue) -> None:
        """Handle an H gate operation."""
        kernel.h(qubit)

    @staticmethod
    def _handle_S(kernel: cudaq.Kernel, gate: S, qubit: cudaq.QuakeValue) -> None:
        """Handle an S gate operation."""
        kernel.s(qubit)

    @staticmethod
    def _handle_T(kernel: cudaq.Kernel, gate: T, qubit: cudaq.QuakeValue) -> None:
        """Handle an T gate operation."""
        kernel.t(qubit)

    @staticmethod
    def _handle_RX(kernel: cudaq.Kernel, gate: RX, qubit: cudaq.QuakeValue) -> None:
        """Handle an RX gate operation."""
        kernel.rx(*gate.parameter_values, qubit)

    @staticmethod
    def _handle_RY(kernel: cudaq.Kernel, gate: RY, qubit: cudaq.QuakeValue) -> None:
        """Handle an RY gate operation."""
        kernel.ry(*gate.parameter_values, qubit)

    @staticmethod
    def _handle_RZ(kernel: cudaq.Kernel, gate: RZ, qubit: cudaq.QuakeValue) -> None:
        """Handle an RZ gate operation."""
        kernel.rz(*gate.parameter_values, qubit)

    @staticmethod
    def _handle_U1(kernel: cudaq.Kernel, gate: U1, qubit: cudaq.QuakeValue) -> None:
        """Handle an U1 gate operation."""
        kernel.u3(theta=0.0, phi=gate.phi, delta=0.0, target=qubit)

    @staticmethod
    def _handle_U2(kernel: cudaq.Kernel, gate: U2, qubit: cudaq.QuakeValue) -> None:
        """Handle an U2 gate operation."""
        kernel.u3(theta=np.pi / 2, phi=gate.phi, delta=gate.gamma, target=qubit)

    @staticmethod
    def _handle_U3(kernel: cudaq.Kernel, gate: U3, qubit: cudaq.QuakeValue) -> None:
        """Handle an U3 gate operation."""
        kernel.u3(theta=gate.theta, phi=gate.phi, delta=gate.gamma, target=qubit)
        kernel.u3(theta=gate.theta, phi=gate.phi, delta=gate.gamma, target=qubit)

    @staticmethod
    def _handle_PauliX(operator: PauliX) -> ElementaryOperator:
        return spin.x(target=operator.qubit)

    @staticmethod
    def _handle_PauliY(operator: PauliY) -> ElementaryOperator:
        return spin.y(target=operator.qubit)

    @staticmethod
    def _handle_PauliZ(operator: PauliZ) -> ElementaryOperator:
        return spin.z(target=operator.qubit)

    @staticmethod
    def _handle_PauliI(operator: PauliI) -> ElementaryOperator:
        return spin.i(target=operator.qubit)
