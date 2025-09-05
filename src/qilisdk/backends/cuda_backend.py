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
from itertools import product
from typing import TYPE_CHECKING, Callable, Type, TypeVar

import cudaq
import numpy as np
from cudaq import ElementaryOperator, OperatorSum, ScalarOperator, State, evolve, spin
from cudaq import Schedule as CudaSchedule
from loguru import logger

from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliOperator, PauliX, PauliY, PauliZ
from qilisdk.backends.backend import Backend
from qilisdk.common.qtensor import QTensor, ket, tensor_prod
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import RX, RY, RZ, U1, U2, U3, Adjoint, BasicGate, Controlled, H, M, S, T, X, Y, Z
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.state_tomography_result import StateTomographyResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

if TYPE_CHECKING:
    from qilisdk.digital.circuit import Circuit
    from qilisdk.functionals.sampling import Sampling
    from qilisdk.functionals.state_tomography import StateTomography
    from qilisdk.functionals.time_evolution import TimeEvolution


TBasicGate = TypeVar("TBasicGate", bound=BasicGate)
BasicGateHandlersMapping = dict[Type[TBasicGate], Callable[[cudaq.Kernel, TBasicGate, cudaq.QuakeValue], None]]

TPauliOperator = TypeVar("TPauliOperator", bound=PauliOperator)
PauliOperatorHandlersMapping = dict[Type[TPauliOperator], Callable[[TPauliOperator], ElementaryOperator]]


class CudaSamplingMethod(str, Enum):
    """
    Enumeration of available simulation methods for the CUDA backend.
    """

    STATE_VECTOR = "state_vector"
    TENSOR_NETWORK = "tensor_network"
    MATRIX_PRODUCT_STATE = "matrix_product_state"


class CudaBackend(Backend):
    """
    Backend implementation using CUDA-based simulation.

    This backend translates a quantum circuit into a CUDA-compatible kernel and executes it
    using the cudaq library. It supports different simulation methods including state vector,
    tensor network, and matrix product state simulations. Gate operations in the circuit are
    mapped to CUDA operations via dedicated handler functions.
    """

    def __init__(self, sampling_method: CudaSamplingMethod = CudaSamplingMethod.STATE_VECTOR) -> None:
        """
        Initialize the CudaBackend.

        Args:
            sampling_simulation_method (CudaSamplingMethod, optional): The simulation method to use for sampling circuits.
                Options include STATE_VECTOR, TENSOR_NETWORK, or MATRIX_PRODUCT_STATE.
                Defaults to STATE_VECTOR.
        """
        super().__init__()
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
        self._sampling_method = sampling_method
        logger.success("CudaBackend initialised (sampling_method={})", sampling_method.value)

    @property
    def sampling_method(self) -> CudaSamplingMethod:
        """
        Get the simulation method currently configured for the backend.

        Returns:
            SimulationMethod: The simulation method to be used for circuit execution.
        """
        return self._sampling_method

    def _apply_digital_simulation_method(self) -> None:
        """
        Configure the cudaq simulation target based on the selected simulation method.

        For the STATE_VECTOR method, it checks for GPU availability and selects an appropriate target.
        For TENSOR_NETWORK and MATRIX_PRODUCT_STATE methods, it explicitly sets the target to use tensor network-based simulations.
        """
        logger.info("Applying sampling simulation method {}", self.sampling_method.value)
        if self.sampling_method == CudaSamplingMethod.STATE_VECTOR:
            if cudaq.num_available_gpus() == 0:
                cudaq.set_target("qpp-cpu")
                logger.debug("No GPU detected, using cudaq's 'qpp-cpu' backend")
            else:
                cudaq.set_target("nvidia")
                logger.debug("GPU detected, using cudaq's 'nvidia' backend")
        elif self.sampling_method == CudaSamplingMethod.TENSOR_NETWORK:
            cudaq.set_target("tensornet")
            logger.debug("Using cudaq's 'tensornet' backend")
        else:
            cudaq.set_target("tensornet-mps")
            logger.debug("Using cudaq's 'tensornet-mps' backend")

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        logger.info("Executing Sampling (shots={})", functional.nshots)
        self._apply_digital_simulation_method()
        kernel = self._get_cuda_kernel(functional.circuit)

        cudaq_result = cudaq.sample(kernel, shots_count=functional.nshots)
        logger.success("Sampling finished; {} distinct bitstrings", len(cudaq_result))
        return SamplingResult(nshots=functional.nshots, samples=dict(cudaq_result.items()))

    def _execute_time_evolution(self, functional: TimeEvolution) -> TimeEvolutionResult:
        logger.info("Executing TimeEvolution (T={}, dt={})", functional.schedule.T, functional.schedule.dt)
        cudaq.set_target("dynamics")

        steps = np.linspace(0, functional.schedule.T, int(functional.schedule.T / functional.schedule.dt))

        cuda_schedule = CudaSchedule(steps, ["t"])

        def get_schedule(key: str) -> Callable:
            return lambda t: (functional.schedule.get_coefficient(t.real, key))

        cuda_hamiltonian = sum(
            (
                ScalarOperator(get_schedule(key)) * self._hamiltonian_to_cuda(ham)
                for key, ham in functional.schedule.hamiltonians.items()
            ),
            cudaq.SpinOperator.empty(),
        )

        logger.trace("Hamiltonian compiled for evolution")

        cuda_observables = []
        for observable in functional.observables:
            if isinstance(observable, PauliOperator):
                cuda_observables.append(self._pauli_operator_handlers[type(observable)](observable))
            elif isinstance(observable, Hamiltonian):
                cuda_observables.append(
                    self._hamiltonian_to_cuda(observable, padding=functional.schedule.nqubits - observable.nqubits)
                )
            else:
                logger.error("Unsupported observable type {}", observable.__class__.__name__)
                raise ValueError(f"unsupported observable type of {observable.__class__}")
        logger.trace("Observables compiled for evolution")
        evolution_result = evolve(
            hamiltonian=cuda_hamiltonian,
            dimensions=dict.fromkeys(range(functional.schedule.nqubits), 2),
            schedule=cuda_schedule,
            initial_state=State.from_data(np.array(functional.initial_state.dense, dtype=np.complex128)),
            observables=cuda_observables,
            collapse_operators=[],
            store_intermediate_results=functional.store_intermediate_results,
        )

        logger.success("TimeEvolution finished")
        return TimeEvolutionResult(
            final_expected_values=np.array(
                [exp_val.expectation() for exp_val in evolution_result.final_expectation_values()]
            ),
            expected_values=(
                np.array(
                    [[val.expectation() for val in exp_vals] for exp_vals in evolution_result.expectation_values()]
                )
                if evolution_result.expectation_values() is not None and functional.store_intermediate_results
                else None
            ),
            final_state=(
                QTensor(np.array(evolution_result.final_state())).adjoint()
                if evolution_result.final_state() is not None
                else None
            ),
            intermediate_states=(
                [QTensor(np.array(state)).adjoint() for state in evolution_result.intermediate_states()]
                if evolution_result.intermediate_states() is not None and functional.store_intermediate_results
                else None
            ),
        )

    def _execute_state_tomography(
        self, functional: StateTomography, initial_state: QTensor | None = None, tol: float = -1e12
    ) -> StateTomographyResult:
        """execute a state tomography protocol.

        Args:
            functional (StateTomography): the state tomography functional.
            initial_state (QTensor | None, optional): the initial state of the circuit. Defaults to None.
                NOTE: this is only used for the quantum reservoir and is not a public attribute.
            tol (float, optional): the tolerance when computing if the state is positive semi-definite. Defaults to -1e12.
                TODO (ameer): Move this somewhere else -> will do that in the reservoir pr.

        Returns:
            StateTomographyResult: the final results of the state tomography
        """
        logger.info("Executing state tomography")
        self._apply_digital_simulation_method()
        kernel = self._get_cuda_kernel(functional.circuit, initial_state, ignore_measurement=True)

        operators: list[tuple[type[PauliOperator], ...]] = list(
            product([PauliI, PauliZ, PauliX, PauliY], repeat=functional.circuit.nqubits)
        )
        _identity_tuple: tuple[type[PauliOperator], ...] = tuple(PauliI for _ in range(functional.circuit.nqubits))
        operators.remove(_identity_tuple)

        def pauli_spin_operator(label: tuple[type[PauliOperator], ...]) -> cudaq.SpinOperatorTerm:
            # Build a cudaq spin operator like X(0)*Y(1)*Z(2)...
            ops = {PauliI: spin.i, PauliX: spin.x, PauliY: spin.y, PauliZ: spin.z}

            term = ops[label[0]](0)
            for idx, ch in enumerate(label[1:], start=1):
                term *= ops[ch](idx)
            return term

        # Build list of spin operators once
        spin_ops = [pauli_spin_operator(op) for op in operators]
        # 2) Get expectations in one call (broadcasting list of SpinOperators)
        res_list = cudaq.observe(kernel, spin_ops)
        # res_list is a list of ObserveResult; pull expectations
        exps = np.array([r.expectation() for r in res_list], dtype=float)

        # 3) Assemble rho = (1/2^n) sum_P <P> P
        def pauli_string_matrix(label: tuple[type[PauliOperator], ...]) -> np.ndarray:
            mats = [ch._MATRIX for ch in label]  # noqa: SLF001
            out = mats[0]
            for m in mats[1:]:
                out = np.kron(out, m)
            return out

        dim = 2**functional.circuit.nqubits
        rho = np.identity(dim, dtype=complex)
        for op, val in zip(operators, exps):
            aux = val * pauli_string_matrix(op)
            rho += aux
        rho /= dim

        # Hermitize & renormalize
        rho = 0.5 * (rho + rho.conj().T)
        rho /= np.trace(rho)
        rho = 0.5 * (rho + rho.conj().T)
        rho /= np.trace(rho)

        eigvals, eigvecs = np.linalg.eigh(rho)
        if any(e < tol for e in eigvals):
            eigvals_clipped = np.clip(eigvals, 0, None)
            rho = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T
            rho /= np.trace(rho)

        logger.success("Sampling finished; ")
        return StateTomographyResult(state=QTensor(rho))

    def _get_cuda_kernel(
        self, circuit: Circuit, initial_state: QTensor | None = None, ignore_measurement: bool = False
    ) -> cudaq.Kernel:
        kernel = cudaq.make_kernel()

        if initial_state is None:
            initial_state = tensor_prod([ket(0)] * circuit.nqubits)
        state = State.from_data(np.array(initial_state.dense, dtype=cudaq.complex()))
        qubits = kernel.qalloc(state)

        for gate in circuit.gates:
            if isinstance(gate, Controlled):
                self._handle_controlled(kernel, gate, qubits[gate.control_qubits[0]], qubits[gate.target_qubits[0]])
            elif isinstance(gate, Adjoint):
                self._handle_adjoint(kernel, gate, qubits[gate.target_qubits[0]])
            elif isinstance(gate, M):
                if not ignore_measurement:
                    self._handle_M(kernel, gate, circuit, qubits)
            else:
                handler = self._basic_gate_handlers.get(type(gate), None)
                if handler is None:
                    raise UnsupportedGateError
                handler(kernel, gate, qubits[gate.target_qubits[0]])
        return kernel

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
            logger.error("Controlled gate with {} control qubits not supported", len(gate.control_qubits))
            raise UnsupportedGateError
        target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
        handler = self._basic_gate_handlers.get(type(gate.basic_gate), None)
        if handler is None:
            logger.error("Unsupported gate inside Controlled: {}", type(gate.basic_gate).__name__)
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
            logger.error("Unsupported gate inside Adjoint: {}", type(gate.basic_gate).__name__)
            raise UnsupportedGateError
        handler(target_kernel, gate.basic_gate, qubit)
        kernel.adjoint(target_kernel, target_qubit)

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
        kernel.rx(*gate.get_parameter_values(), qubit)

    @staticmethod
    def _handle_RY(kernel: cudaq.Kernel, gate: RY, qubit: cudaq.QuakeValue) -> None:
        """Handle an RY gate operation."""
        kernel.ry(*gate.get_parameter_values(), qubit)

    @staticmethod
    def _handle_RZ(kernel: cudaq.Kernel, gate: RZ, qubit: cudaq.QuakeValue) -> None:
        """Handle an RZ gate operation."""
        kernel.rz(*gate.get_parameter_values(), qubit)

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

    def _hamiltonian_to_cuda(self, hamiltonian: Hamiltonian, padding: int = 0) -> OperatorSum:  # type: ignore
        out = None
        for offset, terms in hamiltonian:
            if out is None:
                out = offset * np.prod([self._pauli_operator_handlers[type(pauli)](pauli) for pauli in terms])
            else:
                out += offset * np.prod([self._pauli_operator_handlers[type(pauli)](pauli) for pauli in terms])
        if padding > 0:
            n = hamiltonian.nqubits
            for p in range(padding):
                out += spin.i(n + p)
        return out

    @staticmethod
    def _handle_PauliX(operator: PauliX) -> ElementaryOperator:  # type: ignore
        return spin.x(target=operator.qubit)

    @staticmethod
    def _handle_PauliY(operator: PauliY) -> ElementaryOperator:  # type: ignore
        return spin.y(target=operator.qubit)

    @staticmethod
    def _handle_PauliZ(operator: PauliZ) -> ElementaryOperator:  # type: ignore
        return spin.z(target=operator.qubit)

    @staticmethod
    def _handle_PauliI(operator: PauliI) -> ElementaryOperator:  # type: ignore
        return spin.i(target=operator.qubit)
