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
from typing import TYPE_CHECKING, Callable, Type, TypeVar

import cudaq
import numpy as np
from cudaq import ElementaryOperator, OperatorSum, ScalarOperator, State, evolve, operators, spin
from cudaq import Schedule as CudaSchedule
from loguru import logger

from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliOperator, PauliX, PauliY, PauliZ
from qilisdk.backends.backend import Backend
from qilisdk.core.qtensor import QTensor
from qilisdk.digital.circuit_transpiler_passes import DecomposeMultiControlledGatesPass
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import (
    RX,
    RY,
    RZ,
    SWAP,
    U1,
    U2,
    U3,
    Adjoint,
    BasicGate,
    Controlled,
    Gate,
    H,
    I,
    M,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.noise import (
    BitFlip,
    Depolarizing,
    LindbladGenerator,
    Noise,
    PhaseFlip,
    ReadoutAssignment,
    SupportsStaticKraus,
    SupportsStaticLindblad,
    SupportsTimeDerivedKraus,
    SupportsTimeDerivedLindblad,
)

if TYPE_CHECKING:
    from qilisdk.analog.schedule import Schedule
    from qilisdk.digital.circuit import Circuit
    from qilisdk.functionals.sampling import Sampling
    from qilisdk.functionals.time_evolution import TimeEvolution
    from qilisdk.noise import NoiseModel


TBasicGate = TypeVar("TBasicGate", bound=BasicGate)
BasicGateHandlersMapping = dict[Type[TBasicGate], Callable[[cudaq.Kernel, TBasicGate, cudaq.QuakeValue], None]]

TPauliOperator = TypeVar("TPauliOperator", bound=PauliOperator)
PauliOperatorHandlersMapping = dict[Type[TPauliOperator], Callable[[TPauliOperator], ElementaryOperator]]


def _to_cuda_noise(noise: Noise) -> cudaq.NoiseChannel | None:
    if isinstance(noise, BitFlip):
        return cudaq.BitFlipChannel(noise.probability)
    if isinstance(noise, PhaseFlip):
        return cudaq.PhaseFlipChannel(noise.probability)
    if isinstance(noise, Depolarizing):
        return cudaq.DepolarizationChannel(noise.probability)
    if isinstance(noise, SupportsTimeDerivedKraus):
        kraus_channel = noise.as_kraus_from_duration(duration=1.0)
        kraus_operators_np = [np.array(operator.dense(), dtype=np.complex128) for operator in kraus_channel.operators]
        return cudaq.KrausChannel(kraus_operators_np)
    if isinstance(noise, SupportsStaticKraus):
        kraus_channel = noise.as_kraus()
        kraus_operators_np = [np.array(operator.dense(), dtype=np.complex128) for operator in kraus_channel.operators]
        return cudaq.KrausChannel(kraus_operators_np)
    return None


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
            sampling_method (CudaSamplingMethod, optional): The simulation method to use for sampling circuits.
                Options include STATE_VECTOR, TENSOR_NETWORK, or MATRIX_PRODUCT_STATE.
                Defaults to STATE_VECTOR.
        """
        super().__init__()
        cudaq.register_operation("i", np.array([1, 0, 0, 1]))
        self._basic_gate_handlers: BasicGateHandlersMapping = {
            I: CudaBackend._handle_I,
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
            SWAP: CudaBackend._handle_SWAP,  # type: ignore[dict-item]
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

    @staticmethod
    def _handle_readout_errors(cudaq_result: dict[str, int], noise_model: NoiseModel, nqubits: int) -> dict[str, int]:
        # Determine which qubits have readout assignment errors
        readout_error_per_qubits = [(0.0, 0.0) for _ in range(nqubits)]
        has_readout_error = False
        for noise in noise_model.global_noise:
            if isinstance(noise, ReadoutAssignment):
                for qubit in range(nqubits):
                    readout_error_per_qubits[qubit] = (noise.p01, noise.p10)
                    has_readout_error = True
        for qubit, noises in noise_model.per_qubit_noise.items():
            for noise in noises:
                if isinstance(noise, ReadoutAssignment):
                    readout_error_per_qubits[qubit] = (noise.p01, noise.p10)
                    has_readout_error = True

        if not has_readout_error:
            return cudaq_result

        # numpy generator
        gen = np.random.default_rng(42)

        # First split into individual shots
        shots = []
        for bitstring, count in cudaq_result.items():
            shots.extend([bitstring] * count)

        # Convert each shot according to the readout error probabilities
        adjusted_counts: dict[str, int] = {}
        for shot in shots:
            adjusted_shot = list(shot)
            for qubit_index in range(nqubits):
                p01, p10 = readout_error_per_qubits[qubit_index]
                if shot[nqubits - 1 - qubit_index] == "0" and gen.random() < p01:
                    adjusted_shot[nqubits - 1 - qubit_index] = "1"
                elif gen.random() < p10:
                    adjusted_shot[nqubits - 1 - qubit_index] = "0"
            adjusted_bitstring = "".join(adjusted_shot)
            adjusted_counts[adjusted_bitstring] = adjusted_counts.get(adjusted_bitstring, 0) + 1

        # Set the new results
        cudaq_result = adjusted_counts

        return cudaq_result

    @staticmethod
    def _add_global_noise(
        noise: Noise,
        cuda_noise_model: cudaq.NoiseModel,
        all_cuda_gate_names: set[str],
        nqubits: int,
    ) -> None:
        if cuda_noise := _to_cuda_noise(noise):
            for gate_name in all_cuda_gate_names:
                # If it's a full size kraus channel, special treatment
                if isinstance(noise, SupportsStaticKraus) and len(noise.as_kraus().operators) > 0:
                    dim = noise.as_kraus().operators[0].dense().shape[0]
                    if dim == 2**nqubits:
                        cuda_noise_model.add_channel(gate_name, list(range(nqubits)), cuda_noise)
                        continue

                # Otherwise, add normally
                cuda_noise_model.add_all_qubit_channel(gate_name, cuda_noise)

    @staticmethod
    def _add_per_gate_noise(
        gate_type: Type[BasicGate] | Type[Gate],
        noises: list[Noise],
        cuda_noise_model: cudaq.NoiseModel,
        all_cuda_gate_names: set[str],
    ) -> None:
        if (gate_name := gate_type.__name__.lower()) in all_cuda_gate_names:
            for noise in noises:
                if cuda_noise := _to_cuda_noise(noise):
                    cuda_noise_model.add_all_qubit_channel(gate_name, cuda_noise)

    @staticmethod
    def _add_per_qubit_noise(
        qubit: int,
        noises: list[Noise],
        cuda_noise_model: cudaq.NoiseModel,
        all_cuda_gate_names: set[str],
    ) -> None:
        for noise in noises:
            if cuda_noise := _to_cuda_noise(noise):
                for gate_name in all_cuda_gate_names:
                    cuda_noise_model.add_channel(gate_name, [qubit], cuda_noise)

    @staticmethod
    def _add_per_gate_per_qubit_noise(
        gate_type: Type[BasicGate] | Type[Gate],
        qubit: int,
        noises: list[Noise],
        cuda_noise_model: cudaq.NoiseModel,
        all_cuda_gate_names: set[str],
    ) -> None:
        if (gate_name := gate_type.__name__.lower()) in all_cuda_gate_names:
            for noise in noises:
                if cuda_noise := _to_cuda_noise(noise):
                    cuda_noise_model.add_channel(gate_name, [qubit], cuda_noise)

    def _noise_model_to_cudaq(self, noise_model: NoiseModel, nqubits: int) -> cudaq.NoiseModel:
        all_cuda_gate_names = {gate.__name__.lower() for gate in self._basic_gate_handlers} - {"u1", "u2", "swap"}
        cuda_noise_model = cudaq.NoiseModel()

        # Global noise
        for noise in noise_model.global_noise:
            self._add_global_noise(noise, cuda_noise_model, all_cuda_gate_names, nqubits)

        # Per gate noise
        for gate_type, noises in noise_model.per_gate_noise.items():
            self._add_per_gate_noise(gate_type, noises, cuda_noise_model, all_cuda_gate_names)

        # Per qubit noise
        for qubit, noises in noise_model.per_qubit_noise.items():
            self._add_per_qubit_noise(qubit, noises, cuda_noise_model, all_cuda_gate_names)

        # Per gate per qubit noise
        for (gate_type, qubit), noises in noise_model.per_gate_per_qubit_noise.items():
            self._add_per_gate_per_qubit_noise(gate_type, qubit, noises, cuda_noise_model, all_cuda_gate_names)

        return cuda_noise_model

    @staticmethod
    def _handle_gate_parameter_perturbations(circuit: Circuit, noise_model: NoiseModel) -> None:
        circuit_parameters = circuit.get_parameters()
        for parameter, perturbations in noise_model.global_perturbations.items():
            if parameter in circuit_parameters:
                for perturbation in perturbations:
                    circuit.set_parameters({parameter: perturbation.perturb(circuit_parameters[parameter])})
        for (gate_type, parameter), perturbations in noise_model.per_gate_perturbations.items():
            for gate in circuit.gates:
                if isinstance(gate, gate_type) and parameter in gate.get_parameter_names():
                    gate_parameters = gate.get_parameters()
                    for perturbation in perturbations:
                        gate.set_parameters({parameter: perturbation.perturb(gate_parameters[parameter])})

    def _execute_sampling(self, functional: Sampling, noise_model: NoiseModel | None = None) -> SamplingResult:
        logger.info("Executing Sampling (shots={})", functional.nshots)
        self._apply_digital_simulation_method()
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(functional.circuit.nqubits)

        # Apply parameter perturbations
        if noise_model:
            self._handle_gate_parameter_perturbations(functional.circuit, noise_model)

        # Transpile the circuit into CUDAQ format
        transpiled_circuit = DecomposeMultiControlledGatesPass().run(functional.circuit)
        for gate in transpiled_circuit.gates:
            if isinstance(gate, Controlled):
                self._handle_controlled(kernel, gate, qubits[gate.control_qubits[0]], qubits[gate.target_qubits[0]])
            elif isinstance(gate, Adjoint):
                self._handle_adjoint(kernel, gate, qubits[gate.target_qubits[0]])
            elif isinstance(gate, M):
                self._handle_M(kernel, gate, transpiled_circuit, qubits)
            else:
                handler = self._basic_gate_handlers.get(type(gate), None)
                if handler is None:
                    raise UnsupportedGateError(f"Unsupported gate {type(gate).__name__}")
                handler(kernel, gate, *(qubits[gate.target_qubits[i]] for i in range(len(gate.target_qubits))))

        if noise_model:
            cuda_noise_model = self._noise_model_to_cudaq(noise_model, functional.circuit.nqubits)
            cudaq_result = cudaq.sample(kernel, shots_count=functional.nshots, noise_model=cuda_noise_model)
            cudaq_result = self._handle_readout_errors(cudaq_result, noise_model, functional.circuit.nqubits)
        else:
            cudaq_result = cudaq.sample(kernel, shots_count=functional.nshots)

        logger.success("Sampling finished; {} distinct bitstrings", len(cudaq_result))
        return SamplingResult(nshots=functional.nshots, samples=dict(cudaq_result.items()))

    @staticmethod
    def _handle_schedule_parameter_perturbations(schedule: Schedule, noise_model: NoiseModel) -> None:
        if noise_model.global_perturbations:
            schedule_parameters = schedule.get_parameters()
            for parameter, perturbations in noise_model.global_perturbations.items():
                if parameter in schedule_parameters:
                    for perturbation in perturbations:
                        schedule.set_parameters({parameter: perturbation.perturb(schedule_parameters[parameter])})

    @staticmethod
    def _add_global_noise_dynamics(
        ops_numpy: list,
        jump_operators: list[OperatorSum],
        hamiltonian_deltas: list[OperatorSum],
        lindblad_generator: LindbladGenerator,
        nqubits: int,
    ) -> None:
        for i, operator in enumerate(lindblad_generator.jump_operators_with_rates):
            op_id = f"jump_op_{i}"
            ops_numpy.append(np.array(operator.dense(), dtype=np.complex128))
            operators.define(
                id=op_id,
                expected_dimensions=[2 for _ in range(nqubits)],
                create=lambda op_np=ops_numpy[-1]: op_np,
                override=True,
            )
            dim = ops_numpy[-1].shape[0]
            if dim == 2**nqubits:
                jump_operators.append(operators.instantiate(op_id, degrees=list(range(nqubits))))
            else:
                for qubit in range(nqubits):
                    jump_operators.append(operators.instantiate(op_id, degrees=qubit))
        if lindblad_generator.hamiltonian is not None:
            hamiltonian_deltas.append(CudaBackend._remove_constant_terms(lindblad_generator.hamiltonian))

    @staticmethod
    def _add_per_qubit_noise_dynamics(
        ops_numpy: list,
        jump_operators: list[OperatorSum],
        hamiltonian_deltas: list[OperatorSum],
        lindblad_generator: LindbladGenerator,
        qubit: int,
    ) -> None:
        for i, operator in enumerate(lindblad_generator.jump_operators_with_rates):
            op_id = f"jump_op_q{qubit}_{i}"
            ops_numpy.append(np.array(operator.dense(), dtype=np.complex128))
            operators.define(
                id=op_id,
                expected_dimensions=[ops_numpy[-1].shape[0]],
                create=lambda op_np=ops_numpy[-1]: op_np,
                override=True,
            )
            jump_operators.append(operators.instantiate(op_id, degrees=qubit))
        if lindblad_generator.hamiltonian is not None:
            hamiltonian_deltas.append(CudaBackend._remove_constant_terms(lindblad_generator.hamiltonian))

    def _noise_model_to_cudaq_dynamics(
        self, noise_model: NoiseModel, nqubits: int, dt: float
    ) -> tuple[list[OperatorSum], list]:
        ops_numpy: list[np.ndarray] = []
        jump_operators: list[OperatorSum] = []
        hamiltonian_deltas: list[OperatorSum] = []

        # Global noise
        for noise in noise_model.global_noise:
            if isinstance(noise, SupportsStaticLindblad):
                lindblad_generator = noise.as_lindblad()
                self._add_global_noise_dynamics(
                    ops_numpy, jump_operators, hamiltonian_deltas, lindblad_generator, nqubits
                )
            elif isinstance(noise, SupportsTimeDerivedLindblad):
                lindblad_generator = noise.as_lindblad_from_duration(duration=dt)
                self._add_global_noise_dynamics(
                    ops_numpy, jump_operators, hamiltonian_deltas, lindblad_generator, nqubits
                )

        # Per qubit noise
        for qubit, noises in noise_model.per_qubit_noise.items():
            for noise in noises:
                if isinstance(noise, SupportsStaticLindblad):
                    lindblad_generator = noise.as_lindblad()
                    self._add_per_qubit_noise_dynamics(
                        ops_numpy, jump_operators, hamiltonian_deltas, lindblad_generator, qubit
                    )
                elif isinstance(noise, SupportsTimeDerivedLindblad):
                    lindblad_generator = noise.as_lindblad_from_duration(duration=dt)
                    self._add_per_qubit_noise_dynamics(
                        ops_numpy, jump_operators, hamiltonian_deltas, lindblad_generator, qubit
                    )

        return jump_operators, hamiltonian_deltas

    @staticmethod
    def _remove_constant_terms(operator_sum: OperatorSum) -> OperatorSum:
        new_operator_sum = None
        for term in operator_sum:
            if not term.is_identity():
                if new_operator_sum is None:
                    new_operator_sum = term
                else:
                    new_operator_sum += term
        if new_operator_sum is None:
            new_operator_sum = ScalarOperator(0.0)
        return new_operator_sum

    def _execute_time_evolution(
        self, functional: TimeEvolution, noise_model: NoiseModel | None = None
    ) -> TimeEvolutionResult:
        logger.info("Executing TimeEvolution (T={}, dt={})", functional.schedule.T, functional.schedule.dt)
        cudaq.set_target("dynamics")

        # Apply parameter perturbations
        if noise_model and noise_model.global_perturbations:
            self._handle_schedule_parameter_perturbations(functional.schedule, noise_model)

        steps = functional.schedule.tlist

        cuda_schedule = CudaSchedule(steps, ["t"])

        def get_schedule(key: str) -> Callable[[complex], float]:
            return lambda t: (functional.schedule.coefficients[key][t.real])

        cuda_hamiltonian = sum(
            ScalarOperator(get_schedule(key)) * self._hamiltonian_to_cuda(ham)
            for key, ham in functional.schedule.hamiltonians.items()
        )

        logger.trace("Hamiltonian compiled for evolution")

        cuda_observables = []
        for observable in functional.observables:
            if isinstance(observable, PauliOperator):
                cuda_observables.append(self._pauli_operator_handlers[type(observable)](observable))
            elif isinstance(observable, Hamiltonian):
                cuda_observables.append(self._hamiltonian_to_cuda(observable))
            else:
                logger.error("Unsupported observable type {}", observable.__class__.__name__)
                raise ValueError(f"unsupported observable type of {observable.__class__}")
        logger.trace("Observables compiled for evolution")

        # Add noise
        jump_operators: list[OperatorSum] = []
        hamiltonian_deltas: list[OperatorSum] = []
        if noise_model:
            jump_operators, hamiltonian_deltas = self._noise_model_to_cudaq_dynamics(
                noise_model, functional.schedule.nqubits, functional.schedule.dt
            )

        # Remove any constant terms from the Hamiltonian, also add the deltas
        cuda_hamiltonian = self._remove_constant_terms(cuda_hamiltonian)
        for delta in hamiltonian_deltas:
            cuda_hamiltonian += delta

        evolution_result = evolve(
            hamiltonian=cuda_hamiltonian,
            dimensions=dict.fromkeys(range(functional.schedule.nqubits), 2),
            schedule=cuda_schedule,
            initial_state=State.from_data(np.array(functional.initial_state.unit().dense(), dtype=np.complex128)),
            observables=cuda_observables,
            collapse_operators=jump_operators,
            store_intermediate_results=functional.store_intermediate_results,
        )

        logger.success("TimeEvolution finished")

        final_expected_values = np.array(
            [exp_val.expectation() for exp_val in evolution_result.final_expectation_values()]
        )
        expected_values = (
            np.array([[val.expectation() for val in exp_vals] for exp_vals in evolution_result.expectation_values()])
            if evolution_result.expectation_values() is not None and functional.store_intermediate_results
            else None
        )
        final_state = (
            QTensor(np.array(evolution_result.final_state()).reshape(-1, 1))
            if evolution_result.final_state() is not None
            else None
        )
        intermediate_states = (
            [QTensor(np.array(state).reshape(-1, 1)) for state in evolution_result.intermediate_states()]
            if evolution_result.intermediate_states() is not None and functional.store_intermediate_results
            else None
        )

        return TimeEvolutionResult(
            final_expected_values=final_expected_values,
            expected_values=expected_values,
            final_state=final_state,
            intermediate_states=intermediate_states,
        )

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
    def _handle_I(kernel: cudaq.Kernel, gate: I, qubit: cudaq.QuakeValue) -> None:
        """Handle an X gate operation."""
        kernel.i(qubit)

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
        kernel.rx(*[float(param) for param in gate.get_parameter_values()], qubit)

    @staticmethod
    def _handle_RY(kernel: cudaq.Kernel, gate: RY, qubit: cudaq.QuakeValue) -> None:
        """Handle an RY gate operation."""
        kernel.ry(*[float(param) for param in gate.get_parameter_values()], qubit)

    @staticmethod
    def _handle_RZ(kernel: cudaq.Kernel, gate: RZ, qubit: cudaq.QuakeValue) -> None:
        """Handle an RZ gate operation."""
        kernel.rz(*[float(param) for param in gate.get_parameter_values()], qubit)

    @staticmethod
    def _handle_U1(kernel: cudaq.Kernel, gate: U1, qubit: cudaq.QuakeValue) -> None:
        """Handle an U1 gate operation."""
        kernel.u3(theta=0.0, phi=float(gate.phi), delta=0.0, target=qubit)

    @staticmethod
    def _handle_U2(kernel: cudaq.Kernel, gate: U2, qubit: cudaq.QuakeValue) -> None:
        """Handle an U2 gate operation."""
        kernel.u3(theta=np.pi / 2, phi=float(gate.phi), delta=float(gate.gamma), target=qubit)

    @staticmethod
    def _handle_U3(kernel: cudaq.Kernel, gate: U3, qubit: cudaq.QuakeValue) -> None:
        """Handle an U3 gate operation."""
        kernel.u3(theta=float(gate.theta), phi=float(gate.phi), delta=float(gate.gamma), target=qubit)

    @staticmethod
    def _handle_SWAP(kernel: cudaq.Kernel, gate: SWAP, qubit_0: cudaq.QuakeValue, qubit_1: cudaq.QuakeValue) -> None:
        kernel.swap(qubit_0, qubit_1)

    def _hamiltonian_to_cuda(self, hamiltonian: Hamiltonian, padding: int = 0) -> OperatorSum:  # type: ignore
        out = None
        for offset, terms in hamiltonian:
            if out is None:
                out = offset * np.prod([self._pauli_operator_handlers[type(pauli)](pauli) for pauli in terms])
            else:
                out += offset * np.prod([self._pauli_operator_handlers[type(pauli)](pauli) for pauli in terms])
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
