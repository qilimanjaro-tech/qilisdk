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

from collections import Counter
from typing import TYPE_CHECKING, Callable, Type, TypeVar

import numpy as np
import qutip_qip.operations as QutipGates
from loguru import logger
from qutip import Qobj, basis, mesolve, qeye, tensor
from qutip_qip.circuit import CircuitSimulator, QubitCircuit
from qutip_qip.operations.gateclass import SingleQubitGate, is_qutip5

from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliOperator
from qilisdk.backends.backend import Backend
from qilisdk.core.qtensor import QTensor, tensor_prod
from qilisdk.digital import RX, RY, RZ, SWAP, U1, U2, U3, Circuit, H, I, M, S, T, X, Y, Z
from qilisdk.digital.circuit_transpiler_passes import DecomposeMultiControlledGatesPass
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import Adjoint, BasicGate, Controlled
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

if TYPE_CHECKING:
    from qilisdk.functionals.sampling import Sampling
    from qilisdk.functionals.time_evolution import TimeEvolution


TBasicGate = TypeVar("TBasicGate", bound=BasicGate)
BasicGateHandlersMapping = dict[Type[TBasicGate], Callable[[QubitCircuit, TBasicGate, int], None]]

TPauliOperator = TypeVar("TPauliOperator", bound=PauliOperator)
PauliOperatorHandlersMapping = dict[Type[TPauliOperator], Callable[[TPauliOperator], Qobj]]


class QutipI(SingleQubitGate):
    """
    Single-qubit I gate.

    Examples
    --------
    >>> from qutip_qip.operations import X
    >>> I(0).get_compact_qobj()  # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0.]
     [0. 1.]]
    """

    def __init__(self, targets, **kwargs) -> None:  # noqa: ANN001, ANN003
        super().__init__(targets=targets, **kwargs)
        self.name = "I"
        self.latex_str = r"I"

    def get_compact_qobj(self):  # noqa: ANN201, PLR6301
        return qeye(2) if not is_qutip5 else qeye(2, dtype="dense")


class QutipBackend(Backend):
    """
    Backend that runs both digital-circuit sampling and analog
    time-evolution experiments using the **QuTiP** simulation library.

    The backend is CPU-only and has no hardware dependencies, which makes it
    ideal for local development, CI pipelines, and educational notebooks.
    """

    def __init__(self, nsteps: int = 10_000) -> None:
        """Instantiate a new :class:`QutipBackend`.
        Args:
            nsteps (int): The maximum number of internal steps for the ODE solver."""
        self.nsteps = nsteps

        super().__init__()
        self._basic_gate_handlers: BasicGateHandlersMapping = {
            I: QutipBackend._handle_I,
            X: QutipBackend._handle_X,
            Y: QutipBackend._handle_Y,
            Z: QutipBackend._handle_Z,
            H: QutipBackend._handle_H,
            S: QutipBackend._handle_S,
            T: QutipBackend._handle_T,
            RX: QutipBackend._handle_RX,
            RY: QutipBackend._handle_RY,
            RZ: QutipBackend._handle_RZ,
            U1: QutipBackend._handle_U1,
            U2: QutipBackend._handle_U2,
            U3: QutipBackend._handle_U3,
            SWAP: QutipBackend._handle_SWAP,  # type: ignore[dict-item]
        }
        logger.success("QutipBackend initialised")

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        """
        Execute a quantum circuit and return the measurement results.

        This method applies the selected simulation method, translates the circuit's gates into
        CUDA operations via their respective handlers, runs the simulation, and returns the result
        as a QutipDigitalResult.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            DigitalResult: A result object containing the measurement samples and computed probabilities.

        """
        logger.info("Executing Sampling (shots={})", functional.nshots)

        init_state = tensor(*[basis(2, 0) for _ in range(functional.circuit.nqubits)])

        measurements_set = set()
        for m in functional.circuit.gates:
            if isinstance(m, M):
                measurements_set.update(list(m.target_qubits))
        measurements = sorted(measurements_set)

        transpiled_circuit = DecomposeMultiControlledGatesPass().run(functional.circuit)
        qutip_circuit = self._get_qutip_circuit(transpiled_circuit)
        sim = CircuitSimulator(qutip_circuit)

        res = sim.run_statistics(init_state)  # runs the full circuit for one shot
        _bits = res.cbits  # classical measurement bits
        bits = []
        probs = res.probabilities

        if sum(probs) != 1:
            probs /= sum(probs)

        if len(measurements) > 0:
            for b in _bits:
                aux = []
                for i in measurements:
                    aux.append(b[i])
                bits.append(aux)
        else:
            bits = _bits

        bits_list = ["".join(map(str, cb)) for cb in bits]

        rng = np.random.default_rng()
        samples = rng.choice(bits_list, size=functional.nshots, p=probs)
        samples_py = map(str, samples)

        counts = Counter(samples_py)

        logger.success("Sampling finished; {} distinct bitstrings", len(counts))
        return SamplingResult(nshots=functional.nshots, samples=dict(counts))

    def _execute_time_evolution(self, functional: TimeEvolution) -> TimeEvolutionResult:
        """computes the time evolution under of an initial state under the given schedule.

        Args:
            functional (TimeEvolution): The TimeEvolution functional to execute.

        Returns:
            TimeEvolutionResult: The results of the evolution.

        Raises:
            ValueError: if the initial state provided is invalid.
        """
        logger.info("Executing TimeEvolution (T={}, dt={})", functional.schedule.T, functional.schedule.dt)
        steps = np.linspace(0, functional.schedule.T, int(functional.schedule.T // functional.schedule.dt))
        tlist = np.array(functional.schedule.tlist)
        steps = np.union1d(steps, tlist)

        qutip_hamiltonians = []
        for hamiltonian in functional.schedule.hamiltonians.values():
            qutip_hamiltonians.append(
                Qobj(
                    hamiltonian.to_matrix().toarray(), dims=[[2 for _ in range(hamiltonian.nqubits)] for _ in range(2)]
                )
            )

        H_t = [
            [
                qutip_hamiltonians[i],
                np.array([functional.schedule.coefficients[h][t] for t in tlist]),
            ]
            for i, h in enumerate(functional.schedule.hamiltonians)
        ]
        state_dim = []
        if functional.initial_state.is_density_matrix():
            state_dim = [[2 for _ in range(functional.initial_state.nqubits)] for _ in range(2)]
        elif functional.initial_state.is_bra():
            state_dim = [[1], [2 for _ in range(functional.initial_state.nqubits)]]
        elif functional.initial_state.is_ket():
            state_dim = [[2 for _ in range(functional.initial_state.nqubits)], [1]]
        else:
            logger.error("Invalid initial state provided")
            raise ValueError("invalid initial state provided.")

        qutip_init_state = Qobj(functional.initial_state.dense, dims=state_dim)

        qutip_obs: list[Qobj] = []

        identity = QTensor(PauliI(0).matrix)
        for obs in functional.observables:
            aux_obs = None
            if isinstance(obs, PauliOperator):
                for i in range(functional.schedule.nqubits):
                    if aux_obs is None:
                        aux_obs = identity if i != obs.qubit else QTensor(obs.matrix)
                    else:
                        aux_obs = (
                            tensor_prod([aux_obs, identity])
                            if i != obs.qubit
                            else tensor_prod([aux_obs, QTensor(obs.matrix)])
                        )
            elif isinstance(obs, Hamiltonian):
                aux_obs = QTensor(obs.to_matrix())
                if obs.nqubits < functional.schedule.nqubits:
                    for _ in range(functional.schedule.nqubits - obs.nqubits):
                        aux_obs = tensor_prod([aux_obs, identity])
            elif isinstance(obs, QTensor):
                aux_obs = obs
            else:
                logger.error("Unsupported observable type {}", obs.__class__.__name__)
                raise ValueError(f"unsupported observable type of {obs.__class__}")
            if aux_obs is not None:
                qutip_obs.append(
                    Qobj(aux_obs.dense, dims=[[2 for _ in range(functional.schedule.nqubits)] for _ in range(2)])
                )

        results = mesolve(
            H=H_t,
            e_ops=qutip_obs,
            rho0=qutip_init_state,
            tlist=tlist,
            options={
                "store_states": functional.store_intermediate_results,
                "store_final_state": True,
                "nsteps": self.nsteps,
            },
        )

        logger.success("TimeEvolution finished")
        return TimeEvolutionResult(
            final_expected_values=np.array([results.expect[i][-1] for i in range(len(qutip_obs))]),
            expected_values=(
                np.array(
                    [
                        [results.expect[val][i] for val in range(len(results.expect))]
                        for i in range(len(results.expect[0]))
                    ]
                )
                if len(results.expect) > 0 and functional.store_intermediate_results
                else None
            ),
            final_state=(QTensor(results.final_state.full()) if results.final_state is not None else None),
            intermediate_states=(
                [QTensor(state.full()) for state in results.states]
                if len(results.states) > 1 and functional.store_intermediate_results
                else None
            ),
        )

    def _get_qutip_circuit(self, circuit: Circuit) -> QubitCircuit:
        """_summary_

        Args:
            circuit (Circuit): the qiliSDK circuit to be translated to qutip.

        Raises:
            UnsupportedGateError: If the circuit contains a gate for which no handler is registered.

        Returns:
            QubitCircuit: the translated qutip circuit.
        """
        qutip_circuit = QubitCircuit(
            circuit.nqubits, num_cbits=circuit.nqubits, input_states=[0 for _ in range(circuit.nqubits)]
        )

        for gate in circuit.gates:
            if isinstance(gate, Controlled):
                self._handle_controlled(qutip_circuit, gate)
            elif isinstance(gate, Adjoint):
                self._handle_adjoint(qutip_circuit, gate)
            elif isinstance(gate, M):
                self._handle_M(qutip_circuit, gate)
            else:
                handler = self._basic_gate_handlers.get(type(gate), None)
                if handler is None:
                    logger.error("Unsupported gate {}", type(gate).__name__)
                    raise UnsupportedGateError(f"Unsupported gate {type(gate).__name__}")
                handler(qutip_circuit, gate, *(qubit for qubit in gate.target_qubits))

        no_measurement = True

        for g in circuit.gates:
            if isinstance(g, M):
                no_measurement = False

        if no_measurement:
            for i in range(circuit.nqubits):
                qutip_circuit.add_measurement(f"M{i}", targets=i, classical_store=i)
        return qutip_circuit

    def _handle_controlled(self, circuit: QubitCircuit, gate: Controlled) -> None:  # noqa: PLR6301
        """
        Handle a controlled gate operation by registering a custom QuTiP gate.

        For non-native controlled gates we construct the block-matrix explicitly, mirroring
        the approach recommended in the QuTiP QIP documentation for custom controlled rotations.

        Raises:
            UnsupportedGateError: If the number of control qubits is not equal to one or if the basic gate is unsupported.
        """
        if len(gate.control_qubits) != 1:
            logger.error("Controlled gate with {} control qubits not supported", len(gate.control_qubits))
            raise UnsupportedGateError

        if gate.name == "CNOT":
            circuit.add_gate("CNOT", targets=[*gate.target_qubits], controls=[*gate.control_qubits])
        else:
            base_matrix = gate.basic_gate.matrix
            dim_target = base_matrix.shape[0]
            dim_total = 2 * dim_target
            dims = [[2] + [2] * len(gate.target_qubits), [2] + [2] * len(gate.target_qubits)]

            def qutip_controlled_gate() -> Qobj:
                mat = np.zeros((dim_total, dim_total), dtype=np.complex128)
                mat[:dim_target, :dim_target] = np.eye(dim_target, dtype=np.complex128)
                mat[dim_target:, dim_target:] = base_matrix
                return Qobj(mat, dims=dims)

            matrix_digest = base_matrix.tobytes().hex()[:16]
            gate_name = f"{gate.name}_{matrix_digest}"
            if gate_name not in circuit.user_gates:
                circuit.user_gates[gate_name] = qutip_controlled_gate
            circuit.add_gate(gate_name, targets=[*gate.control_qubits, *gate.target_qubits])

    def _handle_adjoint(self, circuit: QubitCircuit, gate: Adjoint) -> None:  # noqa: PLR6301
        """
        Handle an adjoint (inverse) gate operation.

        This method creates a temporary kernel for the basic gate wrapped by the adjoint,
        applies the corresponding handler, and then integrates it into the main kernel as an adjoint operation.
        """

        def qutip_adjoined_gate() -> Qobj:
            return Qobj(gate.matrix)

        gate_name = "Adjoint_" + gate.name
        if gate_name not in circuit.user_gates:
            circuit.user_gates[gate_name] = qutip_adjoined_gate
        circuit.add_gate(gate_name, targets=[*gate.target_qubits])

    @staticmethod
    def _handle_M(qutip_circuit: QubitCircuit, gate: M) -> None:
        """
        Handle a measurement gate.

        Depending on whether the measurement targets all qubits or a subset,
        this method applies measurement operations accordingly.
        """
        for i in gate.target_qubits:
            qutip_circuit.add_measurement(f"M{i}", targets=[i], classical_store=i)

    @staticmethod
    def _handle_I(circuit: QubitCircuit, gate: I, qubit: int) -> None:
        """Handle an X gate operation."""
        circuit.add_gate(QutipI(targets=qubit))

    @staticmethod
    def _handle_X(circuit: QubitCircuit, gate: X, qubit: int) -> None:
        """Handle an X gate operation."""
        circuit.add_gate(QutipGates.X(targets=qubit))

    @staticmethod
    def _handle_Y(circuit: QubitCircuit, gate: Y, qubit: int) -> None:
        """Handle an Y gate operation."""
        circuit.add_gate(QutipGates.Y(targets=qubit))

    @staticmethod
    def _handle_Z(circuit: QubitCircuit, gate: Z, qubit: int) -> None:
        """Handle an Z gate operation."""
        circuit.add_gate(QutipGates.Z(targets=qubit))

    @staticmethod
    def _handle_H(circuit: QubitCircuit, gate: H, qubit: int) -> None:
        """Handle an H gate operation."""
        circuit.add_gate(QutipGates.H(targets=qubit))

    @staticmethod
    def _handle_S(circuit: QubitCircuit, gate: S, qubit: int) -> None:
        """Handle an S gate operation."""
        circuit.add_gate(QutipGates.S(targets=qubit))

    @staticmethod
    def _handle_T(circuit: QubitCircuit, gate: T, qubit: int) -> None:
        """Handle an T gate operation."""
        circuit.add_gate(QutipGates.T(targets=qubit))

    @staticmethod
    def _handle_RX(circuit: QubitCircuit, gate: RX, qubit: int) -> None:
        """Handle an RX gate operation."""
        circuit.add_gate(QutipGates.RX(targets=[qubit], arg_value=gate.get_parameter_values()[0]))

    @staticmethod
    def _handle_RY(circuit: QubitCircuit, gate: RY, qubit: int) -> None:
        """Handle an RY gate operation."""
        circuit.add_gate(QutipGates.RY(targets=[qubit], arg_value=gate.get_parameter_values()[0]))

    @staticmethod
    def _handle_RZ(circuit: QubitCircuit, gate: RZ, qubit: int) -> None:
        """Handle an RZ gate operation."""
        circuit.add_gate(QutipGates.RZ(targets=[qubit], arg_value=gate.get_parameter_values()[0]))

    @staticmethod
    def _qutip_U1(phi: float) -> Qobj:
        mat = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
        return Qobj(mat, dims=[[2], [2]])

    @staticmethod
    def _handle_U1(circuit: QubitCircuit, gate: U1, qubit: int) -> None:
        """Handle an U1 gate operation."""
        U1_label = "U1"

        if U1_label not in circuit.user_gates:
            circuit.user_gates[U1_label] = QutipBackend._qutip_U1
        circuit.add_gate(U1_label, targets=qubit, arg_value=gate.phi)

    @staticmethod
    def _qutip_U2(angles: list[float]) -> Qobj:
        phi = angles[0]
        gamma = angles[1]
        mat = (1 / np.sqrt(2)) * np.array(
            [
                [1, -np.exp(1j * gamma)],
                [np.exp(1j * phi), np.exp(1j * (phi + gamma))],
            ],
            dtype=complex,
        )
        return Qobj(mat, dims=[[2], [2]])

    @staticmethod
    def _handle_U2(circuit: QubitCircuit, gate: U2, qubit: int) -> None:
        """Handle an U2 gate operation."""
        U2_label = "U2"

        if U2_label not in circuit.user_gates:
            circuit.user_gates[U2_label] = QutipBackend._qutip_U2
        circuit.add_gate(U2_label, targets=qubit, arg_value=[gate.phi, gate.gamma])

    @staticmethod
    def _qutip_U3(angles: list[float]) -> Qobj:
        phi = angles[0]
        gamma = angles[1]
        theta = angles[2]
        mat = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * gamma) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + gamma)) * np.cos(theta / 2)],
            ],
            dtype=complex,
        )
        return Qobj(mat, dims=[[2], [2]])

    @staticmethod
    def _handle_U3(circuit: QubitCircuit, gate: U3, qubit: int) -> None:
        """Handle an U3 gate operation."""
        U3_label = "U3"

        if U3_label not in circuit.user_gates:
            circuit.user_gates[U3_label] = QutipBackend._qutip_U3
        circuit.add_gate(U3_label, targets=qubit, arg_value=[gate.phi, gate.gamma, gate.theta])

    @staticmethod
    def _handle_SWAP(circuit: QubitCircuit, gate: SWAP, qubit_0: int, qubit_1: int) -> None:
        """Handle a SWAP gate operation."""
        circuit.add_gate(QutipGates.SWAP(targets=[qubit_0, qubit_1]))
