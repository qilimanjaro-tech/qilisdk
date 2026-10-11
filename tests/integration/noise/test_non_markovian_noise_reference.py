# Copyright 2026 Qilimanjaro Quantum Tech
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

"""Digital propagation with non-Markovian noise, checked against an independent NumPy Lindblad solver.

The reference writes the environment out explicitly: every gate becomes its generator on the system,
the coupling and environment Hamiltonian are added on the enlarged register, and the state is evolved
with the exact exponential of the Lindblad superoperator before the environment is traced out.
"""

from functools import reduce

import numpy as np
import pytest
from scipy.linalg import expm, schur

from qilisdk.analog import X as PauliX
from qilisdk.analog import Y as PauliY
from qilisdk.analog import Z as PauliZ
from qilisdk.backends import QiliSim
from qilisdk.backends.backend_config import ExecutionConfig
from qilisdk.core import ket
from qilisdk.digital import CNOT, CZ, RX, RY, RZ, SWAP, Circuit, H, X, Y, Z
from qilisdk.functionals import DigitalPropagation
from qilisdk.noise import AmplitudeDamping, Dephasing, EnvironmentNoise, NoiseModel
from qilisdk.readout import Readout

EXECUTION_CONFIG = ExecutionConfig(seed=42, num_threads=1)
PLUS = (ket(0) + ket(1)).unit()
SIGMA = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _on(nqubits: int, operators: dict[int, np.ndarray]) -> np.ndarray:
    return reduce(np.kron, [operators.get(q, SIGMA["I"]) for q in range(nqubits)])


def _generator(unitary: np.ndarray, duration: float) -> np.ndarray:
    # Same convention as QiliSim: eigenphases in (-pi, pi]
    upper, vectors = schur(unitary, output="complex")
    phases = np.angle(np.diag(upper))
    phases[phases <= -np.pi + 1e-9] += 2 * np.pi
    return -vectors @ np.diag(phases) @ vectors.conj().T / duration


def _evolve(rho: np.ndarray, hamiltonian: np.ndarray, jumps: list[np.ndarray], time: float) -> np.ndarray:
    # Column-stacking vectorisation, vec(A X B) = (B^T (x) A) vec(X)
    dim = rho.shape[0]
    eye = np.eye(dim)
    superoperator = -1j * (np.kron(eye, hamiltonian) - np.kron(hamiltonian.T, eye))
    for jump in jumps:
        jdj = jump.conj().T @ jump
        superoperator += np.kron(jump.conj(), jump) - 0.5 * (np.kron(eye, jdj) + np.kron(jdj.T, eye))
    return (expm(superoperator * time) @ rho.reshape(-1, order="F")).reshape(dim, dim, order="F")


def _trace_out_environment(rho: np.ndarray, n_system: int) -> np.ndarray:
    dim_system = 2**n_system
    dim_environment = rho.shape[0] // dim_system
    return np.einsum("ikjk->ij", rho.reshape(dim_system, dim_environment, dim_system, dim_environment))


def _reference(steps, rho_0, coupling, jumps, n_system):
    n_environment = int(np.log2(coupling.shape[0])) - n_system
    rho = rho_0
    for unitary, duration in steps:
        hamiltonian = np.kron(_generator(unitary, duration), np.eye(2**n_environment)) + coupling
        rho = _evolve(rho, hamiltonian, jumps, duration)
    return _trace_out_environment(rho, n_system)


def _local_jump(noise) -> np.ndarray:
    return noise.as_lindblad().jump_operators_with_rates[0].dense()


def _simulate(circuit: Circuit, noise_model: NoiseModel) -> np.ndarray:
    state = (
        QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
        .execute(DigitalPropagation(circuit), readout=Readout().with_state_tomography())
        .get_state()
    )
    dense = state.dense()
    return dense @ dense.conj().T if state.is_ket() else dense


def _gate_steps(circuit: Circuit, noise_model: NoiseModel) -> list[tuple[np.ndarray, float]]:
    steps = []
    for gate in circuit.gates:
        qubits = [*gate.control_qubits, *gate.target_qubits]
        local = gate.matrix
        if qubits == list(range(circuit.nqubits)):
            full = local
        else:
            full = _on(circuit.nqubits, {qubits[0]: local}) if len(qubits) == 1 else None
        assert full is not None, "the reference only places single-qubit gates or gates on the whole register"
        steps.append((full, noise_model.noise_config.get_gate_time(type(gate))))
    return steps


def test_single_qubit_circuit_with_driven_damped_environment():
    circuit = Circuit(nqubits=1)
    for gate in [H(0), RX(0, theta=0.7), Z(0), RY(0, theta=1.1), RZ(0, phi=-0.4)]:
        circuit.add(gate)
    environment = EnvironmentNoise(
        n_environment_qubits=1,
        couplings=[(0.9, PauliX(0), PauliZ(0))],
        environment_hamiltonian=0.6 * PauliX(0),
        environment_noise={0: [AmplitudeDamping(t1=1.5)]},
        environment_state=ket(1),
    )
    noise_model = NoiseModel()
    noise_model.add(environment)
    noise_model.add(Dephasing(t_phi=2.0))
    for gate_type, time in [(H, 0.3), (RX, 0.5), (Z, 0.2), (RY, 0.4), (RZ, 0.25)]:
        noise_model.noise_config.set_gate_time(gate_type, time)

    coupling = 0.9 * _on(2, {0: SIGMA["X"], 1: SIGMA["Z"]}) + 0.6 * _on(2, {1: SIGMA["X"]})
    jumps = [
        _on(2, {1: _local_jump(AmplitudeDamping(t1=1.5))}),
        _on(2, {0: _local_jump(Dephasing(t_phi=2.0))}),
    ]
    rho_0 = np.kron(np.diag([1, 0]), np.diag([0, 1])).astype(complex)
    expected = _reference(_gate_steps(circuit, noise_model), rho_0, coupling, jumps, n_system=1)

    np.testing.assert_allclose(_simulate(circuit, noise_model), expected, atol=1e-6)


@pytest.mark.parametrize("two_qubit_gate", [CNOT(0, 1), CZ(0, 1), SWAP(0, 1)])
def test_two_qubit_circuit_with_environment_on_one_qubit(two_qubit_gate):
    circuit = Circuit(nqubits=2)
    for gate in [H(0), RY(1, theta=0.3), two_qubit_gate, X(0), RX(1, theta=-0.8)]:
        circuit.add(gate)
    environment = EnvironmentNoise(
        n_environment_qubits=1,
        couplings=[(0.8, PauliZ(1), PauliX(0))],
        environment_noise={0: [Dephasing(t_phi=1.0)]},
        environment_state=PLUS,
    )
    noise_model = NoiseModel()
    noise_model.add(environment)
    noise_model.add(AmplitudeDamping(t1=3.0), qubits=[0])
    noise_model.noise_config.set_default_gate_time(0.4)
    noise_model.noise_config.set_gate_time(type(two_qubit_gate), 0.9)

    coupling = 0.8 * _on(3, {1: SIGMA["Z"], 2: SIGMA["X"]})
    jumps = [
        _on(3, {2: _local_jump(Dephasing(t_phi=1.0))}),
        _on(3, {0: _local_jump(AmplitudeDamping(t1=3.0))}),
    ]
    plus = PLUS.dense()
    rho_0 = np.kron(np.diag([1, 0, 0, 0]), plus @ plus.conj().T).astype(complex)
    expected = _reference(_gate_steps(circuit, noise_model), rho_0, coupling, jumps, n_system=2)

    np.testing.assert_allclose(_simulate(circuit, noise_model), expected, atol=1e-6)


def test_two_environments_with_internal_dynamics():
    circuit = Circuit(nqubits=1)
    for gate in [H(0), Y(0), RX(0, theta=1.3)]:
        circuit.add(gate)
    noise_model = NoiseModel()
    noise_model.add(
        EnvironmentNoise(
            n_environment_qubits=1,
            couplings=[(0.5, PauliZ(0), PauliX(0))],
            environment_noise={0: [AmplitudeDamping(t1=0.8)]},
        )
    )
    noise_model.add(
        EnvironmentNoise(
            n_environment_qubits=2,
            couplings=[(0.7, PauliX(0), PauliZ(1))],
            environment_hamiltonian=0.4 * PauliX(0) * PauliX(1) + 0.2 * PauliZ(0),
            environment_state=ket(1, 0),
        )
    )
    noise_model.noise_config.set_default_gate_time(0.6)

    # Register: system 0, first environment 1, second environment 2 and 3
    coupling = (
        0.5 * _on(4, {0: SIGMA["Z"], 1: SIGMA["X"]})
        + 0.7 * _on(4, {0: SIGMA["X"], 3: SIGMA["Z"]})
        + 0.4 * _on(4, {2: SIGMA["X"], 3: SIGMA["X"]})
        + 0.2 * _on(4, {2: SIGMA["Z"]})
    )
    jumps = [_on(4, {1: _local_jump(AmplitudeDamping(t1=0.8))})]
    rho_0 = np.zeros((16, 16), dtype=complex)
    rho_0[2, 2] = 1.0
    expected = _reference(_gate_steps(circuit, noise_model), rho_0, coupling, jumps, n_system=1)

    np.testing.assert_allclose(_simulate(circuit, noise_model), expected, atol=1e-6)


@pytest.mark.parametrize("coupling_strength", [0.0, 0.3, 2.0, 10.0])
def test_coupling_strength_sweep(coupling_strength):
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    circuit.add(RX(0, theta=0.9))
    noise_model = NoiseModel()
    noise_model.add(EnvironmentNoise(n_environment_qubits=1, couplings=[(coupling_strength, PauliY(0), PauliY(0))]))

    coupling = coupling_strength * _on(2, {0: SIGMA["Y"], 1: SIGMA["Y"]})
    rho_0 = np.zeros((4, 4), dtype=complex)
    rho_0[0, 0] = 1.0
    expected = _reference(_gate_steps(circuit, noise_model), rho_0, coupling, [], n_system=1)

    np.testing.assert_allclose(_simulate(circuit, noise_model), expected, atol=1e-6)


@pytest.mark.parametrize("gate_time", [1e-3, 0.1, 1.0, 5.0])
def test_gate_time_sweep(gate_time):
    # Longer gates give the environment more time to act during each gate
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))
    circuit.add(H(0))
    noise_model = NoiseModel()
    noise_model.add(
        EnvironmentNoise(
            n_environment_qubits=1,
            couplings=[(1.0, PauliZ(0), PauliX(0))],
            environment_noise={0: [AmplitudeDamping(t1=2.0)]},
        )
    )
    noise_model.noise_config.set_default_gate_time(gate_time)

    coupling = _on(2, {0: SIGMA["Z"], 1: SIGMA["X"]})
    jumps = [_on(2, {1: _local_jump(AmplitudeDamping(t1=2.0))})]
    rho_0 = np.zeros((4, 4), dtype=complex)
    rho_0[0, 0] = 1.0
    expected = _reference(_gate_steps(circuit, noise_model), rho_0, coupling, jumps, n_system=1)

    np.testing.assert_allclose(_simulate(circuit, noise_model), expected, atol=1e-6)
