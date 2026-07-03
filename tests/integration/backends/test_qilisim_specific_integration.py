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

import numpy as np
import pytest

pytest.importorskip("qilisim_module", reason="QiliSim integration tests require the 'qilisim_module' C++ extension")

import random

from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Y as pauli_y
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.analog.schedule import Schedule
from qilisdk.backends.backend_config import AnalogMethod, DigitalMethod, ExecutionConfig, MonteCarloConfig
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core.qtensor import InitialState, QTensor, ket
from qilisdk.digital import CNOT, RX, RY, RZ, U1, U2, U3, Circuit, H, M, X, Y, Z
from qilisdk.functionals import AnalogEvolution, DigitalPropagation, FunctionalResult
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.noise import AmplitudeDamping, Dephasing, NoiseModel
from qilisdk.readout import Readout, SamplingReadout

random.seed(42)

analog_methods = [
    AnalogMethod.direct(),
    AnalogMethod.arnoldi(),
    AnalogMethod.arnoldi(matrix_free=True),
    AnalogMethod.integrator(),
    AnalogMethod.integrator(matrix_free=True),
]
digital_methods = [DigitalMethod.statevector(matrix_free=False), DigitalMethod.statevector(matrix_free=True)]

_MONTE_CARLO_CONFIG = MonteCarloConfig(trajectories=100)


@pytest.mark.parametrize("method", digital_methods)
def test_seed_same(method):
    backend1 = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1), digital_simulation_method=method)
    backend2 = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1), digital_simulation_method=method)
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    readout = Readout().with_sampling(nshots=100)
    result1 = backend1.execute(DigitalPropagation(circuit=circuit), readout=readout)
    result2 = backend2.execute(DigitalPropagation(circuit=circuit), readout=readout)
    assert result1.get_samples() == result2.get_samples()


def test_no_seed():
    backend = QiliSim(execution_config=ExecutionConfig(seed=None, num_threads=1))
    seed = backend.get_config()["seed"]
    assert isinstance(seed, int)
    assert 0 <= seed < 2**15


@pytest.mark.parametrize("method", digital_methods)
def test_monte_carlo_circuit(method):
    p = 0.2
    initial_state_1 = ket(0).unit()
    initial_state_2 = ket(1).unit()
    initial_state = (initial_state_1.to_density_matrix() * (1 - p) + initial_state_2.to_density_matrix() * p).unit()
    backend = QiliSim(
        digital_simulation_method=method,
        execution_config=ExecutionConfig(seed=42, num_threads=1, monte_carlo=MonteCarloConfig(trajectories=100)),
    )
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    readout = [SamplingReadout(nshots=100)]
    result = backend._execute_digital_propagation(
        DigitalPropagation(circuit=circuit), readout=readout, initial_state=initial_state
    )
    assert isinstance(result, FunctionalResult)
    samples = result.get_samples()
    assert "0" in samples
    assert "1" in samples


@pytest.mark.parametrize("method", digital_methods)
def test_seed_different(method):
    backend1 = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1), digital_simulation_method=method)
    backend2 = QiliSim(execution_config=ExecutionConfig(seed=43, num_threads=1), digital_simulation_method=method)
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    readout = Readout().with_sampling(nshots=100)
    result1 = backend1.execute(DigitalPropagation(circuit=circuit), readout=readout)
    result2 = backend2.execute(DigitalPropagation(circuit=circuit), readout=readout)
    assert result1.get_samples() != result2.get_samples()


@pytest.mark.parametrize("method", analog_methods)
def test_row_vec_ordering(method):
    dt = 0.5
    T = 100

    coeff = (1 + 1j) / np.sqrt(2)
    hamiltonian = coeff * pauli_x(0) + np.conj(coeff) * pauli_y(0)

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": hamiltonian},
        coefficients={"h1": {(0, T): 1}},
    )

    psi0 = (ket(0) - ket(1)).unit()
    psi0 = psi0.to_density_matrix()

    backend = QiliSim(
        analog_simulation_method=method,
        execution_config=ExecutionConfig(seed=42, num_threads=1, monte_carlo=_MONTE_CARLO_CONFIG),
    )
    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0),
        readout=Readout().with_expectation(observables=[pauli_y(0)]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    assert res.get_state().shape == (2, 2)

    # check that it's hermitian
    final_rho = res.get_state().dense()
    assert np.allclose(final_rho, final_rho.conj().T, rtol=1e-6)


@pytest.mark.parametrize("method", analog_methods)
def test_monte_carlo_time_evolution(method):
    o = 1.0
    dt = 0.1
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = ((ket(0) - ket(1)).unit()).to_density_matrix()
    psi1 = ket(1).to_density_matrix()
    mix = 0.2
    psi0 = ((1 - mix) * psi0 + mix * psi1).unit()

    backend = QiliSim(
        analog_simulation_method=method,
        execution_config=ExecutionConfig(seed=42, num_threads=1, monte_carlo=_MONTE_CARLO_CONFIG),
    )
    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0),
        readout=Readout().with_expectation(observables=[pauli_z(0)]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    expect_z = res.get_expectation_values()[0]
    assert res.get_state().shape == (2, 2)
    assert np.isclose(expect_z, -0.8, rtol=1e-1)


@pytest.mark.parametrize(
    ("initial_state", "noise"),
    [
        (ket(0), None),  # unitary evolution on a state vector
        (ket(0).to_density_matrix(), None),  # unitary evolution on a (pure) density matrix
        ((ket(0, 0)).to_density_matrix(), None),  # unitary two-qubit density matrix
        (ket(1), "amplitude_damping"),  # non-unitary (superoperator) path
        ((ket(0) + ket(1)).unit(), "dephasing"),  # non-unitary (superoperator) path
    ],
)
def test_arnoldi_matrix_free_matches_explicit_matrix(initial_state, noise):
    """The matrix-free Arnoldi evolution must reproduce the explicit-matrix Arnoldi
    evolution to numerical precision across the unitary state-vector, unitary
    density-matrix and non-unitary (Lindblad superoperator) code paths."""
    nqubits = 2 if initial_state.data.shape[0] == 4 else 1
    hamiltonians = {"hx": sum(pauli_x(q) for q in range(nqubits))}
    hamiltonians["hz"] = sum(pauli_z(q) for q in range(nqubits))
    schedule = Schedule(
        dt=0.1,
        hamiltonians=hamiltonians,
        coefficients={"hx": {(0, 3): lambda t: 1 - t / 3}, "hz": {(0, 3): lambda t: t / 3}},
    )

    noise_model = None
    if noise == "amplitude_damping":
        noise_model = NoiseModel()
        noise_model.add(AmplitudeDamping(t1=0.5))
    elif noise == "dephasing":
        noise_model = NoiseModel()
        noise_model.add(Dephasing(t_phi=0.4))

    observables = [pauli_z(q) for q in range(nqubits)] + [pauli_x(q) for q in range(nqubits)]

    def _run(matrix_free: bool):
        backend = QiliSim(
            analog_simulation_method=AnalogMethod.arnoldi(dim=12, num_substeps=2, matrix_free=matrix_free),
            noise_model=noise_model,
            execution_config=ExecutionConfig(seed=42, num_threads=2),
        )
        res = backend.execute(
            AnalogEvolution(schedule=schedule, initial_state=initial_state),
            readout=Readout().with_state_tomography().with_expectation(observables=observables),
        )
        return res.get_state().dense(), np.asarray(res.get_expectation_values())

    explicit_state, explicit_ev = _run(matrix_free=False)
    mf_state, mf_ev = _run(matrix_free=True)

    np.testing.assert_allclose(mf_state, explicit_state, atol=1e-9)
    np.testing.assert_allclose(mf_ev, explicit_ev, atol=1e-9)


@pytest.mark.parametrize("method", digital_methods)
def test_exponential_gates(method):
    backend = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1), digital_simulation_method=method)
    circuit = Circuit(nqubits=1)
    circuit.add(X(0).exponential())
    result = backend.execute(DigitalPropagation(circuit=circuit), readout=Readout().with_sampling(nshots=100))
    assert isinstance(result, FunctionalResult)
    samples = result.get_samples()
    assert "1" in samples
    assert "0" in samples


def _counts_similar(counts1, counts2, total_shots, tol=0.1):
    all_keys = set(counts1.keys()) | set(counts2.keys())
    for key in all_keys:
        c1 = counts1.get(key, 0)
        c2 = counts2.get(key, 0)
        if abs(c1 - c2) > tol * total_shots:
            return False
    return True


def test_matrix_free_circuit_versus_normal():
    nqubits = 3
    c = Circuit.random(
        nqubits=nqubits, ngates=1000, single_qubit_gates=[H, X, Y, Z, RX, RY, RZ, U1, U2, U3], two_qubit_gates=[CNOT]
    )
    backend_statevector = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=1), digital_simulation_method=DigitalMethod.statevector()
    )
    backend_matrix_free = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=1),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=True),
    )
    readout = Readout().with_sampling(nshots=1000)
    res_statevector = backend_statevector.execute(DigitalPropagation(circuit=c), readout=readout)
    res_matrix_free = backend_matrix_free.execute(DigitalPropagation(circuit=c), readout=readout)
    assert _counts_similar(res_statevector.get_samples(), res_matrix_free.get_samples(), total_shots=1000, tol=0.1)


@pytest.mark.parametrize("matrix_free", [True, False])
def test_combine_single_qubit_gates(matrix_free):
    nqubits = 3
    c = Circuit.random(
        nqubits=nqubits, ngates=1000, single_qubit_gates=[H, X, Y, Z, RX, RY, RZ, U1, U2, U3], two_qubit_gates=[CNOT]
    )
    backend_combined = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=1),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=matrix_free, combine_single_qubit_gates=True),
    )
    backend_uncombined = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=1),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=matrix_free, combine_single_qubit_gates=False),
    )
    readout = Readout().with_sampling(nshots=1000)
    res_combined = backend_combined.execute(DigitalPropagation(circuit=c), readout=readout)
    res_uncombined = backend_uncombined.execute(DigitalPropagation(circuit=c), readout=readout)
    assert _counts_similar(res_combined.get_samples(), res_uncombined.get_samples(), total_shots=1000, tol=0.1)


@pytest.mark.parametrize("max_fused_qubits", [2, 3, 4, 5])
def test_fuse_gates_matches_unfused(max_fused_qubits):
    # Gate fusion only activates with enough threads (it is a memory-bandwidth vs
    # compute trade), so use num_threads=4 to exercise the fused path. The final
    # statevector must match the unfused result up to numerical noise.
    nqubits = 6
    c = Circuit.random(
        nqubits=nqubits, ngates=400, single_qubit_gates=[H, X, Y, Z, RX, RY, RZ, U1, U2, U3], two_qubit_gates=[CNOT]
    )
    backend_unfused = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=4),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=True, fuse_gates=False),
    )
    backend_fused = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=4),
        digital_simulation_method=DigitalMethod.statevector(
            matrix_free=True, fuse_gates=True, max_fused_qubits=max_fused_qubits
        ),
    )
    readout = Readout().with_state_tomography()
    state_unfused = np.asarray(
        backend_unfused.execute(DigitalPropagation(circuit=c), readout=readout).get_state().dense()
    ).reshape(-1)
    state_fused = np.asarray(
        backend_fused.execute(DigitalPropagation(circuit=c), readout=readout).get_state().dense()
    ).reshape(-1)
    fidelity = np.abs(np.vdot(state_unfused, state_fused)) / (
        np.linalg.norm(state_unfused) * np.linalg.norm(state_fused)
    )
    assert fidelity == pytest.approx(1.0, abs=1e-4)
    assert np.max(np.abs(state_unfused - state_fused)) < 1e-3


def _statevectors_close_up_to_global_phase(a, b, tol=1e-3):
    # Two statevectors describing the same physical state may differ by an overall
    # phase, so compare via the (phase-insensitive) fidelity and then align the
    # global phase before the element-wise check.
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    fidelity = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    if fidelity != pytest.approx(1.0, abs=1e-4):
        return False
    # Align b onto a using the phase of their largest overlapping component.
    pivot = int(np.argmax(np.abs(a)))
    if np.abs(b[pivot]) > 0:
        phase = (a[pivot] / b[pivot]) / np.abs(a[pivot] / b[pivot])
        b = b * phase
    return bool(np.max(np.abs(a - b)) < tol)


@pytest.mark.parametrize("nqubits_max_fused_qubits", [(4, 2), (6, 3), (7, 4), (8, 5)])
def test_fused_matches_normal_statevector_large_random(nqubits_max_fused_qubits):
    nqubits, max_fused_qubits = nqubits_max_fused_qubits
    # The fused matrix-free path must produce the same statevector as the normal
    # (non-matrix-free) dense path, which applies each gate via its full matrix.
    # This is the path that builds scattered multi-qubit gate matrices, so the test
    # guards the gate-expansion logic shared by fusion and the dense simulator.
    # Fusion only activates with >= 4 threads.
    c = Circuit.random(
        nqubits=nqubits, ngates=500, single_qubit_gates=[H, X, Y, Z, RX, RY, RZ, U1, U2, U3], two_qubit_gates=[CNOT]
    )
    backend_normal = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=4),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=False),
    )
    backend_fused = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=4),
        digital_simulation_method=DigitalMethod.statevector(
            matrix_free=True, fuse_gates=True, max_fused_qubits=max_fused_qubits
        ),
    )
    readout = Readout().with_state_tomography()
    state_normal = backend_normal.execute(DigitalPropagation(circuit=c), readout=readout).get_state().dense()
    state_fused = backend_fused.execute(DigitalPropagation(circuit=c), readout=readout).get_state().dense()
    assert _statevectors_close_up_to_global_phase(state_normal, state_fused)


def test_fused_matches_normal_sampling_large_random():
    # Same comparison as above but for a larger register via sampling (phase
    # insensitive and cheaper than full state tomography at high qubit counts).
    nqubits = 10
    c = Circuit.random(
        nqubits=nqubits, ngates=800, single_qubit_gates=[H, X, Y, Z, RX, RY, RZ, U1, U2, U3], two_qubit_gates=[CNOT]
    )
    backend_normal = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=4),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=False),
    )
    backend_fused = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=4),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=True, fuse_gates=True, max_fused_qubits=4),
    )
    readout = Readout().with_sampling(nshots=2000)
    res_normal = backend_normal.execute(DigitalPropagation(circuit=c), readout=readout)
    res_fused = backend_fused.execute(DigitalPropagation(circuit=c), readout=readout)
    assert _counts_similar(res_normal.get_samples(), res_fused.get_samples(), total_shots=2000, tol=0.1)


def test_fuse_gates_with_mid_circuit_measurement():
    # Fusion must treat measurements as barriers; verify sampling still works.
    circuit = Circuit(nqubits=3)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(M(0))
    circuit.add(H(1))
    circuit.add(CNOT(1, 2))
    circuit.add(M(1))
    circuit.add(M(2))
    backend = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=4),
        digital_simulation_method=DigitalMethod.statevector(matrix_free=True, fuse_gates=True),
    )
    result = backend.execute(DigitalPropagation(circuit=circuit), readout=Readout().with_sampling(nshots=500))
    assert sum(result.get_samples().values()) == 500


def test_matrix_free_time_evolution_versus_normal():
    dt = 0.1
    T = 10
    hamiltonian = pauli_x(0) + pauli_z(0)
    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": hamiltonian},
        coefficients={"h1": {(0, T): 1}},
    )
    psi0 = (ket(0) - ket(1)).unit()
    readout = Readout().with_expectation(observables=[pauli_y(0)])
    backend_normal = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=1), analog_simulation_method=AnalogMethod.integrator()
    )
    backend_matrix_free = QiliSim(
        execution_config=ExecutionConfig(seed=42, num_threads=1),
        analog_simulation_method=AnalogMethod.integrator(matrix_free=True),
    )
    res_normal = backend_normal.execute(AnalogEvolution(schedule=schedule, initial_state=psi0), readout=readout)
    res_matrix_free = backend_matrix_free.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0), readout=readout
    )
    assert np.isclose(res_normal.get_expectation_values()[0], res_matrix_free.get_expectation_values()[0], rtol=0.01)


def test_mid_circuit_samples():
    backend = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1))
    c = Circuit(2)
    c.add(X(0))
    c.add(M(0))
    c.add(X(0))
    c.add(M(0))
    result = backend.execute(DigitalPropagation(circuit=c), Readout().with_sampling(nshots=50))
    assert isinstance(result, FunctionalResult)
    samples = result.get_samples()
    assert "0_" in samples
    assert samples["0_"] == 50
    inter = result.get_intermediate_samples()
    assert len(inter) == 1


def test_mid_circuit_measurement_collapse():
    backend = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1, measurement_collapse=True))
    c = Circuit(2)
    c.add(H(0))
    c.add(M(0))
    c.add(H(0))
    c.add(M(0, 1))
    result = backend.execute(DigitalPropagation(circuit=c), Readout().with_sampling(nshots=50))
    assert isinstance(result, FunctionalResult)
    samples = result.get_samples()
    assert "00" in samples
    assert "10" in samples
    assert samples["00"] + samples["10"] == 50


# --- Variational annealing (variational_exponential method) ---


def _make_annealing_schedule():
    """Simple 1-qubit X→Z annealing schedule."""
    T = 10
    steps = 100
    return Schedule(
        dt=T / steps,
        hamiltonians={"h_x": -pauli_x(0), "h_z": pauli_z(0)},
        coefficients={
            "h_x": {(0, T): lambda t: 1 - t / T},
            "h_z": {(0, T): lambda t: t / T},
        },
    )


def _make_many_qubit_annealing_schedule(nqubits):
    """n-qubit X→ZZ annealing schedule."""
    hamiltonians = {"h_x": -sum(pauli_x(i) for i in range(nqubits)), "h_z": sum(pauli_z(i) for i in range(nqubits))}
    T = 10
    steps = 100
    coefficients = {
        "h_x": {(0, T): lambda t: 1 - t / T},
        "h_z": {(0, T): lambda t: t / T},
    }
    return Schedule(dt=T / steps, hamiltonians=hamiltonians, coefficients=coefficients)


@pytest.mark.parametrize(
    "readout",
    [
        Readout().with_expectation(observables=[pauli_z(0)]),
        Readout().with_sampling(nshots=50),
    ],
)
def test_variational_annealing_runs(readout):

    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=100, warmups=5),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    result = backend.execute(
        AnalogEvolution(schedule=_make_annealing_schedule(), initial_state=InitialState.UNIFORM),
        readout=readout,
    )
    assert isinstance(result, FunctionalResult)


def test_variational_annealing_expectation_value_bounded():
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=200, warmups=10),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    result = backend.execute(
        AnalogEvolution(schedule=_make_annealing_schedule(), initial_state=InitialState.UNIFORM),
        readout=Readout().with_expectation(observables=[pauli_z(0)]),
    )
    assert isinstance(result, FunctionalResult)
    ev = result.get_expectation_values()[0]
    assert -1.00001 <= ev.real <= 1.00001


def test_variational_annealing_wrong_initial_state_raises():
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=50, warmups=0),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    with pytest.raises(ValueError, match="Initial state must be"):
        backend.execute(
            AnalogEvolution(schedule=_make_annealing_schedule(), initial_state=ket(0)),
            readout=Readout().with_expectation(observables=[pauli_z(0)]),
        )


def test_variational_annealing_non_x_first_hamiltonian_raises():

    bad_schedule = Schedule(
        dt=1,
        hamiltonians={"h_z1": pauli_z(0), "h_z2": pauli_z(0)},
        coefficients={"h_z1": {(0, 4): lambda t: 1 - t / 4}, "h_z2": {(0, 4): lambda t: t / 4}},
    )
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=50, warmups=0),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    with pytest.raises(ValueError, match="The first Hamiltonian"):
        backend.execute(
            AnalogEvolution(schedule=bad_schedule, initial_state=InitialState.UNIFORM),
            readout=Readout().with_expectation(observables=[pauli_z(0)]),
        )


def test_variational_annealing_non_z_final_hamiltonian_raises():

    bad_schedule = Schedule(
        dt=1,
        hamiltonians={"h_x": pauli_x(0), "h_y": pauli_y(0)},
        coefficients={"h_x": {(0, 4): lambda t: 1 - t / 4}, "h_y": {(0, 4): lambda t: t / 4}},
    )
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=50, warmups=0),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    with pytest.raises(ValueError, match="The last Hamiltonian"):
        backend.execute(
            AnalogEvolution(schedule=bad_schedule, initial_state=InitialState.UNIFORM),
            readout=Readout().with_expectation(observables=[pauli_z(0)]),
        )


def test_variational_annealing_config_validation_raises():
    """Negative warmups should raise a validation error."""

    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=50, warmups=0),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    # Override internal solver config to trigger validation failure
    backend._solver_config["variational_warmups"] = -1
    with pytest.raises(ValueError, match="Warmups cannot be negative"):
        backend.execute(
            AnalogEvolution(schedule=_make_annealing_schedule(), initial_state=InitialState.UNIFORM),
            readout=Readout().with_expectation(observables=[pauli_z(0)]),
        )


def test_variational_annealing_single_qubit_correct():
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    result = backend.execute(
        AnalogEvolution(schedule=_make_annealing_schedule(), initial_state=InitialState.UNIFORM),
        readout=Readout().with_expectation(observables=[pauli_z(0)]),
    )
    assert isinstance(result, FunctionalResult)
    ev = result.get_expectation_values()[0]
    assert np.isclose(ev.real, -1.0, atol=0.2)


def test_variational_annealing_many_qubit_correct():
    nqubits = 5
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    observable = sum(pauli_z(i) for i in range(nqubits))
    result = backend.execute(
        AnalogEvolution(schedule=_make_many_qubit_annealing_schedule(nqubits), initial_state=InitialState.UNIFORM),
        readout=Readout().with_expectation(observables=[observable]),
    )
    assert isinstance(result, FunctionalResult)
    ev = result.get_expectation_values()[0]
    assert np.isclose(ev.real, -nqubits, atol=0.2)


def test_matrix_free_complex_gate_on_mixed_state_stays_hermitian():
    """Regression: ``rho -> U rho U†`` on the matrix-free path must use ``U†`` (conjugate
    transpose), not ``U*`` (conjugate).

    A qubit reset turns the state into a *mixed* density matrix, so gate application takes
    the density-matrix (``LeftAndRight``) path. With a complex, non-symmetric single-qubit
    gate (``U2``), the buggy right-multiplication computed ``U rho U*`` instead of ``U rho U†``,
    producing a non-Hermitian state and raising "imaginary expectation value". Real-symmetric
    (X/Z/H) and diagonal (S/T) gates are unaffected, which is why this slipped through.

    This guards the regression three ways: the run must succeed with real expectation values,
    the matrix-free path must agree with the explicit-matrix path, and the tomographed state
    must remain a valid (Hermitian) density matrix.
    """

    def _build() -> QuantumReservoir:
        pre = Circuit(2)
        pre.add(U2(1, phi=ReservoirInput("phi", 0.1), gamma=ReservoirInput("gamma", 0.1)))
        layer = ReservoirLayer(
            evolution_dynamics=Schedule(
                hamiltonians={"h": pauli_z(0) + pauli_z(1) + pauli_z(0) * pauli_z(1) + 0.5 * (pauli_x(0) + pauli_x(1))},
                total_time=1.0,
                dt=0.1,
            ),
            input_encoding=pre,
            qubits_to_reset=[0],
        )
        return QuantumReservoir(
            initial_state=QTensor.uniform(2).to_density_matrix(),
            reservoir_layer=layer,
            input_per_layer=[{"phi": 0.2, "gamma": 0.1}, {"phi": 0.3, "gamma": 0.2}],
        )

    readout = (
        Readout()
        .with_expectation(observables=[pauli_z(0), pauli_z(1), pauli_z(0) * pauli_z(1)])
        .with_state_tomography()
    )

    # The buggy matrix-free path raised "imaginary expectation value" here.
    matrix_free = QiliSim(digital_simulation_method=DigitalMethod(matrix_free=True)).execute(_build(), readout)
    explicit = QiliSim(digital_simulation_method=DigitalMethod(matrix_free=False)).execute(_build(), readout)

    mf_values = np.asarray(matrix_free.get_expectation_values())
    explicit_values = np.asarray(explicit.get_expectation_values())

    # Matrix-free must match the explicit-matrix ground truth.
    np.testing.assert_allclose(mf_values, explicit_values, atol=1e-9)

    def _dense(qtensor) -> np.ndarray:
        data = qtensor.data
        return np.asarray(data.todense() if hasattr(data, "todense") else data)

    # The state after reset + complex gate must stay Hermitian and trace-1. The bug produced
    # states off by ~17 from their adjoint; here we require agreement to numerical precision.
    mf_state = _dense(matrix_free.get_state())
    np.testing.assert_allclose(mf_state, mf_state.conj().T, atol=1e-9)
    assert abs(np.trace(mf_state) - 1.0) < 1e-9

    # And it must match the explicit-matrix path's state.
    np.testing.assert_allclose(mf_state, _dense(explicit.get_state()), atol=1e-9)


@pytest.mark.parametrize("method", [AnalogMethod.direct(), AnalogMethod.arnoldi(), AnalogMethod.integrator()])
def test_reservoir_schedule_hamiltonians_of_differing_qubit_support(method):
    """Regression: a schedule whose Hamiltonian terms touch different qubits produces
    ``to_matrix()`` matrices of *different* dimensions (e.g. ``Z0*Z1`` -> 4x4 while
    ``Z0*Z5`` -> 64x64). The explicit-matrix analog path took its dimension from only the
    first Hamiltonian and then summed the terms, reading out of bounds and segfaulting.

    Each term must be expanded to the full ``nqubits`` space (identity on untouched qubits),
    so the run must succeed and agree with the matrix-free ground truth.
    """
    nqubits = 6

    def _build() -> QuantumReservoir:
        pre = Circuit(nqubits)
        pre.add(RX(0, theta=ReservoirInput("phi", 0.1)))
        layer = ReservoirLayer(
            evolution_dynamics=Schedule(
                hamiltonians={"h1": pauli_z(0) * pauli_z(1), "h2": pauli_z(0) * pauli_z(5) + pauli_x(3)},
                coefficients={"h1": {(0.0, 1.0): 1.0}, "h2": {(0.0, 1.0): 0.5}},
                total_time=2.0,
                dt=0.1,
            ),
            input_encoding=pre,
            qubits_to_reset=[0],
        )
        return QuantumReservoir(
            initial_state=QTensor.uniform(nqubits).to_density_matrix(),
            reservoir_layer=layer,
            input_per_layer=[{"phi": 0.2}, {"phi": 0.3}],
        )

    readout = Readout().with_expectation(observables=[pauli_z(0), pauli_z(3), pauli_z(5)])
    explicit = QiliSim(analog_simulation_method=method).execute(_build(), readout)
    reference = QiliSim(analog_simulation_method=AnalogMethod.integrator(matrix_free=True)).execute(_build(), readout)

    np.testing.assert_allclose(
        np.asarray(explicit.get_expectation_values()),
        np.asarray(reference.get_expectation_values()),
        atol=1e-6,
    )
