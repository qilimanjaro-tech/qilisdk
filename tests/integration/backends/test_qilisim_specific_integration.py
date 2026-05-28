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

from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Y as pauli_y
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.analog.schedule import Schedule
from qilisdk.backends.backend_config import AnalogMethod, DigitalMethod, ExecutionConfig, MonteCarloConfig
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core.qtensor import ket
from qilisdk.digital import CNOT, RX, RY, RZ, U1, U2, U3, Circuit, H, M, X, Y, Z
from qilisdk.functionals import AnalogEvolution, DigitalPropagation, FunctionalResult
from qilisdk.readout import Readout, SamplingReadout

analog_methods = [
    AnalogMethod.direct(),
    AnalogMethod.arnoldi(),
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


# --- Variational annealing (variational_exponential method) ---


def _make_annealing_schedule():
    """Simple 1-qubit X→Z annealing schedule."""
    return Schedule(
        dt=1,
        hamiltonians={"h_x": pauli_x(0), "h_z": pauli_z(0)},
        coefficients={
            "h_x": {(0, 4): lambda t: 1 - t / 4},
            "h_z": {(0, 4): lambda t: t / 4},
        },
    )


@pytest.mark.parametrize(
    "readout",
    [
        Readout().with_expectation(observables=[pauli_z(0)]),
        Readout().with_sampling(nshots=50),
    ],
)
def test_variational_annealing_runs(readout):
    from qilisdk.core.qtensor import InitialState

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
    """After full X→Z annealing the ground state of Z is near |0⟩, so <Z> > 0."""
    from qilisdk.core.qtensor import InitialState

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
    assert -1.0 <= ev <= 1.0


def test_variational_annealing_wrong_initial_state_raises():
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=50, warmups=0),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    with pytest.raises(Exception):
        backend.execute(
            AnalogEvolution(schedule=_make_annealing_schedule(), initial_state=ket(0)),
            readout=Readout().with_expectation(observables=[pauli_z(0)]),
        )


def test_variational_annealing_non_x_first_hamiltonian_raises():
    from qilisdk.core.qtensor import InitialState

    bad_schedule = Schedule(
        dt=1,
        hamiltonians={"h_z1": pauli_z(0), "h_z2": pauli_z(0)},
        coefficients={"h_z1": {(0, 4): lambda t: 1 - t / 4}, "h_z2": {(0, 4): lambda t: t / 4}},
    )
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=50, warmups=0),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    with pytest.raises(Exception):
        backend.execute(
            AnalogEvolution(schedule=bad_schedule, initial_state=InitialState.UNIFORM),
            readout=Readout().with_expectation(observables=[pauli_z(0)]),
        )


def test_variational_annealing_config_validation_raises():
    """Negative warmups should raise a validation error."""
    from qilisdk.core.qtensor import InitialState

    backend = QiliSim(
        analog_simulation_method=AnalogMethod.variational_annealing(order=1, shots=50, warmups=0),
        execution_config=ExecutionConfig(seed=42, num_threads=1),
    )
    # Override internal solver config to trigger validation failure
    backend._solver_config["warmups"] = -1
    with pytest.raises(Exception):
        backend.execute(
            AnalogEvolution(schedule=_make_annealing_schedule(), initial_state=InitialState.UNIFORM),
            readout=Readout().with_expectation(observables=[pauli_z(0)]),
        )
    assert samples["00"] + samples["10"] == 50
