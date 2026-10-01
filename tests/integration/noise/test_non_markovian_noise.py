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

import numpy as np
import pytest

from qilisdk.analog import Schedule
from qilisdk.analog import X as PauliX
from qilisdk.analog import Y as PauliY
from qilisdk.analog import Z as PauliZ
from qilisdk.backends import QiliSim
from qilisdk.backends.backend_config import ExecutionConfig
from qilisdk.core import QTensor, ket
from qilisdk.core.interpolator import Interpolation
from qilisdk.digital import RX, Circuit, H, I, X
from qilisdk.functionals import AnalogEvolution, DigitalPropagation
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.noise import AmplitudeDamping, BitFlip, Dephasing, EnvironmentNoise, NoiseModel
from qilisdk.readout import Readout

EXECUTION_CONFIG = ExecutionConfig(seed=42, num_threads=1)
PLUS = (ket(0) + ket(1)).unit()


def _density_matrix(state: QTensor) -> np.ndarray:
    dense = state.dense()
    return dense @ dense.conj().T if state.is_ket() else dense


def _purity(state: QTensor) -> float:
    rho = _density_matrix(state)
    return float(np.real(np.trace(rho @ rho)))


def _idle_circuit(idle_time: float, noise_model: NoiseModel) -> Circuit:
    # Prepare |+> almost instantly, then idle so only the environment acts on the system
    noise_model.noise_config.set_gate_time(H, 1e-6)
    noise_model.noise_config.set_gate_time(I, idle_time)
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    circuit.add(I(0))
    return circuit


def _zz_environment(coupling: float, environment_state: QTensor = PLUS, **kwargs) -> EnvironmentNoise:
    return EnvironmentNoise(
        n_environment_qubits=1,
        couplings=[(coupling, PauliZ(0), PauliZ(0))],
        environment_state=environment_state,
        **kwargs,
    )


def _run_idle(environment: EnvironmentNoise, idle_time: float, readout: Readout):
    noise_model = NoiseModel()
    noise_model.add(environment)
    circuit = _idle_circuit(idle_time, noise_model)
    return QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        DigitalPropagation(circuit), readout=readout
    )


@pytest.mark.parametrize("idle_time", [np.pi / 8, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
def test_digital_zz_environment_coherence_revives(idle_time):
    # A ZZ-coupled environment in |+> gives <X> = cos(2 J t): it collapses and then revives, which
    # no Markovian dephasing can reproduce
    coupling = 1.0
    result = _run_idle(
        _zz_environment(coupling), idle_time, Readout().with_expectation(observables=[PauliX(0), PauliY(0)])
    )

    expectation_x, expectation_y = result.get_expectation_values()
    assert np.isclose(expectation_x, np.cos(2 * coupling * idle_time), atol=1e-4)
    assert np.isclose(expectation_y, 0.0, atol=1e-4)


def test_digital_zz_environment_purity_drops_and_returns():
    readout = Readout().with_state_tomography()

    mixed = _run_idle(_zz_environment(1.0), np.pi / 4, readout).get_state()
    revived = _run_idle(_zz_environment(1.0), np.pi / 2, readout).get_state()

    assert np.isclose(_purity(mixed), 0.5, atol=1e-4)
    assert np.isclose(_purity(revived), 1.0, atol=1e-4)


def test_digital_environment_in_eigenstate_stays_coherent():
    # An environment in |0>, kept there by its own damping, only shifts the system frequency
    environment = _zz_environment(1.0, environment_state=ket(0), environment_noise={0: [AmplitudeDamping(t1=0.1)]})

    state = _run_idle(environment, np.pi / 4, Readout().with_state_tomography()).get_state()

    assert np.isclose(_purity(state), 1.0, atol=1e-4)


def test_digital_environment_dephasing_commuting_with_coupling_has_no_effect():
    # Dephasing on the environment commutes with a ZZ coupling, so the system sees the same dynamics
    idle_time = np.pi / 8
    readout = Readout().with_state_tomography()
    with_noise = _zz_environment(1.0, environment_noise={0: [Dephasing(t_phi=0.01)]})

    state = _run_idle(with_noise, idle_time, readout).get_state()
    reference = _run_idle(_zz_environment(1.0), idle_time, readout).get_state()

    np.testing.assert_allclose(_density_matrix(state), _density_matrix(reference), atol=1e-4)


def test_digital_zero_coupling_matches_noiseless():
    circuit = Circuit(nqubits=1)
    circuit.add(RX(0, theta=0.7))
    readout = Readout().with_state_tomography()
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(0.0))

    state = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        DigitalPropagation(circuit), readout=readout
    )
    reference = QiliSim(execution_config=EXECUTION_CONFIG).execute(DigitalPropagation(circuit), readout=readout)

    np.testing.assert_allclose(
        _density_matrix(state.get_state()), _density_matrix(reference.get_state()), atol=1e-6
    )


def test_digital_results_only_contain_system_qubits():
    circuit = Circuit(nqubits=2)
    circuit.add(X(0))
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(0.0, environment_state=ket(0)))

    result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=100).with_state_tomography()
    )

    assert result.get_samples() == {"10": 100}
    assert result.get_state().shape[0] == 4


def test_digital_per_gate_noise_with_environment_raises():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))
    noise_model.add(BitFlip(probability=0.1), gate=X)
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    backend = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
    with pytest.raises(ValueError, match=r"per-gate"):
        backend.execute(DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=10))


def _analog_evolution(hamiltonian, initial_state, total_time=1.0):
    schedule = Schedule(
        hamiltonians={"h": hamiltonian},
        coefficients={"h": {0.0: 1.0, total_time: 1.0}},
        dt=0.01,
        interpolation=Interpolation.LINEAR,
    )
    return AnalogEvolution(schedule=schedule, initial_state=initial_state)


def test_analog_matches_explicit_environment():
    # The same physics simulated with the environment as a visible qubit, then traced out by hand
    system_hamiltonian = 0.5 * PauliX(0)
    environment = EnvironmentNoise(
        n_environment_qubits=1,
        couplings=[(0.7, PauliZ(0), PauliX(0))],
        environment_hamiltonian=0.3 * PauliZ(0),
        environment_noise={0: [AmplitudeDamping(t1=2.0)]},
        environment_state=ket(1),
    )
    noise_model = NoiseModel()
    noise_model.add(environment)
    noise_model.add(Dephasing(t_phi=3.0))
    readout = Readout().with_state_tomography()

    state = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(system_hamiltonian, ket(0)), readout=readout
    )

    reference_noise_model = NoiseModel()
    reference_noise_model.add(AmplitudeDamping(t1=2.0), qubits=[1])
    reference_noise_model.add(Dephasing(t_phi=3.0), qubits=[0])
    reference_hamiltonian = system_hamiltonian + 0.7 * PauliZ(0) * PauliX(1) + 0.3 * PauliZ(1)
    reference = QiliSim(noise_model=reference_noise_model, execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(reference_hamiltonian, ket(0, 1)), readout=readout
    )

    np.testing.assert_allclose(
        _density_matrix(state.get_state()),
        _density_matrix(reference.get_state().partial_trace({0})),
        atol=1e-6,
    )


def test_analog_two_environments_match_explicit_environments():
    environments = [
        EnvironmentNoise(n_environment_qubits=1, couplings=[(0.4, PauliZ(0), PauliX(0))]),
        EnvironmentNoise(
            n_environment_qubits=1,
            couplings=[(0.9, PauliX(0), PauliZ(0))],
            environment_state=PLUS.to_density_matrix(),
        ),
    ]
    noise_model = NoiseModel()
    for environment in environments:
        noise_model.add(environment)
    readout = Readout().with_state_tomography()

    state = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(0.5 * PauliY(0), ket(0)), readout=readout
    )

    # The first environment sits at qubit 1, the second at qubit 2
    reference_hamiltonian = 0.5 * PauliY(0) + 0.4 * PauliZ(0) * PauliX(1) + 0.9 * PauliX(0) * PauliZ(2)
    reference_initial_state = QTensor(np.kron(ket(0, 0).dense(), PLUS.dense()))
    reference = QiliSim(execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(reference_hamiltonian, reference_initial_state), readout=readout
    )

    np.testing.assert_allclose(
        _density_matrix(state.get_state()),
        _density_matrix(reference.get_state().partial_trace({0})),
        atol=1e-6,
    )


def test_analog_expectation_on_system_observable():
    environment = EnvironmentNoise(
        n_environment_qubits=1,
        couplings=[(1.0, PauliZ(0), PauliZ(0))],
        environment_state=PLUS,
    )
    noise_model = NoiseModel()
    noise_model.add(environment)

    # Each environment branch precesses at 1 +- J, so <X> = cos(2 t) cos(2 J t)
    total_time = np.pi / 8
    result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(PauliZ(0), PLUS, total_time=total_time),
        readout=Readout().with_expectation(observables=[PauliX(0)]),
    )

    assert np.isclose(result.get_expectation_values()[0], np.cos(2 * total_time) ** 2, atol=1e-4)


def test_quantum_reservoir_with_environment_raises():
    schedule = Schedule(
        hamiltonians={"hz": PauliZ(0)},
        coefficients={"hz": {0.0: 1.0, 1.0: 1.0}},
        dt=0.1,
        interpolation=Interpolation.LINEAR,
    )
    encoding = Circuit(1)
    encoding.add(RX(0, theta=ReservoirInput("u", 0.1)))
    reservoir = QuantumReservoir(
        initial_state=QTensor.uniform(1).to_density_matrix(),
        reservoir_layer=ReservoirLayer(evolution_dynamics=schedule, input_encoding=encoding),
        input_per_layer=[{"u": 0.2}],
    )
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))

    backend = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
    with pytest.raises(ValueError, match=r"Non-Markovian noise is not supported for quantum reservoirs"):
        backend.execute(reservoir, readout=Readout().with_expectation(observables=[PauliZ(0)]))
