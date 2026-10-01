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

from itertools import pairwise

import numpy as np
import pytest

from qilisdk.analog import I as PauliI
from qilisdk.analog import Schedule
from qilisdk.analog import X as PauliX
from qilisdk.analog import Y as PauliY
from qilisdk.analog import Z as PauliZ
from qilisdk.backends import QiliSim
from qilisdk.backends.backend_config import AnalogMethod, ExecutionConfig, MonteCarloConfig
from qilisdk.core import QTensor, ket
from qilisdk.core.interpolator import Interpolation
from qilisdk.digital import RX, Circuit, H, I, M, X
from qilisdk.functionals import AnalogEvolution, DigitalPropagation
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.noise import (
    AmplitudeDamping,
    BitFlip,
    Dephasing,
    EnvironmentNoise,
    KrausChannel,
    LindbladGenerator,
    NoiseModel,
    OffsetPerturbation,
    ReadoutAssignment,
)
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

    np.testing.assert_allclose(_density_matrix(state.get_state()), _density_matrix(reference.get_state()), atol=1e-6)


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


def test_digital_non_positive_gate_time_with_environment_raises():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))
    # Bypasses NoiseConfig.set_gate_time, which rejects non-positive times
    noise_model.noise_config._gate_times[X] = 0.0
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    backend = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
    with pytest.raises(ValueError, match=r"positive gate times"):
        backend.execute(DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=10))


def test_digital_readout_assignment_with_environment():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(0.0, environment_state=ket(0)))
    noise_model.add(ReadoutAssignment(p01=0.0, p10=1.0))
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=50)
    )

    assert result.get_samples() == {"0": 50}


def test_noise_without_lindblad_form_with_environment_raises():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))
    noise_model.add(KrausChannel(operators=[QTensor(np.eye(2))]))
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    backend = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
    with pytest.raises(ValueError, match=r"Lindblad form"):
        backend.execute(DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=10))


@pytest.mark.parametrize("measurement_collapse", [False, True])
def test_digital_mid_circuit_measurement_with_environment(measurement_collapse):
    # H M H: without collapse the two H cancel, with collapse the final state stays mixed
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(0.0, environment_state=ket(0)))
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    circuit.add(M(0))
    circuit.add(H(0))
    config = ExecutionConfig(seed=42, num_threads=1, measurement_collapse=measurement_collapse)

    result = QiliSim(noise_model=noise_model, execution_config=config).execute(
        DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=200)
    )

    assert len(result.intermediate_results) == 1
    assert set(result.intermediate_results[0].get_samples()) == {"0", "1"}
    if measurement_collapse:
        assert set(result.get_samples()) == {"0", "1"}
    else:
        assert result.get_samples() == {"0": 200}


def _analog_evolution(hamiltonian, initial_state, total_time=1.0, store_intermediate_results=False):
    schedule = Schedule(
        hamiltonians={"h": hamiltonian},
        coefficients={"h": {0.0: 1.0, total_time: 1.0}},
        dt=0.01,
        interpolation=Interpolation.LINEAR,
    )
    return AnalogEvolution(
        schedule=schedule, initial_state=initial_state, store_intermediate_results=store_intermediate_results
    )


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


@pytest.mark.parametrize(
    "method", [AnalogMethod.integrator(), AnalogMethod.integrator(matrix_free=False), AnalogMethod.direct()]
)
def test_analog_per_qubit_noise_matches_explicit_environment(method):
    environment = EnvironmentNoise(n_environment_qubits=1, couplings=[(0.8, PauliX(0), PauliX(0))])
    noise_model = NoiseModel()
    noise_model.add(environment)
    noise_model.add(AmplitudeDamping(t1=1.5), qubits=[0])
    readout = Readout().with_state_tomography()

    state = QiliSim(
        noise_model=noise_model, analog_simulation_method=method, execution_config=EXECUTION_CONFIG
    ).execute(_analog_evolution(0.5 * PauliZ(0), ket(1)), readout=readout)

    reference_noise_model = NoiseModel()
    reference_noise_model.add(AmplitudeDamping(t1=1.5), qubits=[0])
    reference = QiliSim(
        noise_model=reference_noise_model, analog_simulation_method=method, execution_config=EXECUTION_CONFIG
    ).execute(_analog_evolution(0.5 * PauliZ(0) + 0.8 * PauliX(0) * PauliX(1), ket(1, 0)), readout=readout)

    np.testing.assert_allclose(
        _density_matrix(state.get_state()),
        _density_matrix(reference.get_state().partial_trace({0})),
        atol=1e-6,
    )


def test_analog_intermediate_results_are_traced_out():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))

    result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(PauliZ(0), PLUS, store_intermediate_results=True), readout=Readout().with_state_tomography()
    )

    assert len(result.intermediate_results) > 0
    for intermediate in result.intermediate_results:
        assert intermediate.get_state().shape == (2, 2)
    np.testing.assert_allclose(
        _density_matrix(result.intermediate_results[-1].get_state()), _density_matrix(result.get_state()), atol=1e-9
    )


def test_analog_variational_method_with_environment_raises():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))
    backend = QiliSim(
        noise_model=noise_model,
        analog_simulation_method=AnalogMethod.variational_annealing(),
        execution_config=EXECUTION_CONFIG,
    )

    with pytest.raises(ValueError, match=r"variational exponential method does not support non-Markovian noise"):
        backend.execute(_analog_evolution(PauliX(0), PLUS), readout=Readout().with_expectation(observables=[PauliZ(0)]))


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


ANALOG_METHODS = [
    AnalogMethod.integrator(),
    AnalogMethod.integrator(matrix_free=False),
    AnalogMethod.adaptive_integrator(),
    AnalogMethod.arnoldi(),
    AnalogMethod.arnoldi(matrix_free=False),
    AnalogMethod.direct(),
]
LOWERING = QTensor(np.array([[0.0, 1.0], [0.0, 0.0]]))


def _echo_circuit(idle_time: float, noise_model: NoiseModel, *, echo: bool) -> Circuit:
    for gate_type in (H, X):
        noise_model.noise_config.set_gate_time(gate_type, 1e-6)
    noise_model.noise_config.set_gate_time(I, idle_time)
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    circuit.add(I(0))
    if echo:
        circuit.add(X(0))
    circuit.add(I(0))
    circuit.add(H(0))
    return circuit


def _run_echo(environment: EnvironmentNoise, *, echo: bool) -> float:
    noise_model = NoiseModel()
    noise_model.add(environment)
    circuit = _echo_circuit(np.pi / 8, noise_model, echo=echo)
    result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        DigitalPropagation(circuit), readout=Readout().with_expectation(observables=[PauliZ(0)])
    )
    return result.get_expectation_values()[0]


def test_digital_spin_echo_refocuses_static_environment():
    # A static ZZ shift is undone by an echo, which no Markovian dephasing allows
    assert np.isclose(_run_echo(_zz_environment(1.0), echo=False), 0.0, atol=1e-4)
    assert np.isclose(_run_echo(_zz_environment(1.0), echo=True), 1.0, atol=1e-4)


def test_digital_spin_echo_fails_for_dynamic_environment():
    # A transverse field makes the environment fluctuate, so the echo no longer refocuses it
    environment = _zz_environment(1.0, environment_hamiltonian=3.0 * PauliX(0))

    assert _run_echo(environment, echo=True) < 0.95


def test_digital_environment_on_one_qubit_leaves_the_other_untouched():
    noise_model = NoiseModel()
    noise_model.add(
        EnvironmentNoise(n_environment_qubits=1, couplings=[(1.0, PauliZ(1), PauliZ(0))], environment_state=PLUS)
    )
    noise_model.noise_config.set_gate_time(H, 1e-6)
    noise_model.noise_config.set_gate_time(I, np.pi / 8)
    circuit = Circuit(nqubits=2)
    circuit.add(H(0))
    circuit.add(H(1))
    circuit.add(I(0))

    result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        DigitalPropagation(circuit), readout=Readout().with_expectation(observables=[PauliX(0), PauliX(1)])
    )

    expectation_0, expectation_1 = result.get_expectation_values()
    assert np.isclose(expectation_0, 1.0, atol=1e-4)
    assert np.isclose(expectation_1, np.cos(np.pi / 4), atol=1e-4)


def test_digital_sampling_statistics_with_environment():
    # At t = pi/4 the coherence has collapsed, so the final H gives an even split
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))
    circuit = _idle_circuit(np.pi / 4, noise_model)
    circuit.add(H(0))
    nshots = 4000

    samples = (
        QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
        .execute(DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=nshots))
        .get_samples()
    )

    assert sum(samples.values()) == nshots
    assert abs(samples.get("0", 0) / nshots - 0.5) < 0.05


def test_digital_gate_parameter_perturbation_with_environment():
    readout = Readout().with_state_tomography()

    def run(theta: float, offset: float | None) -> np.ndarray:
        noise_model = NoiseModel()
        noise_model.add(EnvironmentNoise(n_environment_qubits=1, couplings=[(0.6, PauliX(0), PauliZ(0))]))
        if offset is not None:
            noise_model.add(OffsetPerturbation(offset=offset), gate=RX, parameter="theta")
        circuit = Circuit(nqubits=1)
        circuit.add(RX(0, theta=theta))
        result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
            DigitalPropagation(circuit), readout=readout
        )
        return _density_matrix(result.get_state())

    np.testing.assert_allclose(run(0.3, 0.4), run(0.7, None), atol=1e-9)


def test_digital_monte_carlo_with_mixed_environment():
    # Every environment basis state gives <X> = cos(2 J t), so each trajectory agrees with the exact result
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0, environment_state=QTensor(np.eye(2) / 2)))
    circuit = _idle_circuit(np.pi / 8, noise_model)
    config = ExecutionConfig(seed=42, num_threads=1, monte_carlo=MonteCarloConfig(trajectories=20))

    result = QiliSim(noise_model=noise_model, execution_config=config).execute(
        DigitalPropagation(circuit), readout=Readout().with_expectation(observables=[PauliX(0)])
    )

    assert np.isclose(result.get_expectation_values()[0], np.cos(np.pi / 4), atol=1e-4)


def test_digital_time_dependent_system_rate_with_environment_raises():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))
    noise_model.add(LindbladGenerator(jump_operators=[LOWERING], rates=[lambda t: 0.1 * t]))
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    backend = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
    with pytest.raises(ValueError, match=r"Time-dependent Lindblad rates are not supported"):
        backend.execute(DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=10))


def test_digital_measurement_only_circuit_with_environment():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))
    circuit = Circuit(nqubits=1)
    circuit.add(M(0))

    result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=20)
    )

    assert result.get_samples() == {"0": 20}


def _time_dependent_schedule(extra_hamiltonian=None, total_time=1.0):
    hamiltonians = {"hx": PauliX(0), "hz": PauliZ(0)}
    coefficients = {"hx": {0.0: 1.0, total_time: 0.0}, "hz": {0.0: 0.0, total_time: 1.0}}
    if extra_hamiltonian is not None:
        hamiltonians["environment"] = extra_hamiltonian
        coefficients["environment"] = {0.0: 1.0, total_time: 1.0}
    return Schedule(hamiltonians=hamiltonians, coefficients=coefficients, dt=0.01, interpolation=Interpolation.LINEAR)


@pytest.mark.parametrize("method", ANALOG_METHODS, ids=lambda method: method.evolution_method)
def test_analog_time_dependent_schedule_matches_explicit_environment(method):
    environment = EnvironmentNoise(
        n_environment_qubits=1,
        couplings=[(0.9, PauliY(0), PauliX(0))],
        environment_hamiltonian=0.5 * PauliZ(0),
        environment_noise={0: [Dephasing(t_phi=1.2)]},
        environment_state=PLUS,
    )
    noise_model = NoiseModel()
    noise_model.add(environment)
    readout = Readout().with_state_tomography()

    state = QiliSim(
        noise_model=noise_model, analog_simulation_method=method, execution_config=EXECUTION_CONFIG
    ).execute(AnalogEvolution(schedule=_time_dependent_schedule(), initial_state=ket(0)), readout=readout)

    reference_noise_model = NoiseModel()
    reference_noise_model.add(Dephasing(t_phi=1.2), qubits=[1])
    reference_schedule = _time_dependent_schedule(0.9 * PauliY(0) * PauliX(1) + 0.5 * PauliZ(1))
    reference = QiliSim(
        noise_model=reference_noise_model, analog_simulation_method=method, execution_config=EXECUTION_CONFIG
    ).execute(
        AnalogEvolution(schedule=reference_schedule, initial_state=QTensor(np.kron(ket(0).dense(), PLUS.dense()))),
        readout=readout,
    )

    np.testing.assert_allclose(
        _density_matrix(state.get_state()),
        _density_matrix(reference.get_state().partial_trace({0})),
        atol=1e-6,
    )


@pytest.mark.parametrize("method", [AnalogMethod.integrator(), AnalogMethod.integrator(matrix_free=False)])
def test_analog_time_dependent_system_rate_matches_explicit_environment(method):
    def rate(t):
        return 0.5 * t

    noise_model = NoiseModel()
    noise_model.add(EnvironmentNoise(n_environment_qubits=1, couplings=[(0.7, PauliX(0), PauliZ(0))]))
    noise_model.add(LindbladGenerator(jump_operators=[LOWERING], rates=[rate]), qubits=[0])
    readout = Readout().with_state_tomography()

    state = QiliSim(
        noise_model=noise_model, analog_simulation_method=method, execution_config=EXECUTION_CONFIG
    ).execute(AnalogEvolution(schedule=_time_dependent_schedule(), initial_state=ket(1)), readout=readout)

    reference_noise_model = NoiseModel()
    reference_noise_model.add(LindbladGenerator(jump_operators=[LOWERING], rates=[rate]), qubits=[0])
    reference = QiliSim(
        noise_model=reference_noise_model, analog_simulation_method=method, execution_config=EXECUTION_CONFIG
    ).execute(
        AnalogEvolution(schedule=_time_dependent_schedule(0.7 * PauliX(0) * PauliZ(1)), initial_state=ket(1, 0)),
        readout=readout,
    )

    np.testing.assert_allclose(
        _density_matrix(state.get_state()),
        _density_matrix(reference.get_state().partial_trace({0})),
        atol=1e-6,
    )


def test_analog_multi_qubit_environment_matches_explicit_environment():
    environment = EnvironmentNoise(
        n_environment_qubits=2,
        couplings=[(0.6, PauliZ(0), PauliX(1)), (0.3, PauliX(1), PauliX(0))],
        environment_hamiltonian=0.8 * PauliZ(0) * PauliZ(1) + 0.4 * PauliX(0),
        environment_noise={1: [AmplitudeDamping(t1=0.7)]},
        environment_state=QTensor(np.diag([0.5, 0.0, 0.25, 0.25])),
    )
    noise_model = NoiseModel()
    noise_model.add(environment)
    readout = Readout().with_state_tomography()
    system_hamiltonian = 0.5 * PauliX(0) + 0.2 * PauliZ(1) + 0.3 * PauliZ(0) * PauliZ(1)

    state = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(system_hamiltonian, ket(0, 1)), readout=readout
    )

    # Register: system 0 and 1, environment 2 and 3
    reference_noise_model = NoiseModel()
    reference_noise_model.add(AmplitudeDamping(t1=0.7), qubits=[3])
    reference_hamiltonian = (
        system_hamiltonian
        + 0.6 * PauliZ(0) * PauliX(3)
        + 0.3 * PauliX(1) * PauliX(2)
        + 0.8 * PauliZ(2) * PauliZ(3)
        + 0.4 * PauliX(2)
    )
    reference_initial_state = QTensor(np.kron(_density_matrix(ket(0, 1)), np.diag([0.5, 0.0, 0.25, 0.25])))
    reference = QiliSim(noise_model=reference_noise_model, execution_config=EXECUTION_CONFIG).execute(
        _analog_evolution(reference_hamiltonian, reference_initial_state), readout=readout
    )

    np.testing.assert_allclose(
        _density_matrix(state.get_state()),
        _density_matrix(reference.get_state().partial_trace({0, 1})),
        atol=1e-6,
    )


def test_analog_sampling_readout_only_contains_system_qubits():
    noise_model = NoiseModel()
    noise_model.add(_zz_environment(1.0))

    samples = (
        QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
        .execute(_analog_evolution(PauliX(0), ket(0)), readout=Readout().with_sampling(nshots=100))
        .get_samples()
    )

    assert sum(samples.values()) == 100
    assert all(len(bitstring) == 1 for bitstring in samples)


def test_analog_time_dependent_environment_rate_raises():
    environment = _zz_environment(
        1.0, environment_noise={0: [LindbladGenerator(jump_operators=[LOWERING], rates=[lambda t: 0.1 * t])]}
    )
    noise_model = NoiseModel()
    noise_model.add(environment)

    backend = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG)
    with pytest.raises(ValueError, match=r"time-dependent"):
        backend.execute(_analog_evolution(PauliX(0), ket(0)), readout=Readout().with_state_tomography())


@pytest.mark.parametrize("flip_rates", [(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0)])
def test_analog_telegraph_environment_revival_fades_with_flip_rate(flip_rates):
    # Random flips of a ZZ-coupled environment wash out the revival at t = pi/2, and very fast
    # flips average the coupling away entirely (motional narrowing)
    flip = QTensor(np.array([[0.0, 1.0], [1.0, 0.0]]))
    revivals = []
    for rate in flip_rates:
        noise = {0: [LindbladGenerator(jump_operators=[flip], rates=[rate])]} if rate > 0 else None
        noise_model = NoiseModel()
        noise_model.add(_zz_environment(1.0, environment_noise=noise))
        result = QiliSim(noise_model=noise_model, execution_config=EXECUTION_CONFIG).execute(
            _analog_evolution(PauliI(0), PLUS, total_time=np.pi / 2),
            readout=Readout().with_expectation(observables=[PauliX(0)]),
        )
        revivals.append(result.get_expectation_values()[0])

    assert np.isclose(revivals[0], -1.0, atol=1e-3)
    assert all(earlier < later for earlier, later in pairwise(revivals))
    assert revivals[-1] > 0.9
