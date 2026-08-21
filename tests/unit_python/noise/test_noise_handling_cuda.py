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

pytest.importorskip("cudaq", reason="CUDA noise tests require the 'cudaq' optional dependency")

from qilisdk.analog import PauliX as pauli_x
from qilisdk.analog import Schedule
from qilisdk.backends.cuda_backend import (
    CudaBackend,
    _compose_kraus,
    _embed_operator,
    _kraus_matrices,
    _to_builtin_cuda_noise,
    cudaq,
)
from qilisdk.core import Parameter
from qilisdk.core.qtensor import QTensor
from qilisdk.digital import RX, Circuit, X
from qilisdk.noise import (
    AmplitudeDamping,
    BitFlip,
    Dephasing,
    Depolarizing,
    KrausChannel,
    LindbladGenerator,
    NoiseModel,
    OffsetPerturbation,
    PauliChannel,
    PhaseFlip,
    ReadoutAssignment,
)


def test_kraus_matrices_of_every_noise_type():
    ops = [
        QTensor(np.array([[1, 0], [0, np.sqrt(0.8)]])),
        QTensor(np.array([[0, np.sqrt(0.2)], [0, 0]])),
    ]
    noises = [
        BitFlip(probability=0.5),
        PhaseFlip(probability=0.5),
        Depolarizing(probability=0.5),
        PauliChannel(pX=0.2, pY=0.3, pZ=0.1),
        KrausChannel(operators=ops),
        AmplitudeDamping(t1=1.0),
        Dephasing(t_phi=1.0),
    ]
    for noise in noises:
        matrices = _kraus_matrices(noise=noise, gate_duration=1.0)
        assert matrices is not None
        assert all(matrix.shape == (2, 2) for matrix in matrices)

    # A certain bit flip is exactly the X error, with no identity branch left.
    certain_flip = _kraus_matrices(noise=BitFlip(probability=1.0), gate_duration=1.0)
    assert certain_flip is not None
    assert np.allclose(certain_flip[-1], np.array([[0, 1], [1, 0]]))

    # Anything that carries no Kraus representation is ignored rather than converted.
    assert _kraus_matrices(noise="bad noise", gate_duration=1.0) is None
    assert _kraus_matrices(noise=ReadoutAssignment(p01=0.1, p10=0.1), gate_duration=1.0) is None


def test_handle_readout_errors():
    cudaq_results = {"01": 100}
    noise_model = NoiseModel()
    noise_model.add(ReadoutAssignment(p01=0.0, p10=1.0))
    noise_model.add(ReadoutAssignment(p01=1.0, p10=0.0), qubits=[1])
    adjusted_results = CudaBackend._handle_readout_errors(cudaq_results, noise_model, nqubits=2)
    assert adjusted_results != cudaq_results
    total_counts = sum(adjusted_results.values())
    assert total_counts == 100
    assert all(bitstring in ["00", "01", "10", "11"] for bitstring in adjusted_results)
    assert adjusted_results["10"] == 100


def test_no_readout_errors():
    cudaq_results = {"01": 100}
    noise_model = NoiseModel()
    adjusted_results = CudaBackend._handle_readout_errors(cudaq_results, noise_model, nqubits=2)
    assert adjusted_results == cudaq_results


def _gate_channel(cuda_noise_model, gate_name, qubits):
    """Return the single channel registered for a gate invocation on the given qubits."""
    channels = cuda_noise_model.get_channels(gate_name, qubits)
    assert len(channels) == 1
    return channels[0]


def _channel_operators(cuda_noise_model, gate_name, qubits):
    """Return the Kraus matrices of the single channel registered for a gate invocation."""
    return [np.array(operator) for operator in _gate_channel(cuda_noise_model, gate_name, qubits).get_ops()]


def test_noise_model_to_cudaq():
    backend = CudaBackend()
    noise_model = NoiseModel()
    single_qubit_kraus = KrausChannel(
        operators=[
            QTensor(np.array([[1, 0], [0, 1]])),
        ]
    )
    two_qubit_kraus = KrausChannel(
        operators=[
            QTensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
        ]
    )
    noise_model.add(single_qubit_kraus)
    noise_model.add(single_qubit_kraus, qubits=[1])
    noise_model.add(single_qubit_kraus, qubits=[0], gate=X)
    noise_model.add(single_qubit_kraus, gate=X)
    noise_model.add(two_qubit_kraus)
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model)

    # Every gate gets a single channel, whose operators live on the space of the gate's qubits.
    for qubit in (0, 1):
        operators = _channel_operators(cuda_noise_model, "x", [qubit])
        assert all(operator.shape == (2, 2) for operator in operators)
    operators = _channel_operators(cuda_noise_model, "x", [0, 1])
    assert all(operator.shape == (4, 4) for operator in operators)


def test_noise_model_to_cudaq_per_qubit_noise_targets_only_its_qubit():
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0), qubits=[0])
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model)

    flip = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)

    # On a single-qubit gate the noise needs no embedding, so CUDA-Q gets its own bit-flip channel,
    # and a gate on another qubit gets nothing to apply.
    assert _gate_channel(cuda_noise_model, "x", [0]).noise_type == cudaq.NoiseModelType.BitFlipChannel
    assert np.allclose(_channel_operators(cuda_noise_model, "x", [1])[0], identity)

    # On a controlled gate the noise is embedded at the position of its qubit, control included.
    # CUDA-Q orders qubits little-endian, so the gate's first qubit is the last tensor factor.
    assert np.allclose(_channel_operators(cuda_noise_model, "x", [0, 1])[0], np.kron(identity, flip))
    assert np.allclose(_channel_operators(cuda_noise_model, "x", [1, 0])[0], np.kron(flip, identity))

    # A gate that does not touch the noisy qubit is left alone.
    assert np.allclose(_channel_operators(cuda_noise_model, "x", [1, 2])[0], np.eye(4, dtype=np.complex128))


def test_noise_model_to_cudaq_global_noise_hits_every_qubit_of_the_gate():
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0))
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model)

    flip = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    assert _gate_channel(cuda_noise_model, "x", [0]).noise_type == cudaq.NoiseModelType.BitFlipChannel
    assert np.allclose(_channel_operators(cuda_noise_model, "x", [0, 1])[0], np.kron(flip, flip))


def test_noise_model_to_cudaq_keeps_builtin_channels():
    backend = CudaBackend()
    for noise, noise_type in (
        (BitFlip(probability=0.1), cudaq.NoiseModelType.BitFlipChannel),
        (PhaseFlip(probability=0.1), cudaq.NoiseModelType.PhaseFlipChannel),
        (Depolarizing(probability=0.1), cudaq.NoiseModelType.DepolarizationChannel),
    ):
        noise_model = NoiseModel()
        noise_model.add(noise)
        cuda_noise_model = backend._noise_model_to_cudaq(noise_model)
        # A lone channel on a single-qubit gate is handed to CUDA-Q as its own optimised channel.
        assert _gate_channel(cuda_noise_model, "x", [0]).noise_type == noise_type
        # Once it has to be embedded or composed it becomes a generic Kraus channel.
        assert _gate_channel(cuda_noise_model, "x", [0, 1]).noise_type == cudaq.NoiseModelType.Unknown

    # Two channels on the same gate have to be composed, so neither built-in survives.
    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=0.1))
    noise_model.add(PhaseFlip(probability=0.1), qubits=[0])
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model)
    assert _gate_channel(cuda_noise_model, "x", [0]).noise_type == cudaq.NoiseModelType.Unknown

    # Noise CUDA-Q has no named channel for goes through the generic path.
    assert _to_builtin_cuda_noise(AmplitudeDamping(t1=1.0)) is None
    noise_model = NoiseModel()
    noise_model.add(AmplitudeDamping(t1=1.0))
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model)
    assert _gate_channel(cuda_noise_model, "x", [0]).noise_type == cudaq.NoiseModelType.Unknown


def test_noise_model_to_cudaq_ignores_noise_without_kraus_operators():
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(ReadoutAssignment(p01=0.1, p10=0.1))
    noise_model.add(BitFlip(probability=1.0), qubits=[0])
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model)

    # Readout errors are applied to the samples, not as a gate channel, so only the bit flip is
    # left on the gate.
    flip = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    assert np.allclose(_channel_operators(cuda_noise_model, "x", [0, 1])[0], np.kron(identity, flip))


def test_noise_model_to_cudaq_skips_unplaceable_channels():
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(
        KrausChannel(operators=[QTensor(np.eye(4))]),
        qubits=[0],
    )
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model)

    # A two-qubit channel cannot be placed on the single qubit of an X gate, so it is skipped
    # rather than silently mis-applied.
    assert np.allclose(_channel_operators(cuda_noise_model, "x", [0])[0], np.eye(2, dtype=np.complex128))


def test_embed_and_compose_kraus():
    flip = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    assert np.allclose(_embed_operator(flip, position=0, num_qubits=1), flip)
    assert np.allclose(_embed_operator(flip, position=0, num_qubits=2), np.kron(identity, flip))
    assert np.allclose(_embed_operator(flip, position=1, num_qubits=2), np.kron(flip, identity))

    composed = _compose_kraus([identity, flip], [flip])
    assert len(composed) == 2
    assert np.allclose(composed[0], flip)
    assert np.allclose(composed[1], identity)


def test_bad_kraus():
    bad_kraus = KrausChannel(
        operators=[
            QTensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.5]])),
        ]
    )
    noise_model = NoiseModel()
    noise_model.add(bad_kraus)
    cuda_noise_model = CudaBackend()._noise_model_to_cudaq(noise_model)
    with pytest.raises(RuntimeError, match="are not completely positive"):
        _ = cuda_noise_model.get_channels("x", [0, 1])


def test_parameter_perturbations():
    backend = CudaBackend()
    noise_model = NoiseModel()

    circuit = Circuit(1)
    param1 = Parameter("test1", 0.5)
    param2 = Parameter("test2", 0.5)
    circuit.add(RX(0, theta=param1))
    circuit.add(RX(0, theta=param2))

    perturb = OffsetPerturbation(offset=0.1)
    noise_model.add(perturb, parameter="test1")
    backend._handle_gate_parameter_perturbations(circuit, noise_model)
    assert np.isclose(circuit.get_parameters()["test1"], 0.6)

    noise_model.add(perturb, parameter="theta", gate=RX)

    backend._handle_gate_parameter_perturbations(circuit, noise_model)

    assert np.isclose(circuit.get_parameters()["test1"], 0.8)
    assert np.isclose(circuit.get_parameters()["test2"], 0.6)


def test_parameter_perturbations_errors():
    backend = CudaBackend()
    noise_model = NoiseModel()

    circuit = Circuit(1)
    param1 = Parameter("test1", 0.5)
    param2 = Parameter("test2", 0.5)
    circuit.add(RX(0, theta=param1))
    circuit.add(RX(0, theta=param2))

    perturb = OffsetPerturbation(offset=0.1)
    noise_model.add(perturb, parameter="test_1")

    with pytest.raises(ValueError, match=r"Perturbing Parameter test_1 that doesn't exist in the circuit."):
        backend._handle_gate_parameter_perturbations(circuit, noise_model)

    noise_model = NoiseModel()
    noise_model.add(perturb, gate=RX, parameter="test1")

    with pytest.raises(ValueError, match=r"Invalid parameter name passed to gate."):
        backend._handle_gate_parameter_perturbations(circuit, noise_model)

    assert np.isclose(circuit.get_parameters()["test1"], 0.5)
    assert np.isclose(circuit.get_parameters()["test2"], 0.5)


def test_schedule_parameter_perturbations():
    backend = CudaBackend()
    dt = 1
    param1 = Parameter("test1", 0.5)
    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": param1 * pauli_x(0)},
        total_time=10.0,
    )
    perturb = OffsetPerturbation(offset=0.1)
    noise_model = NoiseModel()
    noise_model.add(perturb, parameter=param1)
    backend._handle_schedule_parameter_perturbations(schedule, noise_model)
    assert np.isclose(schedule.get_parameters()["test1"], 0.6)


def test_noise_model_to_cudaq_dynamics():
    backend = CudaBackend()
    noise_model = NoiseModel()
    ham_noise = 0.1 * pauli_x(0)
    time_derived_lindblad = PauliChannel(pX=0.2, pY=0.0, pZ=0.0)
    single_qubit_jump = LindbladGenerator(
        jump_operators=[
            QTensor(np.array([[0, 1], [0, 0]])),
        ],
        hamiltonian=ham_noise,
    )
    two_qubit_jump = LindbladGenerator(
        jump_operators=[
            QTensor(np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
        ]
    )
    noise_model.add(time_derived_lindblad)  # adds 2 jumps (to qubits 0 and 1)
    noise_model.add(time_derived_lindblad, qubits=[1])  # adds 1 jump (to qubit 1)
    noise_model.add(single_qubit_jump)  # adds 2 jumps and the delta (to qubits 0 and 1)
    noise_model.add(single_qubit_jump, qubits=[1])  # adds 1 jump and the delta (to qubit 1)
    noise_model.add(two_qubit_jump)  # adds 1 jump (to both qubits)
    cuda_noise_model = backend._noise_model_to_cudaq_dynamics(noise_model, nqubits=2, dt=1.0)
    assert len(cuda_noise_model[0]) == 7  # jump operators
    assert len(cuda_noise_model[1]) == 2  # hamiltonian deltas


def test_global_single_qubit_lindblad_uses_single_degree_operator(monkeypatch):
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(Dephasing(t_phi=1.0))

    expected_dimensions_per_id = {}
    instantiate_calls = []

    def fake_define(id, expected_dimensions, create, override):
        expected_dimensions_per_id[id] = list(expected_dimensions)

    def fake_instantiate(id, degrees):
        degree_list = [degrees] if isinstance(degrees, int) else list(degrees)
        assert len(expected_dimensions_per_id[id]) == len(degree_list)
        instantiate_calls.append((id, degree_list))
        return object()

    monkeypatch.setattr("qilisdk.backends.cuda_backend.operators.define", fake_define)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.operators.instantiate", fake_instantiate)

    jump_operators, hamiltonian_deltas = backend._noise_model_to_cudaq_dynamics(noise_model, nqubits=2, dt=1.0)

    assert len(jump_operators) == 2
    assert len(hamiltonian_deltas) == 0
    assert len(instantiate_calls) == 2
    assert all(expected_dimensions == [2] for expected_dimensions in expected_dimensions_per_id.values())
