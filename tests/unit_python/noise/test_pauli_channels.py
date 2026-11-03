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
from loguru_caplog import loguru_caplog as caplog  # ruff: ignore[unused-import]
from scipy.linalg import expm

from qilisdk.noise import BitFlip, Depolarizing, KrausChannel, PauliChannel, PhaseFlip
from qilisdk.noise.protocols import AttachmentScope


def test_pauli_channel_properties_and_kraus():
    x_rate = 0.25
    y_rate = 0.1
    z_rate = 0.15
    identity_rate = 1.0 - (x_rate + y_rate + z_rate)
    channel = PauliChannel(pX=x_rate, pY=y_rate, pZ=z_rate)

    assert np.isclose(channel.pX, x_rate)
    assert np.isclose(channel.pY, y_rate)
    assert np.isclose(channel.pZ, z_rate)

    kraus = channel.as_kraus()
    assert isinstance(kraus, KrausChannel)
    assert len(kraus.operators) == 4

    expected_identity = np.sqrt(identity_rate) * np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    expected_x = np.sqrt(x_rate) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    expected_y = np.sqrt(y_rate) * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    expected_z = np.sqrt(z_rate) * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    np.testing.assert_allclose(kraus.operators[0].dense(), expected_identity)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_x)
    np.testing.assert_allclose(kraus.operators[2].dense(), expected_y)
    np.testing.assert_allclose(kraus.operators[3].dense(), expected_z)

    # need to re.escape the regex
    with pytest.raises(ValueError, match=r"pX must be in \[0, 1\]."):
        PauliChannel(pX=-0.1)
    with pytest.raises(ValueError, match=r"pX \+ pY \+ pZ must be <= 1."):
        PauliChannel(pX=0.6, pY=0.6)

    jump_operators = channel.as_lindblad_from_duration(duration=1.0)
    assert len(jump_operators.jump_operators) == 3
    with pytest.raises(ValueError, match="Duration must be positive"):
        channel.as_lindblad_from_duration(duration=-1.0)


def test_bit_flip_probability_and_scopes():
    noise = BitFlip(probability=0.2)

    assert np.isclose(noise.probability, 0.2)
    assert np.isclose(noise.pX, 0.2)
    assert np.isclose(noise.pY, 0.0)
    assert np.isclose(noise.pZ, 0.0)

    scopes = BitFlip.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT in scopes

    kraus = noise.as_kraus()
    assert len(kraus.operators) == 2

    expected_x = np.sqrt(0.2) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_x)

    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        BitFlip(probability=1.2)


def test_phase_flip_probability_and_scopes():
    noise = PhaseFlip(probability=0.3)

    assert np.isclose(noise.probability, 0.3)
    assert np.isclose(noise.pX, 0.0)
    assert np.isclose(noise.pY, 0.0)
    assert np.isclose(noise.pZ, 0.3)
    scopes = PhaseFlip.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT in scopes

    kraus = noise.as_kraus()
    assert len(kraus.operators) == 2

    expected_z = np.sqrt(0.3) * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_z)

    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        PhaseFlip(probability=-0.1)


def test_depolarizing_probability_distribution():
    noise = Depolarizing(probability=0.3)

    scopes = Depolarizing.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT in scopes

    assert np.isclose(noise.probability, 0.3)
    assert np.isclose(noise.pX, 0.1)
    assert np.isclose(noise.pY, 0.1)
    assert np.isclose(noise.pZ, 0.1)

    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        Depolarizing(probability=-0.1)


def test_repr():
    noise = PauliChannel(pX=0.2, pY=0.1, pZ=0.15)
    repr_str = str(noise)
    assert "PauliChannel" in repr_str
    assert "pX=0.2" in repr_str
    assert "pY=0.1" in repr_str
    assert "pZ=0.15" in repr_str


def _kraus_apply(channel, rho):
    return sum(K.dense() @ rho @ K.dense().conj().T for K in channel.operators)


def _lindblad_evolve(generator, rho, duration):
    identity = np.eye(rho.shape[0])
    superoperator = np.zeros((rho.size, rho.size), dtype=complex)
    for jump, rate in zip(generator.jump_operators, generator.rates):
        L = jump.dense()
        LdL = L.conj().T @ L
        superoperator += rate * (np.kron(L, L.conj()) - 0.5 * (np.kron(LdL, identity) + np.kron(identity, LdL.T)))
    return (expm(superoperator * duration) @ rho.reshape(-1)).reshape(rho.shape)


@pytest.mark.parametrize(
    "channel",
    [
        PauliChannel(pZ=0.4),
        PauliChannel(pX=0.1),
        PauliChannel(pX=0.25, pY=0.1, pZ=0.15),
        BitFlip(probability=0.3),
        PhaseFlip(probability=0.01),
        Depolarizing(probability=0.6),
    ],
)
@pytest.mark.parametrize("duration", [0.5, 2.0])
def test_lindblad_from_duration_matches_kraus(channel, duration):
    rng = np.random.default_rng(1234)
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    kraus_rho = _kraus_apply(channel.as_kraus(), rho)
    lindblad_rho = _lindblad_evolve(channel.as_lindblad_from_duration(duration=duration), rho, duration)

    np.testing.assert_allclose(lindblad_rho, kraus_rho, atol=1e-10)


def test_lindblad_from_duration_single_axis_rate():
    generator = PhaseFlip(probability=0.4).as_lindblad_from_duration(duration=2.0)

    assert len(generator.jump_operators) == 1
    np.testing.assert_allclose(generator.jump_operators[0].dense(), np.diag([1.0, -1.0]))
    assert np.isclose(generator.rates[0], -np.log(1 - 2 * 0.4) / (2 * 2.0))


@pytest.mark.parametrize(
    "channel",
    [Depolarizing(probability=0.75), BitFlip(probability=0.5), PhaseFlip(probability=0.5), PauliChannel(pX=0.5)],
)
def test_lindblad_from_duration_full_decoherence_boundary(channel, caplog):  # ruff: ignore[redefined-while-unused]
    rho = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex) / 2
    generator = channel.as_lindblad_from_duration(duration=1.0)

    kraus_rho = _kraus_apply(channel.as_kraus(), rho)
    lindblad_rho = _lindblad_evolve(generator, rho, 1.0)

    np.testing.assert_allclose(lindblad_rho, kraus_rho, atol=1e-10)
    assert "not reachable by a Lindbladian" not in caplog.text


@pytest.mark.parametrize(
    "channel",
    [Depolarizing(probability=0.9), BitFlip(probability=0.7), PauliChannel(pX=1.0), PauliChannel(pX=0.3, pY=0.3)],
)
def test_lindblad_from_duration_approximates_inverting_channels(channel, caplog):  # ruff: ignore[redefined-while-unused]
    generator = channel.as_lindblad_from_duration(duration=1.0)

    assert "not reachable by a Lindbladian" in caplog.text
    assert len(generator.jump_operators) > 0
    assert all(np.isfinite(rate) and rate > 0 for rate in generator.rates)


def test_lindblad_from_duration_depolarizing_beyond_boundary_fully_mixes():
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    generator = Depolarizing(probability=0.9).as_lindblad_from_duration(duration=1.0)

    np.testing.assert_allclose(_lindblad_evolve(generator, rho, 1.0), np.eye(2) / 2, atol=1e-10)


def test_lindblad_from_duration_clamps_negative_rates(caplog):  # ruff: ignore[redefined-while-unused]
    generator = PauliChannel(pX=0.2, pY=0.2).as_lindblad_from_duration(duration=1.0)

    assert "not reachable by a Lindbladian" in caplog.text
    assert len(generator.jump_operators) == 2
    assert all(rate > 0 for rate in generator.rates)


def test_lindblad_from_duration_warns_once_per_channel(caplog):  # ruff: ignore[redefined-while-unused]
    channel = PauliChannel(pX=0.2, pY=0.2)
    for _ in range(3):
        channel.as_lindblad_from_duration(duration=1.0)

    assert caplog.text.count("not reachable by a Lindbladian") == 1


def test_lindblad_from_duration_short_duration_drops_cancelled_rate(caplog):  # ruff: ignore[redefined-while-unused]
    q1, q2 = 0.37, 0.2
    channel = PauliChannel(pX=q1 * (1 - q2), pY=q2 * (1 - q1), pZ=q1 * q2)
    generator = channel.as_lindblad_from_duration(duration=1e-9)

    assert "not reachable by a Lindbladian" not in caplog.text
    assert len(generator.jump_operators) == 2
    np.testing.assert_allclose(generator.jump_operators[0].dense(), np.array([[0.0, 1.0], [1.0, 0.0]]))
    np.testing.assert_allclose(generator.jump_operators[1].dense(), np.array([[0.0, -1j], [1j, 0.0]]))


def test_lindblad_from_duration_keeps_tiny_probabilities():
    generator = PauliChannel(pX=1e-17).as_lindblad_from_duration(duration=1.0)

    assert len(generator.jump_operators) == 1
    assert np.isclose(generator.rates[0], 1e-17, rtol=1e-6)
