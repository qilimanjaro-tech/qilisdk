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

from qilisdk.analog import X as PauliX
from qilisdk.analog import Z as PauliZ
from qilisdk.analog.hamiltonian import PauliX as PauliXOperator
from qilisdk.analog.hamiltonian import PauliZ as PauliZOperator
from qilisdk.backends import CudaqBackend, QutipBackend
from qilisdk.core import QTensor, ket
from qilisdk.digital import Circuit, X
from qilisdk.functionals import DigitalPropagation
from qilisdk.noise import AmplitudeDamping, BitFlip, Dephasing, EnvironmentNoise, LindbladGenerator, NoiseModel
from qilisdk.noise.protocols import (
    AttachmentScope,
    SupportsStaticKraus,
    SupportsStaticLindblad,
    SupportsTimeDerivedKraus,
    SupportsTimeDerivedLindblad,
)
from qilisdk.readout import Readout


def _environment(**kwargs):
    defaults = {"n_environment_qubits": 1, "couplings": [(1.0, PauliZ(0), PauliZ(0))]}
    return EnvironmentNoise(**(defaults | kwargs))


def test_properties():
    hamiltonian = 0.5 * PauliX(0)
    noise = {0: [AmplitudeDamping(t1=1.0)]}
    environment = _environment(environment_hamiltonian=hamiltonian, environment_noise=noise)

    assert environment.n_environment_qubits == 1
    assert len(environment.couplings) == 1
    assert environment.environment_hamiltonian is hamiltonian
    assert environment.environment_noise == noise


def test_defaults():
    environment = _environment()

    assert environment.environment_hamiltonian is None
    assert environment.environment_noise == {}


def test_default_environment_state_is_all_zeros():
    environment = _environment(n_environment_qubits=2)

    np.testing.assert_allclose(environment.environment_state.dense(), ket(0, 0).dense())


def test_given_environment_state_is_kept():
    state = (ket(0) + ket(1)).unit()
    environment = _environment(environment_state=state)

    np.testing.assert_allclose(environment.environment_state.dense(), state.dense())


def test_only_global_scope_allowed():
    assert EnvironmentNoise.allowed_scopes() == frozenset({AttachmentScope.GLOBAL})

    noise_model = NoiseModel()
    with pytest.raises(ValueError, match=r"cannot be added with scope"):
        noise_model.add(_environment(), qubits=[0])
    with pytest.raises(ValueError, match=r"cannot be added with scope"):
        noise_model.add(_environment(), gate=X)


def test_not_seen_as_markovian_noise():
    # The C++ noise parser dispatches on these protocols, so EnvironmentNoise must match none of them
    environment = _environment()

    for protocol in (
        SupportsStaticKraus,
        SupportsTimeDerivedKraus,
        SupportsStaticLindblad,
        SupportsTimeDerivedLindblad,
    ):
        assert not isinstance(environment, protocol)


def test_repr():
    representation = repr(_environment())

    assert "EnvironmentNoise" in representation
    assert "n_environment_qubits=1" in representation


@pytest.mark.parametrize("n_environment_qubits", [0, -1])
def test_non_positive_environment_qubits_raises(n_environment_qubits):
    with pytest.raises(ValueError, match=r"n_environment_qubits"):
        _environment(n_environment_qubits=n_environment_qubits)


def test_coupling_environment_index_out_of_range_raises():
    with pytest.raises(ValueError, match=r"out of range"):
        _environment(couplings=[(1.0, PauliZ(0), PauliZ(1))])


def test_environment_hamiltonian_index_out_of_range_raises():
    with pytest.raises(ValueError, match=r"out of range"):
        _environment(environment_hamiltonian=PauliX(1))


def test_environment_noise_index_out_of_range_raises():
    with pytest.raises(ValueError, match=r"out of range"):
        _environment(environment_noise={1: [AmplitudeDamping(t1=1.0)]})


def test_environment_noise_without_lindblad_form_raises():
    with pytest.raises(ValueError, match=r"Lindblad"):
        _environment(environment_noise={0: [BitFlip(probability=0.1)]})


def test_environment_state_wrong_dimension_raises():
    with pytest.raises(ValueError, match=r"dimension"):
        _environment(environment_state=ket(0, 0))


def test_hamiltonian_with_environment():
    environment = _environment(
        n_environment_qubits=2,
        couplings=[(2.0, PauliZ(0), PauliX(1))],
        environment_hamiltonian=0.5 * PauliZ(0),
    )

    # Two system qubits, environment qubits at 2 and 3
    expected = 2.0 * PauliZ(0) * PauliX(3) + 0.5 * PauliZ(2)
    np.testing.assert_allclose(
        environment.as_hamiltonian_with_environment(nqubits=2).to_matrix().toarray(),
        expected.to_matrix().toarray(),
    )


def test_hamiltonian_with_environment_offset():
    environment = _environment(couplings=[(1.0, PauliZ(0), PauliZ(0))])

    # A previous environment already holds qubit 1, so this one starts at qubit 2
    expected = PauliZ(0) * PauliZ(2)
    np.testing.assert_allclose(
        environment.as_hamiltonian_with_environment(nqubits=1, offset=1).to_matrix().toarray(),
        expected.to_matrix().toarray(),
    )


def test_hamiltonian_with_environment_accepts_pauli_operators():
    with_operators = _environment(couplings=[(1.5, PauliZOperator(0), PauliXOperator(0))])
    with_hamiltonians = _environment(couplings=[(1.5, PauliZ(0), PauliX(0))])

    np.testing.assert_allclose(
        with_operators.as_hamiltonian_with_environment(nqubits=1).to_matrix().toarray(),
        with_hamiltonians.as_hamiltonian_with_environment(nqubits=1).to_matrix().toarray(),
    )


def test_hamiltonian_with_environment_system_index_out_of_range_raises():
    environment = _environment(couplings=[(1.0, PauliZ(1), PauliZ(0))])

    with pytest.raises(ValueError, match=r"system qubits"):
        environment.as_hamiltonian_with_environment(nqubits=1)


def test_lindblad_with_environment():
    environment = _environment(n_environment_qubits=2, environment_noise={1: [AmplitudeDamping(t1=4.0)]})

    generator = environment.as_lindblad_with_environment(nqubits=1)

    # Damping on environment qubit 1 is qubit 2 of the full register: I (x) I (x) L
    local = AmplitudeDamping(t1=4.0).as_lindblad().jump_operators_with_rates[0].dense()
    expected = np.kron(np.eye(4), local)
    assert len(generator.jump_operators_with_rates) == 1
    np.testing.assert_allclose(generator.jump_operators_with_rates[0].dense(), expected)


def test_lindblad_with_environment_several_noises():
    environment = _environment(environment_noise={0: [AmplitudeDamping(t1=1.0), Dephasing(t_phi=1.0)]})

    generator = environment.as_lindblad_with_environment(nqubits=1, offset=0)

    assert len(generator.jump_operators_with_rates) == 2
    for operator in generator.jump_operators_with_rates:
        assert operator.shape == (4, 4)


def test_lindblad_with_environment_generator_without_rates():
    lowering = QTensor(np.array([[0.0, 1.0], [0.0, 0.0]]))
    environment = _environment(environment_noise={0: [LindbladGenerator(jump_operators=[lowering])]})

    generator = environment.as_lindblad_with_environment(nqubits=1)

    np.testing.assert_allclose(generator.jump_operators_with_rates[0].dense(), np.kron(np.eye(2), lowering.dense()))


def test_lindblad_with_environment_multi_qubit_jump_raises():
    environment = _environment(environment_noise={0: [LindbladGenerator(jump_operators=[QTensor(np.eye(4))])]})

    with pytest.raises(ValueError, match=r"single-qubit"):
        environment.as_lindblad_with_environment(nqubits=1)


def test_lindblad_with_environment_no_noise():
    generator = _environment().as_lindblad_with_environment(nqubits=1)

    assert generator.jump_operators_with_rates == []


def test_noise_model_non_markovian_noise():
    environment = _environment()
    noise_model = NoiseModel()
    noise_model.add(AmplitudeDamping(t1=1.0))
    assert noise_model.non_markovian_noise == []

    noise_model.add(environment)
    assert noise_model.non_markovian_noise == [environment]


@pytest.mark.parametrize("backend_class", [QutipBackend, CudaqBackend])
def test_unsupported_backends_raise(backend_class):
    noise_model = NoiseModel()
    noise_model.add(_environment())
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    backend = backend_class(noise_model=noise_model)
    with pytest.raises(NotImplementedError, match=r"does not support non-Markovian noise"):
        backend.execute(DigitalPropagation(circuit), readout=Readout().with_sampling(nshots=10))
