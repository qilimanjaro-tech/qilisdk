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

"""Non-Markovian noise on the external backends, which must reject it. Runs in the external backend CI jobs."""

import pytest

pytest.importorskip("qutip", reason="QuTiP backend tests require the 'qutip' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "qutip_qip",
    reason="QuTiP backend tests require the 'qutip' optional dependency",
    exc_type=ImportError,
)
pytest.importorskip("cudaq", reason="CUDA backend tests require the 'cuda' optional dependency", exc_type=ImportError)

from qilisdk.analog import Schedule
from qilisdk.analog import X as PauliX
from qilisdk.analog import Z as PauliZ
from qilisdk.backends import CudaqBackend, QutipBackend
from qilisdk.core import ket
from qilisdk.digital import Circuit, X
from qilisdk.functionals import AnalogEvolution, DigitalPropagation
from qilisdk.noise import AmplitudeDamping, EnvironmentNoise, NoiseModel
from qilisdk.readout import Readout


def _environment():
    return EnvironmentNoise(n_environment_qubits=1, couplings=[(1.0, PauliZ(0), PauliZ(0))])


def _digital_propagation():
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))
    return DigitalPropagation(circuit)


def _analog_evolution():
    schedule = Schedule(hamiltonians={"h": PauliX(0)}, coefficients={"h": {0.0: 1.0, 1.0: 1.0}}, dt=0.1)
    return AnalogEvolution(schedule=schedule, initial_state=ket(0))


@pytest.mark.parametrize("backend_class", [QutipBackend, CudaqBackend])
@pytest.mark.parametrize("make_functional", [_digital_propagation, _analog_evolution])
def test_unsupported_backends_raise(backend_class, make_functional):
    noise_model = NoiseModel()
    noise_model.add(_environment())

    backend = backend_class(noise_model=noise_model)
    functional = make_functional()
    readout = Readout().with_sampling(nshots=10)
    with pytest.raises(NotImplementedError, match=r"does not support non-Markovian noise"):
        backend.execute(functional, readout=readout)


def test_backend_support_flags():
    assert not QutipBackend._supports_non_markovian_noise
    assert not CudaqBackend._supports_non_markovian_noise


def test_unsupported_backend_accepts_markovian_only_noise_model():
    noise_model = NoiseModel()
    noise_model.add(AmplitudeDamping(t1=1.0))

    result = QutipBackend(noise_model=noise_model).execute(
        _digital_propagation(), readout=Readout().with_sampling(nshots=10)
    )

    assert sum(result.get_samples().values()) == 10


@pytest.mark.parametrize("backend_class", [QutipBackend, CudaqBackend])
def test_unsupported_backends_raise_for_environment_added_after_construction(backend_class):
    noise_model = NoiseModel()
    noise_model.add(AmplitudeDamping(t1=1.0))
    backend = backend_class(noise_model=noise_model)

    noise_model.add(_environment())

    functional = _digital_propagation()
    readout = Readout().with_sampling(nshots=10)
    with pytest.raises(NotImplementedError, match=r"does not support non-Markovian noise"):
        backend.execute(functional, readout=readout)
