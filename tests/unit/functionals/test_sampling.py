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

from qilisdk.core import Parameter
from qilisdk.digital import RZ, Circuit
from qilisdk.functionals.sampling import Sampling
from qilisdk.settings import get_settings


def _isclose(lhs: float, rhs: float) -> bool:
    return bool(np.isclose(lhs, rhs, atol=get_settings().atol, rtol=get_settings().rtol))


def test_sampling():
    circuit = Circuit(2)
    param = Parameter("theta", 0.5)
    circuit.add(RZ(0, phi=param))
    nshots = 1024
    sampling = Sampling(circuit=circuit, nshots=nshots)

    sampling.set_parameters({"theta": 0.3})
    sampling.set_parameter_values([1.0])
    sampling.set_parameter_bounds({"theta": (0.0, 2.0)})

    assert sampling.circuit == circuit
    assert sampling.nshots == nshots
    assert sampling.nparameters == 1
    assert _isclose(sampling.get_parameters()["theta"], 1.0)
    assert sampling.get_parameter_names() == ["theta"]
    assert _isclose(sampling.get_parameter_bounds()["theta"][0], 0.0)
    assert _isclose(sampling.get_parameter_bounds()["theta"][1], 2.0)
    assert _isclose(sampling.get_parameter_values()[0], 1.0)
    assert sampling.get_constraints() == []


def test_sampling_parameter_sync_with_circuit_child():
    circuit = Circuit(1)
    theta = Parameter("theta", 0.5, bounds=(0.0, 2.0))
    phi_nt = Parameter("phi_nt", 0.2, trainable=False)
    circuit.add(RZ(0, phi=theta))
    circuit.add(RZ(0, phi=phi_nt))
    sampling = Sampling(circuit=circuit, nshots=10)

    sampling.set_parameters({"theta": 0.7, "phi_nt": 0.25})
    assert _isclose(circuit.get_parameters()["theta"], 0.7)
    assert _isclose(circuit.get_parameters()["phi_nt"], 0.25)

    circuit.set_parameters({"theta": 0.9, "phi_nt": 0.3})
    assert _isclose(sampling.get_parameters()["theta"], 0.9)
    assert _isclose(sampling.get_parameters()["phi_nt"], 0.3)
    assert _isclose(sampling.get_parameters(where=lambda param: param.is_trainable)["theta"], 0.9)
    assert _isclose(sampling.get_parameters(where=lambda param: not param.is_trainable)["phi_nt"], 0.3)

    sampling.set_parameter_values([1.1], where=lambda param: param.is_trainable)
    assert _isclose(circuit.get_parameters()["theta"], 1.1)
    assert _isclose(circuit.get_parameters()["phi_nt"], 0.3)

    sampling.set_parameter_bounds({"theta": (-1.0, 2.0), "phi_nt": (0.3, 0.3)})
    assert _isclose(circuit.get_parameter_bounds()["theta"][0], -1.0)
    assert _isclose(circuit.get_parameter_bounds()["theta"][1], 2.0)
    assert _isclose(sampling.get_parameter_bounds(where=lambda param: not param.is_trainable)["phi_nt"][0], 0.3)
    assert _isclose(sampling.get_parameter_bounds(where=lambda param: not param.is_trainable)["phi_nt"][1], 0.3)
