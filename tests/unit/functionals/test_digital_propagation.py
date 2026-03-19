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
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.settings import get_settings


def _isclose(lhs: float, rhs: float) -> bool:
    return bool(np.isclose(lhs, rhs, atol=get_settings().atol, rtol=get_settings().rtol))


def test_digital_propagation():
    circuit = Circuit(2)
    param = Parameter("theta", 0.5)
    circuit.add(RZ(0, phi=param))
    functional = DigitalPropagation(circuit=circuit)

    functional.set_parameters({"theta": 0.3})
    functional.set_parameter_values([1.0])
    functional.set_parameter_bounds({"theta": (0.0, 2.0)})

    assert functional.circuit == circuit
    assert functional.nparameters == 1
    assert _isclose(functional.get_parameters()["theta"], 1.0)
    assert functional.get_parameter_names() == ["theta"]
    assert _isclose(functional.get_parameter_bounds()["theta"][0], 0.0)
    assert _isclose(functional.get_parameter_bounds()["theta"][1], 2.0)
    assert _isclose(functional.get_parameter_values()[0], 1.0)
    assert functional.get_constraints() == []


def test_digital_propagation_parameter_sync_with_circuit_child():
    circuit = Circuit(1)
    theta = Parameter("theta", 0.5, bounds=(0.0, 2.0))
    phi_nt = Parameter("phi_nt", 0.2, trainable=False)
    circuit.add(RZ(0, phi=theta))
    circuit.add(RZ(0, phi=phi_nt))
    functional = DigitalPropagation(circuit=circuit)

    functional.set_parameters({"theta": 0.7, "phi_nt": 0.25})
    assert _isclose(circuit.get_parameters()["theta"], 0.7)
    assert _isclose(circuit.get_parameters()["phi_nt"], 0.25)

    circuit.set_parameters({"theta": 0.9, "phi_nt": 0.3})
    assert _isclose(functional.get_parameters()["theta"], 0.9)
    assert _isclose(functional.get_parameters()["phi_nt"], 0.3)
    assert _isclose(functional.get_parameters(where=lambda param: param.is_trainable)["theta"], 0.9)
    assert _isclose(functional.get_parameters(where=lambda param: not param.is_trainable)["phi_nt"], 0.3)

    functional.set_parameter_values([1.1], where=lambda param: param.is_trainable)
    assert _isclose(circuit.get_parameters()["theta"], 1.1)
    assert _isclose(circuit.get_parameters()["phi_nt"], 0.3)

    functional.set_parameter_bounds({"theta": (-1.0, 2.0), "phi_nt": (0.3, 0.3)})
    assert _isclose(circuit.get_parameter_bounds()["theta"][0], -1.0)
    assert _isclose(circuit.get_parameter_bounds()["theta"][1], 2.0)
    assert _isclose(functional.get_parameter_bounds(where=lambda param: not param.is_trainable)["phi_nt"][0], 0.3)
    assert _isclose(functional.get_parameter_bounds(where=lambda param: not param.is_trainable)["phi_nt"][1], 0.3)


def test_repr():
    circuit = Circuit(2)
    param = Parameter("theta", 0.5)
    circuit.add(RZ(0, phi=param))
    functional = DigitalPropagation(circuit=circuit)
    repr_str = str(functional)
    assert "DigitalPropagation" in repr_str
    assert "circuit=" in repr_str
