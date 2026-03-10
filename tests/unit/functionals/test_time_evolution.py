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

from qilisdk.analog.hamiltonian import Hamiltonian, PauliZ
from qilisdk.analog.schedule import Schedule
from qilisdk.core import Parameter
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.settings import get_settings


def _isclose(lhs: float, rhs: float) -> bool:
    return bool(np.isclose(lhs, rhs, atol=get_settings().atol, rtol=get_settings().rtol))


def test_time_evolution_initial_state_qubits_mismatch_raises():
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = tensor_prod([ket(0), ket(0)]).unit()

    with pytest.raises(ValueError, match=r"The initial state provided acts on 2 qubits"):
        TimeEvolution(schedule=schedule, observables=[PauliZ(0)], initial_state=initial_state)


def test_time_evolution_properties():
    param = Parameter("theta", 0.5)
    hamiltonian = Hamiltonian({(PauliZ(0),): param})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = ket(0).unit()
    observables = [PauliZ(0)]

    time_evolution = TimeEvolution(schedule=schedule, observables=observables, initial_state=initial_state)

    time_evolution.set_parameters({"theta": 0.3})
    time_evolution.set_parameter_values([1.0])
    time_evolution.set_parameter_bounds({"theta": (0.0, 2.0)})

    assert time_evolution.schedule == schedule
    assert time_evolution.initial_state == initial_state
    assert time_evolution.observables == observables
    assert time_evolution.nparameters == 1
    assert _isclose(time_evolution.get_parameters()["theta"], 1.0)
    assert time_evolution.get_parameter_names() == ["theta"]
    assert _isclose(time_evolution.get_parameter_bounds()["theta"][0], 0.0)
    assert _isclose(time_evolution.get_parameter_bounds()["theta"][1], 2.0)
    assert _isclose(time_evolution.get_parameter_values()[0], 1.0)
    assert time_evolution.get_parameter_names(where=lambda param: param.is_trainable) == ["theta"]
    assert _isclose(time_evolution.get_parameters(where=lambda param: param.is_trainable)["theta"], 1.0)
    assert _isclose(time_evolution.get_parameter_bounds(where=lambda param: param.is_trainable)["theta"][0], 0.0)
    assert _isclose(time_evolution.get_parameter_bounds(where=lambda param: param.is_trainable)["theta"][1], 2.0)
    assert time_evolution.get_constraints() == []


def test_time_evolution_parameter_sync_with_schedule_child():
    theta = Parameter("theta", 0.5, bounds=(0.0, 2.0))
    coeff = Parameter("coeff", 0.2, trainable=False)
    hamiltonian = Hamiltonian({(PauliZ(0),): theta})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, coefficients={"h": {0: coeff, 1: 1.0}}, dt=0.1)
    time_evolution = TimeEvolution(schedule=schedule, observables=[PauliZ(0)], initial_state=ket(0).unit())

    time_evolution.set_parameters({"theta": 0.7, "coeff": 0.3})
    assert _isclose(schedule.get_parameters()["theta"], 0.7)
    assert _isclose(schedule.get_parameters()["coeff"], 0.3)

    schedule.set_parameters({"theta": 0.8, "coeff": 0.4})
    assert _isclose(time_evolution.get_parameters()["theta"], 0.8)
    assert _isclose(time_evolution.get_parameters()["coeff"], 0.4)
    assert _isclose(time_evolution.get_parameters(where=lambda param: param.is_trainable)["theta"], 0.8)
    assert _isclose(time_evolution.get_parameters(where=lambda param: not param.is_trainable)["coeff"], 0.4)

    time_evolution.set_parameter_values([1.1], where=lambda param: param.is_trainable)
    assert _isclose(schedule.get_parameters()["theta"], 1.1)
    assert _isclose(schedule.get_parameters()["coeff"], 0.4)
