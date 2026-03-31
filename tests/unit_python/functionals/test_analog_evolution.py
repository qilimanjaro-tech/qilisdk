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
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.settings import get_settings


def _isclose(lhs: float, rhs: float) -> bool:
    return bool(np.isclose(lhs, rhs, atol=get_settings().atol, rtol=get_settings().rtol))


def test_analog_evolution_initial_state_qubits_mismatch_raises():
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = tensor_prod([ket(0), ket(0)]).unit()

    with pytest.raises(ValueError, match=r"The initial state provided acts on 2 qubits"):
        AnalogEvolution(schedule=schedule, initial_state=initial_state)


def test_analog_evolution_properties():
    param = Parameter("theta", 0.5)
    hamiltonian = Hamiltonian({(PauliZ(0),): param})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = ket(0).unit()

    functional = AnalogEvolution(schedule=schedule, initial_state=initial_state)

    functional.set_parameters({"theta": 0.3})
    functional.set_parameter_values([1.0])
    functional.set_parameter_bounds({"theta": (0.0, 2.0)})

    assert functional.schedule == schedule
    assert functional.initial_state == initial_state
    assert functional.nparameters == 1
    assert _isclose(functional.get_parameters()["theta"], 1.0)
    assert functional.get_parameter_names() == ["theta"]
    assert _isclose(functional.get_parameter_bounds()["theta"][0], 0.0)
    assert _isclose(functional.get_parameter_bounds()["theta"][1], 2.0)
    assert _isclose(functional.get_parameter_values()[0], 1.0)
    assert functional.get_parameter_names(where=lambda param: param.is_trainable) == ["theta"]
    assert _isclose(functional.get_parameters(where=lambda param: param.is_trainable)["theta"], 1.0)
    assert _isclose(functional.get_parameter_bounds(where=lambda param: param.is_trainable)["theta"][0], 0.0)
    assert _isclose(functional.get_parameter_bounds(where=lambda param: param.is_trainable)["theta"][1], 2.0)
    assert functional.get_constraints() == []


def test_analog_evolution_parameter_sync_with_schedule_child():
    theta = Parameter("theta", 0.5, bounds=(0.0, 2.0))
    coeff = Parameter("coeff", 0.2, trainable=False)
    hamiltonian = Hamiltonian({(PauliZ(0),): theta})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, coefficients={"h": {0: coeff, 1: 1.0}}, dt=0.1)
    functional = AnalogEvolution(schedule=schedule, initial_state=ket(0).unit())

    functional.set_parameters({"theta": 0.7, "coeff": 0.3})
    assert _isclose(schedule.get_parameters()["theta"], 0.7)
    assert _isclose(schedule.get_parameters()["coeff"], 0.3)

    schedule.set_parameters({"theta": 0.8, "coeff": 0.4})
    assert _isclose(functional.get_parameters()["theta"], 0.8)
    assert _isclose(functional.get_parameters()["coeff"], 0.4)
    assert _isclose(functional.get_parameters(where=lambda param: param.is_trainable)["theta"], 0.8)
    assert _isclose(functional.get_parameters(where=lambda param: not param.is_trainable)["coeff"], 0.4)

    functional.set_parameter_values([1.1], where=lambda param: param.is_trainable)
    assert _isclose(schedule.get_parameters()["theta"], 1.1)
    assert _isclose(schedule.get_parameters()["coeff"], 0.4)


def test_repr():
    param = Parameter("theta", 0.5)
    hamiltonian = Hamiltonian({(PauliZ(0),): param})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = ket(0).unit()

    functional = AnalogEvolution(schedule=schedule, initial_state=initial_state)
    repr_str = str(functional)
    assert "AnalogEvolution" in repr_str
    assert "schedule=" in repr_str
    assert "initial_state=" in repr_str


# --- Parameter Cache Invalidation Tests ---


def _isclose_cache(lhs: float, rhs: float) -> bool:
    return bool(np.isclose(lhs, rhs, atol=get_settings().atol, rtol=get_settings().rtol))


def test_analog_evolution_set_parameters_clears_interpolator_cache():
    """AnalogEvolution.set_parameters must propagate through Schedule to clear
    the Interpolator's _cached_time, not just update the Parameter value."""
    p = Parameter("coeff", 2.0)
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(
        hamiltonians={"H": hamiltonian},
        coefficients={"H": {0: p, 1.0: 0.0}},
        dt=0.1,
    )
    initial_state = ket(0).unit()
    evolution = AnalogEvolution(schedule=schedule, initial_state=initial_state)

    # Populate _cached_time: linear interp at t=0.5 → p*0.5 = 1.0
    assert _isclose_cache(schedule.coefficients["H"][0.5], 1.0)

    evolution.set_parameters({"coeff": 4.0})

    # Stale cache would return 1.0; correct answer is 4.0*0.5 = 2.0.
    assert _isclose_cache(schedule.coefficients["H"][0.5], 2.0)


def test_analog_evolution_set_parameter_values_clears_interpolator_cache():
    """AnalogEvolution.set_parameter_values must clear the Interpolator cache."""
    p = Parameter("coeff", 2.0)
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(
        hamiltonians={"H": hamiltonian},
        coefficients={"H": {0: p, 1.0: 0.0}},
        dt=0.1,
    )
    initial_state = ket(0).unit()
    evolution = AnalogEvolution(schedule=schedule, initial_state=initial_state)

    assert _isclose_cache(schedule.coefficients["H"][0.5], 1.0)

    evolution.set_parameter_values([4.0])

    assert _isclose_cache(schedule.coefficients["H"][0.5], 2.0)


def test_analog_evolution_set_parameter_bounds_propagates_to_schedule():
    """AnalogEvolution.set_parameter_bounds must update bounds in the Schedule."""
    p = Parameter("coeff", 2.0, bounds=(0.0, 3.0))
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(
        hamiltonians={"H": hamiltonian},
        coefficients={"H": {(0, 1.0): p}},
        dt=0.1,
    )
    initial_state = ket(0).unit()
    evolution = AnalogEvolution(schedule=schedule, initial_state=initial_state)

    assert schedule.get_parameter_bounds()["coeff"] == (0.0, 3.0)

    evolution.set_parameter_bounds({"coeff": (0.0, 5.0)})

    assert schedule.get_parameter_bounds()["coeff"] == (0.0, 5.0)
