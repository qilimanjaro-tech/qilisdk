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

import pytest
from numpy import isclose

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import LEQ, Parameter


class DummyParameterizable(Parameterizable):
    def __init__(self) -> None:
        super().__init__()
        self._parameters = {
            "alpha": Parameter("alpha", 1.0, bounds=(0, 2)),
            "beta": Parameter("beta", 2.0, bounds=(1, 3)),
        }
        self._parameter_constraints = [LEQ(self._parameters["alpha"], self._parameters["beta"])]


@pytest.fixture
def parameterizable() -> DummyParameterizable:
    return DummyParameterizable()


def test_parameter_getters(parameterizable: DummyParameterizable):
    assert parameterizable.nparameters == 2
    assert parameterizable.get_parameter_names() == ["alpha", "beta"]
    assert parameterizable.get_parameter_values() == [1.0, 2.0]
    assert parameterizable.get_parameters() == {"alpha": 1.0, "beta": 2.0}
    assert parameterizable.get_parameter_bounds() == {"alpha": (0, 2), "beta": (1, 3)}


def test_set_parameter_values_updates_all(parameterizable: DummyParameterizable):
    parameterizable.set_parameter_values([1.5, 2.5])

    assert parameterizable.get_parameters() == {"alpha": 1.5, "beta": 2.5}


def test_set_parameter_values_length_mismatch(parameterizable: DummyParameterizable):
    with pytest.raises(ValueError, match=r"Provided 1 but this object has 2 parameters."):
        parameterizable.set_parameter_values([1.0])


def test_set_parameters_unknown_label(parameterizable: DummyParameterizable):
    with pytest.raises(ValueError, match=r"Parameter gamma is not defined for this object."):
        parameterizable.set_parameters({"gamma": 1.0})


def test_constraints_enforced_on_set(parameterizable: DummyParameterizable):
    assert parameterizable.check_constraints({"alpha": 1.5})
    assert not parameterizable.check_constraints({"alpha": 2, "beta": 1})

    parameterizable.set_parameter_values([0.5, 1])
    with pytest.raises(
        ValueError,
        match=r"New assignation of the parameters breaks the parameter constraints:",
    ):
        parameterizable.set_parameters({"alpha": 2})

    parameterizable.set_parameters({"alpha": 1.5, "beta": 2})
    assert isclose(parameterizable.get_parameters()["alpha"], 1.5)


def test_check_constraints_unknown_parameter(parameterizable: DummyParameterizable):
    with pytest.raises(ValueError, match=r"Parameter unknown is not defined for this object."):
        parameterizable.check_constraints({"unknown": 0.0})


def test_set_parameter_bounds(parameterizable: DummyParameterizable):
    parameterizable.set_parameter_bounds({"alpha": (0.0, 1.5)})
    assert parameterizable.get_parameter_bounds()["alpha"] == (0.0, 1.5)

    with pytest.raises(
        ValueError,
        match=r"The provided parameter label gamma is not defined in the list of parameters in this object.",
    ):
        parameterizable.set_parameter_bounds({"gamma": (0, 1)})


def test_set_nonexistent_parameter(monkeypatch, parameterizable: DummyParameterizable):
    # Bypass constraint checking to test nonexistent parameter handling.
    monkeypatch.setattr(
        parameterizable,
        "check_constraints",
        lambda params: True,
    )

    with pytest.raises(ValueError, match=r"not defined for this object"):
        parameterizable.set_parameters({"gamma": 1.0})
