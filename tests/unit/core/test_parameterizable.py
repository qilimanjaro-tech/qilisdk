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

from typing import Iterator

import numpy as np
import pytest

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import LEQ, Parameter
from qilisdk.settings import get_settings


def _isclose(lhs: float, rhs: float) -> bool:
    return bool(np.isclose(lhs, rhs, atol=get_settings().atol, rtol=get_settings().rtol))


class DummyParameterizable(Parameterizable):
    def __init__(self) -> None:
        super().__init__()
        self._parameters = {
            "alpha": Parameter("alpha", 1.0, bounds=(0, 2)),
            "beta": Parameter("beta", 2.0, bounds=(1, 3)),
        }
        self._parameter_constraints = [LEQ(self._parameters["alpha"], self._parameters["beta"])]


class LeafParameterizable(Parameterizable):
    def __init__(self, label: str, value: float, *, trainable: bool = True) -> None:
        super().__init__()
        self._parameters = {
            label: Parameter(label, value, bounds=(-10, 10), trainable=trainable),
        }


class CompositeParameterizable(Parameterizable):
    def __init__(self, left: Parameterizable, right: Parameterizable) -> None:
        super().__init__()
        self._left = left
        self._right = right

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        yield self._left
        yield self._right


@pytest.fixture
def parameterizable() -> DummyParameterizable:
    return DummyParameterizable()


def test_parameter_getters(parameterizable: DummyParameterizable):
    assert parameterizable.nparameters == 2
    assert parameterizable.get_parameter_names() == ["alpha", "beta"]
    assert _isclose(parameterizable.get_parameter_values()[0], 1.0)
    assert _isclose(parameterizable.get_parameter_values()[1], 2.0)
    assert _isclose(parameterizable.get_parameters()["alpha"], 1.0)
    assert _isclose(parameterizable.get_parameters()["beta"], 2.0)
    assert _isclose(parameterizable.get_parameter_bounds()["alpha"][0], 0.0)
    assert _isclose(parameterizable.get_parameter_bounds()["alpha"][1], 2.0)
    assert _isclose(parameterizable.get_parameter_bounds()["beta"][0], 1.0)
    assert _isclose(parameterizable.get_parameter_bounds()["beta"][1], 3.0)


def test_set_parameter_values_updates_all(parameterizable: DummyParameterizable):
    parameterizable.set_parameter_values([1.5, 2.5])

    assert _isclose(parameterizable.get_parameters()["alpha"], 1.5)
    assert _isclose(parameterizable.get_parameters()["beta"], 2.5)


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
    assert _isclose(parameterizable.get_parameters()["alpha"], 1.5)


def test_check_constraints_unknown_parameter(parameterizable: DummyParameterizable):
    with pytest.raises(ValueError, match=r"Parameter unknown is not defined for this object."):
        parameterizable.check_constraints({"unknown": 0.0})


def test_set_parameter_bounds(parameterizable: DummyParameterizable):
    parameterizable.set_parameter_bounds({"alpha": (0.0, 1.5)})
    assert _isclose(parameterizable.get_parameter_bounds()["alpha"][0], 0.0)
    assert _isclose(parameterizable.get_parameter_bounds()["alpha"][1], 1.5)

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


def test_composite_parent_child_parameter_sync():
    left = LeafParameterizable("left", 1.0, trainable=True)
    right = LeafParameterizable("right", 2.0, trainable=False)
    parent = CompositeParameterizable(left, right)

    assert _isclose(parent.get_parameters()["left"], 1.0)
    assert _isclose(parent.get_parameters()["right"], 2.0)
    assert _isclose(parent.get_parameters(trainable=True)["left"], 1.0)
    assert _isclose(parent.get_parameters(trainable=False)["right"], 2.0)

    parent.set_parameters({"left": 1.5, "right": 2.5})
    assert _isclose(left.get_parameters()["left"], 1.5)
    assert _isclose(right.get_parameters()["right"], 2.5)

    right.set_parameters({"right": 3.5})
    assert _isclose(parent.get_parameters()["right"], 3.5)

    parent.set_parameter_values([4.0], trainable=True)
    assert _isclose(left.get_parameters()["left"], 4.0)

    parent.set_parameter_bounds({"left": (-1.0, 5.0), "right": (0.0, 4.0)})
    assert _isclose(left.get_parameter_bounds()["left"][0], -1.0)
    assert _isclose(left.get_parameter_bounds()["left"][1], 5.0)
    assert _isclose(right.get_parameter_bounds()["right"][0], 3.5)
    assert _isclose(right.get_parameter_bounds()["right"][1], 3.5)

    left.set_parameter_bounds({"left": (-2.0, 5.0)})
    assert _isclose(parent.get_parameter_bounds()["left"][0], -2.0)
    assert _isclose(parent.get_parameter_bounds()["left"][1], 5.0)
