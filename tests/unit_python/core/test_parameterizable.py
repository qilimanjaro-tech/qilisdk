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
from qilisdk.core.variables import LEQ, Domain, Parameter, Variable
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


class EmptyParameterizable(Parameterizable):
    pass


class ParentWithLocalAndChild(Parameterizable):
    def __init__(self) -> None:
        super().__init__()
        self._parameters = {
            "local_trainable": Parameter("local_trainable", 0.5, trainable=True),
            "local_fixed": Parameter("local_fixed", 0.25, trainable=False),
        }
        self.child = LeafParameterizable("child_theta", 0.1, trainable=False)

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        yield self.child


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
    assert _isclose(parent.get_parameters(where=lambda param: param.is_trainable)["left"], 1.0)
    assert _isclose(parent.get_parameters(where=lambda param: not param.is_trainable)["right"], 2.0)

    parent.set_parameters({"left": 1.5, "right": 2.5})
    assert _isclose(left.get_parameters()["left"], 1.5)
    assert _isclose(right.get_parameters()["right"], 2.5)

    right.set_parameters({"right": 3.5})
    assert _isclose(parent.get_parameters()["right"], 3.5)

    parent.set_parameter_values([4.0], where=lambda param: param.is_trainable)
    assert _isclose(left.get_parameters()["left"], 4.0)

    parent.set_parameter_bounds({"left": (-1.0, 5.0), "right": (0.0, 4.0)})
    assert _isclose(left.get_parameter_bounds()["left"][0], -1.0)
    assert _isclose(left.get_parameter_bounds()["left"][1], 5.0)
    assert _isclose(right.get_parameter_bounds()["right"][0], 0.0)
    assert _isclose(right.get_parameter_bounds()["right"][1], 4.0)

    left.set_parameter_bounds({"left": (-2.0, 5.0)})
    assert _isclose(parent.get_parameter_bounds()["left"][0], -2.0)
    assert _isclose(parent.get_parameter_bounds()["left"][1], 5.0)


def test_parameter_filter_predicate_supports_custom_queries():
    input_param = LeafParameterizable("input_encoding_theta", 0.1, trainable=False)
    output_param = LeafParameterizable("output_encoding_theta", 0.9, trainable=True)
    parent = CompositeParameterizable(input_param, output_param)

    def starts_with_input(param: Parameter) -> bool:
        return param.label.startswith("input_encoding_")

    def value_in_range(param: Parameter) -> bool:
        return 0.5 <= param.value <= 1.0

    assert parent.get_parameter_names(where=starts_with_input) == ["input_encoding_theta"]
    assert parent.get_parameters(where=value_in_range) == {"output_encoding_theta": 0.9}
    assert parent.get_parameters(where=lambda p: starts_with_input(p) and not p.is_trainable) == {
        "input_encoding_theta": 0.1
    }

    parent.set_parameter_values([0.7], where=lambda param: param.label.startswith("output_encoding_"))
    assert _isclose(parent.get_parameters()["input_encoding_theta"], 0.1)
    assert _isclose(parent.get_parameters()["output_encoding_theta"], 0.7)


def test_private_helpers_add_and_query_parameter_names():
    source = DummyParameterizable()
    target = EmptyParameterizable()
    target.set_prefix("pref_")

    target._add_parameter("gamma", Parameter("gamma", 3.0))
    target._add_parameter_from("alpha", source)
    target._add_parameter_from("beta", source, new_label="beta_alias")

    assert "pref_gamma" in target.get_parameter_names()
    assert target._parameters["pref_alpha"] is source._parameters["alpha"]
    assert target._parameters["pref_beta_alias"] is source._parameters["beta"]
    assert target._query_parameter_original_name(target, "pref_beta_alias") == "beta"


def test_private_helpers_update_and_link_share_parameter_references():
    source = DummyParameterizable()
    linked = LeafParameterizable("theta", 0.4)
    target = EmptyParameterizable()

    target._update_parameters({"alpha_copy": source._parameters["alpha"]})
    target._link_parameters(linked)

    assert target.get_parameter_names() == ["alpha_copy", "theta"]
    assert target._parameters["theta"] is linked._parameters["theta"]

    target.set_parameters({"theta": 0.9})
    assert _isclose(linked.get_parameters()["theta"], 0.9)


def test_set_prefix_where_is_local_only_but_children_are_recursive():
    parent = ParentWithLocalAndChild()

    parent.set_prefix("p_", where=lambda param: param.is_trainable)
    assert parent.get_parameter_names() == ["local_fixed", "p_local_trainable", "p_child_theta"]
    assert parent.get_prefix() == "p_"
    assert parent.child.get_prefix() == "p_"

    parent.set_prefix("q_", where=lambda _: True)
    assert set(parent.get_parameter_names()) == {"q_local_trainable", "q_local_fixed", "q_child_theta"}
    assert parent.child.get_parameter_names() == ["q_child_theta"]
    assert parent.get_prefix() == "q_"
    assert parent.child.get_prefix() == "q_"


def test_add_parameter_constraint_rejects_non_parameter_variables():
    parameterizable = DummyParameterizable()
    generic_variable = Variable("x", domain=Domain.REAL, bounds=(0.0, 1.0))

    with pytest.raises(
        ValueError,
        match=r"The constraint should only contain parameters and having generic variables is not allowed.",
    ):
        parameterizable.add_parameter_constraint(LEQ(parameterizable._parameters["alpha"], generic_variable))


def test_pop_removes_local_or_child_parameters_and_errors_on_unknown_label():
    parent = ParentWithLocalAndChild()

    popped_local = parent._pop("local_fixed")
    assert popped_local.label == "local_fixed"
    assert "local_fixed" not in parent.get_parameter_names()

    popped_child = parent._pop("child_theta")
    assert popped_child.label == "child_theta"
    assert "child_theta" not in parent.child.get_parameter_names()

    with pytest.raises(
        ValueError,
        match=r"Parameter unknown is not defined in the current object or any of its children.",
    ):
        parent._pop("unknown")
