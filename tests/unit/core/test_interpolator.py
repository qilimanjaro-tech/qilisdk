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

import qilisdk
from qilisdk.core.interpolator import Interpolation, Interpolator, _process_callable
from qilisdk.core.variables import Domain, Parameter, Sin, Variable


@pytest.mark.parametrize(
    ("points", "interpolation", "measured_expected"),
    [
        (
            {0: 0, 1: 1},
            Interpolation.LINEAR,
            [(0, 0), (1, 1), (0.5, 0.5), (0.7, 0.7)],
        ),
        ({0: 0, 10: 10}, Interpolation.LINEAR, [(0, 0), (10, 10), (5, 5), (7, 7)]),
        ({0: 0, 10: 1}, Interpolation.LINEAR, [(0, 0), (10, 1), (5, 0.5), (7, 0.7)]),
    ],
)
def test_linear_interpolation(points, interpolation, measured_expected):
    interp = Interpolator(
        points,
        interpolation,
    )

    for point, exp_val in measured_expected:
        assert interp[point] == exp_val


@pytest.mark.parametrize(
    ("points", "interpolation", "measured_expected"),
    [
        (
            {0: 0, 1: 1},
            Interpolation.STEP,
            [(0, 0), (1, 1), (0.5, 0), (0.7, 0)],
        ),
        ({0: 0, 10: 10}, Interpolation.STEP, [(0, 0), (10, 10), (5, 0), (7, 0)]),
        ({0: 0, 10: 1}, Interpolation.STEP, [(0, 0), (10, 1), (5, 0), (7, 0)]),
    ],
)
def test_step_interpolation(points, interpolation, measured_expected):
    interp = Interpolator(
        points,
        interpolation,
    )

    for point, exp_val in measured_expected:
        assert interp[point] == exp_val


@pytest.mark.parametrize(
    ("points", "interpolation", "measured_expected", "max_time"),
    [
        ({0: 0, 1: 1}, Interpolation.LINEAR, [(0, 0), (10, 1), (0.5, 0.05), (0.7, 0.07)], 10),
        ({0: 0, 10: 10}, Interpolation.LINEAR, [(0, 0), (1, 10), (0.5, 5), (0.7, 7)], 1),
        ({0: 0, 10: 1}, Interpolation.LINEAR, [(0, 0), (10, 1), (5, 0.5), (7, 0.7)], 10),
        ({0: 0, 10: 1}, Interpolation.STEP, [(0, 0), (10, 1), (5, 0), (7, 0)], 10),
    ],
)
def test_set_max_time(points, interpolation, measured_expected, max_time):
    interp = Interpolator(
        points,
        interpolation,
    )
    interp.set_max_time(max_time)

    assert interp.total_time == max_time

    for point, exp_val in measured_expected:
        assert np.isclose(interp[point], exp_val)


@pytest.mark.parametrize(
    ("points", "interpolation", "measured_expected", "max_time"),
    [
        ({(0, 1): lambda t: t}, Interpolation.LINEAR, [(0, 0), (10, 1), (0.5, 0.05), (0.7, 0.07)], 10),
        (
            {(0, 1): lambda t: Sin(t), (1.1, 2): lambda t: t},  # noqa: PLW0108
            Interpolation.LINEAR,
            [(0, np.sin(0)), (1, 2), (0.5, np.sin(1)), (0.7, 1.4)],
            1,
        ),
        ({(0, 10): 1}, Interpolation.LINEAR, [(0, 1), (10, 1), (5, 1), (7, 1)], 10),
    ],
)
def test_intervals(points, interpolation, measured_expected, max_time):
    interp = Interpolator(
        points,
        interpolation,
    )
    interp.set_max_time(max_time)

    for point, exp_val in measured_expected:
        assert np.isclose(interp[point], exp_val)


@pytest.mark.parametrize(
    ("points", "interpolation", "measured_expected", "max_time"),
    [
        (
            {(0, 4): lambda t: Sin(t), (Parameter("o", 5), 10): lambda t: t},  # noqa: PLW0108
            Interpolation.LINEAR,
            [(0, np.sin(0)), (1, 10), (0.7, 7)],
            1,
        ),
    ],
)
def test_parameterization(points, interpolation, measured_expected, max_time):
    interp = Interpolator(
        points,
        interpolation,
    )
    interp.set_max_time(max_time)

    for point, exp_val in measured_expected:
        assert np.isclose(interp[point], exp_val)

    with pytest.raises(
        ValueError,
        match=r"New assignation of the parameters breaks the parameter constraints:",
    ):
        interp.set_parameter_values([3])


def test_interpolator_items():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    items = list(interp.items())
    expected_items = [(0, 0), (10, 10)]
    assert items == expected_items


def test_interpolator_items_max_time():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    interp.set_max_time(5)
    expected_items = {(0, 0), (5, 10)}
    assert set(interp.items()) == expected_items
    assert set(interp.items()) == expected_items  # twice to make sure it doesn't change


def test_interpolator_fixed_items():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    fixed_items = list(interp.fixed_items())
    expected_fixed_items = [(0, 0), (10, 10)]
    assert fixed_items == expected_fixed_items


def test_interpolator_coefficients():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    coefficients = interp.coefficients
    expected_coefficients = [0, 10]
    assert coefficients == expected_coefficients


def test_interpolator_coefficients_dict():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    coefficients_dict = interp.coefficients_dict
    expected_coefficients_dict = {0: 0, 10: 10}
    assert coefficients_dict == expected_coefficients_dict


def test_interpolator_fixed_coefficients():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    fixed_coefficients = interp.fixed_coefficients
    expected_fixed_coefficients = [0, 10]
    assert fixed_coefficients == expected_fixed_coefficients


def test_interpolator_parameters():
    points = {0: Parameter("a", 1.0), 10: Parameter("b", 2.0)}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    parameters = interp.parameters
    expected_parameters = {"a": Parameter("a", 1.0), "b": Parameter("b", 2.0)}
    assert parameters == expected_parameters


def test_interpolator_set_bad_max_time():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    with pytest.raises(ValueError, match="set the max time to zero"):
        interp.set_max_time(0)


def test_add_time_point():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    param = Parameter("c", 5)

    def callable(t, param: Parameter = param):
        return param

    interp.add_time_point(5, callable)
    expected_items = {(0, 0), (5, param), (10, 10)}
    assert set(interp.items()) == expected_items


def test_add_time_point_bad_coeff():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    with pytest.raises(ValueError, match="Coefficient must be a number"):
        interp.add_time_point(5, "bad_coefficient")


def test_interpolar_set_parameter_bounds():
    points = {0: Parameter("a", 1.0), 10: Parameter("b", 2.0)}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    interp.set_parameter_bounds({"a": (0.5, 1.5)})
    param_a = interp.parameters["a"]
    assert np.isclose(param_a.lower_bound, 0.5)
    assert np.isclose(param_a.upper_bound, 1.5)


def test_interpolator_get_coefficient(monkeypatch):
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )

    # make it so we don't cache the time scaling between calls
    def new_get_coefficient_expression(self, time_step):
        return_val = self.old_get_coefficient_expression(time_step)
        self._delete_cache()
        return return_val

    interp.old_get_coefficient_expression = interp.get_coefficient_expression
    monkeypatch.setattr(interp, "get_coefficient_expression", new_get_coefficient_expression.__get__(interp))

    interp.set_max_time(5)
    coeff_0 = interp.get_coefficient(0)
    coeff_10 = interp.get_coefficient(10)
    assert coeff_0 == 0
    assert coeff_10 == 20


def test_interpolator_get_coefficient_expression(monkeypatch):
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    interp.set_max_time(5)

    expr_0 = interp.get_coefficient_expression(0)
    expr_1_1 = interp.get_coefficient_expression(1)
    expr_1_2 = interp.get_coefficient_expression(1)
    expr_10 = interp.get_coefficient_expression(10)
    assert expr_0 == 0
    assert expr_1_1 == 2
    assert expr_1_2 == 2
    assert expr_10 == 20


def test_bad_interpolar_type(monkeypatch):
    class FakeType:
        def __init__(self):
            self.value = "BAD_TYPE"

    Interpolation.BAD_TYPE = FakeType()
    points = {0: 0, 10: 10}
    interp = Interpolator(points, Interpolation.BAD_TYPE)
    with pytest.raises(ValueError, match="Interpolation type BAD_TYPE"):
        interp.get_coefficient(5)


def test_interpolator_len():
    points = {0: 0, 10: 10, 5: 5}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    assert len(interp) == 3


def test_interpolator_iter():
    points = {0: 0, 10: 10, 5: 5}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    expected_points = [0, 5, 10]
    for t, expected_t in zip(interp, expected_points):
        assert t == expected_t


def test_interpolator_get_linear(monkeypatch):
    a = Parameter("a", 0)
    b = Parameter("b", 1)
    points = {a: 5, b: 5}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    interp.set_parameter_values([0, 0])
    with pytest.raises(ValueError, match="Ambiguous evaluation"):
        interp._get_coefficient_expression_linear(5)


def test_interpolator_terms():
    a = Parameter("a", 1)
    b = Parameter("b", 2)
    t1 = 2 * a + 3
    t2 = 2 * b + 4
    not_term = 3
    points = {0: t1, 5: t2, 10: not_term, 15: t1 + t2}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    coeff_2_5 = interp.get_coefficient(2.5)
    coeff_7_5 = interp.get_coefficient(7.5)
    coeff_12_5 = interp.get_coefficient(12.5)
    assert np.isclose(coeff_2_5, 6.5)
    assert np.isclose(coeff_7_5, 5.5)
    assert np.isclose(coeff_12_5, 8.0)


def test_interpolator_single_time():
    points = {0: 0}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    coeff_0 = interp.get_coefficient(0)
    coeff_1 = interp.get_coefficient(1)
    assert np.isclose(coeff_0, 0)
    assert np.isclose(coeff_1, 0)


def test_interpolator_no_time():
    points = {}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    coeff_0 = interp.get_coefficient(0)
    coeff_1 = interp.get_coefficient(1)
    assert coeff_0 == 0
    assert coeff_1 == 0


def test_interpolator_weird(monkeypatch):
    # always return 0 so we always hit the first point
    def fake_bisect_right(a, x, key):
        return 0

    monkeypatch.setattr("qilisdk.core.interpolator.bisect_right", fake_bisect_right)

    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    coeff_1 = interp._get_coefficient_expression_linear(1)
    assert coeff_1 == 1

    points = {0: 0}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    coeff_1 = interp._get_coefficient_expression_linear(1)
    assert coeff_1 == 0


def test_interpolator_get_value():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    a = Parameter(qilisdk.core.interpolator._TIME_PARAMETER_NAME, 5)
    assert interp._get_value(5 + 0j) == 5
    with pytest.raises(ValueError, match="Can't evaluate Parameter"):
        interp._get_value(a)
    with pytest.raises(ValueError, match="Invalid value"):
        interp._get_value("bad value")


def test_interpolator_bad_init():
    a = Parameter("a", 1)
    b = Parameter("b", 1)
    with pytest.raises(ValueError, match="The time point"):
        Interpolator(
            {a: 1, b: 2},
            Interpolation.LINEAR,
        )
    with pytest.raises(ValueError, match="Can't provide a point"):
        Interpolator(
            {(1, 3): 1, b: 2},
            Interpolation.LINEAR,
        )
    with pytest.raises(ValueError, match="need to be defined by two points"):
        Interpolator(
            {(2, 3, 4): 1, b: 2},
            Interpolation.LINEAR,
        )


def test_interpolator_variables():
    var = Variable("var", Domain.REAL)
    term = 2 * var + 1
    with pytest.raises(ValueError, match="contains objects other than parameters"):
        Interpolator(
            {1: term, 10: 2},
            Interpolation.LINEAR,
        )


def test_interpolator_callables():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )

    def callable(t, param: Parameter):
        return param

    interp.add_time_point(5, callable)
    assert interp[2.5] == 0
    term, params = _process_callable(callable, 2.5, param=Parameter("param", 3))
    assert term == 3
    assert "param" in params
    var = Variable("param", Domain.REAL)
    term = 3 * var + 1
    with pytest.raises(ValueError, match="contains variables that are not time"):
        _process_callable(callable, 2.5, param=term)
    with pytest.raises(ValueError, match="contains variables that are not time"):
        _process_callable(callable, 2.5, param=var)


def test_time_scale_caching():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )

    # first call without max_time set
    assert np.isclose(interp._time_scale, 1.0)

    # second call should use cache
    assert np.isclose(interp._time_scale, 1.0)

    # set max_time and check scale updates
    interp.set_max_time(5)
    assert np.isclose(interp._time_scale, 0.5)

    # subsequent call should use cache
    assert np.isclose(interp._time_scale, 0.5)


def test_add_time_point_with_scaling():
    points = {0: 0, 10: 10}
    interp = Interpolator(
        points,
        Interpolation.LINEAR,
    )
    interp.set_max_time(5)
    time = 2.5
    coeff = 3
    interp.add_time_point(time, coeff)
    expected_time_dict = {(0, 0), (5, 3), (10, 10)}
    expected_items = {(0, 0), (2.5, 3), (5, 10)}
    assert set(interp._time_dict.items()) == expected_time_dict
    assert set(interp.items()) == expected_items
