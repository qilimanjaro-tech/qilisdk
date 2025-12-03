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

from qilisdk.core.interpolator import Interpolation, Interpolator
from qilisdk.core.variables import Parameter, Sin


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
            {(0, 1): lambda t: Sin(t), (1.1, 2): lambda t: t},
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
            {(0, 4): lambda t: Sin(t), (Parameter("o", 5), 10): lambda t: t},
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
