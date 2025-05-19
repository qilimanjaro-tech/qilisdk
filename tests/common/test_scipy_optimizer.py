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
from unittest.mock import MagicMock, patch

import numpy as np

from qilisdk.common.optimizer import SciPyOptimizer


def dummy_cost(params):
    return sum(params)


def test_optimize_sets_optimal_parameters():
    with patch("scipy.optimize.minimize") as mock_minimize:
        # Create a fake result with a known optimal parameter set.
        fake_result = MagicMock()
        fake_result.fun = -0.5
        fake_result.x = np.array([1.0, 2.0, 3.0])
        mock_minimize.return_value = fake_result

        optimizer = SciPyOptimizer(method="BFGS")
        initial_parameters = [0.0, 0.0, 0.0]
        optimizer_result = optimizer.optimize(dummy_cost, initial_parameters)

        # Check that the optimizer returned the fake result.
        assert optimizer_result.optimal_cost == fake_result.fun
        assert optimizer_result.optimal_parameters == fake_result.x.tolist()

        # Verify that the patched scipy.optimize.minimize was called with the correct parameters.
        mock_minimize.assert_called_once_with(
            dummy_cost,
            x0=initial_parameters,
            method="BFGS",
            jac=None,
            hess=None,
            hessp=None,
            bounds=None,
            constraints=(),
            tol=None,
            options=None,
            callback=None,
        )


def test_extra_arguments_are_propagated():
    with patch("scipy.optimize.minimize") as mock_minimize:
        fake_result = MagicMock()
        fake_result.fun = -0.5
        fake_result.x = np.array([0.5, 1.5])
        mock_minimize.return_value = fake_result

        # Pass extra keyword arguments.
        optimizer = SciPyOptimizer(method="BFGS", jac="dummy_jac", bounds=[(0, 1), (0, 2)])
        initial_parameters = [0.0, 0.0]
        optimizer.optimize(dummy_cost, initial_parameters)

        # Assert that extra arguments were forwarded to scipy.optimize.minimize.
        mock_minimize.assert_called_once_with(
            dummy_cost,
            x0=initial_parameters,
            method="BFGS",
            jac="dummy_jac",
            hess=None,
            hessp=None,
            bounds=[(0, 1), (0, 2)],
            constraints=(),
            tol=None,
            options=None,
            callback=None,
        )
