# Copyright 2026 Qilimanjaro Quantum Tech
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

from qilisdk.core import Model
from qilisdk.core.variables import BaseVariable, Number, RealNumber
from qilisdk.settings import get_settings


def _assert_real(number: complex) -> float:
    if isinstance(number, complex):
        if abs(number.imag) < get_settings().atol:
            return number.real
        raise ValueError("Complex Number encountered when expecting only real values to be present.")
    return number


def _variable_bounds(variable: BaseVariable) -> tuple[float, float]:
    lower, upper = variable.bounds
    lower = variable.domain.min() if lower is None else lower
    upper = variable.domain.max() if upper is None else upper
    return float(lower), float(upper)


class ClassicalSolver:
    """Base class for classical solvers."""

    def solve(self, model: Model) -> tuple[dict[str, Number], dict[BaseVariable, RealNumber]]:
        """Solve the given model."""
        raise NotImplementedError("ClassicalSolver is an abstract base class.")
