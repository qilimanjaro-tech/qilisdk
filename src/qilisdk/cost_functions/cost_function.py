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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qilisdk.core.types import Number
    from qilisdk.functionals.functional_result import FunctionalResult


class CostFunction(ABC):
    """
    Base class that maps functional results into scalar costs.
    """

    @abstractmethod
    def compute_cost(self, results: FunctionalResult) -> Number:
        """
        Compute a scalar cost from functional execution results.

        Args:
            results (FunctionalResult): Output of a functional execution.

        Returns:
            Number: Scalar cost extracted from the results.
        """
