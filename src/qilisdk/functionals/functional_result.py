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
from abc import abstractmethod

from qilisdk.common.model import Model
from qilisdk.common.result import Result


class FunctionalResult(Result):
    @abstractmethod
    def compute_cost(self, cost_model: Model) -> float:
        """Compute the cost of the functional given a cost model.

        Args:
            cost_model (Model): The Model object used to represent the cost of different states.

        Returns:
            float: the cost of the results.
        """
