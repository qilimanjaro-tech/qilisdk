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

from qilisdk.common.model import Model
from qilisdk.functionals.functional import Functional
from qilisdk.optimizers.optimizer import Optimizer


class ParameterizedProgram:
    def __init__(self, functional: Functional, optimizer: Optimizer, cost_model: Model) -> None:
        """The Parameterized Program provides a way to optimized a parameterized functional.

        Args:
            functional (Functional): the parameterized functional.
            optimizer (Optimizer): the optimizer to be used.
            cost_model (Model): the model that evaluates the cost for a given set of parameters.
        """
        self.functional = functional
        self.optimizer = optimizer
        self.cost_model = cost_model
