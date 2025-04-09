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
from typing import Callable

from qilisdk.common.algorithm import Algorithm
from qilisdk.common.optimizer import Optimizer
from qilisdk.digital.ansatz import Ansatz
from qilisdk.digital.digital_backend import DigitalBackend
from qilisdk.digital.digital_result import DigitalResult


class DigitalAlgorithm(Algorithm):
    """
    Abstract base class for digital quantum algorithms.
    """


class VQE(DigitalAlgorithm):

    def __init__(
        self, ansatz: Ansatz, initial_params: list[float], cost_function: Callable[[dict[str, float]], float]
    ) -> None:
        self._ansatz = ansatz
        self._initial_params = initial_params
        self._cost_function = cost_function

    def obtain_cost(self, params: list[float], backend: DigitalBackend, nshots: int = 1000) -> float:
        circuit = self._ansatz.get_circuit(params)
        results = backend.execute(circuit=circuit, nshots=nshots)
        return self._cost_function(results.get_probabilities())

    def execute(self, backend: DigitalBackend, optimizer: Optimizer, nshots: int = 1000) -> DigitalResult:
        optimizer.optimize(lambda x: self.obtain_cost(x, backend=backend, nshots=nshots), self._initial_params)
        circuit = self._ansatz.get_circuit(optimizer.optimal_parameters)
        return backend.execute(circuit=circuit, nshots=nshots)
