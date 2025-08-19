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
from typing import ClassVar

from qilisdk.common.variables import Number
from qilisdk.digital.circuit import Circuit
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.yaml import yaml


@yaml.register_class
class Sampling(PrimitiveFunctional[SamplingResult]):
    result_type: ClassVar[type[SamplingResult]] = SamplingResult

    def __init__(self, circuit: Circuit, nshots: int = 1000) -> None:
        self.circuit = circuit
        self.nshots = nshots

    def set_parameters(self, parameters: dict[str, Number]) -> None:
        self.circuit.set_parameters(parameters)

    def get_parameters(self) -> dict[str, Number]:
        return self.circuit.get_parameters()

    def get_parameter_names(self) -> list[str]:
        return list(self.circuit.get_parameters().keys())

    def get_parameter_values(self) -> list[Number]:
        return list(self.circuit.get_parameters().values())
