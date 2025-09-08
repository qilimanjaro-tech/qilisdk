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

from qilisdk.common.variables import RealNumber
from qilisdk.digital.circuit import Circuit
from qilisdk.functionals import SamplingResult
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.yaml import yaml


@yaml.register_class
class StateTomography(PrimitiveFunctional[SamplingResult]):
    """State Tomography functional reconstructs the state at the end of the circuit execution."""

    result_type: ClassVar[type[SamplingResult]] = SamplingResult

    def __init__(self, circuit: Circuit) -> None:
        """
        Args:
            circuit (Circuit): The circuit to be executed.
        """
        self.circuit = circuit

    @property
    def nparameters(self) -> int:
        return self.circuit.nparameters

    def set_parameters(self, parameters: dict[str, RealNumber]) -> None:
        self.circuit.set_parameters(parameters)

    def get_parameters(self) -> dict[str, RealNumber]:
        return self.circuit.get_parameters()

    def get_parameter_names(self) -> list[str]:
        return list(self.circuit.get_parameters().keys())

    def get_parameter_values(self) -> list[RealNumber]:
        return list(self.circuit.get_parameters().values())

    def set_parameter_values(self, values: list[float]) -> None:
        self.circuit.set_parameter_values(values)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self.circuit.get_parameter_bounds()

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        self.circuit.set_parameter_bounds(ranges)
