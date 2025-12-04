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

from qilisdk.core.variables import ComparisonTerm, RealNumber
from qilisdk.digital.circuit import Circuit
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.yaml import yaml


@yaml.register_class
class Sampling(PrimitiveFunctional[SamplingResult]):
    """
    Execute a digital circuit and collect bitstring samples.

    Example:
        .. code-block:: python

            from qilisdk.digital.circuit import Circuit
            from qilisdk.functionals.sampling import Sampling

            circuit = Circuit(nqubits=2)
            circuit.h(0)
            sampler = Sampling(circuit, nshots=1024)
    """

    result_type: ClassVar[type[SamplingResult]] = SamplingResult

    def __init__(self, circuit: Circuit, nshots: int = 1000) -> None:
        """
        Args:
            circuit (Circuit): Circuit to execute for sampling.
            nshots (int, optional): Number of repetitions used to estimate probabilities. Defaults to 1000.
        """
        self.circuit = circuit
        self.nshots = nshots

    @property
    def nparameters(self) -> int:
        """Return the number of parameters exposed by the underlying circuit."""
        return self.circuit.nparameters

    def set_parameters(self, parameters: dict[str, RealNumber]) -> None:
        """Assign a subset of circuit parameters provided by label."""
        self.circuit.set_parameters(parameters)

    def get_parameters(self) -> dict[str, RealNumber]:
        """Return a mapping of all circuit parameters to their current value."""
        return self.circuit.get_parameters()

    def get_parameter_names(self) -> list[str]:
        """Return circuit parameter labels in deterministic order."""
        return list(self.circuit.get_parameters().keys())

    def get_parameter_values(self) -> list[RealNumber]:
        """Return the current parameter values following ``get_parameter_names`` order."""
        return list(self.circuit.get_parameters().values())

    def set_parameter_values(self, values: list[float]) -> None:
        """Update all circuit parameters using the order defined by ``get_parameter_names``."""
        self.circuit.set_parameter_values(values)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return lower and upper bounds registered for each circuit parameter."""
        return self.circuit.get_parameter_bounds()

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Update the admissible range for selected circuit parameters."""
        self.circuit.set_parameter_bounds(ranges)

    def get_constraints(self) -> list[ComparisonTerm]:
        """Expose parameter constraints defined on the circuit.
        Returns:
            list[ComparisonTerm]: a list of constraints on the circuit parameters.
        """
        return self.circuit.get_constraints()
