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
from typing import ClassVar, Iterator

from qilisdk.core.parameterizable import Parameterizable
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
        super().__init__()
        self.circuit = circuit
        self.nshots = nshots

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        yield self.circuit
