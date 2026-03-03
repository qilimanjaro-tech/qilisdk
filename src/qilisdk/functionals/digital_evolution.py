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
from typing import Iterator

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.digital.circuit import Circuit
from qilisdk.functionals.functional import PrimitiveFunctional, ReadoutMethod
from qilisdk.yaml import yaml


@yaml.register_class
class DigitalEvolution(PrimitiveFunctional):
    """
    Execute a digital circuit.

    Example:
        .. code-block:: python

            from qilisdk.digital.circuit import Circuit
            from qilisdk.functionals.sampling import Sampling

            circuit = Circuit(nqubits=2)
            circuit.h(0)
            sampler = DigitalEvolution(circuit, readout=ReadoutMethod.sample(nshots=1024))
    """

    def __init__(self, circuit: Circuit, readout: ReadoutMethod | list[ReadoutMethod]) -> None:
        """
        Args:
            circuit (Circuit): Circuit to execute for sampling.
            readout (ReadoutMethod | list[ReadoutMethod], optional): Readout method to be applied at the end of the execution.
        """
        super().__init__(readout=readout)
        self.circuit = circuit

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        yield self.circuit
