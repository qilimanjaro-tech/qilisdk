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
from typing import Callable, ClassVar, Iterator

from qilisdk.core import Parameter
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.qtensor import InitialState, QTensor
from qilisdk.digital.circuit import Circuit
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class DigitalPropagation(PrimitiveFunctional):
    """
    Propagate a quantum state through a digital circuit.

    The circuit is executed and results are returned based on the :class:`~qilisdk.readout.Readout`
    passed to :meth:`~qilisdk.backends.Backend.execute`.

    Example:
        .. code-block:: python

            from qilisdk.digital.circuit import Circuit
            from qilisdk.functionals import DigitalPropagation
            from qilisdk.readout import Readout

            circuit = Circuit(nqubits=2)
            circuit.h(0)
            functional = DigitalPropagation(circuit)
            result = backend.execute(functional, readout=Readout().with_sampling(nshots=1024))
            counts = result.samples  # dict[str, int]
    """

    result_type: ClassVar[type[FunctionalResult]] = FunctionalResult

    def __init__(self, 
                 circuit: Circuit, 
                 initial_state: QTensor | InitialState | Circuit = InitialState.ZERO,
                ) -> None:
        """
        Args:
            circuit (Circuit): Circuit to propagate.
            initial_state (QTensor | InitialState | Circuit): Quantum state used as the simulation starting point.
        """
        super().__init__()
        # Circuit init just prepends it, that way if it's parameterized it will be handled correctly
        if isinstance(initial_state, Circuit):
            self.initial_state = InitialState.ZERO
            self.circuit = initial_state + circuit
        # Otherwise we leave it as is and let the backend handle it
        else:
            self.initial_state = initial_state
            self.circuit = circuit

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        """Yield the circuit as the sole parameterizable child.

        Yields:
            Iterator[Parameterizable]: The underlying ``Circuit``.
        """
        yield self.circuit

    def __repr__(self) -> str:
        return f"DigitalPropagation(circuit={self.circuit}, initial_state={self.initial_state})"

    def set_parameter_values(
        self,
        values: list[float],
        where: Callable[[Parameter], bool] | None = None,
    ) -> None:
        """
        Assign parameter values by position and clear caches.

        Args:
            values (list[float]): New values ordered consistently with ``get_parameter_names()``.
            where (Callable[[Parameter], bool] | None): Optional predicate selecting parameters to update.
        """
        self.circuit.set_parameter_values(values=values, where=where)

    def set_parameters(self, parameters: dict[str, int | float]) -> None:
        """
        Assign parameter values by name and clear caches.

        Args:
            parameters (dict[str, int | float]): Mapping from parameter labels to numeric values.
        """
        self.circuit.set_parameters(parameters)

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """
        Update parameter bounds and clear caches.

        Args:
            ranges (dict[str, tuple[float, float]]): Bounds keyed by parameter label.
        """
        self.circuit.set_parameter_bounds(ranges)
