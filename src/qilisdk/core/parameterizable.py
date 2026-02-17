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

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qilisdk.core.variables import BaseVariable, ComparisonTerm, Parameter

    from .types import RealNumber


class Parameterizable(ABC):
    """Mixin for objects that expose tunable parameters and constraints."""

    def __init__(self) -> None:
        super(Parameterizable, self).__init__()
        self._parameters: dict[str, Parameter] = {}
        self._parameter_constraints: list[ComparisonTerm] = []

    def _iter_parameter_children(self) -> Iterable[Parameterizable]:  # noqa: PLR6301
        """Yield parameterizable children to compose this object's parameter interface.

        Returns:
            Iterable[Parameterizable]: Child objects that contribute parameters to this instance.
        """
        return ()

    def _iter_parameter_items(self) -> Iterable[tuple[str, Parameter]]:
        """Yield ``(label, parameter)`` items exposed by this object."""
        local_params = getattr(self, "_parameters", {})
        yield from local_params.items()
        for child in self._iter_parameter_children():
            yield from child._iter_parameter_items()  # noqa: SLF001

    def _parameter_map(self) -> dict[str, Parameter]:
        params: dict[str, Parameter] = {}
        for label, parameter in self._iter_parameter_items():
            params[label] = parameter
        return params

    def _filtered_parameter_map(self, trainable: bool | None = None) -> dict[str, Parameter]:
        params = self._parameter_map()
        if trainable is None:
            return params
        return {label: param for label, param in params.items() if param.is_trainable is trainable}

    @property
    def nparameters(self) -> int:
        """Number of tunable parameters defined by the object."""
        return len(self._parameter_map())

    def get_parameter_values(self, trainable: bool | None = None) -> list[float]:
        """Return the current numerical values of the parameters."""
        return list(self.get_parameters(trainable=trainable).values())

    def get_parameter_names(self, trainable: bool | None = None) -> list[str]:
        """Return the ordered list of parameter labels."""
        return list(self.get_parameters(trainable=trainable).keys())

    def get_parameters(self, trainable: bool | None = None) -> dict[str, RealNumber]:
        """Return a mapping from parameter labels to their current numerical values."""
        return {label: param.value for label, param in self._filtered_parameter_map(trainable=trainable).items()}

    def set_parameter_values(self, values: list[float], trainable: bool | None = None) -> None:
        """
        Update all parameter values at once.

        Args:
            values (list[float]): New parameter values ordered consistently with ``get_parameter_names()``.

        Raises:
            ValueError: If ``values`` does not contain exactly ``nparameters`` entries.
        """
        param_names = self.get_parameter_names(trainable=trainable)
        if len(values) != len(param_names):
            raise ValueError(f"Provided {len(values)} but this object has {len(param_names)} parameters.")
        value_dict = {param_names[i]: values[i] for i in range(len(values))}
        self.set_parameters(value_dict)

    def set_parameters(self, parameters: dict[str, float]) -> None:
        """
        Update a subset of parameters by label.

        Args:
            parameters (dict[str, float]): Mapping from parameter labels to updated numeric values.

        Raises:
            ValueError: If an unknown parameter label is provided or constraints are violated.
        """
        available_parameters = self._parameter_map()
        if not self.check_constraints(parameters):
            raise ValueError(
                f"New assignation of the parameters breaks the parameter constraints: \n{self.get_constraints()}"
            )
        for label, param in parameters.items():
            if label not in available_parameters:
                raise ValueError(f"Parameter {label} is not defined for this object.")
            available_parameters[label].set_value(param)

    def get_parameter_bounds(self, trainable: bool | None = None) -> dict[str, tuple[float, float]]:
        """Return the ``(lower, upper)`` bounds associated with each parameter."""
        return {label: param.bounds for label, param in self._filtered_parameter_map(trainable=trainable).items()}

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """
        Update the allowable ranges for the specified parameters.

        Args:
            ranges (dict[str, tuple[float, float]]): Mapping from parameter label to ``(lower, upper)`` bounds.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
        available_parameters = self._parameter_map()
        for label, bound in ranges.items():
            if label not in available_parameters:
                raise ValueError(
                    f"The provided parameter label {label} is not defined in the list of parameters in this object."
                )
            available_parameters[label].set_bounds(bound[0], bound[1])

    def get_constraints(self) -> list[ComparisonTerm]:
        """Get all constraints on the parameters.

        Returns:
            list[ComparisonTerm]: A list of comparison terms involving the parameters of the Object.
        """
        constraints = list(getattr(self, "_parameter_constraints", []))
        for child in self._iter_parameter_children():
            constraints.extend(child.get_constraints())
        return constraints

    def check_constraints(self, parameters: dict[str, float]) -> bool:
        """Validate that proposed parameter updates satisfy all constraints.

        Args:
            parameters (dict[str, float]): Candidate parameter values keyed by label.

        Returns:
            bool: True if every constraint evaluates to True for the provided values.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
        available_parameters = self._parameter_map()
        evaluate_dict: dict[BaseVariable, float] = {}
        for label, value in parameters.items():
            if label not in available_parameters:
                raise ValueError(f"Parameter {label} is not defined for this object.")
            evaluate_dict[available_parameters[label]] = value
        constraints = self.get_constraints()
        valid = all(con.evaluate(evaluate_dict) for con in constraints)
        return valid
