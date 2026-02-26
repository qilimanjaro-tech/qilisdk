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
    from collections.abc import Callable, Iterable

    from qilisdk.core.variables import BaseVariable, ComparisonTerm, Parameter

    from .types import RealNumber


class Parameterizable(ABC):
    """Mixin for objects that expose tunable parameters and constraints."""

    def __init__(self) -> None:
        super(Parameterizable, self).__init__()
        self._parameters: dict[str, Parameter] = {}
        self._parameter_constraints: list[ComparisonTerm] = []
        self._prefix = ""

    def _iter_parameter_children(self) -> Iterable[Parameterizable]:  # noqa: PLR6301
        """Yield parameterizable children to compose this object's parameter interface.

        Returns:
            Iterable[Parameterizable]: Child objects that contribute parameters to this instance.
        """
        return ()

    def _iter_parameter_items(self) -> Iterable[tuple[str, Parameter]]:
        """Yield ``(label, parameter)`` items exposed by this object."""
        local_params = self._parameters or {}
        yield from local_params.items()
        for child in self._iter_parameter_children():
            yield from child._iter_parameter_items()  # noqa: SLF001

    def set_prefix(
        self,
        prefix: str,
        parameter_filter: Callable[[Parameter], bool] | None = None,
    ) -> None:
        """Sets a prefix to all existing parameters in the object.

        Args:
            prefix (str): the prefix to be set.
        """
        if parameter_filter:
            old_keys: list[str] = [key for key, value in self._parameters.items() if parameter_filter(value)]
        else:
            old_keys: list[str] = list(self._parameters.keys())
        for name in old_keys:
            if not name.startswith(prefix):
                _name = name.removeprefix(self._prefix) if self._prefix and name.startswith(self._prefix) else name
                self._parameters[prefix + _name] = self._parameters.pop(name)
        for child in self._iter_parameter_children():
            child.set_prefix(prefix)
        self._prefix = prefix

    def get_prefix(self) -> str:
        return self._prefix

    def _filtered_parameter_map(
        self,
        parameter_filter: Callable[[Parameter], bool] | None = None,
    ) -> dict[str, Parameter]:
        if parameter_filter is None:
            return dict(self._iter_parameter_items())
        return {label: param for label, param in self._iter_parameter_items() if parameter_filter(param)}

    @property
    def nparameters(self) -> int:
        """Number of tunable parameters defined by the object."""
        return len(dict(self._iter_parameter_items()))

    def get_parameter_values(
        self,
        parameter_filter: Callable[[Parameter], bool] | None = None,
    ) -> list[float]:
        """Return the current numerical values of the parameters.

        Args:
            parameter_filter (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return list(self.get_parameters(parameter_filter=parameter_filter).values())

    def get_parameter_names(
        self,
        parameter_filter: Callable[[Parameter], bool] | None = None,
    ) -> list[str]:
        """Return the ordered list of parameter labels.

        Args:
            parameter_filter (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return list(self.get_parameters(parameter_filter=parameter_filter).keys())

    def get_parameters(
        self,
        parameter_filter: Callable[[Parameter], bool] | None = None,
    ) -> dict[str, RealNumber]:
        """Return a mapping from parameter labels to their current numerical values.

        Args:
            parameter_filter (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return {
            label: param.value
            for label, param in self._filtered_parameter_map(parameter_filter=parameter_filter).items()
        }

    def set_parameter_values(
        self,
        values: list[float],
        parameter_filter: Callable[[Parameter], bool] | None = None,
    ) -> None:
        """
        Update all parameter values at once.

        Args:
            values (list[float]): New parameter values ordered consistently with ``get_parameter_names()``.
            parameter_filter (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.

        Raises:
            ValueError: If ``values`` does not contain exactly ``nparameters`` entries.
        """
        param_names = self.get_parameter_names(parameter_filter=parameter_filter)
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
        available_parameters = self._filtered_parameter_map()
        if not self.check_constraints(parameters):
            raise ValueError(
                f"New assignation of the parameters breaks the parameter constraints: \n{self.get_constraints()}"
            )
        for label, param in parameters.items():
            if label not in available_parameters:
                raise ValueError(f"Parameter {label} is not defined for this object.")
            available_parameters[label].set_value(param)

    def get_parameter_bounds(
        self,
        parameter_filter: Callable[[Parameter], bool] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Return the ``(lower, upper)`` bounds associated with each parameter.

        Args:
            parameter_filter (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return {
            label: param.bounds
            for label, param in self._filtered_parameter_map(parameter_filter=parameter_filter).items()
        }

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """
        Update the allowable ranges for the specified parameters.

        Args:
            ranges (dict[str, tuple[float, float]]): Mapping from parameter label to ``(lower, upper)`` bounds.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
        available_parameters = self._filtered_parameter_map()
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
        constraints = list((self._parameter_constraints or []))
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
        available_parameters = self._filtered_parameter_map()
        evaluate_dict: dict[BaseVariable, float] = {}
        for label, value in parameters.items():
            if label not in available_parameters:
                raise ValueError(f"Parameter {label} is not defined for this object.")
            evaluate_dict[available_parameters[label]] = value
        constraints = self.get_constraints()
        valid = all(con.evaluate(evaluate_dict) for con in constraints)
        return valid
