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
    """Mixin for objects that expose tunable parameters and constraints.

    Subclasses can compose parameter interfaces by yielding child
    :class:`Parameterizable` instances from :meth:`_iter_parameter_children`.
    """

    def __init__(self) -> None:
        super(Parameterizable, self).__init__()
        self._parameters: dict[str, Parameter] = {}
        self._parameter_constraints: list[ComparisonTerm] = []
        self._prefix = ""

    # Private Methods

    def _iter_parameter_children(self) -> Iterable[Parameterizable]:  # noqa: PLR6301
        """Yield child objects that compose this object's parameter interface.

        Override this method in subclasses that expose nested
        :class:`Parameterizable` objects.

        Returns:
            Iterable[Parameterizable]: Child objects that contribute parameters to this instance.
        """
        return ()

    def _iter_parameter_items(self) -> Iterable[tuple[str, Parameter]]:
        """Yield ``(label, parameter)`` items from this object and its children."""
        local_params = self._parameters or {}
        yield from local_params.items()
        for child in self._iter_parameter_children():
            yield from child._iter_parameter_items()  # noqa: SLF001

    def _add_parameter(self, label: str, parameter: Parameter) -> None:
        """Add a parameter under the current prefix.

        Args:
            label (str): Parameter label before prefixing.
            parameter (Parameter): Parameter instance to register.
        """
        self._parameters[self._prefix + label] = parameter

    def _add_parameter_from(self, parameter_label: str, other: Parameterizable, new_label: str | None = None) -> None:
        """Add a parameter from another :class:`Parameterizable`.

        Args:
            parameter_label (str): Label of the source parameter in ``other``.
            other (Parameterizable): Object to pull the parameter from.
            new_label (str | None, optional): Optional label to use in this object.
        """
        if new_label:
            self._parameters[self._prefix + new_label] = other._parameters[parameter_label]
        else:
            self._parameters[self._prefix + parameter_label] = other._parameters[parameter_label]

    @staticmethod
    def _query_parameter_original_name(parameterizable: Parameterizable, label: str) -> str:
        """Return the underlying parameter label stored on the :class:`Parameter`.

        Args:
            parameterizable (Parameterizable): Object containing the parameter.
            label (str): Parameter label used in ``parameterizable``.

        Returns:
            str: Original parameter label.
        """
        return parameterizable._parameters[label].label

    def _update_parameters(self, parameters: dict[str, Parameter]) -> None:
        """Update local parameters with the provided mapping.

        Args:
            parameters (dict[str, Parameter]): Parameters to merge into this object.
        """
        self._parameters.update(parameters)

    def _link_parameters(self, other: Parameterizable) -> None:
        """Link all parameters from another object into this one.

        Parameters are shared by reference, so updates in this object
        affect the same underlying :class:`Parameter` instances.

        Args:
            other (Parameterizable): Object to copy parameter references from.
        """
        for label, p in other._parameters.items():
            self._add_parameter(label, p)

    def _filtered_parameter_map(
        self,
        where: Callable[[Parameter], bool] | None = None,
    ) -> dict[str, Parameter]:
        """Return parameters, optionally filtered by a predicate.

        Args:
            where (Callable[[Parameter], bool] | None, optional): Predicate applied to each parameter.

        Returns:
            dict[str, Parameter]: Filtered parameter mapping.
        """
        if where is None:
            return dict(self._iter_parameter_items())
        return {label: param for label, param in self._iter_parameter_items() if where(param)}

    # Public Methods

    @property
    def nparameters(self) -> int:
        """Number of tunable parameters defined by the object."""
        return len(dict(self._iter_parameter_items()))

    def set_prefix(
        self,
        prefix: str,
        where: Callable[[Parameter], bool] | None = None,
    ) -> None:
        """Set a prefix on parameter labels.

        Args:
            prefix (str): Prefix to prepend to selected parameter labels.
            where (Callable[[Parameter], bool] | None): Optional predicate selecting local parameters.

        Notes:
            The ``where`` predicate is applied to local parameters only. Child parameterizable
            objects always receive the same prefix operation recursively.
        """
        old_keys: list[str] = list(self._filtered_parameter_map(where=where))
        for name in old_keys:
            if not name.startswith(prefix):
                _name = name.removeprefix(self._prefix) if self._prefix and name.startswith(self._prefix) else name
                self._parameters[prefix + _name] = self._parameters.pop(name)
        for child in self._iter_parameter_children():
            child.set_prefix(prefix)
        self._prefix = prefix

    def get_prefix(self) -> str:
        """Return the currently configured parameter prefix for this object."""
        return self._prefix

    def add_parameter_constraint(self, constraint: ComparisonTerm) -> None:
        """Add a constraint on a single or a set of parameters

        Args:
            constraint (ComparisonTerm): The comparison term to specify the constraint. Only Parameter objects are allowed in the constraint.

        Raises:
            ValueError: If Generic Variables are present in the constraint.
        """
        if not (constraint.lhs.is_parameterized_term() and constraint.rhs.is_parameterized_term()):
            raise ValueError(
                "The constraint should only contain parameters and having generic variables is not allowed."
            )

        self._parameter_constraints.append(constraint)

    def get_parameter_values(
        self,
        where: Callable[[Parameter], bool] | None = None,
    ) -> list[float]:
        """Return the current numerical values of the parameters.

        Args:
            where (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return list(self.get_parameters(where=where).values())

    def get_parameter_names(
        self,
        where: Callable[[Parameter], bool] | None = None,
    ) -> list[str]:
        """Return the ordered list of parameter labels.

        Args:
            where (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return list(self.get_parameters(where=where).keys())

    def get_parameters(
        self,
        where: Callable[[Parameter], bool] | None = None,
    ) -> dict[str, RealNumber]:
        """Return a mapping from parameter labels to their current numerical values.

        Args:
            where (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return {label: param.value for label, param in self._filtered_parameter_map(where=where).items()}

    def set_parameter_values(
        self,
        values: list[float],
        where: Callable[[Parameter], bool] | None = None,
    ) -> None:
        """
        Update all parameter values at once.

        Args:
            values (list[float]): New parameter values ordered consistently with ``get_parameter_names()``.
            where (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.

        Raises:
            ValueError: If ``values`` does not match the number of parameters selected by ``where``.
        """
        param_names = self.get_parameter_names(where=where)
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
        where: Callable[[Parameter], bool] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Return the ``(lower, upper)`` bounds associated with each parameter.

        Args:
            where (Callable[[Parameter], bool] | None): Optional predicate over ``Parameter`` objects.
        """
        return {label: param.bounds for label, param in self._filtered_parameter_map(where=where).items()}

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
            list[ComparisonTerm]: Comparison terms defined locally and by child parameterizable objects.
        """
        constraints = list((self._parameter_constraints or []))
        for child in self._iter_parameter_children():
            constraints.extend(child.get_constraints())
        return constraints

    def check_constraints(self, parameters: dict[str, float]) -> bool:
        """Validate that proposed parameter updates satisfy all constraints.

        Args:
            parameters (dict[str, float]): Candidate updates keyed by parameter label.

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
