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

import inspect
from bisect import bisect_right
from collections import defaultdict
from copy import copy
from enum import Enum
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import Domain, Number, Parameter, Term
from qilisdk.utils.visualization import ScheduleStyle
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

_TIME_PARAMETER_NAME = "t"


class Interpolation(str, Enum):
    STEP = "Step function interpolation between schedule points"
    LINEAR = "linear interpolation between schedule points"


def _process_callable(
    function: Callable, current_time: Parameter, **kwargs: Any
) -> tuple[float | Parameter | Term, dict[str, Parameter]]:

    # Define variables
    c = function
    parameters: dict[str, Parameter] = {}

    # get callable parameters
    c_params = inspect.signature(c).parameters
    EMPTY = inspect.Parameter.empty
    # process callable parameters
    for param_name, param_info in c_params.items():
        # parameter type extraction
        if param_info.annotation is not EMPTY and param_info.annotation is Parameter:
            if param_info.default is not EMPTY:
                parameters[param_info.default.label] = copy(param_info.default)
            else:
                value = kwargs.get(param_name, 0)
                if isinstance(value, (float, int)):
                    parameters[param_name] = Parameter(param_name, value)
                elif isinstance(value, Parameter):
                    parameters[value.label] = value

    if _TIME_PARAMETER_NAME in c_params:
        kwargs[_TIME_PARAMETER_NAME] = current_time
    return copy(c(**kwargs)), parameters


@yaml.register_class
class Schedule(Parameterizable):
    """
    Builds a set of time-dependent coefficients applied to a collection of Hamiltonians.

    A Schedule defines the evolution of a system by associating time steps with a set
    of Hamiltonian coefficients. It maintains a dictionary of Hamiltonian objects and a
    corresponding schedule that specifies the coefficients (weights) for each Hamiltonian
    at discrete time steps.

    Example:
        .. code-block:: python

            import numpy as np
            from qilisdk.analog import Schedule, X, Z

            T, dt = 10, 1
            steps = np.linspace(0, T, int(T / dt))

            h1 = X(0) + X(1) + X(2)
            h2 = -Z(0) - Z(1) - 2 * Z(2) + 3 * Z(0) * Z(1)

            schedule = Schedule(
                T=T,
                dt=dt,
                hamiltonians={"driver": h1, "problem": h2},
                schedule={i: {"driver": 1 - t / T, "problem": t / T} for i, t in enumerate(steps)},
            )
            schedule.draw()
    """

    def __init__(
        self,
        tlist: list[float | Parameter | Term],
        hamiltonians: dict[str, Hamiltonian] | None = None,
        coefficients: (
            dict[int, dict[str, float | Term | Parameter | Callable[..., float | Term | Parameter]]] | None
        ) = None,
        interpolation: Interpolation = Interpolation.STEP,
        **kwargs: dict,
    ) -> None:
        """
        Args:
            tlist (list[float]): a list of the time steps to evaluate the schedule.
            hamiltonians (dict[str, Hamiltonian] | None, optional): Mapping from labels to Hamiltonian instances that
                define the building blocks of the schedule. Defaults to None, which creates an empty mapping.
            coefficients (dict[int, dict[str, float | Term | Parameter]] | None, optional): Predefined coefficients for
                specific time steps. Each inner dictionary maps Hamiltonian labels to numerical or symbolic
                coefficients. Defaults to {0: {}} if None.

        Raises:
            ValueError: If the provided schedule references Hamiltonians that have not been defined.
        """
        self._hamiltonians: dict[str, Hamiltonian] = hamiltonians if hamiltonians is not None else {}
        self._coefficients: dict[int, dict[str, int | float | Term | Parameter]] = {0: defaultdict(int)}
        self._coefficient_index_cache: dict[str, list[int]] = {}
        self._coefficient_index_cache_dirty = True
        self._tlist: list[float | Parameter | Term] = tlist
        self._interpolation = interpolation
        self._parameters: dict[str, Parameter] = {}
        self._current_time: Parameter = Parameter(_TIME_PARAMETER_NAME, 0, Domain.REAL)
        self.iter_time_step = 0
        self._nqubits = 0
        self._max_time: float | Parameter | Term | None = None
        self._max_coefficient: float | Parameter | Term | None = None

        if coefficients is not None and max(coefficients.keys()) > len(tlist) - 1:
            raise ValueError(
                f"Can't index tlist point ({max(coefficients.keys())}) that is outside of range (len(tlist) = {len(tlist)})"
            )

        # --- Gather the parameters from the Hamiltonians ---
        for hamiltonian in self._hamiltonians.values():
            self._nqubits = max(self._nqubits, hamiltonian.nqubits)
            for param in hamiltonian.parameters.values():
                self._parameters[param.label] = param

        # --- Gather the parameters from the tlist ---
        for t in self._tlist:
            self._extract_parameters(t)

        # --- resolve the coefficients ---
        lambda_cache: dict[str, Callable] = {}
        if coefficients is not None:
            for index in range(len(tlist)):
                ham_map = coefficients.get(index, {})
                aux = {}
                for ham in self.hamiltonians:
                    if ham not in ham_map:
                        if ham in lambda_cache:
                            self._current_time.set_value(self.tlist[index])
                            aux[ham], _params = _process_callable(lambda_cache[ham], self._current_time, **kwargs)
                            if len(_params) > 0:
                                self._parameters.update(_params)
                        continue
                    current = ham_map[ham]
                    if callable(current):
                        lambda_cache[ham] = current
                        self._current_time.set_value(self.tlist[index])
                        aux[ham], _params = _process_callable(current, self._current_time, **kwargs)
                        if len(_params) > 0:
                            self._parameters.update(_params)
                    elif isinstance(current, (int, float, Parameter, Term)):
                        if ham in lambda_cache:
                            lambda_cache.pop(ham)
                        self._extract_parameters(current)
                        aux[ham] = copy(current)
                    else:
                        raise ValueError
                self._coefficients[index] = copy(aux)
                self._mark_coefficient_index_cache_dirty()

    @property
    def hamiltonians(self) -> dict[str, Hamiltonian]:
        """
        Return the Hamiltonians managed by the schedule.

        Returns:
            dict[str, Hamiltonian]: Mapping of labels to Hamiltonian instances.
        """
        return self._hamiltonians

    @property
    def coefficients(self) -> dict[int, dict[str, float | Parameter | Term]]:
        """
        Return the evaluated schedule of Hamiltonian coefficients.

        Returns:
            dict[int, dict[str, Number]]: Mapping from time indices to evaluated coefficients.
        """
        # out_dict: dict[int, dict[str, Number]] = {}
        # for k, v in self._coefficients.items():
        #     out_dict[k] = {ham: self._get_value(coeff, self.tlist[k]) for ham, coeff in v.items()}
        # return dict(sorted(out_dict.items()))
        return dict(sorted(self._coefficients.items()))

    def _mark_coefficient_index_cache_dirty(self) -> None:
        self._coefficient_index_cache_dirty = True

    def _get_coefficient_indices(self, hamiltonian_key: str) -> list[int]:
        if self._coefficient_index_cache_dirty:
            self._coefficient_index_cache.clear()
            self._coefficient_index_cache_dirty = False
        if hamiltonian_key not in self._coefficient_index_cache:
            indices = sorted(idx for idx, coeffs in self._coefficients.items() if hamiltonian_key in coeffs)
            self._coefficient_index_cache[hamiltonian_key] = indices
        return self._coefficient_index_cache[hamiltonian_key]

    @property
    def T(self) -> float:
        """Total annealing time of the schedule."""
        return max(self.tlist)

    @property
    def tlist(self) -> list[float]:
        tlist = sorted([self._get_value(t) for t in self._tlist])
        if self._max_time is not None:
            max_t = max(tlist)
            tlist = [t * self._get_value(self._max_time) / max_t for t in tlist]
        return tlist

    @property
    def dt(self) -> float:
        """Duration of a single time step in the annealing grid."""
        np_tlist = [
            self._get_value(
                t,
            )
            for t in self.tlist
        ]
        return np.min(np.diff(np.sort(np_tlist)))

    @property
    def nqubits(self) -> int:
        """Maximum number of qubits affected by Hamiltonians contained in the schedule."""
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """Number of symbolic parameters introduced by the Hamiltonians or coefficients."""
        return len(self._parameters)

    def _get_value(self, value: Number | Parameter | Term, t: float | None = None) -> float:
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, complex):
            return value.real
        if isinstance(value, Parameter):
            if value.label == _TIME_PARAMETER_NAME:
                if t is None:
                    raise ValueError("Can't evaluate Parameter because time is not provided.")
                value.set_value(t)
            return float(value.evaluate())
        if isinstance(value, Term):
            if t is None:
                raise ValueError("Can't evaluate term because time is not provided.")
            aux = value.evaluate({self._current_time: t})
            if isinstance(aux, complex):
                return aux.real
            return aux
        raise ValueError(f"Invalid value of type {type(value)} is being evaluated.")

    def _extract_parameters(self, element: float | Parameter | Term) -> None:
        if isinstance(element, Parameter):
            self._parameters[element.label] = element
        elif isinstance(element, Term):
            if not element.is_parameterized_term():
                raise ValueError(
                    f"Tlist can only contain parameters and no variables, but the term {element} contains objects other than parameters."
                )
            for p in element.variables():
                if isinstance(p, Parameter):
                    self._parameters[p.label] = p

    def get_parameter_values(self) -> list[float]:
        """Return the current values associated with the schedule parameters."""
        return [param.value for param in self._parameters.values()]

    def get_parameter_names(self) -> list[str]:
        """Return the ordered list of parameter labels managed by the schedule."""
        return list(self._parameters.keys())

    def get_parameters(self) -> dict[str, float]:
        """Return a mapping from parameter labels to their current numerical values."""
        return {label: param.value for label, param in self._parameters.items()}

    def set_parameter_values(self, values: list[float]) -> None:
        """
        Update the numerical values of all parameters referenced by the schedule.

        Args:
            values (list[float]): New parameter values ordered according to ``get_parameter_names()``.

        Raises:
            ValueError: If the number of provided values does not match ``nparameters``.
        """
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but Schedule has {self.nparameters} parameters.")
        for i, parameter in enumerate(self._parameters.values()):
            parameter.set_value(values[i])

    def set_parameters(self, parameter_dict: dict[str, int | float]) -> None:
        """
        Update a subset of parameters by label.

        Args:
            parameter_dict (dict[str, float]): Mapping from parameter labels to new values.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
        for label, param in parameter_dict.items():
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this Schedule.")
            self._parameters[label].set_value(param)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return the bounds registered for each schedule parameter."""
        return {k: v.bounds for k, v in self._parameters.items()}

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """
        Update the bounds of existing parameters.

        Args:
            ranges (dict[str, tuple[float, float]]): Mapping from label to ``(lower, upper)`` bounds.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
        for label, bound in ranges.items():
            if label not in self._parameters:
                raise ValueError(
                    f"The provided parameter label {label} is not defined in the list of parameters in this object."
                )
            self._parameters[label].set_bounds(bound[0], bound[1])

    def set_max_time(self, max_time: float | Parameter | Term) -> None:
        self._extract_parameters(max_time)
        self._max_time = max_time

    def _add_hamiltonian_from_dict(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: dict[int, float | Term | Parameter | Callable[..., float | Term | Parameter]],
        **kwargs: Any,
    ) -> None:
        if label in self._hamiltonians:
            raise ValueError(f"Can't add Hamiltonian because label {label} is already associated with a Hamiltonian.")
        self._hamiltonians[label] = hamiltonian

        for index, current in coefficients.items():
            if callable(current):
                self._current_time.set_value(self.tlist[index])
                self._coefficients[index][label], _params = _process_callable(current, self._current_time, **kwargs)
                if len(_params) > 0:
                    self._parameters.update(_params)
                self._mark_coefficient_index_cache_dirty()
            elif isinstance(current, (int, float, Parameter, Term)):
                self._extract_parameters(current)
                self._coefficients[index][label] = copy(current)
                self._mark_coefficient_index_cache_dirty()
            else:
                raise ValueError

    def _add_hamiltonian_from_lambda(
        self, label: str, hamiltonian: Hamiltonian, coefficients: Callable[..., float | Term | Parameter], **kwargs: Any
    ) -> None:
        if label in self._hamiltonians:
            raise ValueError(f"Can't add Hamiltonian because label {label} is already associated with a Hamiltonian.")
        self._hamiltonians[label] = hamiltonian
        self._current_time.set_value(self.tlist[0])
        self._coefficients[0][label], _params = _process_callable(coefficients, self._current_time, **kwargs)
        if len(_params) > 0:
            self._parameters.update(_params)
        self._mark_coefficient_index_cache_dirty()

    @overload
    def add_hamiltonian(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: dict[int, float | Term | Parameter | Callable[..., float | Term | Parameter]],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def add_hamiltonian(
        self, label: str, hamiltonian: Hamiltonian, coefficients: Callable[..., float | Term | Parameter], **kwargs: Any
    ) -> None: ...

    def add_hamiltonian(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: (
            Callable[..., float | Term | Parameter]
            | dict[int, float | Term | Parameter | Callable[..., float | Term | Parameter]]
        ),
        **kwargs: Any,
    ) -> None:
        if callable(coefficients):
            self._add_hamiltonian_from_lambda(label, hamiltonian, coefficients, **kwargs)
        elif isinstance(coefficients, dict):
            self._add_hamiltonian_from_dict(label, hamiltonian, coefficients, **kwargs)
        else:
            raise ValueError("Unsupported type of coefficient.")

    def _update_hamiltonian_from_dict(
        self,
        label: str,
        new_coefficients: dict[int, float | Term | Parameter | Callable[..., float | Term | Parameter]] | None = None,
        **kwargs: Any,
    ) -> None:
        if new_coefficients is not None:
            for index, current in new_coefficients.items():
                if callable(current):
                    self._current_time.set_value(self.tlist[index])
                    self._coefficients[index][label], _params = _process_callable(current, self._current_time, **kwargs)
                    if len(_params) > 0:
                        self._parameters.update(_params)
                    self._mark_coefficient_index_cache_dirty()
                elif isinstance(current, (int, float, Parameter, Term)):
                    self._extract_parameters(current)
                    self._coefficients[index][label] = copy(current)
                    self._mark_coefficient_index_cache_dirty()
                else:
                    raise ValueError

    def _update_hamiltonian_from_lambda(
        self, label: str, new_coefficients: Callable[..., float | Term | Parameter] | None = None, **kwargs: Any
    ) -> None:
        if new_coefficients is not None:
            self._current_time.set_value(self.tlist[0])
            self._coefficients[0][label], _params = _process_callable(new_coefficients, self._current_time, **kwargs)
            if len(_params) > 0:
                self._parameters.update(_params)
            self._mark_coefficient_index_cache_dirty()

    @overload
    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: dict[int, float | Term | Parameter | Callable[..., float | Term | Parameter]] | None = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: Callable[..., float | Term | Parameter] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: (
            Callable[..., float | Term | Parameter]
            | dict[int, float | Term | Parameter | Callable[..., float | Term | Parameter]]
            | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        if new_hamiltonian is not None:
            self._hamiltonians[label] = new_hamiltonian
        if new_coefficients is not None:
            if callable(new_coefficients):
                self._update_hamiltonian_from_lambda(label, new_coefficients, **kwargs)
            elif isinstance(new_coefficients, dict):
                self._update_hamiltonian_from_dict(label, new_coefficients, **kwargs)
            else:
                raise ValueError("Unsupported type of coefficient.")

    def __getitem__(self, time_step: float) -> Hamiltonian:
        """
        Retrieve the effective Hamiltonian at a given time step.

        The effective Hamiltonian is computed by summing the contributions of all Hamiltonians,
        using the latest defined coefficients at or before the given time step.

        Args:
            time_step (float): Time step for which to retrieve the Hamiltonian (``time_step * dt`` in units).

        Returns:
            Hamiltonian: The effective Hamiltonian at the specified time step with coefficients evaluated to numbers.
        """
        ham = Hamiltonian()
        for ham_label in self._hamiltonians:
            coeff = self.get_coefficient(time_step, ham_label)
            ham += coeff * self._hamiltonians[ham_label]
        return ham.get_static_hamiltonian()

    def get_coefficient(self, time_step: float, hamiltonian_key: str) -> float:
        """
        Retrieve the coefficient of a specified Hamiltonian at a given time.

        This function searches backwards in time (by multiples of dt) until it finds a defined
        coefficient for the given Hamiltonian.

        Args:
            time_step (float): The time (in the same units as ``T``) at which to query the coefficient.
            hamiltonian_key (str): The label of the Hamiltonian.

        Returns:
            Number: The coefficient of the Hamiltonian at the specified time, or 0 if not defined.
        """
        time_step = float(time_step)
        val = self.get_coefficient_expression(time_step=time_step, hamiltonian_key=hamiltonian_key)
        return self._get_value(val, time_step)

    def get_coefficient_expression(self, time_step: float, hamiltonian_key: str) -> Number | Term | Parameter:
        """
        Retrieve the expression of a specified Hamiltonian at a given time. If any parameters are
        present in the expression they will be printed in the expression.

        This function searches backwards in time (by multiples of dt) until it finds a defined
        coefficient for the given Hamiltonian.

        Args:
            time_step (float): The time (in the same units as ``T``) at which to query the coefficient.
            hamiltonian_key (str): The label of the Hamiltonian.

        Returns:
            Number | Term: The coefficient expression of the Hamiltonian at the specified time, or 0 if not defined.

        Raises:
            ValueError: If the the interpolation method is not supported.
        """

        if self._interpolation is Interpolation.STEP:
            return self._get_coefficient_expression_step(time_step, hamiltonian_key)
        if self._interpolation is Interpolation.LINEAR:
            return self._get_coefficient_expression_linear(time_step, hamiltonian_key)

        raise ValueError(f"interpolation {self._interpolation.value} is not supported!")

    def _get_coefficient_expression_step(self, time_step: float, hamiltonian_key: str) -> Number | Term | Parameter:
        """
        Retrieve the expression of a specified Hamiltonian at a given time. If any parameters are
        present in the expression they will be printed in the expression.

        This function searches backwards in time (by multiples of dt) until it finds a defined
        coefficient for the given Hamiltonian.

        Args:
            time_step (float): The time (in the same units as ``T``) at which to query the coefficient.
            hamiltonian_key (str): The label of the Hamiltonian.

        Returns:
            Number | Term: The coefficient expression of the Hamiltonian at the specified time, or 0 if not defined.
        """
        time_idx = 0
        for i, t in enumerate(self.tlist):
            if time_step < t:
                break
            time_idx = i
        while time_idx >= 0:
            if time_idx in self._coefficients and hamiltonian_key in self._coefficients[time_idx]:
                val = self._coefficients[time_idx][hamiltonian_key]
                return val
            time_idx -= 1
        return 0

    def _get_coefficient_expression_linear(self, time_step: float, hamiltonian_key: str) -> Number | Term | Parameter:
        """
        Retrieve the expression of a specified Hamiltonian at a given time. If any parameters are
        present in the expression they will be printed in the expression.

        This function searches backwards in time (by multiples of dt) until it finds a defined
        coefficient for the given Hamiltonian.

        Args:
            time_step (float): The time (in the same units as ``T``) at which to query the coefficient.
            hamiltonian_key (str): The label of the Hamiltonian.

        Returns:
            Number | Term: The coefficient expression of the Hamiltonian at the specified time, or 0 if not defined.

        Raises:
            ValueError: If something unexpected happens during coefficient retrieval.
        """
        t_idx = 0
        for i, t in enumerate(self.tlist):
            if time_step < t:
                break
            t_idx = i

        # if t_idx in self._coefficients and hamiltonian_key in self._coefficients[t_idx]:
        #     return self._coefficients[t_idx][hamiltonian_key]

        coeff_indices = self._get_coefficient_indices(hamiltonian_key)
        if not coeff_indices:
            return 0
        insert_pos = bisect_right(coeff_indices, t_idx)
        prev_idx = coeff_indices[insert_pos - 1] if insert_pos else None
        next_idx = coeff_indices[insert_pos] if insert_pos < len(coeff_indices) else None
        prev_expr = self._coefficients[prev_idx][hamiltonian_key] if prev_idx is not None else None
        next_expr = self._coefficients[next_idx][hamiltonian_key] if next_idx is not None else None

        # cases
        if prev_expr is None and next_expr is not None:
            return next_expr
        if next_expr is None and prev_expr is not None:
            return prev_expr
        if prev_expr is None and next_expr is None:
            return 0

        # linear interpolation (keeps expressions if they are Terms/Parameters)
        if next_idx is None or prev_idx is None or prev_expr is None or next_expr is None:
            raise ValueError("Something unexpected happened while retrieving the coefficient.")
        alpha: float = (time_step - self.tlist[prev_idx]) / (self.tlist[next_idx] - self.tlist[prev_idx])
        next_is_term = isinstance(next_expr, (Term, Parameter))
        prev_is_term = isinstance(prev_expr, (Term, Parameter))
        if next_is_term and prev_is_term and next_expr != prev_expr:
            next_expr = self._get_value(next_expr, self.tlist[next_idx])
            prev_expr = self._get_value(prev_expr, self.tlist[prev_idx])
        elif next_is_term and not prev_is_term:
            next_expr = self._get_value(next_expr, self.tlist[next_idx])
        elif prev_is_term and not next_is_term:
            prev_expr = self._get_value(prev_expr, self.tlist[prev_idx])

        e1 = next_expr * alpha
        e2 = prev_expr * (1 - alpha)
        return e1 + e2

    def __len__(self) -> int:
        """
        Get the total number of discrete time steps in the annealing process.

        Returns:
            int: The number of time steps, calculated as T / dt.
        """
        return len(self.tlist)

    def __iter__(self) -> Schedule:
        """
        Return an iterator over the schedule's time steps.

        Returns:
            Schedule: The schedule instance itself as an iterator.
        """
        self.iter_time_step = 0
        return self

    def __next__(self) -> Hamiltonian:
        """
        Retrieve the next effective Hamiltonian in the schedule during iteration.

        Returns:
            Hamiltonian: The effective Hamiltonian at the current time step.

        Raises:
            StopIteration: When the iteration has reached beyond the total number of time steps.
        """
        if self.iter_time_step < self.__len__():
            result = self[self.tlist[self.iter_time_step]]
            self.iter_time_step += 1
            return result
        raise StopIteration

    def draw(
        self, style: ScheduleStyle | None = None, filepath: str | None = None, time_precision: float = 0.01
    ) -> None:
        """Render a plot of the schedule using matplotlib and optionally save it to a file.

        The schedule is rendered using the provided style configuration. If ``filepath`` is
        given, the resulting figure is saved to disk (the output format is inferred
        from the file extension, e.g. ``.png``, ``.pdf``, ``.svg``).

        Args:
            style (ScheduleStyle, optional): Customization options for the plot appearance.
                Defaults to ScheduleStyle().
            filepath (str | None, optional): If provided, saves the plot to the specified file path.
        """
        from qilisdk.utils.visualization.schedule_renderers import MatplotlibScheduleRenderer  # noqa: PLC0415

        style = style or ScheduleStyle()
        renderer = MatplotlibScheduleRenderer(self, style=style, time_precision=time_precision)
        renderer.plot()
        if filepath:
            renderer.save(filepath)
        else:
            renderer.show()
