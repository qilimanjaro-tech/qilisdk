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
from collections.abc import Callable
from copy import copy
from enum import Enum
from typing import Any, overload

import numpy as np

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import Domain, Number, Parameter, Term
from qilisdk.utils.visualization import ScheduleStyle
from qilisdk.yaml import yaml

_TIME_PARAMETER_NAME = "t"
PARAMETERIZED_NUMBER = float | Parameter | Term

# type aliases just to keep this short
TimeDict = dict[PARAMETERIZED_NUMBER | tuple[float, float], PARAMETERIZED_NUMBER | Callable[..., PARAMETERIZED_NUMBER]]
CoeffDict = dict[str, TimeDict]
InterpDict = dict[str, "Interpolator"]


class Interpolation(str, Enum):
    STEP = "Step function interpolation between schedule points"
    LINEAR = "linear interpolation between schedule points"


def _process_callable(
    function: Callable, current_time: Parameter, **kwargs: Any
) -> tuple[PARAMETERIZED_NUMBER, dict[str, Parameter]]:

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
        hamiltonians: dict[str, Hamiltonian] | None = None,
        coefficients: InterpDict | CoeffDict | None = None,
        dt: float = 0.1,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        # THIS is the only runtime implementation
        self._hamiltonians = hamiltonians if hamiltonians is not None else {}
        self._coefficients: dict[str, Interpolator] = {}
        self._interpolation = None
        self._parameters: dict[str, Parameter] = {}
        self._current_time: Parameter = Parameter(_TIME_PARAMETER_NAME, 0, Domain.REAL)
        self.iter_time_step = 0
        self._nqubits = 0
        self._max_time: PARAMETERIZED_NUMBER | None = None
        self._dt = dt

        coefficients = coefficients or {}

        for ham in self._hamiltonians:
            if ham not in coefficients:
                self._coefficients[ham] = Interpolator({0: 0}, interpolation=interpolation)
                continue
            coeff = copy(coefficients[ham])
            if isinstance(coeff, Interpolator):
                self._coefficients[ham] = coeff
            elif isinstance(coeff, dict):
                self._coefficients[ham] = Interpolator(coeff, interpolation, **kwargs)

            for p_name, p_value in self._coefficients[ham].parameters.items():
                self._parameters[p_name] = p_value

    @property
    def hamiltonians(self) -> dict[str, Hamiltonian]:
        """
        Return the Hamiltonians managed by the schedule.

        Returns:
            dict[str, Hamiltonian]: Mapping of labels to Hamiltonian instances.
        """
        return self._hamiltonians

    @property
    def coefficients(self) -> dict[str, dict[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER]]:
        return {ham: self._coefficients[ham].coefficients_dict for ham in self._hamiltonians}

    @property
    def T(self) -> float:
        """Total annealing time of the schedule."""
        return max(self.tlist)

    @property
    def tlist(self) -> list[float]:
        _tlist: set[float] = set()
        for ham in self._hamiltonians:
            _tlist.update(self._coefficients[ham].fixed_tlist)
        tlist = sorted(_tlist)
        if self._max_time is not None:
            max_t = max(tlist)
            tlist = [t * self._get_value(self._max_time) / max_t for t in tlist]
        return tlist

    @property
    def dt(self) -> float:
        return self._dt

    def set_dt(self, dt: float) -> None:
        if not isinstance(dt, float):
            raise ValueError(f"dt is only allowed to be a float but {type(dt)} was provided")
        self._dt = dt

    @property
    def nqubits(self) -> int:
        """Maximum number of qubits affected by Hamiltonians contained in the schedule."""
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """Number of symbolic parameters introduced by the Hamiltonians or coefficients."""
        return len(self._parameters)

    def _get_value(self, value: PARAMETERIZED_NUMBER | complex, t: float | None = None) -> float:
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
            for v in value.variables():
                if v.label == _TIME_PARAMETER_NAME and t is None:
                    raise ValueError("Can't evaluate Parameter because time is not provided.")
            aux = value.evaluate({self._current_time: t}) if t is not None else value.evaluate({})
            if isinstance(aux, complex):
                return aux.real
            return aux
        raise ValueError(f"Invalid value of type {type(value)} is being evaluated.")

    def _extract_parameters(self, element: PARAMETERIZED_NUMBER) -> None:
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
        return [param.value for param in self._parameters.values()]

    def get_parameter_names(self) -> list[str]:
        return list(self._parameters.keys())

    def get_parameters(self) -> dict[str, float]:
        return {label: param.value for label, param in self._parameters.items()}

    def set_parameter_values(self, values: list[float]) -> None:
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but Schedule has {self.nparameters} parameters.")
        for i, parameter in enumerate(self._parameters.values()):
            parameter.set_value(values[i])

    def set_parameters(self, parameters: dict[str, int | float]) -> None:
        for label, param in parameters.items():
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

    def set_max_time(self, max_time: PARAMETERIZED_NUMBER) -> None:  # FIX!
        self._extract_parameters(max_time)
        self._max_time = max_time

    def _add_hamiltonian_from_dict(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: TimeDict,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        if label in self._hamiltonians:
            raise ValueError(f"Can't add Hamiltonian because label {label} is already associated with a Hamiltonian.")
        self._hamiltonians[label] = hamiltonian
        self._coefficients[label] = Interpolator(coefficients, interpolation, **kwargs)

    def _add_hamiltonian_from_interpolator(
        self, label: str, hamiltonian: Hamiltonian, coefficients: Interpolator
    ) -> None:
        if label in self._hamiltonians:
            raise ValueError(f"Can't add Hamiltonian because label {label} is already associated with a Hamiltonian.")
        self._hamiltonians[label] = hamiltonian
        self._coefficients[label] = coefficients

    @overload
    def add_hamiltonian(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: TimeDict,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def add_hamiltonian(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: Interpolator,
        **kwargs: Any,
    ) -> None: ...

    def add_hamiltonian(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: Interpolator | TimeDict,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        if isinstance(coefficients, Interpolator):
            self._add_hamiltonian_from_interpolator(label, hamiltonian, coefficients)
        elif isinstance(coefficients, dict):
            self._add_hamiltonian_from_dict(label, hamiltonian, coefficients, interpolation, **kwargs)
        else:
            raise ValueError("Unsupported type of coefficient.")

    def _update_hamiltonian_from_dict(
        self,
        label: str,
        new_coefficients: TimeDict | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        if new_coefficients is not None:
            self._coefficients[label] = Interpolator(new_coefficients, interpolation, **kwargs)

    def _update_hamiltonian_from_lambda(self, label: str, new_coefficients: Interpolator | None = None) -> None:
        if new_coefficients is not None:
            self._coefficients[label] = new_coefficients

    @overload
    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: TimeDict | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: Interpolator | None = None,
        **kwargs: Any,
    ) -> None: ...

    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: Interpolator | TimeDict | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        if new_hamiltonian is not None:
            self._hamiltonians[label] = new_hamiltonian
        if new_coefficients is not None:
            if callable(new_coefficients):
                self._update_hamiltonian_from_lambda(label, new_coefficients)
            elif isinstance(new_coefficients, dict):
                self._update_hamiltonian_from_dict(label, new_coefficients, interpolation, **kwargs)
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
        final_ham = Hamiltonian()
        for label, ham in self._hamiltonians.items():
            final_ham += ham * self._coefficients[label][time_step]
        return final_ham

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

    def draw(self, style: ScheduleStyle | None = None, filepath: str | None = None) -> None:
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
        renderer = MatplotlibScheduleRenderer(self, style=style, time_precision=self.dt)
        renderer.plot()
        if filepath:
            renderer.save(filepath)
        else:
            renderer.show()


class Interpolator(Parameterizable):
    """It's a dictionary that can interpolate between defined indecies."""

    def __init__(
        self,
        time_dict: TimeDict,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        self._interpolation = interpolation
        self._time_dict: dict[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER] = {}
        self._current_time = Parameter("t", 0)
        self._parameters: dict[str, Parameter] = {}
        self._total_time: float | None = None
        self.iter_time_step = 0
        self._cached = False
        self._cached_time: dict[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER] = {}
        self._tlist: list[PARAMETERIZED_NUMBER] | None = None
        self._fixed_tlist: list[float] | None = None

        for time, coefficient in time_dict.items():
            if isinstance(time, tuple):
                if len(time) != 2:  # noqa: PLR2004
                    raise ValueError(
                        f"time intervals need to be defined by two points, but this interval was provided: {time}"
                    )
                self.add_time_point(time[0], coefficient, **kwargs)
                self.add_time_point(time[1], coefficient, **kwargs)
            else:
                self.add_time_point(time, coefficient, **kwargs)

    @property
    def tlist(self) -> list[PARAMETERIZED_NUMBER]:
        if self._tlist:
            return self._tlist
        tlist = list(self._time_dict.keys())
        self._tlist = sorted(tlist, key=self._get_value)
        return self._tlist

    @property
    def fixed_tlist(self) -> list[float]:
        if self._fixed_tlist:
            return self._fixed_tlist
        self._fixed_tlist = [self._get_value(k) for k in self.tlist]
        return self._fixed_tlist

    @property
    def total_time(self) -> float:
        if not self._total_time:
            self._total_time = max(self.fixed_tlist)
        return self._total_time

    @property
    def nparameters(self) -> int:
        """Number of symbolic parameters introduced by the Hamiltonians or coefficients."""
        return len(self._parameters)

    def items(self) -> list[tuple[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER]]:
        return list(self._time_dict.items())

    def fixed_items(self) -> list[tuple[float, float]]:
        return [(t, self._get_value(self[t], t)) for t in self.fixed_tlist]

    @property
    def coefficients(self) -> list[PARAMETERIZED_NUMBER]:
        return [self._time_dict[t] for t in self.tlist]

    @property
    def coefficients_dict(self) -> dict[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER]:
        return copy(self._time_dict)

    @property
    def fixed_coefficeints(self) -> list[float]:
        return [self._get_value(self[t]) for t in self.fixed_tlist]

    @property
    def parameters(self) -> dict[str, Parameter]:
        return self._parameters

    def _delete_cache(self) -> None:
        self._cached = False
        self._total_time = None
        self._cached_time = {}
        self._tlist = None
        self._fixed_tlist = None

    def _get_value(self, value: PARAMETERIZED_NUMBER | complex, t: float | None = None) -> float:
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
            for v in value.variables():
                if v.label == _TIME_PARAMETER_NAME and t is None:
                    raise ValueError("Can't evaluate Parameter because time is not provided.")
            aux = value.evaluate({self._current_time: t}) if t is not None else value.evaluate({})
            if isinstance(aux, complex):
                return aux.real
            return aux
        raise ValueError(f"Invalid value of type {type(value)} is being evaluated.")

    def _extract_parameters(self, element: PARAMETERIZED_NUMBER) -> None:
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

    def add_time_point(
        self,
        time: PARAMETERIZED_NUMBER,
        coefficient: PARAMETERIZED_NUMBER | Callable[..., PARAMETERIZED_NUMBER],
        **kwargs: Any,
    ) -> None:
        self._extract_parameters(time)
        coeff = copy(coefficient)
        if callable(coeff):
            self._current_time.set_value(self._get_value(time))
            coeff, _params = _process_callable(coeff, self._current_time, **kwargs)
            if len(_params) > 0:
                self._parameters.update(_params)
        elif isinstance(coeff, (int, float, Parameter, Term)):
            self._extract_parameters(coeff)
            coeff = copy(coeff)
        else:
            raise ValueError
        self._time_dict[time] = coeff
        self._delete_cache()

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
        self._fixed_tlist = None
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but Schedule has {self.nparameters} parameters.")
        param_names = self.get_parameter_names()
        value_dict = {param_names[i]: values[i] for i in range(len(values))}
        self.set_parameters(value_dict)

    def set_parameters(self, parameters: dict[str, int | float]) -> None:
        """
        Update a subset of parameters by label.

        Args:
            parameters (dict[str, float]): Mapping from parameter labels to new values.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
        self._fixed_tlist = None
        for label, param in parameters.items():
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
        self._fixed_tlist = None
        for label, bound in ranges.items():
            if label not in self._parameters:
                raise ValueError(
                    f"The provided parameter label {label} is not defined in the list of parameters in this object."
                )
            self._parameters[label].set_bounds(bound[0], bound[1])

    def get_coefficient(self, time_step: float) -> float:
        time_step = time_step.item() if isinstance(time_step, np.generic) else self._get_value(time_step)

        val = self.get_coefficient_expression(time_step=time_step)
        return self._get_value(val, time_step)

    def get_coefficient_expression(self, time_step: float) -> Number | Term | Parameter:
        time_step = time_step.item() if isinstance(time_step, np.generic) else self._get_value(time_step)

        if time_step in self.fixed_tlist:
            indx = self.fixed_tlist.index(time_step)
            return self._time_dict[self.tlist[indx]]
        if time_step in self._cached_time:
            return self._cached_time[time_step]

        if self._interpolation is Interpolation.STEP:
            return self._get_coefficient_expression_step(time_step)
        if self._interpolation is Interpolation.LINEAR:
            return self._get_coefficient_expression_linear(time_step)

        raise ValueError(f"interpolation {self._interpolation.value} is not supported!")

    def _get_coefficient_expression_step(self, time_step: float) -> Number | Term | Parameter:

        prev_indx = bisect_right(self.tlist, time_step, key=self._get_value)
        if prev_indx >= len(self.tlist):
            prev_indx = -1
        prev_time_step = self.tlist[prev_indx]
        self._cached_time[time_step] = self._time_dict[prev_time_step]
        return self._cached_time[time_step]

    def _get_coefficient_expression_linear(self, time_step: float) -> Number | Term | Parameter:
        insert_pos = bisect_right(self.tlist, time_step, key=self._get_value)

        prev_idx = self.tlist[insert_pos - 1] if insert_pos else None
        next_idx = self.tlist[insert_pos] if insert_pos < len(self.tlist) else None
        prev_expr = self._time_dict[prev_idx] if prev_idx is not None else None
        next_expr = self._time_dict[next_idx] if next_idx is not None else None

        def _linear_value(
            t0: PARAMETERIZED_NUMBER, v0: PARAMETERIZED_NUMBER, t1: PARAMETERIZED_NUMBER, v1: PARAMETERIZED_NUMBER
        ) -> PARAMETERIZED_NUMBER:
            t0_val = self._get_value(t0)
            t1_val = self._get_value(t1)
            if t0_val == t1_val:
                raise ValueError(
                    f"Ambigous evaluation: The same time step {t0_val} has two different coefficient assignation ({v0} and {v1})."
                )
            alpha: float = (time_step - t0_val) / (t1_val - t0_val)
            next_is_term = isinstance(v1, (Term, Parameter))
            prev_is_term = isinstance(v0, (Term, Parameter))
            if next_is_term and prev_is_term and v1 != v0:
                v1 = self._get_value(v1, t1_val)
                v0 = self._get_value(v0, t0_val)
            elif next_is_term and not prev_is_term:
                v1 = self._get_value(v1, t1_val)
            elif prev_is_term and not next_is_term:
                v0 = self._get_value(v0, t0_val)

            return v1 * alpha + v0 * (1 - alpha)

        if prev_expr is None and next_expr is not None:
            if len(self.tlist) == 1:
                return next_expr
            first_idx = self.tlist[0]
            second_idx = self.tlist[1]
            return _linear_value(first_idx, self._time_dict[first_idx], second_idx, self._time_dict[second_idx])

        if next_expr is None and prev_expr is not None:
            if len(self.tlist) == 1:
                return prev_expr
            last_idx = self.tlist[-1]
            penultimate_idx = self.tlist[-2]
            return _linear_value(penultimate_idx, self._time_dict[penultimate_idx], last_idx, self._time_dict[last_idx])
        if prev_expr is None and next_expr is None:
            return 0

        if next_idx is None or prev_idx is None or prev_expr is None or next_expr is None:
            raise ValueError("Something unexpected happened while retrieving the coefficient.")
        return _linear_value(prev_idx, prev_expr, next_idx, next_expr)

    def __getitem__(self, time_step: float) -> float:
        return self.get_coefficient(time_step)

    def __len__(self) -> int:
        return len(self.tlist)

    def __iter__(self) -> "Interpolator":
        self.iter_time_step = 0
        return self

    def __next__(self) -> float:
        if self.iter_time_step < self.__len__():
            result = self[self.fixed_tlist[self.iter_time_step]]
            self.iter_time_step += 1
            return result
        raise StopIteration
        raise StopIteration
