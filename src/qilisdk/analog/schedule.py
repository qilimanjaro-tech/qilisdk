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
from itertools import chain
from typing import Any, Mapping, overload

import numpy as np

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import LEQ, BaseVariable, ComparisonTerm, Domain, Number, Parameter, Term
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
    term = copy(c(**kwargs))
    if isinstance(term, Term) and not all(
        (isinstance(v, Parameter) or v.label == _TIME_PARAMETER_NAME) for v in term.variables()
    ):
        raise ValueError("function contains variables that are not time. Only Parameters are allowed.")
    if isinstance(term, BaseVariable) and not (isinstance(term, Parameter) or term.label == _TIME_PARAMETER_NAME):
        raise ValueError("function contains variables that are not time. Only Parameters are allowed.")
    return term, parameters


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
        total_time: PARAMETERIZED_NUMBER | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        # THIS is the only runtime implementation
        super(Schedule, self).__init__()
        self._hamiltonians = hamiltonians if hamiltonians is not None else {}
        self._coefficients: dict[str, Interpolator] = {}
        self._interpolation = None
        self._parameters: dict[str, Parameter] = {}
        self._current_time: Parameter = Parameter(_TIME_PARAMETER_NAME, 0, Domain.REAL)
        self.iter_time_step = 0
        self._max_time: PARAMETERIZED_NUMBER | None = None
        if dt <= 0:
            raise ValueError("dt must be greater than zero.")
        self._dt = dt

        coefficients = coefficients or {}

        if coefficients.keys() > self._hamiltonians.keys():
            missing = coefficients.keys() - self._hamiltonians.keys()
            raise ValueError(f"Missing keys in hamiltonians: {missing}")

        for ham, hamiltonian in self._hamiltonians.items():
            # Gather Hamiltonian parameters and nqubits
            for param in hamiltonian.parameters.values():
                self._parameters[param.label] = param

            # Build hamiltonian schedule
            if ham not in coefficients:
                self._coefficients[ham] = Interpolator({0: 1}, interpolation=interpolation, nsamples=int(1 / dt))
                continue
            coeff = copy(coefficients[ham])
            if isinstance(coeff, Interpolator):
                self._coefficients[ham] = coeff
            elif isinstance(coeff, dict):
                self._coefficients[ham] = Interpolator(coeff, interpolation, nsamples=int(1 / dt), **kwargs)

            for p_name, p_value in self._coefficients[ham].parameters.items():
                self._parameters[p_name] = p_value

        if total_time is not None:
            self.set_max_time(total_time)

    @property
    def hamiltonians(self) -> dict[str, Hamiltonian]:
        """
        Return the Hamiltonians managed by the schedule.

        Returns:
            dict[str, Hamiltonian]: Mapping of labels to Hamiltonian instances.
        """
        return self._hamiltonians

    @property
    def coefficients_dict(self) -> dict[str, dict[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER]]:
        return {ham: self._coefficients[ham].coefficients_dict for ham in self._hamiltonians}

    @property
    def coefficients(self) -> dict[str, Interpolator]:
        return {ham: self._coefficients[ham] for ham in self._hamiltonians}

    @property
    def T(self) -> float:
        """Total annealing time of the schedule."""
        return max(self.tlist)

    @property
    def tlist(self) -> list[float]:
        _tlist: set[float] = set()
        if len(self._hamiltonians) == 0:
            tlist = [0.0]
        else:
            for ham in self._hamiltonians:
                _tlist.update(self._coefficients[ham].fixed_tlist)
            tlist = list(_tlist)
        if self._max_time is not None:
            max_t = max(tlist) or 1
            max_t = max_t if max_t != 0 else 1
            T = self._get_value(self._max_time)
            tlist = [t * T / max_t for t in tlist]
            if T not in tlist:
                tlist.append(T)
        return sorted(tlist)

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
        if len(self._hamiltonians) == 0:
            return 0
        return max(self._hamiltonians.values(), key=lambda v: v.nqubits).nqubits

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
            ctx: Mapping[BaseVariable, list[int] | int | float] = {self._current_time: t} if t is not None else {}
            aux = value.evaluate(ctx)

            return aux.real if isinstance(aux, complex) else float(aux)
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

    def set_parameters(self, parameters: dict[str, int | float]) -> None:
        for label in parameters:
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this Schedule.")
        for h in self._hamiltonians:
            self._coefficients[h].set_parameters(
                {p: parameters[p] for p in self._coefficients[h].get_parameter_names() if p in parameters}
            )
        super().set_parameters(parameters)

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        for label in ranges:
            if label not in self._parameters:
                raise ValueError(
                    f"The provided parameter label {label} is not defined in the list of parameters in this object."
                )
        for h in self._hamiltonians:
            self._coefficients[h].set_parameter_bounds(
                {p: ranges[p] for p in self._coefficients[h].get_parameter_names() if p in ranges}
            )
        super().set_parameter_bounds(ranges)

    def get_constraints(self) -> list[ComparisonTerm]:
        const_lists = [coeff.get_constraints() for coeff in self._coefficients.values()]
        combined_list = chain.from_iterable(const_lists)
        return list(set(combined_list))

    def set_max_time(self, max_time: PARAMETERIZED_NUMBER) -> None:  # FIX!
        self._extract_parameters(max_time)
        self._max_time = max_time
        for ham in self._hamiltonians:
            self._coefficients[ham].set_max_time(max_time)

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
        self._coefficients[label] = Interpolator(coefficients, interpolation, nsamples=int(1 / self.dt), **kwargs)

        for p_name, p_value in self._coefficients[label].parameters.items():
            self._parameters[p_name] = p_value

    def _add_hamiltonian_from_interpolator(
        self, label: str, hamiltonian: Hamiltonian, coefficients: Interpolator
    ) -> None:
        if label in self._hamiltonians:
            raise ValueError(f"Can't add Hamiltonian because label {label} is already associated with a Hamiltonian.")
        self._hamiltonians[label] = hamiltonian
        self._coefficients[label] = coefficients

        for p_name, p_value in self._coefficients[label].parameters.items():
            self._parameters[p_name] = p_value

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

        if not isinstance(hamiltonian, Hamiltonian):
            raise ValueError(f"Expecting a Hamiltonian object but received {type(hamiltonian)} instead.")

        if isinstance(coefficients, Interpolator):
            self._add_hamiltonian_from_interpolator(label, hamiltonian, coefficients)
        elif isinstance(coefficients, dict):
            self._add_hamiltonian_from_dict(label, hamiltonian, coefficients, interpolation, **kwargs)
        else:
            raise ValueError("Unsupported type of coefficient.")
        if self._max_time is not None:
            self._coefficients[label].set_max_time(self._max_time)

    def _update_hamiltonian_from_dict(
        self,
        label: str,
        new_coefficients: TimeDict | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
        **kwargs: Any,
    ) -> None:
        if new_coefficients is not None:
            self._coefficients[label] = Interpolator(
                new_coefficients, interpolation, nsamples=int(1 / self.dt), **kwargs
            )  # TODO (ameer): allow for partial updates of the coefficients

            for p_name, p_value in self._coefficients[label].parameters.items():
                self._parameters[p_name] = p_value

    def _update_hamiltonian_from_interpolator(self, label: str, new_coefficients: Interpolator | None = None) -> None:
        if new_coefficients is not None:
            self._coefficients[label] = new_coefficients

            for p_name, p_value in self._coefficients[label].parameters.items():
                self._parameters[p_name] = p_value

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
        if label not in self._hamiltonians:
            raise ValueError(f"Can't update unknown hamiltonian {label}. Did you mean `add_hamiltonian`?")
        if new_hamiltonian is not None:
            if not isinstance(new_hamiltonian, Hamiltonian):
                raise ValueError(f"Expecting a Hamiltonian object but received {type(new_hamiltonian)} instead.")
            self._hamiltonians[label] = new_hamiltonian
        if new_coefficients is not None:
            if isinstance(new_coefficients, Interpolator):
                self._update_hamiltonian_from_interpolator(label, new_coefficients)
            elif isinstance(new_coefficients, dict):
                self._update_hamiltonian_from_dict(label, new_coefficients, interpolation, **kwargs)
            else:
                raise ValueError("Unsupported type of coefficient.")

        if self._max_time is not None:
            self._coefficients[label].set_max_time(self._max_time)

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
        renderer = MatplotlibScheduleRenderer(self, style=style)
        renderer.plot()
        if filepath:
            renderer.save(filepath)
        else:
            renderer.show()


@yaml.register_class
class Interpolator(Parameterizable):
    """It's a dictionary that can interpolate between defined indecies."""

    def __init__(
        self,
        time_dict: TimeDict,
        interpolation: Interpolation = Interpolation.LINEAR,
        nsamples: int = 100,
        **kwargs: Any,
    ) -> None:
        super(Interpolator, self).__init__()
        self._interpolation = interpolation
        self._time_dict: dict[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER] = {}
        self._current_time = Parameter("t", 0)
        self._total_time: float | None = None
        self.iter_time_step = 0
        self._cached = False
        self._cached_time: dict[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER | Number] = {}
        self._tlist: list[PARAMETERIZED_NUMBER] | None = None
        self._fixed_tlist: list[float] | None = None
        self._max_time: PARAMETERIZED_NUMBER | None = None
        self._time_scale_cache: float | None = None

        for time, coefficient in time_dict.items():
            if isinstance(time, tuple):
                if len(time) != 2:  # noqa: PLR2004
                    raise ValueError(
                        f"time intervals need to be defined by two points, but this interval was provided: {time}"
                    )
                for t in np.linspace(0, 1, (max(2, nsamples))):
                    self.add_time_point((1 - t) * time[0] + t * time[1], coefficient, **kwargs)

            else:
                self.add_time_point(time, coefficient, **kwargs)
        self._tlist = self._generate_tlist()

        time_insertion_list = sorted(
            [k for item in time_dict for k in (item if isinstance(item, tuple) else (item,))],
            key=self._get_value,
        )
        l = len(time_insertion_list)
        for i in range(l):
            t = time_insertion_list[i]
            if isinstance(t, (Parameter, Term)):
                term = None
                if i > 0:
                    term = LEQ(time_insertion_list[i - 1], t)
                if i < l - 1:
                    term = LEQ(t, time_insertion_list[i + 1])

                if term is not None and term not in self._parameter_constraints:
                    self._parameter_constraints.append(term)

    def _generate_tlist(self) -> list[PARAMETERIZED_NUMBER]:
        return sorted((self._time_dict.keys()), key=self._get_value)

    @property
    def tlist(self) -> list[PARAMETERIZED_NUMBER]:
        if self._tlist is None:
            self._tlist = self._generate_tlist()
        if self._max_time is not None:
            if self._time_scale_cache is None:
                max_t = self._get_value(max(self._tlist, key=self._get_value)) or 1
                max_t = max_t if max_t != 0 else 1
                self._time_scale_cache = self._get_value(self._max_time) / max_t
            return [t * self._time_scale_cache for t in self._tlist]
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

    def items(self) -> list[tuple[PARAMETERIZED_NUMBER, PARAMETERIZED_NUMBER]]:
        if self._max_time is not None and self._tlist is not None:
            if self._time_scale_cache is None:
                self._time_scale_cache = self._get_value(self._max_time) / self._get_value(
                    max(self._tlist, key=self._get_value)
                )
            return [(k * self._time_scale_cache, v) for k, v in self._time_dict.items()]
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

    def set_max_time(self, max_time: PARAMETERIZED_NUMBER) -> None:
        self._delete_cache()
        self._max_time = max_time

    def _delete_cache(self) -> None:
        self._cached = False
        self._total_time = None
        self._cached_time = {}
        self._tlist = None
        self._fixed_tlist = None
        self._time_scale_cache = None

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
            ctx: Mapping[BaseVariable, list[int] | int | float] = {self._current_time: t} if t is not None else {}
            aux = value.evaluate(ctx)

            return aux.real if isinstance(aux, complex) else float(aux)
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
        coeff = coefficient
        if callable(coeff):
            self._current_time.set_value(self._get_value(time))
            coeff, _params = _process_callable(coeff, self._current_time, **kwargs)
            if len(_params) > 0:
                self._parameters.update(_params)
        elif isinstance(coeff, (int, float, Parameter, Term)):
            self._extract_parameters(coeff)
        else:
            raise ValueError
        if self._max_time is not None and self._tlist is not None:
            if self._time_scale_cache is None:
                self._time_scale_cache = self._get_value(self._max_time) / self._get_value(
                    max(self._tlist, key=self._get_value)
                )
            time /= self._time_scale_cache
        self._time_dict[time] = coeff
        self._delete_cache()

    def set_parameter_values(self, values: list[float]) -> None:
        self._delete_cache()
        super().set_parameter_values(values)

    def set_parameters(self, parameters: dict[str, int | float]) -> None:
        self._delete_cache()
        super().set_parameters(parameters)

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        self._delete_cache()
        super().set_parameter_bounds(ranges)

    def get_coefficient(self, time_step: float) -> float:
        time_step = time_step.item() if isinstance(time_step, np.generic) else self._get_value(time_step)
        val = self.get_coefficient_expression(time_step=time_step)

        if self._max_time is not None and self._tlist is not None:
            if self._time_scale_cache is None:
                self._time_scale_cache = self._get_value(self._max_time) / self._get_value(
                    max(self._tlist, key=self._get_value)
                )
            time_step /= self._time_scale_cache

        return self._get_value(val, time_step)

    def get_coefficient_expression(self, time_step: float) -> Number | Term | Parameter:
        time_step = time_step.item() if isinstance(time_step, np.generic) else self._get_value(time_step)

        # generate the tlist
        self._tlist = self._generate_tlist()

        if time_step in self.fixed_tlist:
            indx = self.fixed_tlist.index(time_step)
            return self._time_dict[self._tlist[indx]]
        if time_step in self._cached_time:
            return self._cached_time[time_step]

        if self._max_time is not None and self._tlist is not None:
            if self._time_scale_cache is None:
                self._time_scale_cache = self._get_value(self._max_time) / self._get_value(
                    max(self._tlist, key=self._get_value)
                )
            time_step /= self._time_scale_cache
        factor = self._time_scale_cache or 1.0

        result = None
        if self._interpolation is Interpolation.STEP:
            result = self._get_coefficient_expression_step(time_step)
        if self._interpolation is Interpolation.LINEAR:
            result = self._get_coefficient_expression_linear(time_step)

        if result is None:
            raise ValueError(f"interpolation {self._interpolation.value} is not supported!")
        self._cached_time[time_step * factor] = result
        return result

    def _get_coefficient_expression_step(self, time_step: float) -> Number | Term | Parameter:

        self._tlist = self._generate_tlist()
        prev_indx = bisect_right(self._tlist, time_step, key=self._get_value) - 1
        if prev_indx >= len(self._tlist):
            prev_indx = -1
        prev_time_step = self._tlist[prev_indx]
        return self._time_dict[prev_time_step]

    def _get_coefficient_expression_linear(self, time_step: float) -> Number | Term | Parameter:
        self._tlist = self._generate_tlist()
        insert_pos = bisect_right(self._tlist, time_step, key=self._get_value)

        prev_idx = self._tlist[insert_pos - 1] if insert_pos else None
        next_idx = self._tlist[insert_pos] if insert_pos < len(self._tlist) else None
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
            if len(self._tlist) == 1:
                return next_expr
            first_idx = self._tlist[0]
            second_idx = self._tlist[1]
            return _linear_value(first_idx, self._time_dict[first_idx], second_idx, self._time_dict[second_idx])

        if next_expr is None and prev_expr is not None:
            if len(self._tlist) == 1:
                return prev_expr
            last_idx = self._tlist[-1]
            penultimate_idx = self._tlist[-2]
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
