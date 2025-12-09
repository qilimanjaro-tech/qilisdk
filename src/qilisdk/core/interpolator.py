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
from typing import Any, Mapping

import numpy as np

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import LEQ, BaseVariable, Number, Parameter, Term
from qilisdk.yaml import yaml

_TIME_PARAMETER_NAME = "t"
PARAMETERIZED_NUMBER = float | Parameter | Term

# type aliases just to keep this short
TimeDict = dict[PARAMETERIZED_NUMBER | tuple[float, float], PARAMETERIZED_NUMBER | Callable[..., PARAMETERIZED_NUMBER]]


class Interpolation(str, Enum):
    STEP = "Step function interpolation between schedule points"
    LINEAR = "linear interpolation between schedule points"


def _process_callable(
    function: Callable[[], PARAMETERIZED_NUMBER], current_time: Parameter, **kwargs: Any
) -> tuple[PARAMETERIZED_NUMBER, dict[str, Parameter]]:
    # Define variables
    parameters: dict[str, Parameter] = {}

    # get callable parameters
    c_params = inspect.signature(function).parameters
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
    term = function(**kwargs)
    if isinstance(term, Term) and not all(
        (isinstance(v, Parameter) or v.label == _TIME_PARAMETER_NAME) for v in term.variables()
    ):
        raise ValueError("function contains variables that are not time. Only Parameters are allowed.")
    if isinstance(term, BaseVariable) and not (isinstance(term, Parameter) or term.label == _TIME_PARAMETER_NAME):
        raise ValueError("function contains variables that are not time. Only Parameters are allowed.")
    return term, parameters


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
        """Initialize an interpolator over discrete points or intervals.

        Args:
            time_dict (TimeDict): Mapping from time points or intervals to coefficients or callables.
            interpolation (Interpolation): Interpolation rule between provided points (``LINEAR`` or ``STEP``).
            nsamples (int): Number of samples used to expand interval definitions.
            **kwargs: Extra arguments forwarded to callable coefficient processing.

        Raises:
            ValueError:if the time intervals contain a number of points different than 2.
        """
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
                    time_point = (1 - float(t)) * time[0] + float(t) * time[1]
                    if isinstance(time_point, (Parameter, Term)):
                        self._extract_parameters(time_point)
                    self.add_time_point(time_point, coefficient, **kwargs)

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
                if i > 0:
                    term = LEQ(time_insertion_list[i - 1], t)
                    if term not in self._parameter_constraints:
                        self._parameter_constraints.append(term)
                if i < l - 1:
                    term = LEQ(t, time_insertion_list[i + 1])
                    if term not in self._parameter_constraints:
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
        """
        Rescale all time points to a new maximum duration while keeping relative spacing.

        Raises:
            ValueError: If the max time is set to zero.
        """
        if self._get_value(max_time) == 0:
            raise ValueError("Setting the max time to zero.")
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
        if isinstance(element, Parameter) and element.label != _TIME_PARAMETER_NAME:
            self._parameters[element.label] = element
        elif isinstance(element, Term):
            if not element.is_parameterized_term():
                raise ValueError(
                    f"Tlist can only contain parameters and no variables, but the term {element} contains objects other than parameters."
                )
            for p in element.variables():
                if isinstance(p, Parameter) and p.label != _TIME_PARAMETER_NAME:
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
            self._extract_parameters(coeff)
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
