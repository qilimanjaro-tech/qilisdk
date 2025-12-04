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

from copy import copy
from itertools import chain
from typing import Any, Mapping, overload

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core.interpolator import Interpolation, Interpolator, TimeDict
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import BaseVariable, ComparisonTerm, Domain, Parameter, Term
from qilisdk.utils.visualization import ScheduleStyle
from qilisdk.yaml import yaml

_TIME_PARAMETER_NAME = "t"
PARAMETERIZED_NUMBER = float | Parameter | Term

# type aliases just to keep this short
CoeffDict = dict[str, TimeDict]
InterpDict = dict[str, "Interpolator"]


@yaml.register_class
class Schedule(Parameterizable):
    """
    Builds a set of time-dependent coefficients applied to a collection of Hamiltonians.

    A Schedule defines the evolution of a system by associating time steps with a set
    of Hamiltonian coefficients. Coefficients can be provided directly, defined as
    functions of time, or specified over time intervals and interpolated (step or linear).

    Example:
        .. code-block:: python

            import numpy as np
            from qilisdk.analog import Schedule, X, Z

            T, dt = 10.0, 1.0

            h1 = X(0) + X(1) + X(2)
            h2 = -Z(0) - Z(1) - 2 * Z(2) + 3 * Z(0) * Z(1)

            schedule = Schedule(
                dt=dt,
                hamiltonians={"driver": h1, "problem": h2},
                coefficients={
                    "driver": {(0, T): lambda t: 1 - t / T},
                    "problem": {(0, T): lambda t: t / T},
                },
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
        """Create a Schedule that assigns time-dependent coefficients to Hamiltonians.

        Args:
            hamiltonians (dict[str, Hamiltonian] | None): Mapping of labels to Hamiltonian objects. If omitted, an empty schedule is created.
            coefficients (InterpDict | CoeffDict | None): Per-Hamiltonian time definitions. Keys are time points or intervals; values are coefficients or callables. If an :class:`Interpolator` is supplied, it is used directly.
            dt (float): Time resolution used for sampling callable/interval definitions and plotting. Must be positive.
            total_time (float | Parameter | Term | None): Optional maximum time that rescales all defined time points proportionally.
            interpolation (Interpolation): How to interpolate between provided time points (``LINEAR`` or ``STEP``).
            **kwargs: Passed to :class:`Interpolator` construction when coefficients are provided as dictionaries.

        Raises:
            ValueError: if the coefficients reference an undefined hamiltonian.
        """
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
            self.scale_max_time(total_time)

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
        """Update parameter values across all Hamiltonian coefficient interpolators.

        Args:
            parameters (dict[str, int | float]): Mapping from parameter labels to numeric values.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
        for label in parameters:
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this Schedule.")
        for h in self._hamiltonians:
            self._coefficients[h].set_parameters(
                {p: parameters[p] for p in self._coefficients[h].get_parameter_names() if p in parameters}
            )
        super().set_parameters(parameters)

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Propagate bound updates to all interpolators and cached parameters.

        Args:
            ranges (dict[str, tuple[float, float]]): Mapping of parameter label to ``(lower, upper)`` bounds.

        Raises:
            ValueError: If an unknown parameter label is provided.
        """
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
        """Return the set of parameter constraints arising from all interpolators."""
        const_lists = [coeff.get_constraints() for coeff in self._coefficients.values()]
        combined_list = chain.from_iterable(const_lists)
        return list(set(combined_list))

    def scale_max_time(self, max_time: PARAMETERIZED_NUMBER) -> None:  # FIX!
        """
        Rescale the schedule to a new maximum time while keeping relative points fixed.

        Raises:
            ValueError: If the max time provided is zero.
        """
        if self._get_value(max_time) == 0:
            raise ValueError("Setting the total time to zero.")
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
            time_step (float): Physical time (same units as the schedule definition) at which to sample.

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
