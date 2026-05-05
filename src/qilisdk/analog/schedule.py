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

import math
from copy import copy
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, overload

from loguru import logger
from numpy import linspace

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core.interpolator import PARAMETERIZED_NUMBER, Interpolation, Interpolator, TimeDict
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import BaseVariable, Cos, Domain, Parameter, Term
from qilisdk.settings import get_settings
from qilisdk.utils.visualization import ScheduleStyle
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from qilisdk.core.qtensor import QTensor

_TIME_PARAMETER_NAME = "t"

# type aliases just to keep this short
CoeffDict = dict[str, TimeDict]
InterpDict = dict[str, "Interpolator"]

_DEFAULT_DT = 0.1


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
        dt: float = _DEFAULT_DT,
        total_time: PARAMETERIZED_NUMBER | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
    ) -> None:
        """Create a Schedule that assigns time-dependent coefficients to Hamiltonians.

        Args:
            hamiltonians (dict[str, Hamiltonian] | None): Mapping of labels to Hamiltonian objects. If omitted, an empty schedule is created.
            coefficients (InterpDict | CoeffDict | None): Per-Hamiltonian time definitions. Keys are time points or intervals; values are coefficients or callables. If an :class:`Interpolator` is supplied, it is used directly.
            dt (float): Time resolution used for sampling callable/interval definitions and plotting. Must be positive.
            total_time (float | Parameter | Term | None): Optional maximum time that rescales all defined time points proportionally.
            interpolation (Interpolation): How to interpolate between provided time points (``LINEAR`` or ``STEP``).

        Raises:
            ValueError: if the coefficients reference an undefined hamiltonian.
        """
        # THIS is the only runtime implementation
        super().__init__()

        self._hamiltonians = hamiltonians if hamiltonians is not None else {}
        self._coefficients: dict[str, Interpolator] = {}
        self._interpolation = None
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
            # Build hamiltonian schedule
            if ham not in coefficients:
                self._coefficients[ham] = Interpolator({0: 1}, interpolation=interpolation, nsamples=int(1 / dt))
                continue
            coeff = copy(coefficients[ham])
            if isinstance(coeff, Interpolator):
                self._coefficients[ham] = coeff
            elif isinstance(coeff, dict):
                self._coefficients[ham] = Interpolator(coeff, interpolation, nsamples=int(1 / dt))

        if total_time is not None:
            self.scale_max_time(total_time)

    @classmethod
    def polynomial(
        cls,
        initial_hamiltonian: Hamiltonian,
        final_hamiltonian: Hamiltonian,
        total_time: float,
        dt: float = _DEFAULT_DT,
        degree: int = 3,
    ) -> Schedule:
        """Convenience constructor for a simple polynomial annealing schedule with two Hamiltonians.

        Args:
            initial_hamiltonian (Hamiltonian): The "driver" Hamiltonian that is dominant at the start of the anneal.
            final_hamiltonian (Hamiltonian): The "problem" Hamiltonian that is dominant at the end of the anneal.
            total_time (float): Total annealing time.
            dt (float): Time resolution used for sampling callable/interval definitions and plotting. Must be positive.
            degree (int): Degree of the polynomial interpolation. Must be at least 1. Defaults to 3.

        Returns:
            Schedule: A schedule instance with polynomially interpolated coefficients for the driver and problem Hamiltonians.

        Raises:
            ValueError: If the degree is less than 1.
        """
        if degree < 1:
            raise ValueError("Degree of polynomial must be at least 1.")
        return cls(
            hamiltonians={"driver": initial_hamiltonian, "problem": final_hamiltonian},
            coefficients={
                "driver": {(0.0, total_time): lambda t: 1 - (t / total_time) ** degree},
                "problem": {(0.0, total_time): lambda t: (t / total_time) ** degree},
            },
            dt=dt,
            interpolation=Interpolation.LINEAR,
        )

    @classmethod
    def linear(
        cls,
        initial_hamiltonian: Hamiltonian,
        final_hamiltonian: Hamiltonian,
        total_time: float,
        dt: float = _DEFAULT_DT,
    ) -> Schedule:
        """Convenience constructor for a simple linear annealing schedule with two Hamiltonians.

        Args:
            initial_hamiltonian (Hamiltonian): The "driver" Hamiltonian that is dominant at the start of the anneal.
            final_hamiltonian (Hamiltonian): The "problem" Hamiltonian that is dominant at the end of the anneal.
            total_time (float): Total annealing time.
            dt (float): Time resolution used for sampling callable/interval definitions and plotting. Must be positive.

        Returns:
            Schedule: A schedule instance with linearly interpolated coefficients for the driver and problem Hamiltonians.
        """
        return cls.polynomial(initial_hamiltonian, final_hamiltonian, total_time, dt, degree=1)

    @classmethod
    def quadratic(
        cls,
        initial_hamiltonian: Hamiltonian,
        final_hamiltonian: Hamiltonian,
        total_time: float,
        dt: float = _DEFAULT_DT,
    ) -> Schedule:
        """Convenience constructor for a simple quadratic annealing schedule with two Hamiltonians.

        Args:
            initial_hamiltonian (Hamiltonian): The "driver" Hamiltonian that is dominant at the start of the anneal.
            final_hamiltonian (Hamiltonian): The "problem" Hamiltonian that is dominant at the end of the anneal.
            total_time (float): Total annealing time.
            dt (float): Time resolution used for sampling callable/interval definitions and plotting. Must be positive.

        Returns:
            Schedule: A schedule instance with quadratically interpolated coefficients for the driver and problem Hamiltonians.
        """
        return cls.polynomial(initial_hamiltonian, final_hamiltonian, total_time, dt, degree=2)

    @classmethod
    def sinusoidal(
        cls,
        initial_hamiltonian: Hamiltonian,
        final_hamiltonian: Hamiltonian,
        total_time: float,
        dt: float = _DEFAULT_DT,
    ) -> Schedule:
        """Convenience constructor for a simple sinusoidal annealing schedule with two Hamiltonians.

        Args:
            initial_hamiltonian (Hamiltonian): The "driver" Hamiltonian that is dominant at the start of the anneal.
            final_hamiltonian (Hamiltonian): The "problem" Hamiltonian that is dominant at the end of the anneal.
            total_time (float): Total annealing time.
            dt (float): Time resolution used for sampling callable/interval definitions and plotting. Must be positive.

        Returns:
            Schedule: A schedule instance with sinusoidally interpolated coefficients for the driver and problem Hamiltonians.
        """
        return cls(
            hamiltonians={"driver": initial_hamiltonian, "problem": final_hamiltonian},
            coefficients={
                "driver": {(0.0, total_time): lambda t: Cos((t / total_time) * math.pi / 2)},
                "problem": {(0.0, total_time): lambda t: 1 - Cos((t / total_time) * math.pi / 2)},
            },
            dt=dt,
            interpolation=Interpolation.LINEAR,
        )

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
        if self._max_time is not None:
            return self._get_value(self._max_time)
        max_t = self._get_coefficients_max_time()
        return self._get_value(max_t)

    @property
    def tlist(self) -> list[float]:
        T = self.T
        return list(linspace(0, T, int(T // self.dt), dtype=float))

    def _get_coefficients_max_time(self) -> float:
        if len(self._hamiltonians) == 0:
            return 0
        _tlist: set[float] = set()
        for ham in self._hamiltonians:
            _tlist.update(self._coefficients[ham].fixed_tlist)
        return max(_tlist)

    @property
    def dt(self) -> float:
        return self._dt

    def set_dt(self, dt: float) -> None:
        """
        Set the time resolution ``dt`` used for sampling callable/interval definitions and plotting.

        Args:
            dt (float): New time resolution. Must be positive.

        Raises:
            ValueError: If ``dt`` is not a positive float.
        """
        if not isinstance(dt, float):
            raise ValueError(f"dt is only allowed to be a float but {type(dt)} was provided")
        self._dt = dt

    @property
    def nqubits(self) -> int:
        """Maximum number of qubits affected by Hamiltonians contained in the schedule."""
        if len(self._hamiltonians) == 0:
            return 0
        return max(self._hamiltonians.values(), key=lambda v: v.nqubits).nqubits

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        """Expose hamiltonians and interpolators to the shared parameter interface.

        Yields:
            Iterator[Parameterizable]: Child parameterizable objects composing the schedule.
        """
        yield from self._hamiltonians.values()
        yield from self._coefficients.values()

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
            self._add_parameter(element.label, element)
        elif isinstance(element, Term):
            if not element.is_parameterized_term():
                raise ValueError(
                    f"Tlist can only contain parameters and no variables, but the term {element} contains objects other than parameters."
                )
            for p in element.variables():
                if isinstance(p, Parameter):
                    self._add_parameter(p.label, p)

    def scale_max_time(self, max_time: PARAMETERIZED_NUMBER) -> None:  # FIX!
        """
        Rescale the schedule to a new maximum time while keeping relative points fixed.

        Raises:
            ValueError: If the max time provided is zero.
        """
        if abs(self._get_value(max_time)) < get_settings().atol:
            raise ValueError("Cannot set the total time to zero.")
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
    ) -> None:
        if label in self._hamiltonians:
            raise ValueError(f"Can't add Hamiltonian because label {label} is already associated with a Hamiltonian.")
        self._hamiltonians[label] = hamiltonian
        self._coefficients[label] = Interpolator(coefficients, interpolation, nsamples=int(1 / self.dt))

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
    ) -> None: ...

    @overload
    def add_hamiltonian(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: Interpolator,
    ) -> None: ...

    def add_hamiltonian(
        self,
        label: str,
        hamiltonian: Hamiltonian,
        coefficients: Interpolator | TimeDict,
        interpolation: Interpolation = Interpolation.LINEAR,
    ) -> None:
        if not isinstance(hamiltonian, Hamiltonian):
            raise ValueError(f"Expecting a Hamiltonian object but received {type(hamiltonian)} instead.")

        if isinstance(coefficients, Interpolator):
            self._add_hamiltonian_from_interpolator(label, hamiltonian, coefficients)
        elif isinstance(coefficients, dict):
            self._add_hamiltonian_from_dict(label, hamiltonian, coefficients, interpolation)
        else:
            raise ValueError("Unsupported type of coefficient.")
        if self._max_time is not None:
            self._coefficients[label].set_max_time(self._max_time)

    def _update_hamiltonian_from_dict(
        self,
        label: str,
        new_coefficients: TimeDict | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
    ) -> None:
        if new_coefficients is not None:
            self._coefficients[label] = Interpolator(
                new_coefficients, interpolation, nsamples=int(1 / self.dt)
            )  # TODO (ameer): allow for partial updates of the coefficients

    def _update_hamiltonian_from_interpolator(self, label: str, new_coefficients: Interpolator | None = None) -> None:
        if new_coefficients is not None:
            self._coefficients[label] = new_coefficients

    @overload
    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: TimeDict | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
    ) -> None: ...

    @overload
    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: Interpolator | None = None,
    ) -> None: ...

    def update_hamiltonian(
        self,
        label: str,
        new_hamiltonian: Hamiltonian | None = None,
        new_coefficients: Interpolator | TimeDict | None = None,
        interpolation: Interpolation = Interpolation.LINEAR,
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
                self._update_hamiltonian_from_dict(label, new_coefficients, interpolation)
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

    def draw_eigenvalues(
        self,
        levels: int = 50,
        style: ScheduleStyle | None = None,
        filepath: str | None = None,
        intermediate_states: list[QTensor] | None = None,
        show_overlaps: bool = False,
    ) -> None:
        """
        Render a plot of the lowest eigenvalues of the schedule's Hamiltonians over time.

        For each Hamiltonian in the schedule, as well as the total Hamiltonian, the specified number of lowest eigenvalues
        are computed at each time step and plotted.

        Args:
            levels (int): The number of lowest eigenvalues to compute and plot for each Hamiltonian.
            style (ScheduleStyle, optional): Customization options for the plot appearance.
            filepath (str | None, optional): If provided, saves the plot to the specified file path.
            intermediate_states (list[QTensor] | None, optional): If provided, these states are plotted alongside the eigenvalues to show their evolution over time.
            show_overlaps (bool): Whether to annotate the plot with the overlaps between the intermediate states and the eigenstates.

        Raises:
            ValueError: If the number of qubits exceeds the supported limit for eigenvalue plotting.
            ValueError: If show_overlaps is True but intermediate_states is not provided.

        """
        from qilisdk.utils.visualization.schedule_renderers import MatplotlibEigenvalueRenderer  # noqa: PLC0415

        # If we try to show overlaps but haven't given intermediate states, raise an error
        if show_overlaps and not intermediate_states:
            logger.warning("Overlaps can't be shown without intermediate states. Setting show_overlaps to False.")
            show_overlaps = False

        renderer = MatplotlibEigenvalueRenderer(
            self, levels=levels, style=style, intermediate_states=intermediate_states, show_overlaps=show_overlaps
        )
        renderer.plot()
        if filepath:
            renderer.save(filepath)
        else:
            renderer.show()

    def eig(self, levels: int = 50) -> tuple[list[list[float]], list[list[QTensor]]]:
        """
        Calculate the lowest eigenvalues and corresponding eigenstates of the schedule's Hamiltonians at each time step.

        Args:
            levels (int): The number of lowest eigenvalues and corresponding eigenstates to compute for each Hamiltonian at each time step.

        Returns:
            tuple[list[list[float]], list[list[QTensor]]]: A tuple containing two lists, the first is a list
            of lists of eigenvalues for each Hamiltonian at each time step, and the second is a
            list of lists of corresponding eigenstates as QTensors.

        Raises:
            ValueError: If the Hamiltonian at any time step is not a valid Hamiltonian object that can be converted to a QTensor for eigenvalue computation.
        """
        _MAX_QUBITS_FOR_EIGENVALUE_PLOTTING = 7
        if self.nqubits > _MAX_QUBITS_FOR_EIGENVALUE_PLOTTING:
            logger.warning(
                f"Calculating eigenvalues with more than {_MAX_QUBITS_FOR_EIGENVALUE_PLOTTING} qubits may be very slow and is not supported. This schedule has {self.nqubits} qubits."
            )
        full_eigenvalues: list[list[float]] = []
        full_eigenstates: list[list[QTensor]] = []
        for i, t in enumerate(self.tlist):
            full_hamiltonian = sum(self.coefficients[h][float(t)] * self.hamiltonians[h] for h in self.hamiltonians)
            if not isinstance(full_hamiltonian, Hamiltonian):
                raise ValueError(f"Expected full_hamiltonian to be a Hamiltonian, got {type(full_hamiltonian)}")
            as_qtensor = full_hamiltonian.to_qtensor()
            vals, vecs = as_qtensor.eig()

            full_eigenvalues.append([float(ev.real) for ev in vals[:levels]])
            full_eigenstates.append(list(vecs[:levels]))

        return full_eigenvalues, full_eigenstates

    def __repr__(self) -> str:
        lines = [
            f"{type(self).__qualname__}(",
            "  hamiltonians={",
            *(f"    '{label}': {ham!r}," for label, ham in self._hamiltonians.items()),
            "  },",
            "  coefficients={",
            *(f"    '{label}': {coeff!r}," for label, coeff in self.coefficients_dict.items()),
            "  },",
            f"  dt={self.dt!r},",
            f"  total_time={self._max_time!r},",
            f"  interpolation={self._interpolation!r}",
            ")",
        ]
        return "\n".join(lines)

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

        Raises:
            ValueError: if the length of the values is not the same as the number of parameters in this object.
        """
        for coeff in self.coefficients.values():
            coeff.delete_cache()
        super().set_parameter_values(values, where)

    def set_parameters(self, parameters: dict[str, int | float]) -> None:
        """
        Assign parameter values by name and clear caches.

        Args:
            parameters (dict[str, int | float]): Mapping from parameter labels to numeric values.
        """
        for coeff in self.coefficients.values():
            coeff.delete_cache()
        super().set_parameters(parameters)

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        """
        Update parameter bounds and clear caches.

        Args:
            ranges (dict[str, tuple[float, float]]): Bounds keyed by parameter label.
        """
        for coeff in self.coefficients.values():
            coeff.delete_cache()
        super().set_parameter_bounds(ranges)
