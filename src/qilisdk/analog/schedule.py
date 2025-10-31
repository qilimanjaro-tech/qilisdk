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

from typing import Callable

from loguru import logger

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import BaseVariable, Number, Parameter, Term
from qilisdk.utils.visualization import ScheduleStyle
from qilisdk.yaml import yaml


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
        T: float,
        dt: float = 1,
        hamiltonians: dict[str, Hamiltonian] | None = None,
        schedule: dict[int, dict[str, float | Term | Parameter]] | None = None,
    ) -> None:
        """
        Args:
            T (float): Total annealing time (in nanoseconds).
            dt (float, optional): Discretization step of the time grid. Defaults to 1.
            hamiltonians (dict[str, Hamiltonian] | None, optional): Mapping from labels to Hamiltonian instances that
                define the building blocks of the schedule. Defaults to None, which creates an empty mapping.
            schedule (dict[int, dict[str, float | Term | Parameter]] | None, optional): Predefined coefficients for
                specific time steps. Each inner dictionary maps Hamiltonian labels to numerical or symbolic
                coefficients. Defaults to {0: {}} if None.

        Raises:
            ValueError: If the provided schedule references Hamiltonians that have not been defined.
        """
        if dt <= 0:
            raise ValueError("dt must be greater than zero.")
        self._hamiltonians: dict[str, Hamiltonian] = hamiltonians if hamiltonians is not None else {}
        self._schedule: dict[int, dict[str, float | Term | Parameter]] = schedule if schedule is not None else {0: {}}
        self._parameters: dict[str, Parameter] = {}
        self._T = T
        self._dt = dt
        self.iter_time_step = 0
        self._nqubits = 0

        for hamiltonian in self._hamiltonians.values():
            self._nqubits = max(self._nqubits, hamiltonian.nqubits)
            for l, param in hamiltonian.parameters.items():
                self._parameters[param.label] = param

        if 0 not in self._schedule:
            self._schedule[0] = dict.fromkeys(self._hamiltonians, 0.0)
        else:
            for label in self._hamiltonians:
                if label not in self._schedule[0]:
                    self._schedule[0][label] = 0

        for time_step in self._schedule.values():
            if not all(s in self._hamiltonians for s in time_step):
                raise ValueError(
                    "All hamiltonians defined in the schedule need to be declared in the hamiltonians dictionary."
                )
            for coeff in time_step.values():
                if isinstance(coeff, Term):
                    for v in coeff.variables():
                        if not isinstance(v, Parameter):
                            raise ValueError(
                                f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                            )
                        self._parameters[v.label] = v
                elif isinstance(coeff, BaseVariable):
                    if not isinstance(coeff, Parameter):
                        raise ValueError(
                            f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                        )
                    self._parameters[coeff.label] = coeff

    @property
    def hamiltonians(self) -> dict[str, Hamiltonian]:
        """
        Return the Hamiltonians managed by the schedule.

        Returns:
            dict[str, Hamiltonian]: Mapping of labels to Hamiltonian instances.
        """
        return self._hamiltonians

    @property
    def schedule(self) -> dict[int, dict[str, Number]]:
        """
        Return the evaluated schedule of Hamiltonian coefficients.

        Returns:
            dict[int, dict[str, Number]]: Mapping from time indices to evaluated coefficients.
        """
        out_dict = {}
        for k, v in self._schedule.items():
            out_dict[k] = {
                ham: (
                    coeff
                    if isinstance(coeff, Number)
                    else (coeff.evaluate() if isinstance(coeff, Parameter) else coeff.evaluate({}))
                )
                for ham, coeff in v.items()
            }
        return dict(sorted(out_dict.items()))

    @property
    def T(self) -> float:
        """Total annealing time of the schedule."""
        return self._T

    @property
    def dt(self) -> float:
        """Duration of a single time step in the annealing grid."""
        return self._dt

    @property
    def nqubits(self) -> int:
        """Maximum number of qubits affected by Hamiltonians contained in the schedule."""
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """Number of symbolic parameters introduced by the Hamiltonians or coefficients."""
        return len(self._parameters)

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

    def add_hamiltonian(
        self, label: str, hamiltonian: Hamiltonian, schedule: Callable | None = None, **kwargs: dict
    ) -> None:
        """
        Add a Hamiltonian to the schedule with an optional coefficient schedule function.

        If a Hamiltonian with the given label already exists, a warning is issued and only
        the schedule is updated if a callable is provided.

        Args:
            label (str): The unique label to identify the Hamiltonian.
            hamiltonian (Hamiltonian): The Hamiltonian object to add.
            schedule (Callable, optional): A function that returns the coefficient of the Hamiltonian at time t.
                It should accept time (and any additional keyword arguments) and return a float.
            **kwargs (dict): Additional keyword arguments to pass to the schedule function.

        Raises:
            ValueError: if the parameterized schedule contains generic variables instead of only Parameters.
        """
        if label in self._hamiltonians:
            logger.warning(
                (f"label {label} is already assigned to a hamiltonian, " + "updating schedule of existing hamiltonian.")
            )
        self._hamiltonians[label] = hamiltonian
        self._schedule[0][label] = 0
        self._nqubits = max(self._nqubits, hamiltonian.nqubits)
        for _, param in hamiltonian.parameters.items():
            self._parameters[param.label] = param

        if schedule is not None:
            for t in range(int(self.T / self.dt)):
                time_step = schedule(float(t), **kwargs)
                if isinstance(time_step, Term):
                    for v in time_step.variables():
                        if not isinstance(v, Parameter):
                            raise ValueError(
                                f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                            )
                        self._parameters[v.label] = v
                elif isinstance(time_step, BaseVariable):
                    if not isinstance(time_step, Parameter):
                        raise ValueError(
                            f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                        )
                    self._parameters[time_step.label] = time_step
                self.update_hamiltonian_coefficient_at_time_step(t, label, time_step)

    def add_schedule_step(
        self, time_step: int, hamiltonian_coefficient_list: dict[str, float | Term | Parameter]
    ) -> None:
        """
        Add or update a schedule step with specified Hamiltonian coefficients.

        Args:
            time_step (int): The time step index at which the Hamiltonian coefficients are updated.
                The actual time is computed as dt * time_step.
            hamiltonian_coefficient_list (dict[str, float | Term | Parameter]): Mapping from Hamiltonian labels to coefficients
                (numeric or symbolic) at this time step.
                If a Hamiltonian is not included in the dictionary, it is assumed its coefficient remains unchanged.

        Raises:
            ValueError: If hamiltonian_coefficient_list references a Hamiltonian that is not defined in the schedule.
        """
        if time_step in self._schedule:
            logger.warning(
                f"time step {time_step} is already defined in the schedule, the values are going to be overwritten.",
            )
        for key, coeff in hamiltonian_coefficient_list.items():
            if key not in self._hamiltonians:
                raise ValueError(f"trying to reference a hamiltonian {key} that is not defined in this schedule.")
            if isinstance(coeff, Term):
                for v in coeff.variables():
                    if not isinstance(v, Parameter):
                        raise ValueError(
                            f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                        )
                    self._parameters[v.label] = v
            if isinstance(coeff, BaseVariable):
                if not isinstance(coeff, Parameter):
                    raise ValueError(
                        f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                    )
                self._parameters[coeff.label] = coeff
        self._schedule[time_step] = hamiltonian_coefficient_list

    def update_hamiltonian_coefficient_at_time_step(
        self, time_step: int, hamiltonian_label: str, new_coefficient: float | Term | Parameter
    ) -> None:
        """
        Update the coefficient value of a specific Hamiltonian at a given time step.

        Args:
            time_step (int): The time step (as an integer multiple of dt) at which to update the coefficient.
            hamiltonian_label (str): The label of the Hamiltonian to update.
            new_coefficient (float | Term | Parameter): The new coefficient value or symbolic expression.

        Raises:
            ValueError: If the specified time step exceeds the total annealing time.
        """
        if not (time_step * self.dt <= self.T):
            raise ValueError("Can't add a time step which happens after the end of the annealing process.")

        if time_step not in self._schedule:
            self._schedule[time_step] = {}
        self._schedule[time_step][hamiltonian_label] = new_coefficient

        if isinstance(new_coefficient, Term):
            for v in new_coefficient.variables():
                if not isinstance(v, Parameter):
                    raise ValueError(
                        f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                    )
                self._parameters[v.label] = v
        if isinstance(new_coefficient, BaseVariable):
            if not isinstance(new_coefficient, Parameter):
                raise ValueError(
                    f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                )
            self._parameters[new_coefficient.label] = new_coefficient

    def __getitem__(self, time_step: int) -> Hamiltonian:
        """
        Retrieve the effective Hamiltonian at a given time step.

        The effective Hamiltonian is computed by summing the contributions of all Hamiltonians,
        using the latest defined coefficients at or before the given time step.

        Args:
            time_step (int): Time step index for which to retrieve the Hamiltonian (``time_step * dt`` in units).

        Returns:
            Hamiltonian: The effective Hamiltonian at the specified time step with coefficients evaluated to numbers.
        """
        ham = Hamiltonian()
        read_labels = []

        if time_step not in self._schedule:
            while time_step > 0:
                time_step -= 1
                if time_step in self._schedule:
                    for ham_label in self._schedule[time_step]:
                        aux = self._schedule[time_step][ham_label]
                        coeff = (
                            aux.evaluate({})
                            if isinstance(aux, Term)
                            else (aux.evaluate() if isinstance(aux, Parameter) else aux)
                        )
                        ham += coeff * self._hamiltonians[ham_label]
                        read_labels.append(ham_label)
                    break
        else:
            for ham_label in self._schedule[time_step]:
                aux = self._schedule[time_step][ham_label]
                coeff = (
                    aux.evaluate({})
                    if isinstance(aux, Term)
                    else (aux.evaluate() if isinstance(aux, Parameter) else aux)
                )
                ham += coeff * self._hamiltonians[ham_label]
                read_labels.append(ham_label)
        if len(read_labels) < len(self._hamiltonians):
            all_labels = self._hamiltonians.keys()
            remaining_labels = list(filter(lambda x: x not in read_labels, all_labels))
            for label in remaining_labels:
                current_time = time_step
                while current_time > 0:
                    current_time -= 1
                    if current_time in self._schedule and label in self._schedule[current_time]:
                        aux = self._schedule[current_time][label]
                        coeff = (
                            aux.evaluate({})
                            if isinstance(aux, Term)
                            else (aux.evaluate() if isinstance(aux, Parameter) else aux)
                        )
                        ham += coeff * self._hamiltonians[label]
                        break
        return ham.get_static_hamiltonian()

    def get_coefficient(self, time_step: float, hamiltonian_key: str) -> Number:
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
        return val.evaluate({}) if isinstance(val, Term) else (val.evaluate() if isinstance(val, Parameter) else val)

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
        """
        time_idx = int(time_step / self.dt)
        while time_idx >= 0:
            if time_idx in self._schedule and hamiltonian_key in self._schedule[time_idx]:
                val = self._schedule[time_idx][hamiltonian_key]
                return val
            time_idx -= 1
        return 0

    def __len__(self) -> int:
        """
        Get the total number of discrete time steps in the annealing process.

        Returns:
            int: The number of time steps, calculated as T / dt.
        """
        return int(self.T / self.dt)

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
        if self.iter_time_step <= self.__len__():
            result = self[self.iter_time_step]
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
