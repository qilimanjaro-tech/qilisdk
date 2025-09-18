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
from qilisdk.common.parameterizable import Parameterizable
from qilisdk.common.variables import BaseVariable, Number, Parameter, Term
from qilisdk.utils.visualization import ScheduleStyle
from qilisdk.yaml import yaml


@yaml.register_class
class Schedule(Parameterizable):
    def __init__(
        self,
        T: int,
        dt: int = 1,
        hamiltonians: dict[str, Hamiltonian] | None = None,
        schedule: dict[int, dict[str, float | Term]] | None = None,
    ) -> None:
        """
        Represents a time-dependent schedule for Hamiltonian coefficients in an annealing process.

        A Schedule defines the evolution of a system by associating time steps with a set
        of Hamiltonian coefficients. It maintains a dictionary of Hamiltonian objects and a
        corresponding schedule that specifies the coefficients (weights) for each Hamiltonian
        at discrete time steps.

        Args:
            T (int): The total annealing time in units of 1ns.
            dt (int): The time step for the annealing process it is defined as multiples of 1ns. Defaults to 1.
            hamiltonians (dict[str, Hamiltonian], optional): A dictionary mapping labels to Hamiltonian objects.
                Defaults to an empty dictionary if None.
            schedule (dict[int, dict[str, float]], optional): A dictionary where keys are time step indices (integers)
                and values are dictionaries mapping Hamiltonian labels to their coefficients at that time step.
                Defaults to {0: {}} if None.

        Raises:
            ValueError: If the provided schedule references Hamiltonians that have not been defined.
        """
        if abs(T % dt) > 1e-12:  # noqa: PLR2004
            raise ValueError("T must be divisible by dt.")
        if not isinstance(T, int):
            raise ValueError("T must be an integer")
        if not isinstance(dt, int):
            raise ValueError("dt must be an integer")
        self._hamiltonians: dict[str, Hamiltonian] = hamiltonians if hamiltonians is not None else {}
        self._schedule: dict[int, dict[str, float | Term]] = schedule if schedule is not None else {0: {}}
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
                if isinstance(coeff, BaseVariable):
                    if not isinstance(coeff, Parameter):
                        raise ValueError(
                            f"The schedule can only contain Parameters, but a generic variable was provided ({time_step})"
                        )
                    self._parameters[coeff.label] = coeff

    @property
    def hamiltonians(self) -> dict[str, Hamiltonian]:
        """
        Get the dictionary of Hamiltonians known to this schedule.

        Returns:
            dict[str, Hamiltonian]: A mapping of Hamiltonian labels to Hamiltonian objects.
        """
        return self._hamiltonians

    @property
    def schedule(self) -> dict[int, dict[str, Number]]:
        """
        Get the full schedule of Hamiltonian coefficients.

        The schedule is returned as a dictionary sorted by time step.

        Returns:
            dict[int, dict[str, float]]: The mapping of time steps to coefficient dictionaries.
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
    def T(self) -> int:
        """
        Get the total annealing time.

        Returns:
            int: The total time T.
        """
        return self._T

    @property
    def dt(self) -> int:
        """
        Get the time step duration.

        Returns:
            int: The duration of each time step.
        """
        return self._dt

    @property
    def nqubits(self) -> int:
        """
        Get the maximum number of qubits among all Hamiltonians in the schedule.

        Returns:
            int: The number of qubits.
        """
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total RealNumber of parameters required by all parameterized gates in the circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """
        return len(self._parameters)

    def get_parameter_values(self) -> list[float]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """
        return [param.value for param in self._parameters.values()]

    def get_parameter_names(self) -> list[str]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """
        return list(self._parameters.keys())

    def get_parameters(self) -> dict[str, float]:
        """
        Retrieve the parameter names and values from all parameterized gates in the circuit.

        Returns:
            dict[str, float]: A dictionary of the parameters with their current values.
        """
        return {label: param.value for label, param in self._parameters.items()}

    def set_parameter_values(self, values: list[float]) -> None:
        """
        Set new parameter values for all parameterized gates in the circuit.

        Args:
            values (list[float]): A list containing new parameter values to assign to the parameterized gates.

        Raises:
            ValueError: If the RealNumber of provided values does not match the expected RealNumber of parameters.
        """
        if len(values) != self.nparameters:
            raise ValueError(f"Provided {len(values)} but Schedule has {self.nparameters} parameters.")
        for i, parameter in enumerate(self._parameters.values()):
            parameter.set_value(values[i])

    def set_parameters(self, parameter_dict: dict[str, int | float]) -> None:
        """Set the parameter values by their label. No need to provide the full list of parameters.

        Args:
            parameter_dict (dict[str, RealNumber]): _description_

        Raises:
            ValueError: _description_
        """
        for label, param in parameter_dict.items():
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this Schedule.")
            self._parameters[label].set_value(param)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {k: v.bounds for k, v in self._parameters.items()}

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
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
        if label not in self._hamiltonians:
            self._hamiltonians[label] = hamiltonian
            self._schedule[0][label] = 0
            self._nqubits = max(self._nqubits, hamiltonian.nqubits)
            for l, param in hamiltonian.parameters.items():
                self._parameters[param.label] = param
        else:
            logger.warning(
                (
                    f"label {label} is already assigned to a hamiltonian, "
                    + "ignoring new hamiltonian and updating schedule of existing hamiltonian."
                )
            )

        if schedule is not None:
            for t in range(int(self.T / self.dt)):
                time_step = schedule(t, **kwargs)
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

    def add_schedule_step(self, time_step: int, hamiltonian_coefficient_list: dict[str, float | Term]) -> None:
        """
        Add or update a schedule step with specified Hamiltonian coefficients.

        Args:
            time_step (int): The time step index at which the Hamiltonian coefficients are updated.
                The actual time is computed as dt * time_step.
            hamiltonian_coefficient_list (dict[str, float]): A dictionary mapping Hamiltonian labels to their
                coefficient values at this time step.
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
        self, time_step: int, hamiltonian_label: str, new_coefficient: float
    ) -> None:
        """
        Update the coefficient value of a specific Hamiltonian at a given time step.

        Args:
            time_step (int): The time step (as an integer multiple of dt) at which to update the coefficient.
            hamiltonian_label (str): The label of the Hamiltonian to update.
            new_coefficient (float): The new coefficient value.

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
            time_step (int): The time step index for which to retrieve the Hamiltonian.

        Returns:
            Hamiltonian: The effective Hamiltonian at the specified time step.
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
                            else (aux.evaluate if isinstance(aux, Parameter) else aux)
                        )
                        ham += coeff * self._hamiltonians[ham_label]
                        read_labels.append(ham_label)
                    break
        else:
            for ham_label in self._schedule[time_step]:
                aux = self._schedule[time_step][ham_label]
                coeff = (
                    aux.evaluate({}) if isinstance(aux, Term) else (aux.evaluate if isinstance(aux, Parameter) else aux)
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
                            else (aux.evaluate if isinstance(aux, Parameter) else aux)
                        )
                        ham += coeff * self._hamiltonians[label]
                        break
        return ham.get_static_hamiltonian()

    def get_coefficient(self, time_step: int, hamiltonian_key: str) -> Number:
        """
        Retrieve the coefficient of a specified Hamiltonian at a given time.

        This function searches backwards in time (by multiples of dt) until it finds a defined
        coefficient for the given Hamiltonian.

        Args:
            time_step (int): The time (in the same units as T) at which to query the coefficient.
            hamiltonian_key (str): The label of the Hamiltonian.

        Returns:
            float: The coefficient of the Hamiltonian at the specified time, or 0 if not defined.
        """
        time_idx = int(time_step / self.dt)
        while time_idx >= 0:
            if time_idx in self._schedule and hamiltonian_key in self._schedule[time_idx]:
                val = self._schedule[time_idx][hamiltonian_key]
                return (
                    val.evaluate({})
                    if isinstance(val, Term)
                    else (val.evaluate() if isinstance(val, Parameter) else val)
                )
            time_idx -= 1
        return 0

    def get_coefficient_expression(self, time_step: int, hamiltonian_key: str) -> Number | Term:
        """
        Retrieve the expression of a specified Hamiltonian at a given time. If any parameters are
        present in the expression they will be printed in the expression.

        This function searches backwards in time (by multiples of dt) until it finds a defined
        coefficient for the given Hamiltonian.

        Args:
            time_step (int): The time (in the same units as T) at which to query the coefficient.
            hamiltonian_key (str): The label of the Hamiltonian.

        Returns:
            float: The coefficient of the Hamiltonian at the specified time, or 0 if not defined.
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

    def draw(self, style: ScheduleStyle | None = None, filepath: str | None = None, title: str | None = None) -> None:
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
        renderer.plot(title=title)
        if filepath:
            renderer.save(filepath)


@yaml.register_class
class LinearSchedule(Schedule):
    """A Schedule that linearly interpolates between defined time steps."""

    def get_coefficient(self, time_step: float, hamiltonian_key: str) -> Number:
        t = time_step / self.dt
        t_idx = int(t)

        # if exactly defined, return directly
        if t_idx in self._schedule and hamiltonian_key in self._schedule[t_idx]:
            val = self._schedule[t_idx][hamiltonian_key]
            return (
                val.evaluate({}) if isinstance(val, Term) else (val.evaluate() if isinstance(val, Parameter) else val)
            )

        # search backwards for last defined
        prev_idx, prev_val = None, None
        for i in range(t_idx, -1, -1):
            if i in self._schedule and hamiltonian_key in self._schedule[i]:
                prev_idx = i
                val = self._schedule[i][hamiltonian_key]
                prev_val = (
                    val.evaluate({})
                    if isinstance(val, Term)
                    else (val.evaluate() if isinstance(val, Parameter) else val)
                )
                break

        # search forwards for next defined
        next_idx, next_val = None, None
        for i in range(t_idx + 1, int(self.T / self.dt) + 1):
            if i in self._schedule and hamiltonian_key in self._schedule[i]:
                next_idx = i
                val = self._schedule[i][hamiltonian_key]
                next_val = (
                    val.evaluate({})
                    if isinstance(val, Term)
                    else (val.evaluate() if isinstance(val, Parameter) else val)
                )
                break

        # cases
        if prev_val is None and next_val is None:
            return 0
        if prev_val is None and next_val is not None:
            return next_val
        if next_val is None and prev_val is not None:
            return prev_val

        # linear interpolation
        if next_idx is None or prev_idx is None or prev_val is None or next_val is None:
            raise ValueError("Something unexpected happened while retrieving the coefficient.")
        alpha: float = (t - prev_idx) / (next_idx - prev_idx)
        return (1 - alpha) * prev_val + alpha * next_val

    def get_coefficient_expression(self, time_step: float, hamiltonian_key: str) -> Number | Term:
        t = time_step / self.dt
        t_idx = int(t)

        if t_idx in self._schedule and hamiltonian_key in self._schedule[t_idx]:
            return self._schedule[t_idx][hamiltonian_key]

        # search backwards
        prev_idx, prev_expr = None, None
        for i in range(t_idx, -1, -1):
            if i in self._schedule and hamiltonian_key in self._schedule[i]:
                prev_idx = i
                prev_expr = self._schedule[i][hamiltonian_key]
                break

        # search forwards
        next_idx, next_expr = None, None
        for i in range(t_idx + 1, int(self.T / self.dt) + 1):
            if i in self._schedule and hamiltonian_key in self._schedule[i]:
                next_idx = i
                next_expr = self._schedule[i][hamiltonian_key]
                break

        # cases
        if prev_expr is None and next_expr is None:
            return 0
        if prev_expr is None and next_expr is not None:
            return next_expr
        if next_expr is None and prev_expr is not None:
            return prev_expr

        # linear interpolation (keeps expressions if they are Terms/Parameters)
        if next_idx is None or prev_idx is None or prev_expr is None or next_expr is None:
            raise ValueError("Something unexpected happened while retrieving the coefficient.")
        alpha: float = (t - prev_idx) / (next_idx - prev_idx)
        return (1 - alpha) * prev_expr + alpha * next_expr

    def __getitem__(self, time_step: int) -> Hamiltonian:
        ham = Hamiltonian()
        for ham_label in self._hamiltonians:
            coeff = self.get_coefficient(time_step * self.dt, ham_label)
            ham += coeff * self._hamiltonians[ham_label]
        return ham.get_static_hamiltonian()
