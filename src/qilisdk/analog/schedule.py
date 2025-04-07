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
from warnings import warn

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.yaml import yaml


@yaml.register_class
class Schedule:
    """
    Represents a time-dependent schedule for Hamiltonian coefficients in an annealing process.

    A Schedule defines the evolution of a system by associating time steps with a set
    of Hamiltonian coefficients. It maintains a dictionary of Hamiltonian objects and a
    corresponding schedule that specifies the coefficients (weights) for each Hamiltonian
    at discrete time steps.

    Attributes:
        _T (float): The total annealing time.
        _dt (float): The time step duration. Total time must be divisible by dt.
        _hamiltonians (dict[str, Hamiltonian]): A mapping of labels to Hamiltonian objects.
        _schedule (dict[int, dict[str, float]]): A mapping of time steps to coefficient dictionaries.
        _nqubits (int): The maximum number of qubits among the Hamiltonians.
        iter_time_step (int): Internal counter for iteration over time steps.
    """

    def __init__(
        self,
        T: float,
        dt: float,
        hamiltonians: dict[str, Hamiltonian] | None = None,
        schedule: dict[int, dict[str, float]] | None = None,
    ) -> None:
        """
        Initialize a Schedule object.

        Args:
            T (float): The total annealing time.
            dt (float): The time step for the annealing process. Note that T needs to be divisible by dt.
            hamiltonians (dict[str, Hamiltonian], optional): A dictionary mapping labels to Hamiltonian objects.
                Defaults to an empty dictionary if None.
            schedule (dict[int, dict[str, float]], optional): A dictionary where keys are time step indices (integers)
                and values are dictionaries mapping Hamiltonian labels to their coefficients at that time step.
                Defaults to {0: {}} if None.

        Raises:
            ValueError: If the provided schedule references Hamiltonians that have not been defined.
        """

        self._hamiltonians: dict[str, Hamiltonian] = hamiltonians if hamiltonians is not None else {}
        self._schedule: dict[int, dict[str, float]] = schedule if schedule is not None else {0: {}}
        self._T = T
        self._dt = dt
        self.iter_time_step = 0
        self._nqubits = 0
        for hamiltonian in self._hamiltonians.values():
            self._nqubits = max(self._nqubits, hamiltonian.nqubits)

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

    @property
    def hamiltonians(self) -> dict[str, Hamiltonian]:
        """
        Get the dictionary of Hamiltonians known to this schedule.

        Returns:
            dict[str, Hamiltonian]: A mapping of Hamiltonian labels to Hamiltonian objects.
        """
        return self._hamiltonians

    @property
    def schedule(self) -> dict[int, dict[str, float]]:
        """
        Get the full schedule of Hamiltonian coefficients.

        The schedule is returned as a dictionary sorted by time step.

        Returns:
            dict[int, dict[str, float]]: The mapping of time steps to coefficient dictionaries.
        """
        return dict(sorted(self._schedule.items()))

    @property
    def T(self) -> float:
        """
        Get the total annealing time.

        Returns:
            float: The total time T.
        """
        return self._T

    @property
    def dt(self) -> float:
        """
        Get the time step duration.

        Returns:
            float: The duration of each time step.
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
        """
        if label not in self._hamiltonians:
            self._hamiltonians[label] = hamiltonian
            self._schedule[0][label] = 0
            self._nqubits = max(self._nqubits, hamiltonian.nqubits)
        else:
            warn(
                (
                    f"label {label} is already assigned to a hamiltonian, "
                    + "ignoring new hamiltonian and updating schedule of existing hamiltonian."
                ),
                RuntimeWarning,
            )

        if schedule is not None:
            for t in range(int(self.T / self.dt) + 1):
                self.update_hamiltonian_coefficient_at_time_step(t, label, schedule(t, **kwargs))

    def add_schedule_step(self, time_step: int, hamiltonian_coefficient_list: dict[str, float]) -> None:
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
            warn(
                f"time step {time_step} is already defined in the schedule, the values are going to be overwritten.",
                RuntimeWarning,
            )
        for key in hamiltonian_coefficient_list:
            if key not in self._hamiltonians:
                raise ValueError(f"trying to reference a hamiltonian {key} that is not defined in this schedule.")
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
                        ham += self._schedule[time_step][ham_label] * self._hamiltonians[ham_label]
                        read_labels.append(ham_label)
                    break
        else:
            for ham_label in self._schedule[time_step]:
                ham += self._schedule[time_step][ham_label] * self._hamiltonians[ham_label]
                read_labels.append(ham_label)
        if len(read_labels) < len(self._hamiltonians):
            all_labels = self._hamiltonians.keys()
            remaining_labels = list(filter(lambda x: x not in read_labels, all_labels))
            for label in remaining_labels:
                current_time = time_step
                while current_time > 0:
                    current_time -= 1
                    if current_time in self._schedule and label in self._schedule[current_time]:
                        ham += self._schedule[current_time][label] * self._hamiltonians[label]
                        break
        return ham

    def get_coefficient(self, time_step: float, hamiltonian_key: str) -> float:
        """
        Retrieve the coefficient of a specified Hamiltonian at a given time.

        This function searches backwards in time (by multiples of dt) until it finds a defined
        coefficient for the given Hamiltonian.

        Args:
            time_step (float): The time (in the same units as T) at which to query the coefficient.
            hamiltonian_key (str): The label of the Hamiltonian.

        Returns:
            float: The coefficient of the Hamiltonian at the specified time, or 0 if not defined.
        """
        time_idx = int(time_step / self.dt)
        while time_idx >= 0:
            if time_idx in self._schedule and hamiltonian_key in self._schedule[time_idx]:
                return self._schedule[time_idx][hamiltonian_key]
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
