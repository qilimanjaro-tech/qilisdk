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
from typing import Callable, Dict
from warnings import warn

from qilisdk.analog.hamiltonian import Hamiltonian


class Schedule:
    def __init__(
        self,
        T: float,
        dt: float,
        hamiltonians: Dict[str, Hamiltonian] | None = None,
        schedule: Dict[int, Dict[str, float]] | None = None,
    ) -> None:
        """Initializes a Schedule object.

        Args:
            T (float): the total annealing time.
            dt (float): the time step of the annealing. Note that `T` needs to be divisible by `dt`
            hamiltonians (Dict[str, Hamiltonian], optional): A dictionary containg the list of the hamiltonians
                known by the schedule. Each hamiltonian is identified by a string. Defaults to None.
            schedule (Dict[int, Dict[str, float]], optional): A dictionary containing the coefficients of the
                hamiltonians at each time step. Defaults to None.
                Note: you don't need to specify the coefficient of the hamiltonian if it has not
                changed over the last time step.
                Note 2: the time steps are integer numbers, and are interpreted as multiples of `dt`.

        Raises:
            ValueError: if the schedule provided references hamiltonians that have not been defined in the schedule.
        """

        self._hamiltonians = hamiltonians if hamiltonians is not None else {}
        self._schedule = schedule if schedule is not None else {0: {}}
        self._T = T
        self._dt = dt
        self.iter_time_step = 0

        if 0 not in self._schedule:
            self._schedule[0] = dict.fromkeys(self._hamiltonians, 0.0)
        else:
            for l in self._hamiltonians:
                if l not in self._schedule[0]:
                    self._schedule[0][l] = 0

        for time_step in self._schedule.values():
            if not all(s in hamiltonians for s in time_step):
                raise ValueError(
                    "All hamiltonians defined in the schedule need to be declared in the hamiltonians dictionary."
                )

    @property
    def hamiltonians(self) -> Hamiltonian:
        return self._hamiltonians

    @property
    def schedule(self) -> dict[int, dict[str, float]]:
        return dict(sorted(self._schedule.items()))

    @property
    def T(self) -> int:
        return self._T

    @property
    def dt(self) -> int:
        return self._dt

    def add_hamiltonian(
        self, label: str, hamiltonian: Hamiltonian, schedule: Callable | None = None, **kwargs: dict
    ) -> None:
        """Adds a hamiltonian object to the list of hamiltonians known by the schedule

        Args:
            label (str): the label used by the schedule to identify the hamiltonian.
            hamiltonian (Hamiltonian): the hamiltonian object.
            schedule (Callable):
                a function that returns the value of the coefficient of the hamiltonian at time t.
        """
        if label not in self._hamiltonians:
            self._hamiltonians[label] = hamiltonian
            self._schedule[0][label] = 0
        else:
            warn(f"label {label} is already assigned to a hamiltonian, ignoring operation", RuntimeWarning)

        if schedule is not None:
            for t in range(int(self.T / self.dt) + 1):
                self.update_hamiltonian_coefficient_at_time_step(t, label, schedule(t, **kwargs))

    def add_schedule_step(self, time_step: int, hamiltonian_coefficient_list: Dict[str, float]) -> None:
        """add a new time step entry.

        Args:
            time_step (int): the time at which the hamiltonian coefficients are updated.
                The global time at which this event occurs is `dt * time_step`.
            hamiltonian_coefficient_list (Dict[str, float]): a dictionary of the hamiltonians
                and their corresponding coefficients at the given time step.
                Note: if a hamiltonian is not present in the dictionary then it's assumed that
                their coefficient value has not changed.
        Raises:
            ValueError: if the hamiltonian_coefficient_list references a hamiltonian that was not defined in the schedule.
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
        """A method used to update the coefficient value of a hamiltonian at a given time step.

        Args:
            time_step (int): the time step (as a multiple of dt).
            hamiltonian_label (str): the hamiltonian label as defined in the hamiltonians dictionary.
            new_coefficient (float): the new hamiltonian coefficient.

        Raises:
            ValueError: if the time step referenced happens after the end of the annealing schedule.
        """
        if not (time_step * self.dt <= self.T):
            raise ValueError("Can't add a time step which happens after the end of the annealing process.")

        if time_step not in self._schedule:
            self._schedule[time_step] = {}
        self._schedule[time_step][hamiltonian_label] = new_coefficient

    def __getitem__(self, time_step: int) -> Hamiltonian:
        ham = 0
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

    def __len__(self) -> int:
        return int(self.T / self.dt)

    def __iter__(self) -> "Schedule":
        self.iter_time_step = 0
        return self

    def __next__(self) -> Hamiltonian:
        if self.iter_time_step <= self.__len__():
            result = self[self.iter_time_step]
            self.iter_time_step += 1
            return result
        raise StopIteration
