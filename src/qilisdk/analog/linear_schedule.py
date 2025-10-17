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

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.analog.schedule import Schedule
from qilisdk.common.variables import Number, Parameter, Term
from qilisdk.yaml import yaml


@yaml.register_class
class LinearSchedule(Schedule):
    """
    Schedule implementation that linearly interpolates coefficients between defined time steps.

    Example:
        .. code-block:: python

            from qilisdk.analog.hamiltonian import Hamiltonian, Z
            from qilisdk.analog.linear_schedule import LinearSchedule

            h = 2 * Z(0)
            schedule = LinearSchedule(T=4.0, dt=1.0, hamiltonians={"hz": h})
            schedule.add_schedule_step(0, {"hz": 0.0})
            schedule.add_schedule_step(4, {"hz": 1.0})
            assert schedule.get_coefficient(2.0, "hz") == 0.5
    """

    def get_coefficient_expression(self, time_step: float, hamiltonian_key: str) -> Number | Term | Parameter:
        """
        Return the symbolic coefficient for a Hamiltonian at an arbitrary time step.

        Args:
            time_step (float): The time at which to evaluate the coefficient.
            hamiltonian_key (str): Label of the Hamiltonian inside the schedule.

        Returns:
            Number | Term: The (possibly symbolic) coefficient associated with ``hamiltonian_key``.

        Raises:
            ValueError: If something unexpected happens during coefficient retrieval.
        """
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

    def get_coefficient(self, time_step: float, hamiltonian_key: str) -> Number:
        """
        Return the numeric coefficient for a Hamiltonian at ``time_step``.

        Args:
            time_step (float): Time at which to evaluate the coefficient.
            hamiltonian_key (str): Label of the Hamiltonian.

        Returns:
            Number: Evaluated coefficient value.
        """
        val = self.get_coefficient_expression(time_step=time_step, hamiltonian_key=hamiltonian_key)
        return val.evaluate({}) if isinstance(val, Term) else (val.evaluate() if isinstance(val, Parameter) else val)

    def __getitem__(self, time_step: int) -> Hamiltonian:
        """
        Retrieve the interpolated Hamiltonian at the specified discrete time step.

        Args:
            time_step (int): Discrete index to evaluate (converted internally to ``time_step * dt``).

        Returns:
            Hamiltonian: Hamiltonian with coefficients interpolated at the requested time step.
        """
        ham = Hamiltonian()
        for ham_label in self._hamiltonians:
            coeff = self.get_coefficient(time_step * self.dt, ham_label)
            ham += coeff * self._hamiltonians[ham_label]
        return ham.get_static_hamiltonian()
