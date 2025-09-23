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
