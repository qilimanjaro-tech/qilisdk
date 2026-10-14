# Copyright 2026 Qilimanjaro Quantum Tech
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

from collections.abc import Mapping
from typing import Self

from qilisdk.core.model import Model
from qilisdk.core.result import Result
from qilisdk.core.variables import BaseVariable, Number, RealNumber
from qilisdk.yaml import yaml


@yaml.register_class
class ModelResult(Result):
    """
    A solution to a :class:`~qilisdk.core.model.Model`, independent of how it was found.

    Both the local classical solvers and SpeQtrum return this type for a solved model.

    Example:
        .. code-block:: python

            from qilisdk.core import Model
            from qilisdk.utils.classical_solvers import BruteForceSolver

            model = Model.random_ising(4)
            result = BruteForceSolver().solve(model)

            result.objective  # the value of the objective at the solution
            result.sample  # the value each variable takes in the solution
            result.results  # the objective and every constraint, by label
    """

    def __init__(
        self,
        results: Mapping[str, Number],
        sample: Mapping[BaseVariable, RealNumber],
        objective_label: str,
    ) -> None:
        """
        Create a new model result.

        Args:
            results (Mapping[str, Number]): the model's objective and constraints, evaluated at
                ``sample`` and keyed by their labels.
            sample (Mapping[BaseVariable, RealNumber]): the value each model variable takes in the
                solution.
            objective_label (str): the label the objective is stored under in ``results``, i.e.
                ``model.objective.label``.
        """
        self._results = dict(results)
        self._sample = dict(sample)
        self._objective_label = objective_label

    @classmethod
    def from_model(cls, model: Model, sample: Mapping[BaseVariable, RealNumber]) -> Self:
        """
        Build a result by evaluating a model at the solution a solver found.

        Args:
            model (Model): the model that was solved.
            sample (Mapping[BaseVariable, RealNumber]): the value each model variable takes in the
                solution.

        Returns:
            Self: the model evaluated at ``sample``, together with ``sample``.
        """
        return cls(model.evaluate(sample), sample, model.objective.label)

    @property
    def results(self) -> dict[str, Number]:
        """The model's objective and constraints, evaluated at :attr:`sample` and keyed by label."""
        return dict(self._results)

    @property
    def sample(self) -> dict[BaseVariable, RealNumber]:
        """The value each model variable takes in the solution."""
        return dict(self._sample)

    @property
    def objective_label(self) -> str:
        """The label the objective is stored under in :attr:`results`."""
        return self._objective_label

    @property
    def objective(self) -> Number:
        """The value of the model's objective at :attr:`sample`.

        Raises:
            KeyError: if the objective's label is absent from :attr:`results`.
        """
        return self._results[self._objective_label]

    @property
    def constraints(self) -> dict[str, Number]:
        """The value of each of the model's constraints at :attr:`sample`, keyed by label.

        A constraint evaluates to zero when it is satisfied, and to its penalty otherwise.
        """
        return {label: value for label, value in self._results.items() if label != self._objective_label}

    def __repr__(self) -> str:
        """Return a formatted string representation for debugging."""
        return f"{type(self).__name__}(objective={self.objective}, sample={self._sample}, results={self._results})"
