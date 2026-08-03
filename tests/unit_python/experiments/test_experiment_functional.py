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

import numpy as np

from qilisdk.experiments import ExperimentFunctional, ExperimentResult
from qilisdk.utils.serialization import deserialize, serialize


class SweepExperimentResult(ExperimentResult):
    plot_title = "Sweep Experiment"


class SweepExperiment(ExperimentFunctional[SweepExperimentResult]):
    """Minimal backend-defined experiment, used to exercise the extension point."""

    result_type = SweepExperimentResult

    def __init__(self, qubit: int, averages: int, sweep_values: np.ndarray) -> None:
        super().__init__(qubit=qubit, averages=averages)
        self.sweep_values = sweep_values


def test_experiment_functional_initialization():
    qubit = 0
    averages = 1000

    functional = ExperimentFunctional(qubit=qubit, averages=averages)

    assert functional.qubit == qubit
    assert functional.averages == averages


def test_experiment_functional_subclass_initialization():
    qubit = 3
    averages = 1000
    values = np.array([0.1, 0.2, 0.3])

    experiment = SweepExperiment(qubit=qubit, averages=averages, sweep_values=values)

    assert experiment.qubit == qubit
    assert experiment.averages == averages
    assert np.array_equal(experiment.sweep_values, values)
    assert experiment.result_type is SweepExperimentResult


def test_experiment_functional_serialization_round_trip():
    experiment = SweepExperiment(qubit=1, averages=500, sweep_values=np.array([10, 20, 30]))

    deserialized = deserialize(serialize(experiment), SweepExperiment)

    assert deserialized.qubit == experiment.qubit
    assert deserialized.averages == experiment.averages
    assert np.array_equal(deserialized.sweep_values, experiment.sweep_values)
