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

from qilisdk.experiments import ExperimentFunctional, RabiExperiment, T1Experiment, T2Experiment, TwoTonesExperiment


def test_experiment_functional_initialization():
    qubit = 0
    functional = ExperimentFunctional(qubit=qubit)
    assert functional.qubit == qubit


def test_rabi_experiment_initialization():
    qubit = 0
    values = np.array([0.1, 0.2, 0.3])
    rabi_exp = RabiExperiment(qubit=qubit, drive_duration_values=values)
    assert rabi_exp.qubit == qubit
    assert np.array_equal(rabi_exp.drive_duration_values, values)


def test_t1_experiment_initialization():
    qubit = 0
    values = np.array([10, 20, 30])
    t1_exp = T1Experiment(qubit=qubit, wait_duration_values=values)
    assert t1_exp.qubit == qubit
    assert np.array_equal(t1_exp.wait_duration_values, values)


def test_t2_experiment_initialization():
    qubit = 0
    values = np.array([5, 15, 25])
    t2_exp = T2Experiment(qubit=qubit, wait_duration_values=values)
    assert t2_exp.qubit == qubit
    assert np.array_equal(t2_exp.wait_duration_values, values)


def test_two_tones_experiment_initialization():
    qubit = 0
    freq_start = 4.0
    freq_stop = 5.0
    freq_step = 5.0
    two_tones_exp = TwoTonesExperiment(
        qubit=qubit, frequency_start=freq_start, frequency_stop=freq_stop, frequency_step=freq_step
    )
    assert two_tones_exp.qubit == qubit
    assert np.isclose(two_tones_exp.frequency_start, freq_start)
    assert np.isclose(two_tones_exp.frequency_stop, freq_stop)
    assert np.isclose(two_tones_exp.frequency_step, freq_step)
