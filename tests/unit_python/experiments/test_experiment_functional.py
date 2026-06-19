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

from qilisdk.experiments import (
    ExperimentFunctional,
    RabiExperiment,
    T1Experiment,
    T1SaturationExperiment,
    T2Experiment,
    TwoToneAtFixedFluxBiasExperiment,
    TwoToneAtFixedFluxBiasSaturationExperiment,
    TwoToneVsFluxBiasRampExperiment,
)


def test_experiment_functional_initialization():
    qubit = 0
    averages = 1000

    functional = ExperimentFunctional(qubit=qubit, averages=averages)

    assert functional.qubit == qubit
    assert functional.averages == averages


def test_rabi_experiment_initialization():
    qubit = 0
    averages = 1000
    values = np.array([0.1, 0.2, 0.3])

    rabi_exp = RabiExperiment(qubit=qubit, averages=averages, drive_duration_values=values)

    assert rabi_exp.qubit == qubit
    assert rabi_exp.averages == averages
    assert np.array_equal(rabi_exp.drive_duration_values, values)


def test_t1_experiment_initialization():
    qubit = 0
    averages = 1000
    values = np.array([10, 20, 30])

    t1_exp = T1Experiment(qubit=qubit, averages=averages, wait_duration_values=values)

    assert t1_exp.qubit == qubit
    assert t1_exp.averages == averages
    assert np.array_equal(t1_exp.wait_duration_values, values)


def test_t2_experiment_initialization():
    qubit = 0
    averages = 1000
    values = np.array([5, 15, 25])

    t2_exp = T2Experiment(qubit=qubit, averages=averages, wait_duration_values=values)

    assert t2_exp.qubit == qubit
    assert t2_exp.averages == averages
    assert np.array_equal(t2_exp.wait_duration_values, values)


def test_two_tones_experiment_initialization():
    qubit = 0
    averages = 1000
    freq_start = 4.0
    freq_stop = 5.0
    freq_step = 5.0

    two_tones_exp = TwoToneAtFixedFluxBiasExperiment(
        qubit=qubit,
        averages=averages,
        frequency_start=freq_start,
        frequency_stop=freq_stop,
        frequency_step=freq_step,
    )

    assert two_tones_exp.qubit == qubit
    assert two_tones_exp.averages == averages
    assert np.isclose(two_tones_exp.frequency_start, freq_start)
    assert np.isclose(two_tones_exp.frequency_stop, freq_stop)
    assert np.isclose(two_tones_exp.frequency_step, freq_step)


def test_two_tones_frequency_vs_flux_qdac_ramp_cw_experiment_initialization():
    qubit = 0
    averages = 1000
    freq_start = 4.0e9
    freq_stop = 6.0e9
    freq_step = 10e6
    flux_start = -0.5
    flux_stop = 0.5
    flux_step = 0.02

    exp = TwoToneVsFluxBiasRampExperiment(
        qubit=qubit,
        averages=averages,
        frequency_start=freq_start,
        frequency_stop=freq_stop,
        frequency_step=freq_step,
        flux_start=flux_start,
        flux_stop=flux_stop,
        flux_step=flux_step,
    )

    assert exp.qubit == qubit
    assert exp.averages == averages
    assert np.isclose(exp.frequency_start, freq_start)
    assert np.isclose(exp.frequency_stop, freq_stop)
    assert np.isclose(exp.frequency_step, freq_step)
    assert np.isclose(exp.flux_start, flux_start)
    assert np.isclose(exp.flux_stop, flux_stop)
    assert np.isclose(exp.flux_step, flux_step)


def test_two_tones_pulsed_soft_experiment_initialization():
    qubit = 0
    averages = 1000
    freq_start = 4.0e9
    freq_stop = 5.0e9
    freq_step = 5e6

    exp = TwoToneAtFixedFluxBiasSaturationExperiment(
        qubit=qubit,
        averages=averages,
        frequency_start=freq_start,
        frequency_stop=freq_stop,
        frequency_step=freq_step,
    )

    assert exp.qubit == qubit
    assert exp.averages == averages
    assert np.isclose(exp.frequency_start, freq_start)
    assert np.isclose(exp.frequency_stop, freq_stop)
    assert np.isclose(exp.frequency_step, freq_step)


def test_t1_soft_saturation_hwl_experiment_initialization():
    qubit = 0
    averages = 1000
    values = np.arange(0, 100000, 2000)

    exp = T1SaturationExperiment(qubit=qubit, averages=averages, wait_duration_values=values)

    assert exp.qubit == qubit
    assert exp.averages == averages
    assert np.array_equal(exp.wait_duration_values, values)
