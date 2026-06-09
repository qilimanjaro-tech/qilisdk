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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qilisdk.experiments import (
    Dimension,
    ExperimentResult,
    RabiExperimentResult,
    T1ExperimentResult,
    T2ExperimentResult,
    TwoTonesAtFluxBiasExperimentResult,
    TwoTonesVsFluxBiasExperimentResult,
)

# Set this to True if you want to actually save the plots during testing
_DEBUG_PLOTS = False


def test_dimension_initialization():
    dim = Dimension(labels=["Drive amplitude"], values=[np.array([0.1, 0.2, 0.3])])
    assert dim.labels == ["Drive amplitude"]
    assert np.array_equal(dim.values[0], np.array([0.1, 0.2, 0.3]))


def test_experiment_result_init():
    data = np.array([[1, 2], [3, 4]])
    qubit = 0
    dims = [Dimension(labels=["Freq"], values=[np.array([1, 2])])]
    exp_result = ExperimentResult(qubit=qubit, data=data, dims=dims)
    assert exp_result.qubit == qubit
    assert np.array_equal(exp_result.data, data)
    assert exp_result.dims == dims


def test_experiment_s21_computation():
    data = np.array([[1, 2], [3, 4]])
    exp_result = ExperimentResult(qubit=0, data=data, dims=[])

    s21 = exp_result.s21
    expected_s21 = np.array([1 + 2j, 3 + 4j])
    assert np.allclose(s21, expected_s21)

    s21_modulus = exp_result.s21_modulus
    expected_modulus = np.abs(expected_s21)
    assert np.allclose(s21_modulus, expected_modulus)

    s21_db = exp_result.s21_db
    expected_db = 20 * np.log10(expected_modulus)
    assert np.allclose(s21_db, expected_db)


def test_experiment_plotting(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # Prevent actual plot display
    monkeypatch.setattr(plt.Figure, "savefig", lambda self, *args, **kwargs: None)  # Prevent file saving

    # 3d should fail
    data3d = np.ones((2, 2, 2, 2))
    dims3d = [
        Dimension(labels=["Dim1", "Freq"], values=[np.array([1, 2]), np.array([10, 20])]),
        Dimension(labels=["Dim2", "Freq2"], values=[np.array([3, 4]), np.array([30, 40])]),
        Dimension(labels=["Dim3", "Freq3"], values=[np.array([5, 6]), np.array([50, 60])]),
    ]
    exp_result_3d = RabiExperimentResult(qubit=0, data=data3d, dims=dims3d)
    with pytest.raises(NotImplementedError, match="3D and higher"):
        exp_result_3d.plot(save_to="./.tmp/")


def test_t1_plotting(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    if not _DEBUG_PLOTS:
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *args, **kwargs: None)

    rng = np.random.default_rng(seed=42)
    tau = np.arange(0, 3200, 100)
    decay = np.exp(-tau / 1000.0)
    noise = rng.normal(0, 0.02, size=(len(tau), 2))
    amplitudes = np.clip(decay[:, np.newaxis] + noise, 0, 1)
    dims = [Dimension(labels=["Time"], values=[tau])]
    t1_result = T1ExperimentResult(qubit=0, data=amplitudes, dims=dims)

    t1_result.plot(save_to="./.tmp/test_t1.png", db=False, fit=False)
    t1_result.plot(save_to="./.tmp/test_t1_db.png", db=True, fit=False)
    t1_result.plot(save_to="./.tmp/test_t1_fit.png", db=False, fit=True)
    t1_result.plot(save_to="./.tmp/test_t1_db_fit.png", db=True, fit=True)


def test_rabi_plotting(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    if not _DEBUG_PLOTS:
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *args, **kwargs: None)

    rng = np.random.default_rng(seed=42)
    drive_durations = np.arange(0, 500, 5)
    f_rabi = 0.01
    I = 0.5 + 0.5 * np.cos(2 * np.pi * f_rabi * drive_durations)
    Q = np.zeros_like(drive_durations, dtype=float)
    I += rng.normal(0, 0.01, size=I.shape)
    data_rabi = np.stack([I, Q], axis=-1)
    dims_rabi = [Dimension(labels=["Drive duration (ns)"], values=[drive_durations])]
    result_rabi = RabiExperimentResult(qubit=0, data=data_rabi, dims=dims_rabi)

    result_rabi.plot(save_to="./.tmp/test_rabi.png", db=False, fit=False)
    result_rabi.plot(save_to="./.tmp/test_rabi_db.png", db=True, fit=False)
    result_rabi.plot(save_to="./.tmp/test_rabi_fit.png", db=False, fit=True)
    result_rabi.plot(save_to="./.tmp/test_rabi_db_fit.png", db=True, fit=True)


def test_two_tones_at_flux_bias_plotting(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    if not _DEBUG_PLOTS:
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *args, **kwargs: None)

    rng = np.random.default_rng(seed=42)
    freqs = np.arange(2.0e8, 4.5e8, 2.5e6)
    f_qubit = 3.9e8
    linewidth = 15e6
    s21_mag = 0.65 + 0.55 / (1 + ((freqs - f_qubit) / (linewidth / 2)) ** 2)
    s21_mag += 0.1 / (1 + ((freqs - 2.9e8) / (5e6)) ** 2)
    s21_mag += rng.normal(0, 0.025, size=s21_mag.shape)
    phase = np.full_like(freqs, np.pi / 6)
    s21_complex = s21_mag * np.exp(1j * phase)
    data_at = np.stack([s21_complex.real, s21_complex.imag], axis=-1)
    dims_at = [Dimension(labels=["IF Frequency (Hz)"], values=[freqs])]
    result_at = TwoTonesAtFluxBiasExperimentResult(qubit=0, data=data_at, dims=dims_at)

    result_at.plot(save_to="./.tmp/test_two_tones_at.png", db=False, fit=False)
    result_at.plot(save_to="./.tmp/test_two_tones_at_db.png", db=True, fit=False)
    result_at.plot(save_to="./.tmp/test_two_tones_at_fit.png", db=False, fit=True)
    result_at.plot(save_to="./.tmp/test_two_tones_at_db_fit.png", db=True, fit=True)


def test_two_tones_vs_flux_bias_plotting(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    if not _DEBUG_PLOTS:
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *args, **kwargs: None)

    rng = np.random.default_rng(seed=42)
    freqs2d = np.arange(4.0e9, 6.0e9, 10e6)
    fluxes = np.arange(-0.5, 0.5, 0.02)
    f_max = 5.5e9
    f_q_vs_flux = f_max * np.sqrt(np.abs(np.cos(np.pi * fluxes)))
    linewidth = 20e6
    PHI, F = np.meshgrid(f_q_vs_flux, freqs2d, indexing="ij")
    s21_mag_2d = 1.0 - 0.7 / (1 + ((F - PHI) / (linewidth / 2)) ** 2)
    s21_mag_2d += rng.normal(0, 0.01, size=s21_mag_2d.shape)
    phase_2d = np.full_like(F, np.pi / 6) + 0.4 / (1 + ((F - PHI) / (linewidth / 2)) ** 2)
    s21_complex_2d = s21_mag_2d * np.exp(1j * phase_2d)
    data_vs = np.stack([s21_complex_2d.real, s21_complex_2d.imag], axis=-1)
    dims_vs = [
        Dimension(labels=["Flux bias (Φ₀)"], values=[fluxes]),
        Dimension(labels=["Drive frequency (Hz)"], values=[freqs2d]),
    ]
    result_vs = TwoTonesVsFluxBiasExperimentResult(qubit=0, data=data_vs, dims=dims_vs)

    result_vs.plot(save_to="./.tmp/test_two_tones_vs.png", db=False, fit=False)
    result_vs.plot(save_to="./.tmp/test_two_tones_vs_db.png", db=True, fit=False)
    result_vs.plot(save_to="./.tmp/test_two_tones_vs_fit.png", db=False, fit=True)
    result_vs.plot(save_to="./.tmp/test_two_tones_vs_db_fit.png", db=True, fit=True)


def test_t2_plotting(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    if not _DEBUG_PLOTS:
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *args, **kwargs: None)

    rng = np.random.default_rng(seed=42)
    tau = np.arange(0, 10, 0.05)
    T2 = 5.0
    detuning = 1.0
    I = 1.0 + 0.1 * np.exp(-tau / T2) * np.cos(2 * np.pi * detuning * tau)
    Q = np.zeros_like(tau, dtype=float)
    I += rng.normal(0, 0.002, size=I.shape)
    data_t2 = np.stack([I, Q], axis=-1)
    dims_t2 = [Dimension(labels=["Wait duration (μs)"], values=[tau])]
    result_t2 = T2ExperimentResult(qubit=0, data=data_t2, dims=dims_t2)

    result_t2.plot(save_to="./.tmp/test_t2.png", db=False, fit=False)
    result_t2.plot(save_to="./.tmp/test_t2_db.png", db=True, fit=False)
    result_t2.plot(save_to="./.tmp/test_t2_fit.png", db=False, fit=True)
    result_t2.plot(save_to="./.tmp/test_t2_db_fit.png", db=True, fit=True)
