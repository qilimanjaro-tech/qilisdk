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
)


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

    data = np.array([[1, 2], [3, 4]])
    dims = [Dimension(labels=["Freq", "Freq2"], values=[np.array([1, 2]), np.array([10, 20])])]
    exp_result = RabiExperimentResult(qubit=0, data=data, dims=dims)
    exp_result.plot(save_to="./")

    # now for 2d
    data2d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dims2d = [
        Dimension(labels=["Drive amplitude", "Freq"], values=[np.array([0.1, 0.2]), np.array([1, 2])]),
        Dimension(labels=["Delay time", "Freq2"], values=[np.array([10, 20]), np.array([100, 200])]),
    ]
    exp_result_2d = RabiExperimentResult(qubit=0, data=data2d, dims=dims2d)
    exp_result_2d.plot(save_to="./")

    # 3d should fail
    data3d = np.ones((2, 2, 2, 2))
    dims3d = [
        Dimension(labels=["Dim1", "Freq"], values=[np.array([1, 2]), np.array([10, 20])]),
        Dimension(labels=["Dim2", "Freq2"], values=[np.array([3, 4]), np.array([30, 40])]),
        Dimension(labels=["Dim3", "Freq3"], values=[np.array([5, 6]), np.array([50, 60])]),
    ]
    exp_result_3d = RabiExperimentResult(qubit=0, data=data3d, dims=dims3d)
    with pytest.raises(NotImplementedError, match="3D and higher"):
        exp_result_3d.plot(save_to="./")
