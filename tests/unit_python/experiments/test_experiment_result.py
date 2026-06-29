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

from qilisdk.experiments import Dimension, ExperimentResult

# Set this to True if you want to actually save the plots during testing
_DEBUG_PLOTS = False


def test_dimension_initialization():
    dim = Dimension(labels=["Drive amplitude"], values=[np.array([0.1, 0.2, 0.3])])
    assert dim.labels == ["Drive amplitude"]
    assert np.array_equal(dim.values[0], np.array([0.1, 0.2, 0.3]))


def test_experiment_result_init():
    data = np.array([[1, 2], [3, 4]])
    qubit = 0
    averages = 1000
    dims = [Dimension(labels=["Freq"], values=[np.array([1, 2])])]

    exp_result = ExperimentResult(qubit=qubit, averages=averages, data=data, dims=dims)

    assert exp_result.qubit == qubit
    assert exp_result.averages == averages
    assert np.array_equal(exp_result.data, data)
    assert exp_result.dims == dims


def test_experiment_printing():
    data = np.array([[1, 2], [3, 4]])
    qubit = 0
    averages = 1000
    dims = [Dimension(labels=["Freq"], values=[np.array([1, 2])])]

    exp_result = ExperimentResult(qubit=qubit, averages=averages, data=data, dims=dims)

    expected_str = (
        "ExperimentResult(qubit=0, averages=1000, data=[[1 2]\n [3 4]], "
        "dims=[Dimension(labels=['Freq'], values=[array([1, 2])])])"
    )
    assert str(exp_result) == expected_str
