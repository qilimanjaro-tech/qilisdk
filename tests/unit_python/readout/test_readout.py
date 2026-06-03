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

import numpy as np
import pytest

from qilisdk.analog import Z as pauli_z
from qilisdk.core import QTensor
from qilisdk.readout import ExpectationReadout, SamplingReadout, StateTomographyReadout


# SamplingReadout
class TestSamplingReadout:
    def test_keyword_constructor(self):
        ro = SamplingReadout(nshots=500)
        assert ro.nshots == 500

    def test_nshots_zero(self):
        with pytest.raises(ValueError, match="The number of shots has to be a positive integer"):
            SamplingReadout(nshots=0)

    def test_negative_nshots_raises(self):
        with pytest.raises(ValueError, match="The number of shots has to be a positive integer"):
            SamplingReadout(nshots=-1)


# ExpectationReadout


class TestExpectationReadout:
    def test_keyword_constructor(self):
        obs = [pauli_z(0)]
        ro = ExpectationReadout(observables=obs)
        assert ro.observables == obs
        assert ro.nshots == 0

    def test_qtensor_observables_auto_set(self):
        obs = [pauli_z(0)]
        ro = ExpectationReadout(observables=obs)
        expanded = ro.expand_observables(nqubits=1)
        assert len(expanded) == 1
        assert isinstance(expanded[0], QTensor)

    def test_qtensor_observable_passthrough(self):
        qt = QTensor(np.array([[1, 0], [0, -1]], dtype=np.complex128))
        ro = ExpectationReadout(observables=[qt])
        assert ro.expand_observables(nqubits=1)[0] == qt

    def test_scale_observables(self):
        ro = ExpectationReadout(observables=[pauli_z(0)])
        assert ro.expand_observables(nqubits=2)[0].shape == (4, 4)

    def test_invalid_observable_type_raises(self):
        with pytest.raises(ValueError, match="Invalid Observable"):
            ExpectationReadout(observables=["not_an_observable"])

    def test_negative_nshots_raises(self):
        with pytest.raises(ValueError, match="The number of shots has to be a positive integer"):
            ExpectationReadout(observables=[pauli_z(0)], nshots=-1)


# StateTomographyReadout


class TestStateTomographyReadout:
    def test_default_constructor(self):
        ro = StateTomographyReadout()
        assert ro.method == "exact"

    def test_keyword_constructor(self):
        ro = StateTomographyReadout(method="exact")
        assert ro.method == "exact"


# ReadoutMethod base class


class TestReadoutMethodIsChecks:
    def test_is_sample(self):
        assert SamplingReadout(nshots=10).is_sampling_readout() is True
        assert ExpectationReadout(observables=[pauli_z(0)]).is_sampling_readout() is False
        assert StateTomographyReadout().is_sampling_readout() is False

    def test_is_expectation_values(self):
        assert SamplingReadout(nshots=10).is_expectation_readout() is False
        assert ExpectationReadout(observables=[pauli_z(0)]).is_expectation_readout() is True
        assert StateTomographyReadout().is_expectation_readout() is False

    def test_is_state_tomography(self):
        assert SamplingReadout(nshots=10).is_state_tomography_readout() is False
        assert ExpectationReadout(observables=[pauli_z(0)]).is_state_tomography_readout() is False
        assert StateTomographyReadout().is_state_tomography_readout() is True
