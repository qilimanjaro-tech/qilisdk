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
from pydantic import ValidationError

from qilisdk.analog import Z as pauli_z
from qilisdk.core import QTensor
from qilisdk.readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout


# SamplingReadout
class TestSamplingReadout:
    def test_keyword_constructor(self):
        ro = SamplingReadout(nshots=500)
        assert ro.nshots == 500

    def test_nshots_zero(self):
        ro = SamplingReadout(nshots=0)
        assert ro.nshots == 0

    def test_negative_nshots_raises(self):
        with pytest.raises(ValidationError):
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
        assert len(ro.qtensor_observables) == 1
        assert isinstance(ro.qtensor_observables[0], QTensor)

    def test_qtensor_observable_passthrough(self):
        qt = QTensor(np.array([[1, 0], [0, -1]], dtype=np.complex128))
        ro = ExpectationReadout(observables=[qt])
        assert ro.qtensor_observables[0] is qt

    def test_scale_observables(self):
        ro = ExpectationReadout(observables=[pauli_z(0)])
        ro.expand_observables(nqubits=2)
        assert ro.qtensor_observables[0].shape == (4, 4)

    def test_invalid_observable_type_raises(self):
        with pytest.raises(ValidationError):
            ExpectationReadout(observables=["not_an_observable"])

    def test_negative_nshots_raises(self):
        with pytest.raises(ValidationError):
            ExpectationReadout(observables=[pauli_z(0)], nshots=-1)


# StateTomographyReadout


class TestStateTomographyReadout:
    def test_default_constructor(self):
        ro = StateTomographyReadout()
        assert ro.state_tomography_method == "exact"

    def test_keyword_constructor(self):
        ro = StateTomographyReadout(state_tomography_method="exact")
        assert ro.state_tomography_method == "exact"


# ReadoutMethod base class


class TestReadoutMethodBase:
    def test_factory_sample(self):
        ro = ReadoutMethod.sample(nshots=42)
        assert isinstance(ro, SamplingReadout)
        assert ro.nshots == 42

    def test_factory_expectation_values(self):
        obs = [pauli_z(0)]
        ro = ReadoutMethod.expectation_values(observables=obs)
        assert isinstance(ro, ExpectationReadout)

    def test_factory_state_tomography(self):
        ro = ReadoutMethod.state_tomography()
        assert isinstance(ro, StateTomographyReadout)

    def test_factory_on_subclass_raises(self):
        with pytest.raises(TypeError, match="only available on ReadoutMethod"):
            SamplingReadout.sample(nshots=10)

        with pytest.raises(TypeError, match="only available on ReadoutMethod"):
            ExpectationReadout.expectation_values(observables=[pauli_z(0)])

        with pytest.raises(TypeError, match="only available on ReadoutMethod"):
            StateTomographyReadout.state_tomography()


class TestReadoutMethodIsChecks:
    def test_is_sample(self):
        assert SamplingReadout(nshots=10).is_sample() is True
        assert ExpectationReadout(observables=[pauli_z(0)]).is_sample() is False
        assert StateTomographyReadout().is_sample() is False

    def test_is_expectation_values(self):
        assert SamplingReadout(nshots=10).is_expectation_values() is False
        assert ExpectationReadout(observables=[pauli_z(0)]).is_expectation_values() is True
        assert StateTomographyReadout().is_expectation_values() is False

    def test_is_state_tomography(self):
        assert SamplingReadout(nshots=10).is_state_tomography() is False
        assert ExpectationReadout(observables=[pauli_z(0)]).is_state_tomography() is False
        assert StateTomographyReadout().is_state_tomography() is True

    def test_is_valid(self):
        assert SamplingReadout(nshots=10).is_valid() is True
        assert ExpectationReadout(observables=[pauli_z(0)]).is_valid() is True
        assert StateTomographyReadout().is_valid() is True
