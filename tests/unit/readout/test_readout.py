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

import pytest

from qilisdk.readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout


def test_readout_factories_still_build_expected_readout_types() -> None:
    sampling = ReadoutMethod.sample(nshots=123)
    assert isinstance(sampling, SamplingReadout)
    assert sampling.nshots == 123

    expectation = ReadoutMethod.expectation_values(observables=[], nshots=3)
    assert isinstance(expectation, ExpectationReadout)
    assert expectation.nshots == 3

    tomography = ReadoutMethod.state_tomography()
    assert isinstance(tomography, StateTomographyReadout)
    assert tomography.state_tomography_method == "exact"


def test_subclasses_cannot_call_readout_factory_methods() -> None:
    with pytest.raises(AttributeError):
        SamplingReadout.sample(nshots=10)

    with pytest.raises(AttributeError):
        ExpectationReadout.expectation_values(observables=[], nshots=0)

    with pytest.raises(AttributeError):
        StateTomographyReadout.state_tomography(method="exact")


def test_specific_readouts_allow_positional_arguments_when_unambiguous() -> None:
    sampling = SamplingReadout(10)
    assert sampling.nshots == 10

    expectation = ExpectationReadout([], 2)
    assert expectation.observables == []
    assert expectation.nshots == 2

    tomography = StateTomographyReadout("exact")
    assert tomography.state_tomography_method == "exact"


def test_specific_readouts_raise_helpful_errors_for_bad_positional_usage() -> None:
    with pytest.raises(TypeError, match="at most 1 positional argument"):
        SamplingReadout(1, 2)

    with pytest.raises(TypeError, match="multiple values for 'nshots'"):
        SamplingReadout(1, nshots=2)

    with pytest.raises(TypeError, match="at most 2 positional argument"):
        ExpectationReadout([], 1, 2)

    with pytest.raises(TypeError, match="multiple values for 'observables'"):
        ExpectationReadout([], observables=[])

    with pytest.raises(TypeError, match="at most 1 positional argument"):
        StateTomographyReadout("exact", "extra")
