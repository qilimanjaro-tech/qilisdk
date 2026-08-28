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

from qilisdk.analog import Z as pauli_z
from qilisdk.readout import Readout


class TestWithSamplingDoesNotMutateReceiver:
    """`Readout.with_sampling` must populate a copy and leave the receiver untouched."""

    def test_receiver_sampling_slot_stays_empty(self):
        base = Readout()
        base.with_sampling(nshots=1000)
        assert base.sampling is None

    def test_returns_a_new_instance(self):
        base = Readout()
        spec = base.with_sampling(nshots=1000)
        assert spec is not base

    def test_returned_instance_carries_the_sampling_readout(self):
        spec = Readout().with_sampling(nshots=750, expand_samples=False)
        assert spec.sampling is not None
        assert spec.sampling.nshots == 750
        assert spec.sampling.expand_samples is False

    def test_receiver_stays_falsy_and_empty(self):
        base = Readout()
        spec = base.with_sampling(nshots=1000)
        assert bool(base) is False
        assert base.to_list() == []
        assert bool(spec) is True
        assert spec.to_list() == [spec.sampling]

    def test_base_can_be_reused_for_several_specifications(self):
        # Regression: while `with_sampling` mutated `self`, the second call raised
        # "Sampling readout already set in this specification." instead of branching.
        base = Readout()
        hundred = base.with_sampling(nshots=100)
        two_hundred = base.with_sampling(nshots=200)

        assert hundred is not two_hundred
        assert hundred.sampling is not None
        assert two_hundred.sampling is not None
        assert hundred.sampling.nshots == 100
        assert two_hundred.sampling.nshots == 200
        assert base.sampling is None

    def test_other_slots_are_carried_over_to_the_copy(self):
        base = Readout().with_expectation(observables=[pauli_z(0)], nshots=0)
        spec = base.with_sampling(nshots=500)

        assert spec.expectation is base.expectation
        assert spec.sampling is not None
        assert spec.sampling.nshots == 500

    def test_receiver_is_untouched_after_a_full_chain(self):
        base = Readout()
        spec = base.with_sampling(nshots=500).with_expectation(observables=[pauli_z(0)]).with_state_tomography()

        assert base.sampling is None
        assert base.expectation is None
        assert base.state_tomography is None
        assert spec.sampling is not None
        assert spec.expectation is not None
        assert spec.state_tomography is not None

    def test_duplicate_sampling_on_the_same_specification_still_raises(self):
        spec = Readout().with_sampling(nshots=100)
        with pytest.raises(ValueError, match=r"Sampling readout already set in this specification."):
            spec.with_sampling(nshots=200)
