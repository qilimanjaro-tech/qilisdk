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

import math

import numpy as np
import pytest

from qilisdk.analog import Z as pauli_z
from qilisdk.backends.backend import Backend
from qilisdk.core import QTensor, ket
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout
from qilisdk.readout.readout_result import (
    ExpectationReadoutResult,
    ReadoutCompositeResults,
    SamplingReadoutResult,
    StateTomographyReadoutResult,
    _real_if_close,
    _samples_from_probabilities,
    _samples_from_state,
    has_expectation_values,
    has_sampling,
    has_state_tomography,
)

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────


class TestRealIfClose:
    def test_real_number_passthrough(self):
        assert _real_if_close(math.pi) == math.pi

    def test_complex_with_zero_imag(self):
        result = _real_if_close(complex(2.5, 0.0))
        assert isinstance(result, float)
        assert np.isclose(result, 2.5)

    def test_complex_with_tiny_imag(self):
        result = _real_if_close(complex(2.5, 1e-15))
        assert isinstance(result, float)
        assert np.isclose(result, 2.5)

    def test_complex_with_large_imag_kept(self):
        c = complex(2.5, 1.0)
        assert _real_if_close(c) == c


class TestSamplesFromProbabilities:
    def test_deterministic_distribution(self):
        probs = {"0": 1.0, "1": 0.0}
        samples = _samples_from_probabilities(probs, nshots=100, seed=42)
        assert samples == {"0": 100}

    def test_total_shots_correct(self):
        probs = {"0": 0.5, "1": 0.5}
        samples = _samples_from_probabilities(probs, nshots=1000, seed=42)
        assert sum(samples.values()) == 1000

    def test_seed_reproducibility(self):
        probs = {"0": 0.5, "1": 0.5}
        s1 = _samples_from_probabilities(probs, nshots=100, seed=123)
        s2 = _samples_from_probabilities(probs, nshots=100, seed=123)
        assert s1 == s2


class TestSamplesFromState:
    def test_ket_zero(self):
        samples = _samples_from_state(ket(0), nshots=50, seed=42)
        assert samples == {"0": 50}


# SamplingReadoutResult


class TestSamplingReadoutResult:
    def test_from_samples(self):
        result = SamplingReadoutResult.from_samples(samples={"0": 60, "1": 40})
        assert result.samples == {"0": 60, "1": 40}
        assert np.isclose(result.probabilities["0"], 0.6)
        assert np.isclose(result.probabilities["1"], 0.4)

    def test_from_state(self):
        readout = SamplingReadout(nshots=100)
        result = SamplingReadoutResult.from_state(sampling_readout=readout, state=ket(0))
        assert sum(result.samples.values()) == 100
        assert "0" in result.samples

    def test_init_no_samples_raises(self):
        with pytest.raises(ValueError, match="samples are not provided"):
            SamplingReadoutResult(samples=None, probabilities=None)

    def test_get_probability(self):
        result = SamplingReadoutResult.from_samples(samples={"00": 60, "01": 40})
        assert np.isclose(result.get_probability("00"), 0.6)
        assert np.isclose(result.get_probability("11"), 0.0)

    def test_get_probabilities_top_n(self):
        result = SamplingReadoutResult.from_samples(samples={"00": 60, "01": 30, "10": 10})
        top = result.get_probabilities(n=2)
        assert len(top) == 2
        assert top[0][0] == "00"

    def test_get_probabilities_all(self):
        result = SamplingReadoutResult.from_samples(samples={"0": 60, "1": 40})
        all_probs = result.get_probabilities()
        assert len(all_probs) == 2

    def test_repr(self):
        result = SamplingReadoutResult.from_samples(samples={"0": 10})
        assert "Sampling Results" in repr(result)
        assert "nshots=10" in repr(result)

    def test_mismatched_bitstring_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            SamplingReadoutResult.from_samples(samples={"0": 50, "01": 50})


# ExpectationReadoutResult


class TestExpectationReadoutResult:
    def test_from_state(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        result = ExpectationReadoutResult.from_state(expectation_readout=readout, state=ket(0))
        assert len(result.expectation_values) == 1
        assert np.isclose(result.expectation_values[0], 1.0)

    def test_init_from_expectation_values(self):
        result = ExpectationReadoutResult(expectation_values=[0.5])
        assert result.expectation_values == [0.5]

    def test_ket_one_z_expectation(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        result = ExpectationReadoutResult.from_state(expectation_readout=readout, state=ket(1))
        assert np.isclose(result.expectation_values[0], -1.0)

    def test_superposition_z_expectation(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        state = (ket(0) + ket(1)).unit()
        result = ExpectationReadoutResult.from_state(expectation_readout=readout, state=state)
        assert np.isclose(result.expectation_values[0], 0.0, atol=1e-10)

    def test_repr(self):
        result = ExpectationReadoutResult(expectation_values=[1.0])
        assert "Expectation Value Results" in repr(result)


# StateTomographyReadoutResult


class TestStateTomographyReadoutResult:
    def test_basic_construction(self):
        result = StateTomographyReadoutResult(state=ket(0))
        assert result.state is not None
        assert result.probabilities is not None

    def test_probabilities_computed(self):
        result = StateTomographyReadoutResult(state=ket(0))
        assert np.isclose(result.probabilities["0"], 1.0)
        assert np.isclose(result.probabilities["1"], 0.0)

    def test_get_probability(self):
        result = StateTomographyReadoutResult(state=ket(0))
        assert np.isclose(result.get_probability("0"), 1.0)
        assert np.isclose(result.get_probability("1"), 0.0)

    def test_get_probabilities_top_n(self):
        state = (ket(0) + ket(1)).unit()
        result = StateTomographyReadoutResult(state=state)
        top = result.get_probabilities(n=1)
        assert len(top) == 1

    def test_repr(self):
        result = StateTomographyReadoutResult(state=ket(0))
        assert "State Tomography Results" in repr(result)

    def test_density_matrix_state(self):
        state = ket(0).to_density_matrix()
        result = StateTomographyReadoutResult(state=state)
        assert np.isclose(result.probabilities["0"], 1.0)


# ReadoutCompositeResults


class TestReadoutCompositeResults:
    @pytest.fixture
    def sampling_result(self):
        return SamplingReadoutResult.from_samples(samples={"0": 60, "1": 40})

    @pytest.fixture
    def expectation_result(self):
        return ExpectationReadoutResult(expectation_values=[1.0])

    @pytest.fixture
    def tomography_result(self):
        return StateTomographyReadoutResult(state=ket(0))

    def test_sampling_field(self, sampling_result):
        composite = ReadoutCompositeResults(sampling=sampling_result)
        assert has_sampling(composite)
        assert composite.sampling.samples == {"0": 60, "1": 40}

    def test_sampling_missing(self, expectation_result):
        composite = ReadoutCompositeResults(expectation_values=expectation_result)
        assert not has_sampling(composite)

    def test_expectation_field(self, expectation_result):
        composite = ReadoutCompositeResults(expectation_values=expectation_result)
        assert has_expectation_values(composite)
        assert composite.expectation_values.expectation_values == [1.0]

    def test_expectation_missing(self, sampling_result):
        composite = ReadoutCompositeResults(sampling=sampling_result)
        assert not has_expectation_values(composite)

    def test_tomography_field(self, tomography_result):
        composite = ReadoutCompositeResults(state_tomography=tomography_result)
        assert has_state_tomography(composite)
        assert composite.state_tomography.state is not None

    def test_tomography_missing(self, sampling_result):
        composite = ReadoutCompositeResults(sampling=sampling_result)
        assert not has_state_tomography(composite)

    def test_empty_composite(self):
        composite = ReadoutCompositeResults()
        assert not has_sampling(composite)
        assert not has_expectation_values(composite)
        assert not has_state_tomography(composite)

    def test_all_results_combined(self, sampling_result, expectation_result, tomography_result):
        composite = ReadoutCompositeResults(
            sampling=sampling_result,
            expectation_values=expectation_result,
            state_tomography=tomography_result,
        )
        assert has_sampling(composite)
        assert has_expectation_values(composite)
        assert has_state_tomography(composite)
        assert composite.sampling.samples == {"0": 60, "1": 40}
        assert composite.expectation_values.expectation_values == [1.0]
        assert composite.state_tomography.state is not None

    def test_from_dict(self, sampling_result, expectation_result, tomography_result):
        composite = ReadoutCompositeResults.from_dict(
            {
                "sampling": sampling_result,
                "expectation_values": expectation_result,
                "state_tomography": tomography_result,
            }
        )
        assert has_sampling(composite)
        assert has_expectation_values(composite)
        assert has_state_tomography(composite)

    def test_from_dict_type_validation(self, sampling_result):
        with pytest.raises(TypeError, match="sampling must be SamplingReadoutResult"):
            ReadoutCompositeResults.from_dict({"sampling": "not_a_result"})

    def test_repr(self, sampling_result):
        composite = ReadoutCompositeResults(sampling=sampling_result)
        text = repr(composite)
        assert isinstance(text, str)
        assert len(text) > 0


class TestSamplingReadoutResultEdgeCases:
    def test_init_no_samples_raises(self):
        with pytest.raises(ValueError, match="samples are not provided"):
            SamplingReadoutResult(samples=None, probabilities=None)

    def test_from_samples_empty_raises(self):
        with pytest.raises(ValueError, match="samples are not provided"):
            SamplingReadoutResult.from_samples(samples={})


class TestStateTomographyEdgeCases:
    def test_get_probabilities(self):
        result = StateTomographyReadoutResult(state=ket(0))
        top = result.get_probabilities(n=1)
        assert len(top) == 1
        assert top[0][0] == "0"

    def test_repr(self):
        result = StateTomographyReadoutResult(state=ket(0))
        assert "State Tomography" in repr(result)


# FunctionalResult integration


def _make_sampling_composite(nshots: int, samples: dict[str, int]) -> ReadoutCompositeResults:
    return ReadoutCompositeResults(sampling=SamplingReadoutResult.from_samples(samples=samples))


def _make_tomography_composite(state: QTensor) -> ReadoutCompositeResults:
    return ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=state))


def _make_expectation_composite(expectation_values: list[float]) -> ReadoutCompositeResults:
    return ReadoutCompositeResults(expectation_values=ExpectationReadoutResult(expectation_values=expectation_values))


class TestFunctionalResult:
    def test_samples(self):
        result = FunctionalResult(readout_results=_make_sampling_composite(100, {"0": 100}))
        assert result.get_samples() == {"0": 100}

    def test_state(self):
        result = FunctionalResult(readout_results=_make_tomography_composite(ket(0)))
        assert result.get_state() is not None

    def test_final_expectation_values(self):
        result = FunctionalResult(readout_results=_make_expectation_composite([1.0]))
        assert result.get_expectation_values() == [1.0]

    def test_final_probabilities_from_sampling(self):
        result = FunctionalResult(readout_results=_make_sampling_composite(100, {"0": 70, "1": 30}))
        assert np.isclose(result.get_probabilities()["0"], 0.7)

    def test_final_probabilities_from_tomography(self):
        result = FunctionalResult(readout_results=_make_tomography_composite(ket(0)))
        assert np.isclose(result.get_probabilities()["0"], 1.0)

    def test_multiple_readout_types(self):
        sampling = SamplingReadoutResult.from_samples(samples={"0": 100})
        tomography = StateTomographyReadoutResult(state=ket(0))
        expectation = ExpectationReadoutResult.from_expectations(expectation_values=[1.0])
        result = FunctionalResult(
            readout_results=ReadoutCompositeResults(
                sampling=sampling, state_tomography=tomography, expectation_values=expectation
            )
        )
        assert result.has_samples()
        assert result.has_state()
        assert result.has_expectation_values()
        assert result.has_probabilities()

    def test_intermediate_results(self):
        final = _make_tomography_composite(ket(0))
        inter1 = _make_tomography_composite(ket(1))
        inter2 = _make_tomography_composite((ket(0) + ket(1)).unit())
        result = FunctionalResult(readout_results=final, intermediate_results=[inter1, inter2])
        assert len(result.intermediate_states) == 3  # 2 intermediate + 1 final
        assert len(result.intermediate_results) == 2

    def test_states_no_intermediate_raises(self):
        result = FunctionalResult(readout_results=_make_tomography_composite(ket(0)))
        with pytest.raises(
            ValueError, match=r"Can't find intermediate states because intermediate Results were not stored."
        ):
            _ = result.intermediate_states

    def test_samples_intermediate(self):
        final = _make_sampling_composite(10, {"0": 10})
        inter = _make_sampling_composite(10, {"1": 10})
        result = FunctionalResult(readout_results=final, intermediate_results=[inter])
        assert len(result.intermediate_samples) == 2

    def test_len_no_intermediate(self):
        result = FunctionalResult(readout_results=_make_sampling_composite(10, {"0": 10}))
        assert len(result) == 1

    def test_len_with_intermediate(self):
        cr = _make_sampling_composite(10, {"0": 10})
        result = FunctionalResult(readout_results=cr, intermediate_results=[cr, cr])
        assert len(result) == 3

    def test_iter(self):
        cr = _make_sampling_composite(10, {"0": 10})
        result = FunctionalResult(readout_results=cr, intermediate_results=[cr])
        items = list(result)
        assert len(items) == 2

    def test_repr(self):
        result = FunctionalResult(readout_results=_make_sampling_composite(10, {"0": 10}))
        assert "Functional Results" in repr(result)

    def test_intermediate_probabilities(self):
        final = _make_tomography_composite(ket(0))
        inter = _make_tomography_composite(ket(1))
        result = FunctionalResult(readout_results=final, intermediate_results=[inter])
        probs = result.intermediate_probabilities
        assert len(probs) == 2
        assert np.isclose(probs[0]["1"], 1.0)
        assert np.isclose(probs[1]["0"], 1.0)

    def test_intermediate_probabilities_no_intermediate_raises(self):
        result = FunctionalResult(readout_results=_make_tomography_composite(ket(0)))
        with pytest.raises(ValueError, match="intermediate probabilities"):
            _ = result.intermediate_probabilities

    def test_intermediate_expectation_values(self):
        final = _make_expectation_composite([1.0])
        inter = _make_expectation_composite([-1.0])
        result = FunctionalResult(readout_results=final, intermediate_results=[inter])
        vals = result.intermediate_expectation_values
        assert len(vals) == 2
        assert np.isclose(vals[0][0], -1.0)
        assert np.isclose(vals[1][0], 1.0)

    def test_intermediate_expectation_values_no_intermediate_raises(self):
        result = FunctionalResult(readout_results=_make_expectation_composite([1.0]))
        with pytest.raises(ValueError, match="intermediate expectation values"):
            _ = result.intermediate_expectation_values

    def test_intermediate_expectation_values_no_readout_returns_empty(self):
        cr = _make_sampling_composite(10, {"0": 10})
        result = FunctionalResult(readout_results=cr, intermediate_results=[cr])
        assert result.intermediate_expectation_values == []

    def test_intermediate_samples_no_readout_returns_empty(self):
        cr = _make_tomography_composite(ket(0))
        result = FunctionalResult(readout_results=cr, intermediate_results=[cr])
        assert result.intermediate_samples == []

    def test_getitem_returns_intermediate(self):
        final = _make_sampling_composite(10, {"0": 10})
        inter = _make_sampling_composite(10, {"1": 10})
        result = FunctionalResult(readout_results=final, intermediate_results=[inter])
        assert has_sampling(result[0])
        assert result[0].sampling.samples == {"1": 10}
        assert has_sampling(result[1])
        assert result[1].sampling.samples == {"0": 10}

    def test_getitem_invalid_index_raises(self):
        result = FunctionalResult(readout_results=_make_sampling_composite(10, {"0": 10}))
        with pytest.raises(IndexError, match="out of range"):
            _ = result[5]

    def test_readout_results_property(self):
        result = FunctionalResult(readout_results=_make_sampling_composite(10, {"0": 10}))
        assert result.readout_results is not None

    def test_intermediate_results_empty_when_not_stored(self):
        result = FunctionalResult(readout_results=_make_sampling_composite(10, {"0": 10}))
        assert result.intermediate_results == []


# Backend._construct_results_list


class TestBackendConstructResultsList:
    def test_sampling_readout(self):
        results = Backend._construct_results_list(final_state=ket(0), readout=[SamplingReadout(nshots=100)])
        assert has_sampling(results)
        assert isinstance(results.sampling, SamplingReadoutResult)
        assert sum(results.sampling.samples.values()) == 100

    def test_state_tomography_readout(self):
        results = Backend._construct_results_list(final_state=ket(0), readout=[StateTomographyReadout()])
        assert has_state_tomography(results)
        assert isinstance(results.state_tomography, StateTomographyReadoutResult)
        assert results.state_tomography.state is not None

    def test_expectation_readout(self):
        results = Backend._construct_results_list(
            final_state=ket(0), readout=[ExpectationReadout(observables=[pauli_z(0)])]
        )
        assert has_expectation_values(results)
        assert isinstance(results.expectation_values, ExpectationReadoutResult)
        assert np.isclose(results.expectation_values.expectation_values[0], 1.0)

    def test_multiple_readouts(self):
        results = Backend._construct_results_list(
            final_state=ket(0),
            readout=[
                SamplingReadout(nshots=100),
                StateTomographyReadout(),
                ExpectationReadout(observables=[pauli_z(0)]),
            ],
        )
        assert has_sampling(results)
        assert has_state_tomography(results)
        assert has_expectation_values(results)
        assert isinstance(results.sampling, SamplingReadoutResult)
        assert isinstance(results.state_tomography, StateTomographyReadoutResult)
        assert isinstance(results.expectation_values, ExpectationReadoutResult)

    def test_unsupported_readout_raises(self):
        with pytest.raises(ValueError, match="Unsupported Readout Method"):
            Backend._construct_results_list(final_state=ket(0), readout=[ReadoutMethod()])
