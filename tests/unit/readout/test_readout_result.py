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
from qilisdk.core import ket, tensor_prod
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout
from qilisdk.readout.readout_result import (
    ExpectationReadoutResult,
    ReadoutCompositeResults,
    SamplingReadoutResult,
    StateTomographyReadoutResult,
    _probabilities_from_state,
    _real_if_close,
    _samples_from_probabilities,
    _samples_from_state,
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


class TestProbabilitiesFromState:
    def test_ket_state(self):
        state = ket(0)
        probs = _probabilities_from_state(state)
        assert probs == {"0": 1.0, "1": 0.0}

    def test_superposition_ket(self):
        state = (ket(0) + ket(1)).unit()
        probs = _probabilities_from_state(state)
        assert np.isclose(probs["0"], 0.5)
        assert np.isclose(probs["1"], 0.5)

    def test_density_matrix(self):
        state = ket(0).to_density_matrix()
        probs = _probabilities_from_state(state)
        assert np.isclose(probs["0"], 1.0)
        assert np.isclose(probs["1"], 0.0)

    def test_bra_state(self):
        state = ket(0).adjoint()
        probs = _probabilities_from_state(state)
        assert np.isclose(probs["0"], 1.0)

    def test_check_normalization(self):
        state = ket(0)
        _probabilities_from_state(state, check_normalization=True)

    def test_two_qubit_state(self):

        state = tensor_prod([ket(0), ket(1)])
        probs = _probabilities_from_state(state)
        assert len(probs) == 4
        assert np.isclose(probs["01"], 1.0)
        assert np.isclose(probs["00"], 0.0)


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
    def test_init_from_samples(self):
        readout = SamplingReadout(nshots=100)
        result = SamplingReadoutResult(readout=readout, samples={"0": 60, "1": 40})
        assert result.samples == {"0": 60, "1": 40}
        assert np.isclose(result.probabilities["0"], 0.6)
        assert np.isclose(result.probabilities["1"], 0.4)

    def test_init_from_state(self):
        readout = SamplingReadout(nshots=100)
        result = SamplingReadoutResult(readout=readout, state=ket(0))
        assert sum(result.samples.values()) == 100
        assert "0" in result.samples

    def test_init_no_args_raises(self):
        readout = SamplingReadout(nshots=100)
        with pytest.raises(ValueError, match="samples and state are not provided"):
            SamplingReadoutResult(readout=readout)

    def test_init_from_probabilities(self):
        readout = SamplingReadout(nshots=100)
        result = SamplingReadoutResult(readout=readout, samples=None, probabilities={"0": 0.7, "1": 0.3})
        assert sum(result.samples.values()) == 100
        assert result.probabilities == {"0": 0.7, "1": 0.3}

    def test_get_probability(self):
        readout = SamplingReadout(nshots=100)
        result = SamplingReadoutResult(readout=readout, samples={"00": 60, "01": 40})
        assert np.isclose(result.get_probability("00"), 0.6)
        assert np.isclose(result.get_probability("11"), 0.0)

    def test_get_probabilities_top_n(self):
        readout = SamplingReadout(nshots=100)
        result = SamplingReadoutResult(readout=readout, samples={"00": 60, "01": 30, "10": 10})
        top = result.get_probabilities(n=2)
        assert len(top) == 2
        assert top[0][0] == "00"

    def test_get_probabilities_all(self):
        readout = SamplingReadout(nshots=100)
        result = SamplingReadoutResult(readout=readout, samples={"0": 60, "1": 40})
        all_probs = result.get_probabilities()
        assert len(all_probs) == 2

    def test_readout_property(self):
        readout = SamplingReadout(nshots=42)
        result = SamplingReadoutResult(readout=readout, samples={"0": 42})
        assert result.readout is readout

    def test_repr(self):
        readout = SamplingReadout(nshots=10)
        result = SamplingReadoutResult(readout=readout, samples={"0": 10})
        assert "Sampling Results" in repr(result)
        assert "nshots=10" in repr(result)

    def test_mismatched_bitstring_lengths_raises(self):
        readout = SamplingReadout(nshots=100)
        with pytest.raises(ValueError, match="same length"):
            SamplingReadoutResult(readout=readout, samples={"0": 50, "01": 50})


# ExpectationReadoutResult


class TestExpectationReadoutResult:
    def test_init_from_state(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        result = ExpectationReadoutResult(readout=readout, state=ket(0))
        assert len(result.expected_values) == 1
        assert np.isclose(result.expected_values[0], 1.0)

    def test_init_from_expected_values(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        result = ExpectationReadoutResult(readout=readout, expected_values=[0.5])
        assert result.expected_values == [0.5]

    def test_init_no_args_raises(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        with pytest.raises(ValueError, match="expected values and the state are not provided"):
            ExpectationReadoutResult(readout=readout)

    def test_readout_property(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        result = ExpectationReadoutResult(readout=readout, expected_values=[1.0])
        assert result.readout is readout

    def test_ket_one_z_expectation(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        result = ExpectationReadoutResult(readout=readout, state=ket(1))
        assert np.isclose(result.expected_values[0], -1.0)

    def test_superposition_z_expectation(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        state = (ket(0) + ket(1)).unit()
        result = ExpectationReadoutResult(readout=readout, state=state)
        assert np.isclose(result.expected_values[0], 0.0, atol=1e-10)

    def test_repr(self):
        readout = ExpectationReadout(observables=[pauli_z(0)])
        result = ExpectationReadoutResult(readout=readout, expected_values=[1.0])
        assert "Expectation Value Results" in repr(result)

    def test_repr_with_nshots(self):
        readout = ExpectationReadout(observables=[pauli_z(0)], nshots=100)
        result = ExpectationReadoutResult(readout=readout, expected_values=[1.0])
        assert "nshots = 100" in repr(result)


# StateTomographyReadoutResult


class TestStateTomographyReadoutResult:
    def test_basic_construction(self):
        readout = StateTomographyReadout()
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        assert result.final_state is not None
        assert result.probabilities is not None

    def test_probabilities_computed(self):
        readout = StateTomographyReadout(compute_probabilities=True)
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        assert np.isclose(result.probabilities["0"], 1.0)
        assert np.isclose(result.probabilities["1"], 0.0)

    def test_probabilities_not_computed(self):
        readout = StateTomographyReadout(compute_probabilities=False)
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        assert result.probabilities is None

    def test_get_probability(self):
        readout = StateTomographyReadout()
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        assert np.isclose(result.get_probability("0"), 1.0)
        assert np.isclose(result.get_probability("1"), 0.0)

    def test_get_probability_not_computed_raises(self):
        readout = StateTomographyReadout(compute_probabilities=False)
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        with pytest.raises(ValueError, match="compute_probabilities"):
            result.get_probability("0")

    def test_get_probabilities_top_n(self):
        readout = StateTomographyReadout()
        state = (ket(0) + ket(1)).unit()
        result = StateTomographyReadoutResult(readout=readout, final_state=state)
        top = result.get_probabilities(n=1)
        assert len(top) == 1

    def test_get_probabilities_not_computed_raises(self):
        readout = StateTomographyReadout(compute_probabilities=False)
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        with pytest.raises(ValueError, match="compute_probabilities"):
            result.get_probabilities()

    def test_readout_property(self):
        readout = StateTomographyReadout()
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        assert result.readout is readout

    def test_repr(self):
        readout = StateTomographyReadout()
        result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
        assert "State Tomography Results" in repr(result)

    def test_density_matrix_state(self):
        readout = StateTomographyReadout()
        state = ket(0).to_density_matrix()
        result = StateTomographyReadoutResult(readout=readout, final_state=state)
        assert np.isclose(result.probabilities["0"], 1.0)


# ReadoutCompositeResults


class TestReadoutCompositeResults:
    @pytest.fixture
    def sampling_result(self):
        return SamplingReadoutResult(readout=SamplingReadout(nshots=100), samples={"0": 60, "1": 40})

    @pytest.fixture
    def expectation_result(self):
        return ExpectationReadoutResult(readout=ExpectationReadout(observables=[pauli_z(0)]), expected_values=[1.0])

    @pytest.fixture
    def tomography_result(self):
        return StateTomographyReadoutResult(readout=StateTomographyReadout(), final_state=ket(0))

    def test_samples_property(self, sampling_result):
        composite = ReadoutCompositeResults([sampling_result])
        assert composite.samples == {"0": 60, "1": 40}

    def test_samples_missing_raises(self, expectation_result):
        composite = ReadoutCompositeResults([expectation_result])
        with pytest.raises(ValueError, match="no Sampling readout"):
            _ = composite.samples

    def test_probabilities_from_sampling(self, sampling_result):
        composite = ReadoutCompositeResults([sampling_result])
        assert np.isclose(composite.probabilities["0"], 0.6)

    def test_probabilities_from_tomography(self, tomography_result):
        composite = ReadoutCompositeResults([tomography_result])
        assert np.isclose(composite.probabilities["0"], 1.0)

    def test_probabilities_missing_raises(self, expectation_result):
        composite = ReadoutCompositeResults([expectation_result])
        with pytest.raises(ValueError, match="no Sampling/State Tomography"):
            _ = composite.probabilities

    def test_final_state_property(self, tomography_result):
        composite = ReadoutCompositeResults([tomography_result])
        assert composite.final_state is not None

    def test_final_state_missing_raises(self, sampling_result):
        composite = ReadoutCompositeResults([sampling_result])
        with pytest.raises(ValueError, match="no State Tomography"):
            _ = composite.final_state

    def test_expected_values_property(self, expectation_result):
        composite = ReadoutCompositeResults([expectation_result])
        assert composite.expected_values == [1.0]

    def test_expected_values_missing_raises(self, sampling_result):
        composite = ReadoutCompositeResults([sampling_result])
        with pytest.raises(ValueError, match="no Expectation readout"):
            _ = composite.expected_values

    def test_has_methods(self, sampling_result, expectation_result, tomography_result):
        composite = ReadoutCompositeResults([sampling_result, expectation_result, tomography_result])
        assert composite.has_samples() is True
        assert composite.has_probabilities() is True
        assert composite.has_final_state() is True
        assert composite.has_expectation_values() is True

    def test_has_methods_empty(self):
        composite = ReadoutCompositeResults([])
        assert composite.has_samples() is False
        assert composite.has_probabilities() is False
        assert composite.has_final_state() is False
        assert composite.has_expectation_values() is False

    def test_len(self, sampling_result, expectation_result):
        composite = ReadoutCompositeResults([sampling_result, expectation_result])
        assert len(composite) == 2

    def test_iter(self, sampling_result, expectation_result):
        composite = ReadoutCompositeResults([sampling_result, expectation_result])
        results = list(composite)
        assert len(results) == 2

    def test_multiple_results_combined(self, sampling_result, expectation_result, tomography_result):
        composite = ReadoutCompositeResults([sampling_result, expectation_result, tomography_result])
        assert composite.samples == {"0": 60, "1": 40}
        assert composite.expected_values == [1.0]
        assert composite.final_state is not None


# FunctionalResult integration


class TestFunctionalResult:
    def test_final_samples(self):
        readout = SamplingReadout(nshots=100)
        result = FunctionalResult([SamplingReadoutResult(readout=readout, samples={"0": 100})])
        assert result.final_samples == {"0": 100}

    def test_final_state(self):
        result = FunctionalResult([StateTomographyReadoutResult(readout=StateTomographyReadout(), final_state=ket(0))])
        assert result.final_state is not None

    def test_final_expected_values(self):
        result = FunctionalResult(
            [ExpectationReadoutResult(readout=ExpectationReadout(observables=[pauli_z(0)]), expected_values=[1.0])]
        )
        assert result.final_expected_values == [1.0]

    def test_final_probabilities_from_sampling(self):
        readout = SamplingReadout(nshots=100)
        result = FunctionalResult([SamplingReadoutResult(readout=readout, samples={"0": 70, "1": 30})])
        assert np.isclose(result.final_probabilities["0"], 0.7)

    def test_final_probabilities_from_tomography(self):
        result = FunctionalResult([StateTomographyReadoutResult(readout=StateTomographyReadout(), final_state=ket(0))])
        assert np.isclose(result.final_probabilities["0"], 1.0)

    def test_duplicate_readout_types_raises(self):
        ro = SamplingReadout(nshots=100)
        r1 = SamplingReadoutResult(readout=ro, samples={"0": 50})
        r2 = SamplingReadoutResult(readout=ro, samples={"1": 50})
        with pytest.raises(ValueError, match="allowed to be specified once"):
            FunctionalResult([r1, r2])

    def test_multiple_readout_types(self):
        sampling = SamplingReadoutResult(readout=SamplingReadout(nshots=100), samples={"0": 100})
        tomo = StateTomographyReadoutResult(readout=StateTomographyReadout(), final_state=ket(0))
        expectation = ExpectationReadoutResult(
            readout=ExpectationReadout(observables=[pauli_z(0)]), expected_values=[1.0]
        )
        result = FunctionalResult([sampling, tomo, expectation])
        assert result.has_samples()
        assert result.has_final_state()
        assert result.has_expectation_values()
        assert result.has_probabilities()

    def test_intermediate_results(self):
        tomo_readout = StateTomographyReadout()
        final = [StateTomographyReadoutResult(readout=tomo_readout, final_state=ket(0))]
        inter1 = [StateTomographyReadoutResult(readout=tomo_readout, final_state=ket(1))]
        inter2 = [StateTomographyReadoutResult(readout=tomo_readout, final_state=(ket(0) + ket(1)).unit())]
        result = FunctionalResult(readout_results=final, intermediate_results=[inter1, inter2])
        assert len(result.states) == 3  # 2 intermediate + 1 final
        assert len(result.intermediate_results) == 2

    def test_states_no_intermediate_raises(self):
        result = FunctionalResult([StateTomographyReadoutResult(readout=StateTomographyReadout(), final_state=ket(0))])
        with pytest.raises(ValueError, match="Intermediate Results were not stored"):
            _ = result.states

    def test_states_no_tomography_raises(self):
        sampling = SamplingReadoutResult(readout=SamplingReadout(nshots=10), samples={"0": 10})
        result = FunctionalResult(readout_results=[sampling], intermediate_results=[[sampling]])
        with pytest.raises(ValueError, match="no State Tomography"):
            _ = result.states

    def test_samples_intermediate(self):
        ro = SamplingReadout(nshots=10)
        final = [SamplingReadoutResult(readout=ro, samples={"0": 10})]
        inter = [SamplingReadoutResult(readout=ro, samples={"1": 10})]
        result = FunctionalResult(readout_results=final, intermediate_results=[inter])
        assert len(result.samples) == 2

    def test_len_no_intermediate(self):
        result = FunctionalResult([SamplingReadoutResult(readout=SamplingReadout(nshots=10), samples={"0": 10})])
        assert len(result) == 1

    def test_len_with_intermediate(self):
        ro = SamplingReadout(nshots=10)
        r = SamplingReadoutResult(readout=ro, samples={"0": 10})
        result = FunctionalResult(readout_results=[r], intermediate_results=[[r], [r]])
        assert len(result) == 3

    def test_iter(self):
        ro = SamplingReadout(nshots=10)
        r = SamplingReadoutResult(readout=ro, samples={"0": 10})
        result = FunctionalResult(readout_results=[r], intermediate_results=[[r]])
        items = list(result)
        assert len(items) == 2

    def test_repr(self):
        result = FunctionalResult([SamplingReadoutResult(readout=SamplingReadout(nshots=10), samples={"0": 10})])
        assert "Functional Results" in repr(result)


# Backend._construct_results_list


class TestBackendConstructResultsList:
    def test_sampling_readout(self):
        results = Backend._construct_results_list(final_state=ket(0), readout=[SamplingReadout(nshots=100)])
        assert len(results) == 1
        assert isinstance(results[0], SamplingReadoutResult)
        assert sum(results[0].samples.values()) == 100

    def test_state_tomography_readout(self):
        results = Backend._construct_results_list(final_state=ket(0), readout=[StateTomographyReadout()])
        assert len(results) == 1
        assert isinstance(results[0], StateTomographyReadoutResult)
        assert results[0].final_state is not None

    def test_expectation_readout(self):
        results = Backend._construct_results_list(
            final_state=ket(0), readout=[ExpectationReadout(observables=[pauli_z(0)])]
        )
        assert len(results) == 1
        assert isinstance(results[0], ExpectationReadoutResult)
        assert np.isclose(results[0].expected_values[0], 1.0)

    def test_multiple_readouts(self):
        results = Backend._construct_results_list(
            final_state=ket(0),
            readout=[
                SamplingReadout(nshots=100),
                StateTomographyReadout(),
                ExpectationReadout(observables=[pauli_z(0)]),
            ],
        )
        assert len(results) == 3
        assert isinstance(results[0], SamplingReadoutResult)
        assert isinstance(results[1], StateTomographyReadoutResult)
        assert isinstance(results[2], ExpectationReadoutResult)

    def test_unsupported_readout_raises(self):
        with pytest.raises(ValueError, match="Unsupported Readout Method"):
            Backend._construct_results_list(final_state=ket(0), readout=[ReadoutMethod()])
