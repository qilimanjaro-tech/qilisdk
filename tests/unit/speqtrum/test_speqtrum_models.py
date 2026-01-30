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

"""
Unit-tests for the SpeQtrum synchronous client.

All tests are *function* based (no classes) so they integrate with plain
pytest discovery.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from qilisdk.analog.hamiltonian import Hamiltonian, PauliZ
from qilisdk.analog.schedule import Schedule
from qilisdk.core.qtensor import ket
from qilisdk.cost_functions.observable_cost_function import ObservableCostFunction
from qilisdk.experiments.experiment_functional import RabiExperiment, T1Experiment, T2Experiment, TwoTonesExperiment
from qilisdk.experiments.experiment_result import (
    RabiExperimentResult,
    T1ExperimentResult,
    T2ExperimentResult,
    TwoTonesExperimentResult,
)
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
from qilisdk.speqtrum.speqtrum_models import (
    ExecuteType,
    TypedJobDetail,
    _require_rabi_experiment_result,
    _require_sampling_result,
    _require_t1_experiment_result,
    _require_t2_experiment_result,
    _require_time_evolution_result,
    _require_two_tones_experiment_result,
    _require_variational_program_result,
)

pytest.importorskip("httpx", reason="SpeQtrum tests require the 'speqtrum' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "keyring",
    reason="SpeQtrum tests require the 'speqtrum' optional dependency",
    exc_type=ImportError,
)


from unittest.mock import MagicMock

from qilisdk.digital import Circuit
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.speqtrum.speqtrum_models import (
    ExecuteResult,
    RabiExperimentPayload,
    SamplingPayload,
    T1ExperimentPayload,
    T2ExperimentPayload,
    TimeEvolutionPayload,
    TwoTonesExperimentPayload,
    VariationalProgramPayload,
)


def test_sampling_payload():
    circ = Circuit(2)
    sampling = Sampling(nshots=1024, circuit=circ)
    payload = SamplingPayload(sampling=sampling)
    serialized_sampling = payload._serialize_sampling(sampling=sampling, _info={})
    deserialized_sampling = payload._load_sampling(serialized_sampling)
    assert deserialized_sampling.nshots == sampling.nshots
    assert deserialized_sampling.circuit.nqubits == sampling.circuit.nqubits


def test_time_evolution_payload():
    # note: these can't be mocked because then YAML throws errors
    hamiltonian = Hamiltonian({(PauliZ(0),): 1})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = ket(0).unit()
    observables = [PauliZ(0)]
    time_evolution = TimeEvolution(schedule=schedule, observables=observables, initial_state=initial_state)
    payload = TimeEvolutionPayload(time_evolution=time_evolution)
    serialized_time_evolution = payload._serialize_time_evolution(time_evolution=time_evolution, _info={})
    deserialized_time_evolution = payload._load_time_evolution(serialized_time_evolution)
    assert deserialized_time_evolution.initial_state == time_evolution.initial_state
    assert deserialized_time_evolution.observables == time_evolution.observables


def test_variational_program_payload():
    circ = Circuit(2)
    sampling = Sampling(nshots=1024, circuit=circ)
    optimizer = SciPyOptimizer(method="Nelder-Mead")
    cost_function = ObservableCostFunction(observable=PauliZ(0))
    variational_program = VariationalProgram(functional=sampling, optimizer=optimizer, cost_function=cost_function)
    payload = VariationalProgramPayload(variational_program=variational_program)
    serialized_variational_program = payload._serialize_variational_program(
        variational_program=variational_program, _info={}
    )
    deserialized_variational_program = payload._load_variational_program(serialized_variational_program)
    assert deserialized_variational_program.functional.circuit.nqubits == variational_program.functional.circuit.nqubits
    assert deserialized_variational_program.optimizer.method == variational_program.optimizer.method
    assert deserialized_variational_program.cost_function.observable == variational_program.cost_function.observable


def test_rabi_experiment_payload():
    experiment = RabiExperiment(qubit=0, drive_duration_values=[10, 20, 30])
    payload = RabiExperimentPayload(rabi_experiment=experiment)
    serialized_experiment = payload._serialize_rabi_experiment(rabi_experiment=experiment, _info={})
    deserialized_experiment = payload._load_rabi_experiment(serialized_experiment)
    assert deserialized_experiment.qubit == experiment.qubit
    assert deserialized_experiment.drive_duration_values == experiment.drive_duration_values


def test_t1_experiment_payload():
    experiment = T1Experiment(qubit=0, wait_duration_values=[10, 20, 30])
    payload = T1ExperimentPayload(t1_experiment=experiment)
    serialized_experiment = payload._serialize_t1_experiment(t1_experiment=experiment, _info={})
    deserialized_experiment = payload._load_t1_experiment(serialized_experiment)
    assert deserialized_experiment.qubit == experiment.qubit
    assert deserialized_experiment.wait_duration_values == experiment.wait_duration_values


def test_t2_experiment_payload():
    experiment = T2Experiment(qubit=0, wait_duration_values=[10, 20, 30])
    payload = T2ExperimentPayload(t2_experiment=experiment)
    serialized_experiment = payload._serialize_t2_experiment(t2_experiment=experiment, _info={})
    deserialized_experiment = payload._load_t2_experiment(serialized_experiment)
    assert deserialized_experiment.qubit == experiment.qubit
    assert deserialized_experiment.wait_duration_values == experiment.wait_duration_values


def test_two_tones_experiment_payload():
    experiment = TwoTonesExperiment(qubit=0, frequency_start=4.9e9, frequency_stop=5.1e9, frequency_step=1e6)
    payload = TwoTonesExperimentPayload(two_tones_experiment=experiment)
    serialized_experiment = payload._serialize_two_tones_experiment(two_tones_experiment=experiment, _info={})
    deserialized_experiment = payload._load_two_tones_experiment(serialized_experiment)
    assert deserialized_experiment.qubit == experiment.qubit
    assert deserialized_experiment.frequency_start == experiment.frequency_start
    assert deserialized_experiment.frequency_stop == experiment.frequency_stop
    assert deserialized_experiment.frequency_step == experiment.frequency_step


def test_execute_result_sampling():
    execute_type = ExecuteType.SAMPLING
    sampling_result = SamplingResult(nshots=1024, samples={"0": 512, "1": 512})
    result = ExecuteResult(
        type=execute_type,
        sampling_result=sampling_result,
    )
    serialized_result = result._serialize_sampling_result(sampling_result=result.sampling_result, _info={})
    deserialized_result = result._load_sampling_result(serialized_result)
    assert deserialized_result.nshots == sampling_result.nshots
    assert deserialized_result.samples == sampling_result.samples


def test_execute_result_time_evolution():
    execute_type = ExecuteType.TIME_EVOLUTION
    time_evolution_result = TimeEvolutionResult(
        final_state=ket(0),
        final_expected_values=[0.1],
    )
    result = ExecuteResult(
        type=execute_type,
        time_evolution_result=time_evolution_result,
    )
    serialized_result = result._serialize_time_evolution_result(
        time_evolution_result=result.time_evolution_result, _info={}
    )
    deserialized_result = result._load_time_evolution_result(serialized_result)
    assert deserialized_result.final_state == time_evolution_result.final_state
    assert deserialized_result.final_expected_values[0] == time_evolution_result.final_expected_values[0]


def test_execute_result_variational_program():
    execute_type = ExecuteType.VARIATIONAL_PROGRAM
    variational_program_result = VariationalProgramResult(
        result=0.3,
        optimizer_result=OptimizerResult(optimal_cost=0.3, optimal_parameters=[0.1, 0.2], intermediate_results=[]),
    )
    result = ExecuteResult(
        type=execute_type,
        variational_program_result=variational_program_result,
    )
    serialized_result = result._serialize_variational_program_result(
        variational_program_result=result.variational_program_result, _info={}
    )
    deserialized_result = result._load_variational_program_result(serialized_result)
    assert deserialized_result.optimal_cost == variational_program_result.optimal_cost
    assert deserialized_result.optimal_parameters == variational_program_result.optimal_parameters
    assert deserialized_result.intermediate_results == variational_program_result.intermediate_results


def test_execute_result_rabi_experiment():
    execute_type = ExecuteType.RABI_EXPERIMENT
    rabi_experiment_result = RabiExperimentResult(
        qubit=0,
        data=[[0.1, 0.2], [0.3, 0.4]],
        dims=[],
    )
    result = ExecuteResult(
        type=execute_type,
        rabi_experiment_result=rabi_experiment_result,
    )
    serialized_result = result._serialize_rabi_experiment_result(
        rabi_experiment_result=result.rabi_experiment_result, _info={}
    )
    deserialized_result = result._load_rabi_experiment_result(serialized_result)
    assert deserialized_result.qubit == rabi_experiment_result.qubit
    assert deserialized_result.data == rabi_experiment_result.data
    assert deserialized_result.dims == rabi_experiment_result.dims


def test_execute_result_t1_experiment():
    execute_type = ExecuteType.T1_EXPERIMENT
    t1_experiment_result = T1ExperimentResult(
        qubit=0,
        data=[[0.1, 0.2], [0.3, 0.4]],
        dims=[],
    )
    result = ExecuteResult(
        type=execute_type,
        t1_experiment_result=t1_experiment_result,
    )
    serialized_result = result._serialize_t1_experiment_result(
        t1_experiment_result=result.t1_experiment_result, _info={}
    )
    deserialized_result = result._load_t1_experiment_result(serialized_result)
    assert deserialized_result.qubit == t1_experiment_result.qubit
    assert deserialized_result.data == t1_experiment_result.data
    assert deserialized_result.dims == t1_experiment_result.dims


def test_execute_result_t2_experiment():
    execute_type = ExecuteType.T2_EXPERIMENT
    t2_experiment_result = T2ExperimentResult(
        qubit=0,
        data=[[0.1, 0.2], [0.3, 0.4]],
        dims=[],
    )
    result = ExecuteResult(
        type=execute_type,
        t2_experiment_result=t2_experiment_result,
    )
    serialized_result = result._serialize_t2_experiment_result(
        t2_experiment_result=result.t2_experiment_result, _info={}
    )
    deserialized_result = result._load_t2_experiment_result(serialized_result)
    assert deserialized_result.qubit == t2_experiment_result.qubit
    assert deserialized_result.data == t2_experiment_result.data
    assert deserialized_result.dims == t2_experiment_result.dims


def test_execute_result_two_tones_experiment():
    execute_type = ExecuteType.TWO_TONES_EXPERIMENT
    two_tones_experiment_result = TwoTonesExperimentResult(
        qubit=0,
        data=[[0.1, 0.2], [0.3, 0.4]],
        dims=[],
    )
    result = ExecuteResult(
        type=execute_type,
        two_tones_experiment_result=two_tones_experiment_result,
    )
    serialized_result = result._serialize_two_tones_experiment_result(
        two_tones_experiment_result=result.two_tones_experiment_result, _info={}
    )
    deserialized_result = result._load_two_tones_experiment_result(serialized_result)
    assert deserialized_result.qubit == two_tones_experiment_result.qubit
    assert deserialized_result.data == two_tones_experiment_result.data
    assert deserialized_result.dims == two_tones_experiment_result.dims


def test_requires():
    good_result = MagicMock()
    good_result.sampling_result = MagicMock()
    good_result.time_evolution_result = MagicMock()
    good_result.variational_program_result = MagicMock()
    good_result.rabi_experiment_result = MagicMock()
    good_result.t1_experiment_result = MagicMock()
    good_result.t2_experiment_result = MagicMock()
    good_result.two_tones_experiment_result = MagicMock()

    bad_result = MagicMock()
    bad_result.sampling_result = None
    bad_result.time_evolution_result = None
    bad_result.variational_program_result = None
    bad_result.rabi_experiment_result = None
    bad_result.t1_experiment_result = None
    bad_result.t2_experiment_result = None
    bad_result.two_tones_experiment_result = None

    assert _require_variational_program_result(good_result) is good_result.variational_program_result
    with pytest.raises(RuntimeError, match="did not return a variational_program_result"):
        _require_variational_program_result(bad_result)

    assert _require_sampling_result(good_result) is good_result.sampling_result
    with pytest.raises(RuntimeError, match="did not return a sampling_result"):
        _require_sampling_result(bad_result)

    assert _require_time_evolution_result(good_result) is good_result.time_evolution_result
    with pytest.raises(RuntimeError, match="did not return a time_evolution_result"):
        _require_time_evolution_result(bad_result)

    assert _require_rabi_experiment_result(good_result) is good_result.rabi_experiment_result
    with pytest.raises(RuntimeError, match="did not return a rabi_experiment_result"):
        _require_rabi_experiment_result(bad_result)

    assert _require_t1_experiment_result(good_result) is good_result.t1_experiment_result
    with pytest.raises(RuntimeError, match="did not return a t1_experiment_result"):
        _require_t1_experiment_result(bad_result)

    assert _require_t2_experiment_result(good_result) is good_result.t2_experiment_result
    with pytest.raises(RuntimeError, match="did not return a t2_experiment_result"):
        _require_t2_experiment_result(bad_result)

    assert _require_two_tones_experiment_result(good_result) is good_result.two_tones_experiment_result
    with pytest.raises(RuntimeError, match="did not return a two_tones_experiment_result"):
        _require_two_tones_experiment_result(bad_result)


def test_typed_job_detail():
    bad_result = MagicMock()
    bad_result.type = ExecuteType.TIME_EVOLUTION

    good_result = MagicMock()
    good_result.type = ExecuteType.SAMPLING

    mock_extractor = MagicMock(return_value=good_result)

    job_detail = TypedJobDetail(
        expected_type=ExecuteType.SAMPLING,
        id=3,
        name="Test Job",
        description="A test job",
        device_id=3,
        created_at=datetime.now(timezone.utc),
        status="pending",
        extractor=mock_extractor,
    )

    job_detail.result = None
    with pytest.raises(RuntimeError, match="without a result payload"):
        job_detail.get_results()

    job_detail.result = bad_result
    with pytest.raises(RuntimeError, match="Expected a result of type"):
        job_detail.get_results()

    job_detail.result = good_result
    result = job_detail.get_results()
    assert result is good_result
    mock_extractor.assert_called_once_with(good_result)
