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
from qilisdk.experiments.experiment_functional import ExperimentFunctional
from qilisdk.experiments.experiment_result import ExperimentResult
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
from qilisdk.speqtrum.speqtrum_models import (
    ExecuteType,
    TypedJobDetail,
    _require_experiment_result,
    _require_functional_result,
    _require_variational_program_result,
)

pytest.importorskip("httpx", reason="SpeQtrum tests require the 'speqtrum' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "keyring",
    reason="SpeQtrum tests require the 'speqtrum' optional dependency",
    exc_type=ImportError,
)


from unittest.mock import MagicMock

from qilisdk.digital import RY, Circuit
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.readout import Readout
from qilisdk.readout.readout_result import ReadoutCompositeResults, SamplingReadoutResult, StateTomographyReadoutResult
from qilisdk.speqtrum.speqtrum_models import (
    AnalogEvolutionPayload,
    DigitalPropagationPayload,
    ExecuteResult,
    ExperimentPayload,
    JobHandle,
    QuantumReservoirPayload,
    VariationalProgramPayload,
    _require_experiment_result_typed,
)


class SweepExperimentResult(ExperimentResult):
    plot_title = "Sweep Experiment"


class SweepExperiment(ExperimentFunctional[SweepExperimentResult]):
    """Minimal backend-defined experiment, standing in for the concrete experiments a backend provides."""

    result_type = SweepExperimentResult

    def __init__(self, qubit, averages, sweep_values):
        super().__init__(qubit=qubit, averages=averages)
        self.sweep_values = sweep_values


def test_digital_propagation_payload():
    circ = Circuit(2)
    digital_propagation = DigitalPropagation(circuit=circ)
    payload = DigitalPropagationPayload(digital_propagation=digital_propagation, readout=Readout())
    serialized = payload._serialize_sampling(digital_propagation=digital_propagation, _info={})
    deserialized = payload._load_sampling(serialized)
    assert deserialized.circuit.nqubits == digital_propagation.circuit.nqubits


def test_analog_evolution_payload():
    # note: these can't be mocked because then YAML throws errors
    hamiltonian = Hamiltonian({(PauliZ(0),): 1})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1, total_time=1.0)
    initial_state = ket(0).unit()
    analog_evolution = AnalogEvolution(schedule=schedule, initial_state=initial_state)
    payload = AnalogEvolutionPayload(analog_evolution=analog_evolution, readout=Readout())
    serialized = payload._serialize_time_evolution(analog_evolution=analog_evolution, _info={})
    deserialized = payload._load_time_evolution(serialized)
    assert deserialized.initial_state == analog_evolution.initial_state


def test_variational_program_payload():
    circ = Circuit(2)
    digital_propagation = DigitalPropagation(circuit=circ)
    optimizer = SciPyOptimizer(method="Nelder-Mead")
    cost_function = ObservableCostFunction(observable=PauliZ(0))
    variational_program = VariationalProgram(
        functional=digital_propagation, optimizer=optimizer, cost_function=cost_function
    )
    payload = VariationalProgramPayload(
        variational_program=variational_program, readout=Readout().with_state_tomography()
    )
    serialized_variational_program = payload._serialize_variational_program(
        variational_program=variational_program, _info={}
    )
    deserialized_variational_program = payload._load_variational_program(serialized_variational_program)
    assert deserialized_variational_program.functional.circuit.nqubits == variational_program.functional.circuit.nqubits
    assert deserialized_variational_program.optimizer.method == variational_program.optimizer.method
    assert deserialized_variational_program.cost_function.observable == variational_program.cost_function.observable


def test_experiment_payload():
    experiment = SweepExperiment(qubit=0, averages=1000, sweep_values=[10, 20, 30])
    payload = ExperimentPayload(experiment=experiment)
    serialized_experiment = payload._serialize_experiment(experiment=experiment, _info={})
    deserialized_experiment = payload._load_experiment(serialized_experiment)
    assert deserialized_experiment.qubit == experiment.qubit
    assert deserialized_experiment.averages == experiment.averages
    assert deserialized_experiment.sweep_values == experiment.sweep_values


def test_experiment_payload_keeps_already_deserialized_experiments():
    experiment = SweepExperiment(qubit=0, averages=1000, sweep_values=[10, 20, 30])
    payload = ExperimentPayload(experiment=experiment)
    assert payload._load_experiment(experiment) is experiment


def _make_quantum_reservoir() -> QuantumReservoir:
    """Build a minimal reservoir out of real objects, since YAML can't serialize mocks."""
    schedule = Schedule(hamiltonians={"h": Hamiltonian({(PauliZ(0),): 1})}, dt=0.1, total_time=1.0)
    encoding = Circuit(1)
    encoding.add(RY(0, theta=ReservoirInput("q0", 0)))
    layer = ReservoirLayer(evolution_dynamics=schedule, input_encoding=encoding, qubits_to_reset=[0])
    return QuantumReservoir(initial_state=ket(0).unit(), reservoir_layer=layer, input_per_layer=[{"q0": 0.5}])


def test_quantum_reservoir_payload():
    quantum_reservoir = _make_quantum_reservoir()
    payload = QuantumReservoirPayload(quantum_reservoir=quantum_reservoir, readout=Readout())
    serialized = payload._serialize_time_evolution(quantum_reservoir=quantum_reservoir, _info={})
    deserialized = payload._load_time_evolution(serialized)
    assert isinstance(deserialized, QuantumReservoir)
    assert deserialized.initial_state == quantum_reservoir.initial_state


def test_quantum_reservoir_payload_round_trips_through_validation():
    """Regression: the validator deserialized as ``AnalogEvolution``, so this always raised."""
    quantum_reservoir = _make_quantum_reservoir()
    payload = QuantumReservoirPayload(quantum_reservoir=quantum_reservoir, readout=Readout())

    restored = QuantumReservoirPayload.model_validate_json(payload.model_dump_json())

    assert isinstance(restored.quantum_reservoir, QuantumReservoir)
    assert restored.quantum_reservoir.initial_state == quantum_reservoir.initial_state


def test_payloads_load_serialized_readout():
    """Every payload's readout validator must accept a serialized string and pass objects through."""
    readout = Readout().with_state_tomography()
    circuit = Circuit(2)
    payloads = [
        DigitalPropagationPayload(digital_propagation=DigitalPropagation(circuit=circuit), readout=readout),
        AnalogEvolutionPayload(
            analog_evolution=AnalogEvolution(
                schedule=Schedule(hamiltonians={"h": Hamiltonian({(PauliZ(0),): 1})}, dt=0.1, total_time=1.0),
                initial_state=ket(0).unit(),
            ),
            readout=readout,
        ),
        QuantumReservoirPayload(quantum_reservoir=_make_quantum_reservoir(), readout=readout),
        VariationalProgramPayload(
            variational_program=VariationalProgram(
                functional=DigitalPropagation(circuit=circuit),
                optimizer=SciPyOptimizer(method="Nelder-Mead"),
                cost_function=ObservableCostFunction(observable=PauliZ(0)),
            ),
            readout=readout,
        ),
    ]

    for payload in payloads:
        serialized = payload._serialize_readout(readout=readout, _info={})
        assert isinstance(serialized, str)
        assert isinstance(payload._load_readout(serialized), Readout)
        assert payload._load_readout(readout) is readout


def test_require_experiment_result_typed_accepts_matching_and_rejects_mismatching():
    extractor = _require_experiment_result_typed(SweepExperimentResult)

    matching = MagicMock()
    matching.experiment_result = SweepExperimentResult(qubit=0, averages=1, data=[[0.0]], dims=[])
    assert extractor(matching) is matching.experiment_result

    class OtherExperimentResult(ExperimentResult):
        plot_title = "Other Experiment"

    mismatching = MagicMock()
    mismatching.experiment_result = OtherExperimentResult(qubit=0, averages=1, data=[[0.0]], dims=[])
    with pytest.raises(RuntimeError, match="does not match the expected"):
        extractor(mismatching)


def test_job_handle_experiment_without_result_type():
    """The untyped path falls back to the plain experiment extractor."""
    handle = JobHandle.experiment(11)

    assert handle.id == 11
    assert handle.execute_type is ExecuteType.EXPERIMENT

    result = MagicMock()
    result.experiment_result = SweepExperimentResult(qubit=0, averages=1, data=[[0.0]], dims=[])
    assert handle.extractor(result) is result.experiment_result


def test_execute_result_sampling():
    execute_type = ExecuteType.DIGITAL_PROPAGATION
    readout_result = SamplingReadoutResult.from_samples(samples={"0": 512, "1": 512})
    functional_result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    result = ExecuteResult(
        type=execute_type,
        functional_result=functional_result,
    )
    serialized_result = result._serialize_sampling_result(functional_result=result.functional_result, _info={})
    deserialized_result = result._load_sampling_result(serialized_result)
    assert deserialized_result.get_samples() == functional_result.get_samples()


def test_execute_result_time_evolution():
    execute_type = ExecuteType.ANALOG_EVOLUTION
    readout_result = StateTomographyReadoutResult(state=ket(0))
    functional_result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=readout_result))
    result = ExecuteResult(
        type=execute_type,
        functional_result=functional_result,
    )
    serialized_result = result._serialize_sampling_result(functional_result=result.functional_result, _info={})
    deserialized_result = result._load_sampling_result(serialized_result)
    assert deserialized_result.get_state() == functional_result.get_state()


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


def test_execute_result_experiment():
    execute_type = ExecuteType.EXPERIMENT
    experiment_result = SweepExperimentResult(
        qubit=0,
        averages=1000,
        data=[[0.1, 0.2], [0.3, 0.4]],
        dims=[],
    )
    result = ExecuteResult(
        type=execute_type,
        experiment_result=experiment_result,
    )
    serialized_result = result._serialize_experiment_result(experiment_result=result.experiment_result, _info={})
    deserialized_result = result._load_experiment_result(serialized_result)
    assert deserialized_result.qubit == experiment_result.qubit
    assert deserialized_result.averages == experiment_result.averages
    assert deserialized_result.data == experiment_result.data
    assert deserialized_result.dims == experiment_result.dims


def test_execute_result_experiment_round_trips_through_validation():
    experiment_result = SweepExperimentResult(qubit=1, averages=10, data=[[0.1, 0.2]], dims=[])
    serialized_result = ExecuteResult(
        type=ExecuteType.EXPERIMENT, experiment_result=experiment_result
    ).model_dump_json()

    result = ExecuteResult.model_validate_json(serialized_result)

    assert isinstance(result.experiment_result, SweepExperimentResult)
    assert result.experiment_result.qubit == experiment_result.qubit
    assert result.experiment_result.data == experiment_result.data


def test_execute_result_experiment_serializes_none():
    result = ExecuteResult(type=ExecuteType.EXPERIMENT)
    assert result._serialize_experiment_result(experiment_result=result.experiment_result, _info={}) is None


def test_requires():
    good_result = MagicMock()
    good_result.functional_result = MagicMock()
    good_result.variational_program_result = MagicMock()
    good_result.experiment_result = MagicMock()

    bad_result = MagicMock()
    bad_result.functional_result = None
    bad_result.variational_program_result = None
    bad_result.experiment_result = None

    assert _require_variational_program_result(good_result) is good_result.variational_program_result
    with pytest.raises(RuntimeError, match="did not return a variational_program_result"):
        _require_variational_program_result(bad_result)

    assert _require_functional_result(good_result) is good_result.functional_result
    with pytest.raises(RuntimeError, match="did not return a functional_result"):
        _require_functional_result(bad_result)

    assert _require_experiment_result(good_result) is good_result.experiment_result
    with pytest.raises(RuntimeError, match="did not return an experiment_result"):
        _require_experiment_result(bad_result)


def test_typed_job_detail():
    bad_result = MagicMock()
    bad_result.type = ExecuteType.ANALOG_EVOLUTION

    good_result = MagicMock()
    good_result.type = ExecuteType.DIGITAL_PROPAGATION

    mock_extractor = MagicMock(return_value=good_result)

    job_detail = TypedJobDetail(
        expected_type=ExecuteType.DIGITAL_PROPAGATION,
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
