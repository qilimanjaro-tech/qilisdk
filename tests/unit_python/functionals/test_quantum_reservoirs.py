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
from copy import copy

import numpy as np
import pytest

from qilisdk.analog import Schedule, Z
from qilisdk.core import Parameter, QTensor, ket
from qilisdk.digital import CNOT, RX, RY, Circuit, M
from qilisdk.functionals import FunctionalResult
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.readout.readout_result import ReadoutCompositeResults, SamplingReadoutResult
from qilisdk.settings import get_settings


def _isclose(lhs: float, rhs: float) -> bool:
    return bool(np.isclose(lhs, rhs, atol=get_settings().atol, rtol=get_settings().rtol))


def _assert_parameter_dict_close(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert _isclose(actual[key], value)


def _assert_bounds_dict_close(actual: dict[str, tuple[float, float]], expected: dict[str, tuple[float, float]]) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert _isclose(actual[key][0], value[0])
        assert _isclose(actual[key][1], value[1])


def _schedule_with_parameter(nqubits: int = 2) -> tuple[Schedule, Parameter]:
    g = Parameter("g", 0.3, bounds=(0.0, 1.0))
    hamiltonian = g * Z(0)
    for qubit in range(1, nqubits):
        hamiltonian += Z(qubit)
    schedule = Schedule(
        dt=0.1,
        hamiltonians={"h": hamiltonian},
        coefficients={"h": {(0.0, 1.0): 1.0}},
    )
    return schedule, g


def test_reservoir_input_is_non_trainable():
    inp = ReservoirInput("u", 0.2)
    assert not inp.is_trainable
    assert _isclose(inp.bounds[0], inp.domain.min())
    assert _isclose(inp.bounds[1], inp.domain.max())


def test_reservoir_layer_properties_and_parameter_interface():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    u = ReservoirInput("u", 0.2)
    p = Parameter("p", 0.4, bounds=(0.0, 1.0))

    pre = Circuit(1)
    pre.add(RX(0, theta=u))

    post = Circuit(1)
    post.add(RY(0, theta=p))

    reservoir_layer = ReservoirLayer(
        evolution_dynamics=schedule,
        input_encoding=pre,
        output_encoding=post,
        qubits_to_reset=[0],
    )

    assert reservoir_layer.nqubits == 2
    assert reservoir_layer.input_encoding is not None
    assert reservoir_layer.input_encoding.nqubits == 2
    assert reservoir_layer.output_encoding is not None
    assert reservoir_layer.output_encoding.nqubits == 2
    assert reservoir_layer.qubits_to_reset == [0]
    assert reservoir_layer.reservoir_dynamics == schedule

    assert reservoir_layer.get_parameter_names() == ["u", "g", "output_encoding_p"]
    assert reservoir_layer.get_parameter_names(where=lambda param: param.is_trainable) == [
        "g",
        "output_encoding_p",
    ]
    assert reservoir_layer.input_parameter_names == ["u"]
    assert reservoir_layer.nparameters == 3

    assert set(reservoir_layer.get_parameters()) == {"u", "g", "output_encoding_p"}
    assert set(reservoir_layer.get_parameters(where=lambda param: param.is_trainable)) == {
        "g",
        "output_encoding_p",
    }
    assert set(reservoir_layer.get_parameter_bounds()) == {"u", "g", "output_encoding_p"}
    assert set(reservoir_layer.get_parameter_bounds(where=lambda param: param.is_trainable)) == {
        "g",
        "output_encoding_p",
    }

    with pytest.raises(ValueError, match="Provided 2 but this object has 3 parameters"):
        reservoir_layer.set_parameter_values([0.1, 0.2])

    reservoir_layer.set_parameters({"u": 0.5, "g": 0.7, "output_encoding_p": 0.8})
    _assert_parameter_dict_close(reservoir_layer.get_parameters(), {"u": 0.5, "g": 0.7, "output_encoding_p": 0.8})

    reservoir_layer.set_parameter_bounds({"u": (-2.0, 2.0), "g": (0.0, 0.9), "output_encoding_p": (0.0, 0.9)})
    _assert_bounds_dict_close(
        reservoir_layer.get_parameter_bounds(where=lambda param: param.is_trainable),
        {"g": (0.0, 0.9), "output_encoding_p": (0.0, 0.9)},
    )

    reservoir_layer.set_parameter_values([0.6, 0.4, 0.3])
    assert _isclose(reservoir_layer.get_parameter_values()[0], 0.6)
    assert _isclose(reservoir_layer.get_parameter_values()[1], 0.4)
    assert _isclose(reservoir_layer.get_parameter_values()[2], 0.3)

    assert len(reservoir_layer) == 3
    steps = list(iter(reservoir_layer))
    assert len(steps) == 3
    assert isinstance(steps[0], Circuit)
    assert isinstance(steps[1], Schedule)
    assert isinstance(steps[2], Circuit)


def test_reservoir_layer_set_parameters_rejects_unknown_labels():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    u = ReservoirInput("u", 0.2)

    pre = Circuit(1)
    pre.add(RX(0, theta=u))

    reservoir_layer = ReservoirLayer(
        evolution_dynamics=schedule,
        input_encoding=pre,
        qubits_to_reset=[0],
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        reservoir_layer.set_parameters({"does_not_exist": 9.9})

    with pytest.raises(ValueError, match="unknown parameter"):
        reservoir_layer.set_parameters({"u": 0.5, "phi": 9.9})
    assert _isclose(reservoir_layer.get_parameters()["u"], 0.2)

    reservoir_layer.set_parameters({"u": 0.5, "g": 0.7})
    _assert_parameter_dict_close(reservoir_layer.get_parameters(), {"u": 0.5, "g": 0.7})


def test_reservoir_layer_validation_errors():
    schedule, _ = _schedule_with_parameter(nqubits=2)

    bad_pre_with_measure = Circuit(1)
    bad_pre_with_measure.add(M(0))
    with pytest.raises(ValueError, match="can't contain measurements"):
        ReservoirLayer(evolution_dynamics=schedule, input_encoding=bad_pre_with_measure)

    bad_post_with_measure = Circuit(1)
    bad_post_with_measure.add(M(0))
    with pytest.raises(ValueError, match="can't contain measurements"):
        ReservoirLayer(evolution_dynamics=schedule, input_encoding=bad_post_with_measure)

    bad_pre_too_wide = Circuit(3)
    with pytest.raises(ValueError, match="acts on more qubits"):
        ReservoirLayer(evolution_dynamics=schedule, input_encoding=bad_pre_too_wide)

    bad_post_too_wide = Circuit(3)
    with pytest.raises(ValueError, match="acts on more qubits"):
        ReservoirLayer(evolution_dynamics=schedule, output_encoding=bad_post_too_wide)

    bad_pre_multi_qubit = Circuit(2)
    bad_pre_multi_qubit.add(CNOT(0, 1))
    with pytest.raises(ValueError, match="Only single qubit gates"):
        ReservoirLayer(evolution_dynamics=schedule, input_encoding=bad_pre_multi_qubit)

    bad_post_multi_qubit = Circuit(2)
    bad_post_multi_qubit.add(CNOT(0, 1))
    with pytest.raises(ValueError, match="Only single qubit gates"):
        ReservoirLayer(evolution_dynamics=schedule, output_encoding=bad_post_multi_qubit)


def test_quantum_reservoir_properties_and_qubit_validation():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    reservoir_layer = ReservoirLayer(evolution_dynamics=schedule)
    initial_state = ket(0, 0)
    qreservoir = QuantumReservoir(
        initial_state=initial_state,
        reservoir_layer=reservoir_layer,
        input_per_layer=[{"g": 0.1}, {"g": 0.2}],
    )

    assert qreservoir.nqubits == 2
    assert qreservoir.initial_state == initial_state
    assert qreservoir.reservoir_layer == reservoir_layer
    assert len(qreservoir.input_per_layer) == 2
    assert _isclose(qreservoir.input_per_layer[0]["g"], 0.1)
    assert _isclose(qreservoir.input_per_layer[1]["g"], 0.2)
    assert qreservoir.input_parameter_names == []

    with pytest.raises(ValueError, match="invalid initial state"):
        QuantumReservoir(
            initial_state=ket(0),
            reservoir_layer=reservoir_layer,
            input_per_layer=[{"g": 0.1}],
        )

    with pytest.raises(ValueError, match="must contain at least one layer"):
        QuantumReservoir(
            initial_state=initial_state,
            reservoir_layer=reservoir_layer,
            input_per_layer=[],
        )


def test_reservoir_layer_parameter_sync_with_children():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    u = ReservoirInput("u", 0.1)
    p = Parameter("p", 0.2)
    pre = Circuit(1)
    pre.add(RX(0, theta=u))
    post = Circuit(1)
    post.add(RY(0, theta=p))
    layer = ReservoirLayer(
        evolution_dynamics=schedule,
        input_encoding=pre,
        output_encoding=post,
    )

    layer.set_parameters({"u": 0.5, "g": 0.6, "output_encoding_p": 0.7})
    assert _isclose(pre.get_parameters()["u"], 0.5)
    assert _isclose(schedule.get_parameters()["g"], 0.6)
    assert _isclose(post.get_parameters()["p"], 0.7)

    pre.set_parameters({"u": 0.8})
    schedule.set_parameters({"g": 0.4})
    post.set_parameters({"p": 0.3})
    _assert_parameter_dict_close(layer.get_parameters(), {"u": 0.8, "g": 0.4, "output_encoding_p": 0.3})

    layer.set_parameter_values([0.9, 1.0], where=lambda param: param.is_trainable)
    assert _isclose(schedule.get_parameters()["g"], 0.9)
    assert _isclose(post.get_parameters()["p"], 1.0)
    assert _isclose(pre.get_parameters()["u"], 0.8)


def _ry_matrix(theta: float) -> np.ndarray:
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    return np.array([[cos_half, -sin_half], [sin_half, cos_half]])


def test_reservoir_layer_set_parameters_invalidates_encoding_gate_matrix():
    """Regression test: ``ReservoirLayer.set_parameters`` must route through each
    child's own ``set_parameters`` so that gate matrix caches are invalidated.

    Before the fix, the inherited ``Parameterizable.set_parameters`` called
    ``Parameter.set_value`` directly, leaving each gate's ``cached_property``
    matrix stale even though ``Parameter.value`` had been updated.
    """
    schedule, _ = _schedule_with_parameter(nqubits=2)
    p_in = Parameter("p_in", 0.2, bounds=(0.0, np.pi))
    p_out = Parameter("p_out", 0.2, bounds=(0.0, np.pi))

    pre = Circuit(1)
    pre.add(RY(0, theta=p_in))
    post = Circuit(1)
    post.add(RY(0, theta=p_out))

    layer = ReservoirLayer(evolution_dynamics=schedule, input_encoding=pre, output_encoding=post)
    pre_gate = pre.gates[0]
    post_gate = post.gates[0]

    np.testing.assert_allclose(pre_gate.matrix, _ry_matrix(0.2), atol=1e-12)
    np.testing.assert_allclose(post_gate.matrix, _ry_matrix(0.2), atol=1e-12)

    layer.set_parameters({"input_encoding_p_in": 0.9, "output_encoding_p_out": 1.1})

    np.testing.assert_allclose(pre_gate.matrix, _ry_matrix(0.9), atol=1e-12)
    np.testing.assert_allclose(post_gate.matrix, _ry_matrix(1.1), atol=1e-12)


def test_reservoir_layer_set_parameter_values_invalidates_encoding_gate_matrix():
    """Regression test for ``ReservoirLayer.set_parameter_values`` updating gate caches."""
    schedule, _ = _schedule_with_parameter(nqubits=2)
    p_out = Parameter("p_out", 0.2, bounds=(0.0, np.pi))

    post = Circuit(1)
    post.add(RY(0, theta=p_out))

    layer = ReservoirLayer(evolution_dynamics=schedule, output_encoding=post)
    post_gate = post.gates[0]
    np.testing.assert_allclose(post_gate.matrix, _ry_matrix(0.2), atol=1e-12)

    layer.set_parameter_values([0.5, 1.3], where=lambda param: param.is_trainable)

    np.testing.assert_allclose(post_gate.matrix, _ry_matrix(1.3), atol=1e-12)


def test_reservoir_layer_set_parameter_bounds_propagates_to_children():
    """``ReservoirLayer.set_parameter_bounds`` must dispatch to each child so that
    bounds-aware setters (e.g. on gates) run their own bookkeeping rather than
    only mutating the underlying ``Parameter`` in place.
    """
    schedule, g = _schedule_with_parameter(nqubits=2)
    p_in = Parameter("p_in", 0.2, bounds=(0.0, 1.0))
    p_out = Parameter("p_out", 0.2, bounds=(0.0, 1.0))

    pre = Circuit(1)
    pre.add(RY(0, theta=p_in))
    post = Circuit(1)
    post.add(RY(0, theta=p_out))

    layer = ReservoirLayer(evolution_dynamics=schedule, input_encoding=pre, output_encoding=post)
    layer.set_parameter_bounds(
        {
            "input_encoding_p_in": (-2.0, 2.0),
            "g": (-3.0, 3.0),
            "output_encoding_p_out": (-4.0, 4.0),
        }
    )

    _assert_bounds_dict_close(layer.input_encoding.get_parameter_bounds(), {"input_encoding_p_in": (-2.0, 2.0)})
    _assert_bounds_dict_close(schedule.get_parameter_bounds(), {"g": (-3.0, 3.0)})
    _assert_bounds_dict_close(layer.output_encoding.get_parameter_bounds(), {"output_encoding_p_out": (-4.0, 4.0)})

    assert _isclose(p_in.bounds[0], -2.0)
    assert _isclose(p_in.bounds[1], 2.0)
    assert _isclose(g.bounds[0], -3.0)
    assert _isclose(g.bounds[1], 3.0)
    assert _isclose(p_out.bounds[0], -4.0)
    assert _isclose(p_out.bounds[1], 4.0)


def test_quantum_reservoir_parameter_sync_with_reservoir_layer_child():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    u = ReservoirInput("u", 0.1)
    p = Parameter("p", 0.2)
    pre = Circuit(1)
    pre.add(RX(0, theta=u))
    post = Circuit(1)
    post.add(RY(0, theta=p))
    layer = ReservoirLayer(
        evolution_dynamics=schedule,
        input_encoding=pre,
        output_encoding=post,
    )
    reservoir = QuantumReservoir(initial_state=ket(0, 0), reservoir_layer=layer, input_per_layer=[{}])

    reservoir.set_parameters({"u": 0.3, "g": 0.4, "output_encoding_p": 0.5})
    _assert_parameter_dict_close(layer.get_parameters(), {"u": 0.3, "g": 0.4, "output_encoding_p": 0.5})

    layer.set_parameters({"u": 0.6, "g": 0.7, "output_encoding_p": 0.8})
    _assert_parameter_dict_close(reservoir.get_parameters(), {"u": 0.6, "g": 0.7, "output_encoding_p": 0.8})

    reservoir.set_parameter_values([0.95, 1.2], where=lambda param: param.is_trainable)
    assert _isclose(layer.get_parameters()["g"], 0.95)
    assert _isclose(layer.get_parameters()["output_encoding_p"], 1.2)
    assert _isclose(layer.get_parameters()["u"], 0.6)


def test_validate_output_encoding_with_measurement():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    bad_post = Circuit(1)
    bad_post.add(M(0))
    with pytest.raises(ValueError, match="can't contain measurements"):
        ReservoirLayer(evolution_dynamics=schedule, output_encoding=bad_post)


def test_process_qtensor_padding():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    single_qubit_obs = QTensor(np.array([[1, 0], [0, -1]]))
    padded = layer._process_qtensor(single_qubit_obs)
    assert padded.nqubits == 2


def test_process_qtensor_too_many_qubits_raises():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)

    big_obs = QTensor(np.identity(8))
    with pytest.raises(ValueError, match="more qubits than the system"):
        layer._process_qtensor(big_obs)


def test_process_qtensor_exact_match():

    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    obs = QTensor(np.identity(4))
    result = layer._process_qtensor(obs)
    assert result.nqubits == 2


def test_reservoir_layer_evolution_dynamics_property():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    assert layer.evolution_dynamics is schedule


def test_quantum_reservoir_none_layer_raises():
    with pytest.raises(ValueError, match="must be provided"):
        QuantumReservoir(initial_state=ket(0), reservoir_layer=None, input_per_layer=[{}])


def test_quantum_reservoir_none_input_raises():
    schedule, _ = _schedule_with_parameter(nqubits=1)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    with pytest.raises(ValueError, match="must be provided"):
        QuantumReservoir(initial_state=ket(0), reservoir_layer=layer, input_per_layer=None)


def test_set_inputs_per_layer():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    reservoir = QuantumReservoir(initial_state=ket(0, 0), reservoir_layer=layer, input_per_layer=[{"g": 0.1}])
    reservoir.set_inputs_per_layer([{"g": 0.5}, {"g": 0.6}])
    assert len(reservoir.input_per_layer) == 2
    assert _isclose(reservoir.input_per_layer[0]["g"], 0.5)


def test_add_inputs_per_layer_list():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    reservoir = QuantumReservoir(initial_state=ket(0, 0), reservoir_layer=layer, input_per_layer=[{"g": 0.1}])
    reservoir.add_inputs_per_layer([{"g": 0.2}, {"g": 0.3}])
    assert len(reservoir.input_per_layer) == 3


def test_add_inputs_per_layer_single_dict():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    reservoir = QuantumReservoir(initial_state=ket(0, 0), reservoir_layer=layer, input_per_layer=[{"g": 0.1}])
    reservoir.add_inputs_per_layer({"g": 0.4})
    assert len(reservoir.input_per_layer) == 2


def test_quantum_reservoir_copy():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule)
    reservoir = QuantumReservoir(initial_state=ket(0, 0), reservoir_layer=layer, input_per_layer=[{"g": 0.1}])
    copied = copy(reservoir)
    assert copied.nqubits == reservoir.nqubits
    assert len(copied.input_per_layer) == len(reservoir.input_per_layer)


def test_reservoir_layer_copy():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    layer = ReservoirLayer(evolution_dynamics=schedule, qubits_to_reset=[0])
    copied = copy(layer)
    assert copied.nqubits == layer.nqubits
    assert copied.qubits_to_reset == [0]


def test_quantum_reservoir_result_is_functional_result():
    r = SamplingReadoutResult.from_samples(samples={"0": 10})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=r))
    assert result.get_samples() == {"0": 10}
    assert isinstance(result, FunctionalResult)
