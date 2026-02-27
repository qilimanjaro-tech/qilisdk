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

from qilisdk.analog import Schedule, Z
from qilisdk.analog.hamiltonian import PauliZ
from qilisdk.core import Parameter, QTensor, ket
from qilisdk.digital import CNOT, RX, RY, Circuit, M
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.functionals.quantum_reservoirs_result import _complex_dtype
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

    observables = [
        PauliZ(0),
        Z(0),
        QTensor(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)),
    ]
    reservoir_layer = ReservoirLayer(
        evolution_dynamics=schedule,
        observables=observables,
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
    assert reservoir_layer.observables == observables
    assert reservoir_layer.reservoir_dynamics == schedule
    assert len(reservoir_layer.observables_as_qtensor) == 3
    assert all(obs.nqubits == 2 for obs in reservoir_layer.observables_as_qtensor)

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


def test_reservoir_layer_validation_errors():
    schedule, _ = _schedule_with_parameter(nqubits=2)

    bad_pre_with_measure = Circuit(1)
    bad_pre_with_measure.add(M(0))
    with pytest.raises(ValueError, match="can't contain measurements"):
        ReservoirLayer(schedule, [PauliZ(0)], input_encoding=bad_pre_with_measure)

    bad_post_with_measure = Circuit(1)
    bad_post_with_measure.add(M(0))
    with pytest.raises(ValueError, match="can't contain measurements"):
        ReservoirLayer(schedule, [PauliZ(0)], input_encoding=bad_post_with_measure)

    bad_pre_too_wide = Circuit(3)
    with pytest.raises(ValueError, match="acts on more qubits"):
        ReservoirLayer(schedule, [PauliZ(0)], input_encoding=bad_pre_too_wide)

    bad_post_too_wide = Circuit(3)
    with pytest.raises(ValueError, match="acts on more qubits"):
        ReservoirLayer(schedule, [PauliZ(0)], output_encoding=bad_post_too_wide)

    bad_pre_multi_qubit = Circuit(2)
    bad_pre_multi_qubit.add(CNOT(0, 1))
    with pytest.raises(ValueError, match="Only single qubit gates"):
        ReservoirLayer(schedule, [PauliZ(0)], input_encoding=bad_pre_multi_qubit)

    bad_post_multi_qubit = Circuit(2)
    bad_post_multi_qubit.add(CNOT(0, 1))
    with pytest.raises(ValueError, match="Only single qubit gates"):
        ReservoirLayer(schedule, [PauliZ(0)], output_encoding=bad_post_multi_qubit)

    with pytest.raises(ValueError, match="Observable acts on more qubits"):
        ReservoirLayer(
            schedule,
            [QTensor(np.eye(8, dtype=np.complex128))],
        )

    with pytest.raises(ValueError, match="Unsupported observable type"):
        ReservoirLayer(
            schedule,
            [object()],  # type: ignore[list-item]
        )


def test_quantum_reservoir_properties_and_qubit_validation():
    schedule, _ = _schedule_with_parameter(nqubits=2)
    reservoir_layer = ReservoirLayer(schedule, [PauliZ(0)])
    initial_state = ket(0, 0)
    qreservoir = QuantumReservoir(
        initial_state=initial_state,
        reservoir_layer=reservoir_layer,
        input_per_layer=[{"g": 0.1}, {"g": 0.2}],
        store_final_state=True,
        store_intermediate_states=True,
        nshots=12,
    )

    assert qreservoir.nqubits == 2
    assert qreservoir.initial_state == initial_state
    assert qreservoir.reservoir_layer == reservoir_layer
    assert len(qreservoir.input_per_layer) == 2
    assert _isclose(qreservoir.input_per_layer[0]["g"], 0.1)
    assert _isclose(qreservoir.input_per_layer[1]["g"], 0.2)
    assert qreservoir.store_final_state
    assert qreservoir.store_intermediate_states
    assert qreservoir.nshots == 12
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


def test_quantum_reservoir_result_complex_dtype_helper():
    assert _complex_dtype() == get_settings().complex_precision.dtype


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
        observables=[PauliZ(0)],
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
        observables=[PauliZ(0)],
        input_encoding=pre,
        output_encoding=post,
    )
    reservoir = QuantumReservoir(initial_state=ket(0, 0), reservoir_layer=layer, input_per_layer=[{}], nshots=1)

    reservoir.set_parameters({"u": 0.3, "g": 0.4, "output_encoding_p": 0.5})
    _assert_parameter_dict_close(layer.get_parameters(), {"u": 0.3, "g": 0.4, "output_encoding_p": 0.5})

    layer.set_parameters({"u": 0.6, "g": 0.7, "output_encoding_p": 0.8})
    _assert_parameter_dict_close(reservoir.get_parameters(), {"u": 0.6, "g": 0.7, "output_encoding_p": 0.8})

    reservoir.set_parameter_values([0.95, 1.2], where=lambda param: param.is_trainable)
    assert _isclose(layer.get_parameters()["g"], 0.95)
    assert _isclose(layer.get_parameters()["output_encoding_p"], 1.2)
    assert _isclose(layer.get_parameters()["u"], 0.6)
