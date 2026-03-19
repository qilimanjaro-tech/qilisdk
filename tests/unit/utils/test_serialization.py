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

from pathlib import Path

import pytest

import qilisdk.utils.serialization
from qilisdk.analog.hamiltonian import Hamiltonian, PauliY, X, Y, Z
from qilisdk.analog.schedule import Schedule
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.utils.serialization import DeserializationError, deserialize, deserialize_from, serialize, serialize_to


def test_analog_evolution_algorithm_serialization():
    T = 100
    dt = 1

    nqubits = 1

    H1 = sum(X(i) for i in range(nqubits))
    H2 = sum(Z(i) for i in range(nqubits))

    assert isinstance(H1, Hamiltonian)
    assert isinstance(H2, Hamiltonian)

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": H1, "h2": H2},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    state = tensor_prod([(ket(0) + ket(1)).unit() for _ in range(nqubits)]).unit()

    analog_evolution = AnalogEvolution(
        schedule=schedule,
        initial_state=state,
    )

    serialized_analog_evolution = serialize(analog_evolution)
    deserialized_analog_evolution = deserialize(serialized_analog_evolution, AnalogEvolution)
    assert isinstance(deserialized_analog_evolution, AnalogEvolution)

    serialize_to(analog_evolution, "analog_evolution.yml")
    deserialized_from_analog_evolution = deserialize_from("analog_evolution.yml", AnalogEvolution)
    assert isinstance(deserialized_from_analog_evolution, AnalogEvolution)

    Path("analog_evolution.yml").unlink()


def test_deserialization_with_wrong_yaml_raises_error():
    not_valid_yaml = "!SomeClass _property: 100"

    with pytest.raises(DeserializationError):
        _ = deserialize(not_valid_yaml)

    Path("not_valid_yaml.yml").write_text(not_valid_yaml, encoding="utf-8")

    with pytest.raises(DeserializationError):
        _ = deserialize_from("not_valid_yaml.yml")

    Path("not_valid_yaml.yml").unlink()


def test_deserialization_with_wrong_cls_raises_error():
    operator = X(0)

    serialized_operator = serialize(operator)

    with pytest.raises(DeserializationError):
        _ = deserialize(serialized_operator, PauliY)

    serialize_to(operator, "pauli_x.yml")

    with pytest.raises(DeserializationError):
        _ = deserialize_from("pauli_x.yml", PauliY)

    Path("pauli_x.yml").unlink()


def test_arbitrary_serialize_errors(monkeypatch):
    def new_dump(*args, **kwargs):
        raise ValueError("Serialization error")

    monkeypatch.setattr(
        qilisdk.utils.serialization.yaml,
        "dump",
        new_dump,
    )

    operator = X(0)

    with pytest.raises(Exception, match="Serialization error"):
        _ = serialize(operator)

    with pytest.raises(Exception, match="Serialization error"):
        serialize_to(operator, "pauli_x.yml")

    Path("pauli_x.yml").unlink(missing_ok=True)
