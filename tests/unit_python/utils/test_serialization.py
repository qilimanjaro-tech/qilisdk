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

import re
import subprocess
import sys
from pathlib import Path

import pytest

from qilisdk.analog.hamiltonian import Hamiltonian, PauliY, X, Z
from qilisdk.analog.schedule import Schedule
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.utils.serialization import (
    _REGISTERING_MODULES,
    DeserializationError,
    SerializationError,
    deserialize,
    deserialize_from,
    serialize,
    serialize_to,
)
from qilisdk.yaml import QiliYAML


def test_analog_evolution_algorithm_serialization(tmp_path):
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
    assert "!lambda" not in serialized_analog_evolution
    assert "!function" not in serialized_analog_evolution
    assert "python/" not in serialized_analog_evolution
    deserialized_analog_evolution = deserialize(serialized_analog_evolution, AnalogEvolution)
    assert isinstance(deserialized_analog_evolution, AnalogEvolution)

    path = str(tmp_path / "analog_evolution.yml")
    serialize_to(analog_evolution, path)
    deserialized_from_analog_evolution = deserialize_from(path, AnalogEvolution)
    assert isinstance(deserialized_from_analog_evolution, AnalogEvolution)


def test_deserialization_with_wrong_yaml_raises_error(tmp_path):
    not_valid_yaml = "!SomeClass _property: 100"

    with pytest.raises(DeserializationError):
        _ = deserialize(not_valid_yaml)

    path = tmp_path / "not_valid_yaml.yml"
    path.write_text(not_valid_yaml, encoding="utf-8")

    with pytest.raises(DeserializationError):
        _ = deserialize_from(str(path))


def test_deserialization_with_wrong_cls_raises_error(tmp_path):
    operator = X(0)

    serialized_operator = serialize(operator)

    with pytest.raises(DeserializationError):
        _ = deserialize(serialized_operator, PauliY)

    path = str(tmp_path / "pauli_x.yml")
    serialize_to(operator, path)

    with pytest.raises(DeserializationError):
        _ = deserialize_from(path, PauliY)


def test_arbitrary_serialize_errors(monkeypatch, tmp_path):
    def new_dump(*args, **kwargs):
        raise ValueError("Serialization error")

    monkeypatch.setattr(QiliYAML, "dump", new_dump)

    operator = X(0)

    with pytest.raises(Exception, match="Serialization error"):
        _ = serialize(operator)

    with pytest.raises(Exception, match="Serialization error"):
        serialize_to(operator, str(tmp_path / "pauli_x.yml"))


def test_serialize_to_keeps_existing_file_on_failure(tmp_path):
    path = tmp_path / "existing.yml"
    path.write_text("previous content", encoding="utf-8")

    target = str(path)

    with pytest.raises(SerializationError, match="not serializable"):
        serialize_to(lambda t: t, target)

    assert path.read_text(encoding="utf-8") == "previous content"


def test_registering_modules_cover_every_registered_class():
    src = Path(__file__).resolve().parents[3] / "src"
    registering = {
        ".".join(path.relative_to(src).with_suffix("").parts)
        for path in (src / "qilisdk").rglob("*.py")
        if re.search(r"^@yaml\.register_class", path.read_text(encoding="utf-8"), re.MULTILINE)
    }

    assert registering == set(_REGISTERING_MODULES)


def test_deserialize_does_not_depend_on_prior_imports():
    code = (
        "from qilisdk.utils.serialization import deserialize\n"
        "print(deserialize('!InitialState zero').name)\n"
        "print(deserialize('!!python/object/apply:qilisdk.core.qtensor.InitialState [one]').name)\n"
        "print(type(deserialize('!QTensor {data: !csr_matrix {data: [1.0], indices: [0], indptr: [0, 1, 1], "
        "shape: !tuple [2, 1]}}')).__name__)\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.split() == ["ZERO", "ONE", "QTensor"]
