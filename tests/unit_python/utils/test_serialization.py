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

import base64
import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pytest
from dill import dumps
from scipy import sparse

import qilisdk.utils.serialization
from qilisdk.analog.hamiltonian import Hamiltonian, PauliY, X, Z
from qilisdk.analog.schedule import Schedule
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.core.variables import Bitwise
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


# ---------------------------------------------------------------------------
# Security: safe-by-default deserialization (QSDK-01)
# ---------------------------------------------------------------------------


def _malicious_code_document(tag: str, sentinel: Path) -> str:
    """Build a `!function`/`!lambda` YAML document whose dill payload would, if
    executed, create *sentinel* on disk."""

    class _Payload:
        def __reduce__(self):
            return (os.system, (f"touch {sentinel}",))

    encoded = base64.b64encode(dumps(_Payload(), recurse=True)).decode("utf-8")
    return f"{tag} {encoded}\n"


@pytest.mark.parametrize("tag", ["!function", "!lambda"])
def test_deserialize_rejects_code_bearing_tags_by_default(tag, tmp_path):
    sentinel = tmp_path / "pwned"
    document = _malicious_code_document(tag, sentinel)

    with pytest.raises(DeserializationError):
        _ = deserialize(document)

    # Proof of non-execution: the dill payload never ran.
    assert not sentinel.exists()


@pytest.mark.parametrize("tag", ["!function", "!lambda"])
def test_deserialize_from_rejects_code_bearing_tags_by_default(tag, tmp_path):
    sentinel = tmp_path / "pwned_file"
    document = _malicious_code_document(tag, sentinel)
    yaml_file = tmp_path / "payload.yml"
    yaml_file.write_text(document, encoding="utf-8")

    with pytest.raises(DeserializationError):
        _ = deserialize_from(str(yaml_file))

    assert not sentinel.exists()


@pytest.mark.parametrize(
    "document",
    [
        "!type os.system\n",
        "!defaultdict\ndefault_factory: os.system\nitems: {}\n",
        "!PydanticModel\ntype: os.system\ndata: {}\n",
        "!!python/object/apply:os.system ['echo pwned']\n",
        "!!python/name:os.system\n",
    ],
)
def test_deserialize_rejects_non_allowlisted_import_tags(document):
    with pytest.raises(DeserializationError):
        _ = deserialize(document)


def test_deserialize_data_round_trip_on_safe_path():
    # All of the pure-data tags the SDK relies on must still round-trip through
    # the default (safe) loader, with no trust_code.
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(deserialize(serialize(arr)), arr)

    csr = sparse.csr_matrix([[0, 1], [2, 0]])
    assert (deserialize(serialize(csr)) != csr).nnz == 0

    scalar = np.int64(42)
    assert deserialize(serialize(scalar)) == scalar

    assert deserialize(serialize(3 + 4j)) == 3 + 4j
    assert deserialize(serialize((1, "a", 3.5))) == (1, "a", 3.5)
    assert list(deserialize(serialize(deque([1, 2, 3])))) == [1, 2, 3]

    dd = defaultdict(list)
    dd["a"].append(1)
    loaded = deserialize(serialize(dd))
    assert isinstance(loaded, defaultdict)
    assert loaded.default_factory is list
    assert loaded["a"] == [1]

    # A defaultdict with no factory exercises the None-factory branch of the safe loader.
    dd_none = defaultdict(None)
    dd_none["x"] = 5
    loaded_none = deserialize(serialize(dd_none))
    assert isinstance(loaded_none, defaultdict)
    assert loaded_none.default_factory is None
    assert loaded_none["x"] == 5

    # Allow-listed `!type` value emitted by the SDK.
    assert deserialize(serialize(Bitwise)) is Bitwise

    # A registered SDK class must still reconstruct on the safe path.
    operator = deserialize(serialize(X(0)), Hamiltonian)
    assert isinstance(operator, Hamiltonian)


def test_deserialize_trust_code_round_trips_function_and_lambda():
    def f(x):
        return x + 1

    g = lambda x: x * 2  # noqa: E731

    loaded_f = deserialize(serialize(f), trust_code=True)
    assert callable(loaded_f)
    assert loaded_f(3) == 4

    loaded_g = deserialize(serialize(g), trust_code=True)
    assert callable(loaded_g)
    assert loaded_g(4) == 8
