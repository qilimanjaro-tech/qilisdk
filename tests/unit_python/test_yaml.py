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
import functools
import pickle
from collections import defaultdict, deque
from enum import Enum
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import BaseModel
from ruamel.yaml import YAML
from ruamel.yaml.constructor import ConstructorError
from ruamel.yaml.representer import RepresenterError
from scipy import sparse

from qilisdk.core.interpolator import Interpolation
from qilisdk.core.model import QUBO, Model
from qilisdk.core.qtensor import InitialState
from qilisdk.core.variables import BinaryVariable
from qilisdk.yaml import (
    QiliYAML,
    pydantic_model_constructor,
    pydantic_model_representer,
    type_constructor,
    type_representer,
    yaml,
)


def dump(obj):
    buf = StringIO()
    QiliYAML().dump(obj, buf)
    return buf.getvalue()


def load(text):
    return QiliYAML().load(text)


def dump_load(obj):
    return load(dump(obj))


# -----------------------
# SciPy CSR matrix
# -----------------------


def test_csr_matrix_roundtrip():
    m = sparse.csr_matrix([[0, 1], [2, 0]])
    loaded = dump_load(m)

    assert isinstance(loaded, sparse.csr_matrix)
    assert (loaded != m).nnz == 0


# -----------------------
# NumPy ndarray
# -----------------------


def test_ndarray_roundtrip():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    loaded = dump_load(arr)

    assert isinstance(loaded, np.ndarray)
    assert np.array_equal(arr, loaded)
    assert arr.dtype == loaded.dtype


def test_ndarray_rejects_object_dtype():
    with pytest.raises(ConstructorError, match="dtype 'object' is not allowed"):
        load("!ndarray {dtype: object, shape: !tuple [1], data: [1]}")


@pytest.mark.parametrize(
    "arr",
    [np.array(["a", "b"]), np.array([1, "a"], dtype=object), np.array(["2024-01-01"], dtype="datetime64[D]")],
    ids=["str", "object", "datetime"],
)
def test_ndarray_with_unsupported_dtype_is_not_serializable(arr):
    with pytest.raises(RepresenterError, match="only numeric and boolean dtypes are supported"):
        dump(arr)


def test_csr_matrix_empty_roundtrip():
    m = sparse.csr_matrix((2, 3))
    loaded = dump_load(m)

    assert loaded.shape == (2, 3)
    assert loaded.nnz == 0


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ("!csr_matrix {data: [a], indices: [0], indptr: [0, 1], shape: !tuple [1, 1]}", "data"),
        ("!csr_matrix {data: [1.0], indices: [0.5], indptr: [0, 1], shape: !tuple [1, 1]}", "indices"),
        ("!csr_matrix {data: [1.0], indices: [0], indptr: [a, b], shape: !tuple [1, 1]}", "indptr"),
    ],
)
def test_csr_matrix_rejects_non_numeric_fields(payload, field):
    with pytest.raises(ConstructorError, match=f"csr_matrix {field} of dtype"):
        load(payload)


# -----------------------
# NumPy scalar
# -----------------------


def test_numpy_scalar_roundtrip():
    x = np.int64(42)
    loaded = dump_load(x)

    assert isinstance(loaded, np.generic)
    assert loaded == x


def test_numpy_scalar_rejects_object_dtype():
    with pytest.raises(ConstructorError, match="dtype 'object' is not allowed"):
        load("!np_scalar {dtype: object, value: 1}")


@pytest.mark.parametrize("scalar", [np.str_("a"), np.datetime64("2024-01-01")], ids=["str", "datetime"])
def test_numpy_scalar_with_unsupported_dtype_is_not_serializable(scalar):
    with pytest.raises(RepresenterError, match="only numeric and boolean dtypes are supported"):
        dump(scalar)


# -----------------------
# defaultdict
# -----------------------


def test_defaultdict_with_factory():
    dd = defaultdict(list)
    dd["a"].append(1)

    loaded = dump_load(dd)

    assert isinstance(loaded, defaultdict)
    assert loaded.default_factory is list
    assert loaded["a"] == [1]


def test_defaultdict_without_factory():
    dd = defaultdict(None)
    dd["x"] = 5

    loaded = dump_load(dd)

    assert isinstance(loaded, defaultdict)
    assert loaded.default_factory is None
    assert loaded["x"] == 5


def test_defaultdict_rejects_unknown_factory_on_load():
    with pytest.raises(ConstructorError, match=r"factory 'os\.system' is not allowed"):
        load("!defaultdict {default_factory: os.system, items: {}}")


class _CallableFactory:
    def __call__(self):
        return 0


@pytest.mark.parametrize(
    "factory",
    [Model, functools.partial(int), _CallableFactory(), lambda: 0],
    ids=["unlisted-type", "partial", "callable-instance", "lambda"],
)
def test_defaultdict_rejects_unknown_factory_on_dump(factory):
    with pytest.raises(RepresenterError, match="cannot be serialized"):
        dump(defaultdict(factory))


# -----------------------
# Callables and code-carrying tags
# -----------------------


def _named_function(x):
    return x + 1


@pytest.mark.parametrize(
    "obj",
    [_named_function, lambda x: x * 2, functools.partial(_named_function, 1), print, Model("m").__repr__],
    ids=["function", "lambda", "partial", "builtin", "method"],
)
def test_callables_are_not_serializable(obj):
    with pytest.raises(RepresenterError, match="functions and lambdas are not serializable"):
        dump(obj)


class _Exploit:
    def __reduce__(self):
        return (eval, ("1 + 1",))


@pytest.mark.parametrize("tag", ["!function", "!lambda"])
def test_pickled_code_tags_are_rejected(tag):
    payload = base64.b64encode(pickle.dumps(_Exploit())).decode()

    with pytest.raises(ConstructorError, match="no longer supported for security reasons"):
        load(f"{tag} {payload}")


@pytest.mark.parametrize(
    "payload",
    [
        "!!python/object/apply:os.getpid []",
        "!!python/object/new:os.getpid []",
        "!!python/object:qilisdk.core.model.Model {}",
        "!!python/name:os.system ''",
        "!!python/module:os ''",
    ],
)
def test_python_tags_are_rejected(payload):
    with pytest.raises(ConstructorError):
        load(payload)


def test_registrations_do_not_leak_into_other_yaml_handlers():
    with pytest.raises(ConstructorError):
        YAML(typ="safe", pure=True).load("!complex {real: 1, imag: 2}")


# -----------------------
# Pydantic model
# -----------------------


@yaml.register_class
class User(BaseModel):
    id: int
    name: str


class UnregisteredUser(BaseModel):
    id: int


def test_pydantic_model_roundtrip():
    u = User(id=1, name="alice")
    loaded = dump_load(u)

    assert isinstance(loaded, User)
    assert loaded == u


def test_pydantic_model_unregistered_is_not_serializable():
    with pytest.raises(RepresenterError, match="is not registered for serialization"):
        dump(UnregisteredUser(id=1))


def test_pydantic_model_unregistered_is_rejected_on_load():
    with pytest.raises(ConstructorError, match="is not a registered QiliSDK class"):
        load(f"!PydanticModel {{type: {UnregisteredUser.__module__}.UnregisteredUser, data: {{id: 1}}}}")


def test_pydantic_model_rejects_registered_non_pydantic_class():
    with pytest.raises(ConstructorError, match="is not a Pydantic model"):
        load("!PydanticModel {type: qilisdk.core.model.Model, data: {}}")


def test_pydantic_model_representer_direct():
    u = User(id=1, name="alice")

    calls = {}

    class DummyRepresenter:
        def represent_mapping(self, tag, value):
            calls["tag"] = tag
            calls["value"] = value
            return (tag, value)

    rep = DummyRepresenter()
    result = pydantic_model_representer(rep, u)

    assert calls["tag"] == "!PydanticModel"
    assert calls["value"]["type"].endswith(".User")
    assert calls["value"]["data"] == {"id": 1, "name": "alice"}
    assert result == ("!PydanticModel", calls["value"])


def test_pydantic_model_constructor_direct():
    node = SimpleNamespace(value=None, tag="!PydanticModel", data=None, start_mark=None)

    mapping = {
        "type": f"{User.__module__}.User",
        "data": {"id": 2, "name": "bob"},
    }

    class DummyConstructor:
        def construct_mapping(self, _node, deep=True):
            return mapping

    loaded = pydantic_model_constructor(DummyConstructor(), node)

    assert isinstance(loaded, User)
    assert loaded.id == 2
    assert loaded.name == "bob"


# -----------------------
# Complex numbers
# -----------------------


def test_complex_roundtrip():
    z = 3 + 4j
    loaded = dump_load(z)

    assert isinstance(loaded, complex)
    assert loaded == z


# -----------------------
# Tuple
# -----------------------


def test_tuple_roundtrip():
    t = (1, "a", 3.5)
    loaded = dump_load(t)

    assert isinstance(loaded, tuple)
    assert loaded == t


# -----------------------
# Model
# -----------------------


def test_model_roundtrip():
    mod = Model("test")
    loaded = dump_load(mod)

    assert isinstance(loaded, Model)
    assert loaded.label == mod.label


def test_linearized_qubo_roundtrip():
    b = [BinaryVariable(f"b{i}") for i in range(3)]
    model = Model("cubic")
    model.set_objective(b[0] * b[1] * b[2])
    qubo = QUBO.from_model(model, linearize=True)

    loaded = dump_load(qubo)

    assert isinstance(loaded, QUBO)
    assert loaded._linearizer is not None
    assert str(loaded.objective) == str(qubo.objective)


# -----------------------
# Type objects
# -----------------------


def test_type_roundtrip():
    loaded = dump_load(Model)
    assert loaded is Model


def test_type_unregistered_is_not_serializable():
    with pytest.raises(RepresenterError, match="is not registered for serialization"):
        dump(dict)


def test_type_unregistered_is_rejected_on_load():
    with pytest.raises(ConstructorError, match="is not a registered QiliSDK class"):
        load("!type os.system")


def test_type_representer_direct():
    calls = {}

    class DummyRepresenter:
        def represent_scalar(self, tag, value):
            calls["tag"] = tag
            calls["value"] = value
            return (tag, value)

    rep = DummyRepresenter()
    result = type_representer(rep, Model)

    assert calls["tag"] == "!type"
    assert calls["value"] == "qilisdk.core.model.Model"
    assert result == ("!type", "qilisdk.core.model.Model")


def test_type_constructor_direct():
    node = SimpleNamespace(value="qilisdk.core.model.Model", start_mark=None)

    loaded = type_constructor(None, node)

    assert loaded is Model


# -----------------------
# deque
# -----------------------


def test_deque_roundtrip():
    d = deque([1, 2, 3])
    loaded = dump_load(d)

    assert isinstance(loaded, deque)
    assert list(loaded) == [1, 2, 3]


# -----------------------
# Enums
# -----------------------


@pytest.mark.parametrize("member", [*Interpolation, *InitialState])
def test_enum_roundtrip(member):
    text = dump(member)

    assert "python/object" not in text
    assert dump_load(member) is member


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            "!!python/object/apply:qilisdk.core.interpolator.Interpolation [linear interpolation between schedule points]",
            Interpolation.LINEAR,
        ),
        ("!!python/object/apply:qilisdk.core.qtensor.InitialState [zero]", InitialState.ZERO),
    ],
)
def test_legacy_enum_tag_is_loaded(payload, expected):
    assert load(payload) is expected


@pytest.mark.parametrize(
    "payload",
    [
        "!!python/object/apply:qilisdk.core.model.Model []",
        "!!python/object/apply:qilisdk.core.qtensor.InitialState {args: [zero]}",
        "!!python/object/apply:qilisdk.core.qtensor.InitialState [zero, one]",
    ],
)
def test_legacy_enum_tag_rejects_other_forms(payload):
    with pytest.raises(ConstructorError, match="Unsupported legacy tag"):
        load(payload)


def test_register_enum_requires_string_values():
    class IntEnum(Enum):
        ONE = 1

    with pytest.raises(TypeError, match="must have string values"):
        QiliYAML().register_class(IntEnum)


# -----------------------
# QiliYAML.register_class
# -----------------------


def test_register_class_decorator_path():
    y = QiliYAML()

    @y.register_class()
    class Foo:
        pass

    assert hasattr(Foo, "yaml_tag")
    assert Foo.yaml_tag.startswith("!")


def test_register_class_shared():
    y = QiliYAML()

    class Bar:
        pass

    y.register_class(Bar, shared=True)

    assert hasattr(Bar, "yaml_tag")
    assert "." in Bar.yaml_tag


def _register_other_clash():
    class Clash:
        pass

    return QiliYAML().register_class(Clash)


def test_register_class_rejects_duplicate_tag():
    class Clash:
        def __init__(self):
            self.value = 1

    QiliYAML().register_class(Clash)

    with pytest.raises(ValueError, match="already used by"):
        _register_other_clash()

    loaded = dump_load(Clash())
    assert type(loaded) is Clash
    assert loaded.value == 1


def test_register_class_allows_reregistering_same_class():
    y = QiliYAML()

    class Again:
        pass

    y.register_class(Again)
    y.register_class(Again)

    assert Again.yaml_tag == "!Again"


def test_unregistered_class_is_not_serializable():
    class NotRegistered:
        pass

    with pytest.raises(RepresenterError, match="cannot represent an object"):
        dump(NotRegistered())
