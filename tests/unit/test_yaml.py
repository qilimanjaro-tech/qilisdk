import logging
import sys
from types import SimpleNamespace

import pytest
from loguru_caplog import loguru_caplog as caplog  # noqa: F401

from qilisdk import _logging
from qilisdk._logging import InterceptHandler
from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import LT, BinaryVariable, Domain, OneHot, Variable

from types import SimpleNamespace
from qilisdk.yaml import type_representer, type_constructor

from ruamel.yaml import YAML
from qilisdk.yaml import QiliYAML

import math
from collections import defaultdict, deque

import numpy as np
import pytest
from pydantic import BaseModel
from scipy import sparse

import types
from collections import defaultdict, deque

import numpy as np
import pytest
from pydantic import BaseModel
from ruamel.yaml import YAML
from scipy import sparse

yaml = YAML(typ="unsafe")

import types
from io import StringIO
from collections import defaultdict, deque

import numpy as np
import pytest
from pydantic import BaseModel
from ruamel.yaml import YAML
from scipy import sparse


def dump_load(obj):
    buf = StringIO()
    yaml.dump(obj, buf)
    return yaml.load(buf.getvalue())


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


# -----------------------
# NumPy scalar
# -----------------------

def test_numpy_scalar_roundtrip():
    x = np.int64(42)
    loaded = dump_load(x)

    assert isinstance(loaded, np.generic)
    assert loaded == x


# -----------------------
# defaultdict (with factory)
# -----------------------

def test_defaultdict_with_factory():
    dd = defaultdict(list)
    dd["a"].append(1)

    loaded = dump_load(dd)

    assert isinstance(loaded, defaultdict)
    assert loaded.default_factory is list
    assert loaded["a"] == [1]


# -----------------------
# defaultdict (factory=None branch)
# -----------------------

def test_defaultdict_without_factory():
    dd = defaultdict(None)
    dd["x"] = 5

    loaded = dump_load(dd)

    assert isinstance(loaded, defaultdict)
    assert loaded.default_factory is None
    assert loaded["x"] == 5


# -----------------------
# Regular function
# -----------------------

def test_function_roundtrip():
    def f(x):
        return x + 1

    loaded = dump_load(f)

    assert isinstance(loaded, types.FunctionType)
    assert loaded(3) == 4


# -----------------------
# Lambda function
# -----------------------

def test_lambda_roundtrip():
    f = lambda x: x * 2  # noqa: E731

    loaded = dump_load(f)

    assert callable(loaded)
    assert loaded(4) == 8


# -----------------------
# Pydantic model
# -----------------------

class User(BaseModel):
    id: int
    name: str


def test_pydantic_model_roundtrip():
    u = User(id=1, name="alice")
    loaded = dump_load(u)

    assert isinstance(loaded, User)
    assert loaded == u


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


# -----------------------
# Type objects
# -----------------------

def test_type_roundtrip():
    t = type(int)
    loaded = dump_load(t)
    assert loaded is type


# -----------------------
# deque
# -----------------------

def test_deque_roundtrip():
    d = deque([1, 2, 3])
    loaded = dump_load(d)

    assert isinstance(loaded, deque)
    assert list(loaded) == [1, 2, 3]


# -----------------------
# QiliYAML.register_class decorator branch
# -----------------------

def test_register_class_decorator_path():
    y = QiliYAML(typ="unsafe")

    @y.register_class()
    class Foo:
        pass

    assert hasattr(Foo, "yaml_tag")
    assert Foo.yaml_tag.startswith("!")


# -----------------------
# QiliYAML.register_class(shared=True) branch
# -----------------------

def test_register_class_shared():
    y = QiliYAML(typ="unsafe")

    class Bar:
        pass

    y.register_class(Bar, shared=True)

    assert hasattr(Bar, "yaml_tag")
    assert "." in Bar.yaml_tag

import base64
from types import SimpleNamespace

from dill import dumps

from qilisdk.yaml import function_representer, function_constructor

def test_function_representer_direct():
    def f(x):
        return x + 1

    calls = {}

    class DummyRepresenter:
        def represent_scalar(self, tag, value):
            calls["tag"] = tag
            calls["value"] = value
            return (tag, value)

    rep = DummyRepresenter()
    result = function_representer(rep, f)

    assert calls["tag"] == "!function"
    assert isinstance(calls["value"], str)

    decoded = base64.b64decode(calls["value"])
    loaded = dumps(f, recurse=True)
    assert decoded == loaded
    assert result == ("!function", calls["value"])
    assert callable(f)
    assert f(3) == 4


def test_function_constructor_direct():
    def f(x):
        return x * 2

    serialized = base64.b64encode(dumps(f, recurse=True)).decode("utf-8")
    node = SimpleNamespace(value=serialized)

    loaded = function_constructor(None, node)

    assert callable(loaded)
    assert loaded(3) == 6

from types import SimpleNamespace

from pydantic import BaseModel

from qilisdk.yaml import (
    pydantic_model_representer,
    pydantic_model_constructor,
)


class User(BaseModel):
    id: int
    name: str


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
    node = SimpleNamespace(
        value=None,
        tag="!PydanticModel",
        data=None,
    )

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



def test_type_representer_direct():
    calls = {}

    class DummyRepresenter:
        def represent_scalar(self, tag, value):
            calls["tag"] = tag
            calls["value"] = value
            return (tag, value)

    rep = DummyRepresenter()
    result = type_representer(rep, dict)

    assert calls["tag"] == "!type"
    assert calls["value"] == "builtins.dict"
    assert result == ("!type", "builtins.dict")


def test_type_constructor_direct():
    node = SimpleNamespace(value="builtins.dict")

    loaded = type_constructor(None, node)

    assert loaded is dict
