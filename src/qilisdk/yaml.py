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

# ruff: noqa: ANN001, ANN201 DOC201, S403

import base64
import types
from collections import defaultdict

import numpy as np
from dill import dumps, loads
from pydantic import BaseModel
from ruamel.yaml import YAML
from scipy import sparse


def csr_representer(representer, data: sparse.csr_matrix):
    """Representer for CSR matrix."""
    value = {
        "data": data.data.tolist(),
        "indices": data.indices.tolist(),
        "indptr": data.indptr.tolist(),
        "shape": data.shape,
    }
    return representer.represent_mapping("!csr_matrix", value)


def csr_constructor(constructor, node):
    """Constructor for CSR matrix."""
    mapping = constructor.construct_mapping(node, deep=True)
    return sparse.csr_matrix(
        (mapping["data"], mapping["indices"], mapping["indptr"]),
        shape=tuple(mapping["shape"]),
    )


def ndarray_representer(representer, data):
    """Representer for ndarray"""
    value = {"dtype": str(data.dtype), "shape": data.shape, "data": data.ravel().tolist()}
    return representer.represent_mapping("!ndarray", value)


def ndarray_constructor(constructor, node):
    """Constructor for ndarray"""
    mapping = constructor.construct_mapping(node, deep=True)
    dtype = np.dtype(mapping["dtype"])
    shape = tuple(mapping["shape"])
    data = mapping["data"]
    return np.array(data, dtype=dtype).reshape(shape)


def np_scalar_representer(representer, data: np.generic):
    """Represent any NumPy scalar (e.g. np.int64, np.float32)."""
    return representer.represent_mapping(
        "!np_scalar",
        {"dtype": str(data.dtype), "value": data.item()},
    )


def np_scalar_constructor(constructor, node):
    """Reconstruct a NumPy scalar."""
    mapping = constructor.construct_mapping(node, deep=True)
    dtype = np.dtype(mapping["dtype"])
    return dtype.type(mapping["value"])


def defaultdict_representer(representer, data: defaultdict):
    """
    Represent a defaultdict by serializing its default_factory
    (as module+qualname) plus its items dict.
    """
    factory = data.default_factory
    factory_name = None if factory is None else f"{factory.__module__}.{factory.__qualname__}"
    return representer.represent_mapping(
        "!defaultdict",
        {"default_factory": factory_name, "items": dict(data)},
    )


def defaultdict_constructor(constructor, node):
    """Reconstruct a defaultdict, restoring its factory and contents."""
    mapping = constructor.construct_mapping(node, deep=True)
    fname = mapping["default_factory"]
    if fname is None:
        factory = None
    else:
        module, qual = fname.rsplit(".", 1)
        mod = __import__(module, fromlist=[qual])
        factory = getattr(mod, qual)
    dd = defaultdict(factory)
    dd.update(mapping["items"])
    return dd


def function_representer(representer, data):
    """Represent a non-lambda function by serializing it."""
    serialized_function = base64.b64encode(dumps(data, recurse=True)).decode("utf-8")
    return representer.represent_scalar("!function", serialized_function)


def function_constructor(constructor, node):
    """Reconstruct a function from the serialized data."""
    serialized_function = base64.b64decode(node.value)
    return loads(serialized_function)  # noqa: S301


def lambda_representer(representer, data):
    """Represent a lambda function by serializing its code."""
    serialized_lambda = base64.b64encode(dumps(data, recurse=True)).decode("utf-8")
    return representer.represent_scalar("!lambda", serialized_lambda)


def lambda_constructor(constructor, node):
    """Reconstruct a lambda function from the serialized data."""
    # Decode the base64-encoded string and load the lambda function
    serialized_lambda = base64.b64decode(node.value)
    return loads(serialized_lambda)  # noqa: S301


def pydantic_model_representer(representer, data):
    """Representer for Pydantic Models."""
    value = {"type": f"{data.__class__.__module__}.{data.__class__.__name__}", "data": data.model_dump()}
    return representer.represent_mapping("!PydanticModel", value)


def pydantic_model_constructor(constructor, node):
    """Constructor for Pydantic Models."""
    mapping = constructor.construct_mapping(node, deep=True)
    model_type_str = mapping["type"]
    data = mapping["data"]
    module_name, class_name = model_type_str.rsplit(".", 1)
    mod = __import__(module_name, fromlist=[class_name])
    model_cls = getattr(mod, class_name)
    return model_cls.model_validate(data)


def complex_representer(representer, data: complex):
    value = {"real": data.real, "imag": data.imag}
    return representer.represent_mapping("!complex", value)


def complex_constructor(constructor, node):
    mapping = constructor.construct_mapping(node, deep=True)
    return complex(mapping["real"], mapping["imag"])


def tuple_representer(representer, data: tuple):
    """Representer for built-in Python tuple."""
    # Emit a tuple as a YAML sequence with tag !tuple
    return representer.represent_sequence("!tuple", list(data))


def tuple_constructor(constructor, node):
    """Constructor for built-in Python tuple."""
    seq = constructor.construct_sequence(node, deep=True)
    return tuple(seq)


def type_representer(representer, data: type):
    """
    Represent any Python class/type by its import path.
    E.g. datetime.datetime → 'datetime.datetime'
    """
    path = f"{data.__module__}.{data.__qualname__}"
    # emit as a simple scalar under !type
    return representer.represent_scalar("!type", path)


def type_constructor(constructor, node):
    """
    Reconstruct a class/type from its import path.
    """
    path = node.value  # e.g. "datetime.datetime"
    module_name, qualname = path.rsplit(".", 1)
    mod = __import__(module_name, fromlist=[qualname])
    return getattr(mod, qualname)


# Create YAML handler and register all custom types
yaml = YAML(typ="unsafe")

# SciPy CSR
yaml.representer.add_representer(sparse.csr_matrix, csr_representer)
yaml.constructor.add_constructor("!csr_matrix", csr_constructor)

# NumPy scalars
yaml.representer.add_multi_representer(np.generic, np_scalar_representer)
yaml.constructor.add_constructor("!np_scalar", np_scalar_constructor)

# defaultdict
yaml.representer.add_representer(defaultdict, defaultdict_representer)
yaml.constructor.add_constructor("!defaultdict", defaultdict_constructor)

# NumPy arrays
yaml.representer.add_representer(np.ndarray, ndarray_representer)
yaml.constructor.add_constructor("!ndarray", ndarray_constructor)

# Python functions and lambdas
yaml.representer.add_representer(types.FunctionType, function_representer)
yaml.constructor.add_constructor("!function", function_constructor)
yaml.representer.add_representer(types.LambdaType, lambda_representer)
yaml.constructor.add_constructor("!lambda", lambda_constructor)

# Pydantic models
yaml.representer.add_representer(BaseModel, pydantic_model_representer)
yaml.constructor.add_constructor("!PydanticModel", pydantic_model_constructor)

# Built-in complex numbers
yaml.representer.add_representer(complex, complex_representer)
yaml.constructor.add_constructor("!complex", complex_constructor)

# Built-in tuples
yaml.representer.add_representer(tuple, tuple_representer)
yaml.constructor.add_constructor("!tuple", tuple_constructor)

# Built-in type
yaml.representer.add_multi_representer(type, type_representer)
yaml.constructor.add_constructor("!type", type_constructor)
