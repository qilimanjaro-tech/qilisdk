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

# ruff: file-ignore[missing-type-function-argument, missing-return-type-undocumented-public-function, docstring-missing-returns]

"""Safe YAML handler for QiliSDK objects.

The loader never imports modules or executes code taken from its input. Only built-in YAML types, the custom tags
registered in this module, and classes registered through :meth:`QiliYAML.register_class` can be reconstructed, so
payloads received from untrusted sources (e.g. sent to or returned by SpeQtrum) cannot trigger code execution.
"""

import functools
import types
from collections import defaultdict, deque
from enum import Enum

import numpy as np
from loguru import logger
from pydantic import BaseModel
from ruamel.yaml import YAML
from ruamel.yaml.constructor import ConstructorError, SafeConstructor
from ruamel.yaml.nodes import SequenceNode
from ruamel.yaml.representer import RepresenterError, SafeRepresenter
from scipy import sparse

_REGISTRY: dict[str, type] = {}
"""Classes registered through :meth:`QiliYAML.register_class`, keyed by ``module.qualname``."""

_TAGS: dict[str, type] = {}
"""Classes registered through :meth:`QiliYAML.register_class`, keyed by YAML tag."""

_ALLOWED_DTYPE_KINDS = frozenset("biufc")
"""NumPy dtype kinds (bool, int, uint, float, complex) accepted when reconstructing arrays and scalars."""

_DEFAULTDICT_FACTORIES: dict[str | None, type | None] = {
    None: None,
    "builtins.complex": complex,
    "builtins.dict": dict,
    "builtins.float": float,
    "builtins.int": int,
    "builtins.list": list,
    "builtins.set": set,
}
"""Factories accepted when reconstructing a ``defaultdict``."""

_LEGACY_APPLY_TAG = "tag:yaml.org,2002:python/object/apply:"
"""Tag prefix emitted for enum members by QiliSDK versions that used the unsafe YAML handler."""


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _lookup_registered(name: str, node) -> type:
    """
    Resolve a registered class by its qualified name.

    Raises:
        ConstructorError: If no class with that name has been registered.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ConstructorError(None, None, f"'{name}' is not a registered QiliSDK class", node.start_mark)
    return cls


def _check_dtype(dtype_name: str, node) -> np.dtype:
    """
    Parse a dtype name, accepting only numeric and boolean dtypes.

    Raises:
        ConstructorError: If the dtype is not numeric or boolean.
    """
    dtype = np.dtype(dtype_name)
    if dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ConstructorError(None, None, f"dtype '{dtype_name}' is not allowed", node.start_mark)
    return dtype


def _check_representable_dtype(dtype: np.dtype) -> None:
    """
    Reject dtypes that the loader would refuse, so they fail when saving rather than when loading.

    Raises:
        RepresenterError: If the dtype is not numeric or boolean.
    """
    if dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise RepresenterError(f"dtype '{dtype}' cannot be serialized; only numeric and boolean dtypes are supported")


def _checked_array(values, kinds: str, name: str, node) -> np.ndarray:
    """
    Convert a loaded list to an array, accepting only the given dtype kinds (empty lists are always accepted).

    Raises:
        ConstructorError: If the values do not form an array of an accepted kind.
    """
    array = np.asarray(values)
    if array.size > 0 and array.dtype.kind not in kinds:
        raise ConstructorError(
            None, None, f"csr_matrix {name} of dtype '{array.dtype}' is not allowed", node.start_mark
        )
    return array


class QiliConstructor(SafeConstructor):
    """Safe YAML constructor holding the QiliSDK tags, isolated from other ``ruamel.yaml`` users."""


class QiliRepresenter(SafeRepresenter):
    """Safe YAML representer holding the QiliSDK tags, isolated from other ``ruamel.yaml`` users."""

    def represent_yaml_object(self, tag, data, cls, flow_style=None):
        """
        Represent a registered object as a mapping of its state.

        ``object.__getstate__`` returns ``None`` for instances without attributes, which is emitted as an empty mapping.
        """
        state = data.__getstate__()
        return self.represent_mapping(tag, {} if state is None else state, flow_style=flow_style)


def csr_representer(representer, data: sparse.csr_matrix):
    """
    Representer for CSR matrix.
    """
    logger.trace("[Yaml] Serializing csr_matrix")
    _check_representable_dtype(data.dtype)
    value = {
        "data": data.data.tolist(),
        "indices": data.indices.tolist(),
        "indptr": data.indptr.tolist(),
        "shape": data.shape,
    }
    return representer.represent_mapping("!csr_matrix", value)


def csr_constructor(constructor, node):
    """
    Constructor for CSR matrix. Only numeric and boolean data and integer indices are accepted.
    """
    logger.trace("[Yaml] Deserializing csr_matrix")
    mapping = constructor.construct_mapping(node, deep=True)
    data = _checked_array(mapping["data"], "".join(_ALLOWED_DTYPE_KINDS), "data", node)
    indices = _checked_array(mapping["indices"], "iu", "indices", node).astype(np.int64)
    indptr = _checked_array(mapping["indptr"], "iu", "indptr", node).astype(np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=tuple(mapping["shape"]))


def ndarray_representer(representer, data):
    """
    Representer for ndarray.
    """
    logger.trace("[Yaml] Serializing ndarray")
    _check_representable_dtype(data.dtype)
    value = {"dtype": str(data.dtype), "shape": data.shape, "data": data.ravel().tolist()}
    return representer.represent_mapping("!ndarray", value)


def ndarray_constructor(constructor, node):
    """
    Constructor for ndarray. Only numeric and boolean dtypes are accepted.
    """
    logger.trace("[Yaml] Deserializing ndarray")
    mapping = constructor.construct_mapping(node, deep=True)
    dtype = _check_dtype(mapping["dtype"], node)
    shape = tuple(mapping["shape"])
    data = mapping["data"]
    return np.array(data, dtype=dtype).reshape(shape)


def np_scalar_representer(representer, data: np.generic):
    """
    Represent any NumPy scalar (e.g. np.int64, np.float32).
    """
    logger.trace("[Yaml] Serializing numpy scalar")
    _check_representable_dtype(data.dtype)
    return representer.represent_mapping(
        "!np_scalar",
        {"dtype": str(data.dtype), "value": data.item()},
    )


def np_scalar_constructor(constructor, node):
    """
    Reconstruct a NumPy scalar. Only numeric and boolean dtypes are accepted.
    """
    logger.trace("[Yaml] Deserializing numpy scalar")
    mapping = constructor.construct_mapping(node, deep=True)
    dtype = _check_dtype(mapping["dtype"], node)
    return dtype.type(mapping["value"])


def defaultdict_representer(representer, data: defaultdict):
    """
    Represent a defaultdict by serializing its default_factory
    (as module+qualname) plus its items dict.

    Raises:
        RepresenterError: If the factory is not in the allowlist.
    """
    logger.trace("[Yaml] Serializing defaultdict")
    factory = data.default_factory
    factory_name = None if factory is None else _qualified_name(factory) if isinstance(factory, type) else None
    if factory is not None and (factory_name is None or _DEFAULTDICT_FACTORIES.get(factory_name) is not factory):
        raise RepresenterError(f"defaultdict factory {factory!r} cannot be serialized")
    return representer.represent_mapping(
        "!defaultdict",
        {"default_factory": factory_name, "items": dict(data)},
    )


def defaultdict_constructor(constructor, node):
    """
    Reconstruct a defaultdict, restoring its factory from a fixed allowlist and its contents.

    Raises:
        ConstructorError: If the factory is not in the allowlist.
    """
    logger.trace("[Yaml] Deserializing defaultdict")
    mapping = constructor.construct_mapping(node, deep=True)
    fname = mapping["default_factory"]
    if fname not in _DEFAULTDICT_FACTORIES:
        raise ConstructorError(None, None, f"defaultdict factory '{fname}' is not allowed", node.start_mark)
    dd = defaultdict(_DEFAULTDICT_FACTORIES[fname])
    dd.update(mapping["items"])
    return dd


def callable_representer(representer, data):
    """
    Refuse to serialize callables, since loading them back would require executing code from the payload.

    Raises:
        RepresenterError: Always.
    """
    raise RepresenterError(
        f"Cannot serialize callable {data!r}: functions and lambdas are not serializable. "
        "Use an Expression or Parameter, or a named option, instead."
    )


def code_constructor(constructor, node):
    """
    Refuse to load legacy ``!function`` / ``!lambda`` payloads, which carry pickled code.

    Raises:
        ConstructorError: Always.
    """
    raise ConstructorError(
        None,
        None,
        f"Tag '{node.tag}' carries pickled code and is no longer supported for security reasons",
        node.start_mark,
    )


def pydantic_model_representer(representer, data):
    """
    Representer for registered Pydantic Models.

    Raises:
        RepresenterError: If the model class is not registered.
    """
    logger.trace("[Yaml] Serializing pydantic model {}", data.__class__.__name__)
    model_type_str = _qualified_name(data.__class__)
    if _REGISTRY.get(model_type_str) is not data.__class__:
        raise RepresenterError(f"Pydantic model '{model_type_str}' is not registered for serialization")
    value = {"type": model_type_str, "data": data.model_dump()}
    return representer.represent_mapping("!PydanticModel", value)


def pydantic_model_constructor(constructor, node):
    """
    Constructor for registered Pydantic Models.

    Raises:
        ConstructorError: If the type is not a registered Pydantic model.
    """
    logger.trace("[Yaml] Deserializing pydantic model")
    mapping = constructor.construct_mapping(node, deep=True)
    model_cls = _lookup_registered(mapping["type"], node)
    if not issubclass(model_cls, BaseModel):
        raise ConstructorError(None, None, f"'{mapping['type']}' is not a Pydantic model", node.start_mark)
    return model_cls.model_validate(mapping["data"])


def complex_representer(representer, data: complex):
    """
    Representer for built-in Python complex numbers.
    """
    logger.trace("[Yaml] Serializing complex")
    value = {"real": data.real, "imag": data.imag}
    return representer.represent_mapping("!complex", value)


def complex_constructor(constructor, node):
    """
    Constructor for built-in Python complex numbers.
    """
    logger.trace("[Yaml] Deserializing complex")
    mapping = constructor.construct_mapping(node, deep=True)
    return complex(mapping["real"], mapping["imag"])


def tuple_representer(representer, data: tuple):
    """
    Representer for built-in Python tuple.
    """
    logger.trace("[Yaml] Serializing tuple")
    # Emit a tuple as a YAML sequence with tag !tuple
    return representer.represent_sequence("!tuple", list(data))


def tuple_constructor(constructor, node):
    """
    Constructor for built-in Python tuple.
    """
    logger.trace("[Yaml] Deserializing tuple")
    seq = constructor.construct_sequence(node, deep=True)
    return tuple(seq)


def type_representer(representer, data: type):
    """
    Represent a registered class by its import path.

    Raises:
        RepresenterError: If the class is not registered.
    """
    logger.trace("[Yaml] Serializing type {}", data.__qualname__)
    path = _qualified_name(data)
    if _REGISTRY.get(path) is not data:
        raise RepresenterError(f"Type '{path}' is not registered for serialization")
    return representer.represent_scalar("!type", path)


def type_constructor(constructor, node):
    """
    Reconstruct a registered class from its import path.
    """
    logger.trace("[Yaml] Deserializing type")
    return _lookup_registered(node.value, node)


def deque_representer(representer, data):
    """
    Representer for deque
    """
    logger.trace("[Yaml] Serializing deque")
    return representer.represent_sequence("!deque", list(data))


def deque_constructor(constructor, node):
    """
    Constructor for deque
    """
    logger.trace("[Yaml] Deserializing deque")
    return deque(constructor.construct_sequence(node))


def legacy_enum_constructor(constructor, tag_suffix, node):
    """
    Load enum members written by QiliSDK versions that used the unsafe YAML handler
    (``!!python/object/apply:<module>.<Enum> [value]``). Only registered enums are accepted.

    Raises:
        ConstructorError: If the tag does not name a registered enum with a single value.
    """
    logger.trace("[Yaml] Deserializing legacy enum {}", tag_suffix)
    enum_cls = _lookup_registered(tag_suffix, node)
    if not issubclass(enum_cls, Enum) or not isinstance(node, SequenceNode) or len(node.value) != 1:
        raise ConstructorError(None, None, f"Unsupported legacy tag '{node.tag}'", node.start_mark)
    (value,) = constructor.construct_sequence(node)
    return enum_cls(value)


class QiliYAML(YAML):
    """
    Safe YAML handler for QiliSDK.

    All instances share the tags registered on :class:`QiliConstructor` and :class:`QiliRepresenter`, so a new
    instance can be created per (de)serialization call without re-registering classes.
    """

    def __init__(self) -> None:
        """
        Initialize the YAML handler with the safe, pure-Python loader and dumper.
        """
        logger.debug("[Yaml] Initializing QiliYAML handler")
        super().__init__(typ="safe", pure=True)
        self.Constructor = QiliConstructor
        self.Representer = QiliRepresenter

    def register_class(self, cls=None, *, shared: bool = False):
        """
        Register a class with the YAML handler, assigning it a unique tag.

        Registered classes are the only ones the loader reconstructs. Pydantic models are serialized through the
        ``!PydanticModel`` tag (and validated on load), and enums without ``to_yaml`` are serialized by value.

        Raises:
            ValueError: If another class is already registered under the same tag.
            TypeError: If an enum without ``to_yaml`` has non-string values.
        """
        if cls is None:

            def decorator(target_cls):  # ruff: ignore[missing-return-type-private-function]
                return self.register_class(target_cls, shared=shared)

            return decorator
        logger.trace("[Yaml] Registering class {}", cls.__name__)
        name = _qualified_name(cls)
        if issubclass(cls, BaseModel):
            _REGISTRY[name] = cls
            return cls
        tag = cls.__dict__.get("yaml_tag", None) or (
            f"!{cls.__module__.split('.')[0]}.{cls.__name__}" if shared else f"!{cls.__name__}"
        )
        # Re-registering a class with the same qualified name (e.g. after a module reload) replaces it.
        existing = _TAGS.get(tag)
        if existing is not None and _qualified_name(existing) != name:
            raise ValueError(
                f"Cannot register {name} under tag '{tag}': already used by {_qualified_name(existing)}. "
                "Rename the class or give it a distinct yaml_tag."
            )
        _REGISTRY[name] = cls
        _TAGS[tag] = cls
        cls.yaml_tag = tag
        if issubclass(cls, Enum) and not hasattr(cls, "to_yaml"):
            if not all(isinstance(member.value, str) for member in cls):
                raise TypeError(f"Enum {cls.__name__} must have string values to be registered")
            self.representer.add_representer(cls, lambda rep, data: rep.represent_scalar(tag, data.value))
            self.constructor.add_constructor(tag, lambda _, node: cls(node.value))
            return cls
        return super().register_class(cls)


yaml = QiliYAML()

# SciPy CSR
QiliRepresenter.add_representer(sparse.csr_matrix, csr_representer)
QiliConstructor.add_constructor("!csr_matrix", csr_constructor)

# NumPy scalars
QiliRepresenter.add_multi_representer(np.generic, np_scalar_representer)
QiliConstructor.add_constructor("!np_scalar", np_scalar_constructor)

# defaultdict
QiliRepresenter.add_representer(defaultdict, defaultdict_representer)
QiliConstructor.add_constructor("!defaultdict", defaultdict_constructor)

# NumPy arrays
QiliRepresenter.add_representer(np.ndarray, ndarray_representer)
QiliConstructor.add_constructor("!ndarray", ndarray_constructor)

# Callables are never serialized; legacy pickled-code tags are rejected
for _callable_type in (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.BuiltinMethodType,
    functools.partial,
):
    QiliRepresenter.add_representer(_callable_type, callable_representer)
QiliConstructor.add_constructor("!function", code_constructor)
QiliConstructor.add_constructor("!lambda", code_constructor)

# Pydantic models
QiliRepresenter.add_multi_representer(BaseModel, pydantic_model_representer)
QiliConstructor.add_constructor("!PydanticModel", pydantic_model_constructor)

# Built-in complex numbers
QiliRepresenter.add_representer(complex, complex_representer)
QiliConstructor.add_constructor("!complex", complex_constructor)

# Built-in tuples
QiliRepresenter.add_representer(tuple, tuple_representer)
QiliConstructor.add_constructor("!tuple", tuple_constructor)

# Built-in type
QiliRepresenter.add_multi_representer(type, type_representer)
QiliConstructor.add_constructor("!type", type_constructor)

# Built-in deque
QiliRepresenter.add_representer(deque, deque_representer)
QiliConstructor.add_constructor("!deque", deque_constructor)

# Enum members written by older QiliSDK versions
QiliConstructor.add_multi_constructor(_LEGACY_APPLY_TAG, legacy_enum_constructor)
