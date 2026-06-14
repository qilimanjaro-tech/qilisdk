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

# ruff: noqa: ANN001, ANN201, ANN202, DOC201, S403

import base64
import types
from collections import defaultdict, deque
from typing import ClassVar

import numpy as np
from dill import dumps, loads
from pydantic import BaseModel
from ruamel.yaml import YAML
from ruamel.yaml.constructor import ConstructorError
from scipy import sparse


def csr_representer(representer, data: sparse.csr_matrix):
    """
    Representer for CSR matrix.
    """
    value = {
        "data": data.data.tolist(),
        "indices": data.indices.tolist(),
        "indptr": data.indptr.tolist(),
        "shape": data.shape,
    }
    return representer.represent_mapping("!csr_matrix", value)


def csr_constructor(constructor, node):
    """
    Constructor for CSR matrix.
    """
    mapping = constructor.construct_mapping(node, deep=True)
    return sparse.csr_matrix(
        (mapping["data"], mapping["indices"], mapping["indptr"]),
        shape=tuple(mapping["shape"]),
    )


def ndarray_representer(representer, data):
    """
    Representer for ndarray.
    """
    value = {"dtype": str(data.dtype), "shape": data.shape, "data": data.ravel().tolist()}
    return representer.represent_mapping("!ndarray", value)


def ndarray_constructor(constructor, node):
    """
    Constructor for ndarray.
    """
    mapping = constructor.construct_mapping(node, deep=True)
    dtype = np.dtype(mapping["dtype"])
    shape = tuple(mapping["shape"])
    data = mapping["data"]
    return np.array(data, dtype=dtype).reshape(shape)


def np_scalar_representer(representer, data: np.generic):
    """
    Represent any NumPy scalar (e.g. np.int64, np.float32).
    """
    return representer.represent_mapping(
        "!np_scalar",
        {"dtype": str(data.dtype), "value": data.item()},
    )


def np_scalar_constructor(constructor, node):
    """
    Reconstruct a NumPy scalar.
    """
    mapping = constructor.construct_mapping(node, deep=True)
    dtype = np.dtype(mapping["dtype"])
    return dtype.type(mapping["value"])


def defaultdict_representer(representer, data: defaultdict):
    """
    Represent a defaultdict by serializing its default_factory
    (as module+qualname) plus its items dict.
    """
    factory = data.default_factory
    factory_name = (
        f"{factory.__module__}.{factory.__qualname__}"
        if factory is not None and hasattr(factory, "__qualname__")
        else None
    )
    return representer.represent_mapping(
        "!defaultdict",
        {"default_factory": factory_name, "items": dict(data)},
    )


def defaultdict_constructor(constructor, node):
    """
    Reconstruct a defaultdict, restoring its factory and contents.
    """
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
    """
    Represent a non-lambda function by serializing it.
    """
    serialized_function = base64.b64encode(dumps(data, recurse=True)).decode("utf-8")
    return representer.represent_scalar("!function", serialized_function)


def function_constructor(constructor, node):
    """
    Reconstruct a function from the serialized data.
    """
    serialized_function = base64.b64decode(node.value)
    return loads(serialized_function)  # noqa: S301


def lambda_representer(representer, data):
    """
    Represent a lambda function by serializing its code.
    """
    serialized_lambda = base64.b64encode(dumps(data, recurse=True)).decode("utf-8")
    return representer.represent_scalar("!lambda", serialized_lambda)


def lambda_constructor(constructor, node):
    """
    Reconstruct a lambda function from the serialized data.
    """
    # Decode the base64-encoded string and load the lambda function
    serialized_lambda = base64.b64decode(node.value)
    return loads(serialized_lambda)  # noqa: S301


def pydantic_model_representer(representer, data):
    """
    Representer for Pydantic Models.
    """
    value = {"type": f"{data.__class__.__module__}.{data.__class__.__name__}", "data": data.model_dump()}
    return representer.represent_mapping("!PydanticModel", value)


def pydantic_model_constructor(constructor, node):
    """
    Constructor for Pydantic Models.
    """
    mapping = constructor.construct_mapping(node, deep=True)
    model_type_str = mapping["type"]
    data = mapping["data"]
    module_name, class_name = model_type_str.rsplit(".", 1)
    mod = __import__(module_name, fromlist=[class_name])
    model_cls = getattr(mod, class_name)
    return model_cls.model_validate(data)


def complex_representer(representer, data: complex):
    """
    Representer for built-in Python complex numbers.
    """
    value = {"real": data.real, "imag": data.imag}
    return representer.represent_mapping("!complex", value)


def complex_constructor(constructor, node):
    """
    Constructor for built-in Python complex numbers.
    """
    mapping = constructor.construct_mapping(node, deep=True)
    return complex(mapping["real"], mapping["imag"])


def tuple_representer(representer, data: tuple):
    """
    Representer for built-in Python tuple.
    """
    # Emit a tuple as a YAML sequence with tag !tuple
    return representer.represent_sequence("!tuple", list(data))


def tuple_constructor(constructor, node):
    """
    Constructor for built-in Python tuple.
    """
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


def deque_representer(representer, data):
    """
    Representer for deque
    """
    return representer.represent_sequence("!deque", list(data))


def deque_constructor(constructor, node):
    """
    Constructor for ndarray
    """
    return deque(constructor.construct_sequence(node))


# --------------------------------------------------------------------------- #
# Safe-by-default loader
# --------------------------------------------------------------------------- #
# The constructors above for `!function`/`!lambda` run `dill.loads` and the ones
# for `!type`/`!PydanticModel`/`!defaultdict` run `__import__` + `getattr` on an
# attacker-controlled fully-qualified name. Both execute arbitrary code at parse
# time, which is an RCE for any caller that feeds untrusted YAML into `yaml.load`
# (CWE-502 / CWE-94). The public `deserialize`/`deserialize_from` API and the
# SpeQtrum server-response validators are exactly such callers.
#
# To make deserialization safe-by-default we build a SECOND loader, `safe_yaml`,
# which:
#   * reuses every pure-data constructor verbatim
#     (`!csr_matrix`, `!ndarray`, `!np_scalar`, `!complex`, `!tuple`, `!deque`),
#   * gates `!type`/`!PydanticModel`/`!defaultdict` behind a strict allow-list of
#     fully-qualified names (resolved/checked BEFORE any import happens), and
#   * REJECTS the code-bearing `!function`/`!lambda` tags instead of running dill.
#
# The original `yaml` instance is preserved unchanged as the dumper and as the
# explicit-opt-in trusted loader (`deserialize(..., trust_code=True)`), so the
# `@yaml.register_class` decorator surface and `serialize` output are untouched.

#: Fully-qualified names that `safe_yaml` is permitted to import for the
#: `!type`, `!PydanticModel` and `!defaultdict` constructors. These are the only
#: factories/classes the SDK legitimately round-trips through those generic tags
#: (the SDK's own classes use the `!ClassName` tags registered via
#: `register_class`, which are data-validated and handled separately). Keep this
#: list explicit and minimal; every addition is a deliberate, reviewed widening
#: of the deserialization attack surface.
SAFE_DESERIALIZATION_ALLOWLIST: frozenset[str] = frozenset(
    {
        # defaultdict factories used by the SDK (e.g. Hamiltonian, noise models)
        "builtins.complex",
        "builtins.list",
        "builtins.dict",
        "builtins.set",
        "builtins.tuple",
        "builtins.int",
        "builtins.float",
        "builtins.str",
        # `!type` values emitted by the SDK
        "qilisdk.core.variables.Bitwise",
        "qilisdk.core.variables.OneHot",
        "qilisdk.core.variables.DomainWall",
        # Targets of ruamel's generic `!!python/...` tags that the SDK legitimately
        # emits (plain enums / classes that do not use the `register_class` data
        # tags). These are gated through the same allow-list so the exact serialized
        # format is preserved without re-opening generic-object RCE.
        "qilisdk.core.interpolator.Interpolation",
        "qilisdk.cost_functions.observable_cost_function.ObservableCostFunction",
    }
)


def _resolve_allowlisted(path: str | None, tag: str):
    """Resolve a fully-qualified name only if it is on the allow-list.

    Args:
        path: The dotted ``module.qualname`` to resolve, or ``None``.
        tag: The YAML tag being constructed (for error messages).

    Returns:
        The resolved attribute, or ``None`` when *path* is ``None``.

    Raises:
        ConstructorError: If *path* is not present in
            :data:`SAFE_DESERIALIZATION_ALLOWLIST`.
    """
    if path is None:
        return None
    if path not in SAFE_DESERIALIZATION_ALLOWLIST:
        raise ConstructorError(
            None,
            None,
            f"refusing to deserialize {tag} pointing at non-allow-listed name {path!r}; "
            "this name is not part of the safe deserialization allow-list",
            None,
        )
    module, qual = path.rsplit(".", 1)
    mod = __import__(module, fromlist=[qual])
    return getattr(mod, qual)


def safe_defaultdict_constructor(constructor, node):
    """Reconstruct a defaultdict, allowing only allow-listed factories."""
    mapping = constructor.construct_mapping(node, deep=True)
    factory = _resolve_allowlisted(mapping["default_factory"], "!defaultdict")
    dd = defaultdict(factory)
    dd.update(mapping["items"])
    return dd


def safe_type_constructor(constructor, node):
    """Reconstruct a class/type, allowing only allow-listed import paths."""
    return _resolve_allowlisted(node.value, "!type")


def safe_pydantic_model_constructor(constructor, node):
    """Reconstruct a Pydantic model, allowing only allow-listed model classes."""
    mapping = constructor.construct_mapping(node, deep=True)
    model_cls = _resolve_allowlisted(mapping["type"], "!PydanticModel")
    return model_cls.model_validate(mapping["data"])


def _reject_code_constructor(tag: str):
    """Build a constructor that refuses a code-bearing tag instead of executing it."""

    def _constructor(constructor, node):
        raise ConstructorError(
            None,
            None,
            f"refusing to deserialize code-bearing tag {tag}; this tag executes "
            "arbitrary code via dill. Pass trust_code=True only for fully trusted input.",
            None,
        )

    return _constructor


def _allowlisted_python_multi_constructor(wrapped, prefix: str):
    """Wrap a ruamel ``!!python/...`` multi-constructor with an allow-list check.

    ruamel's built-in ``!!python/object``, ``!!python/object/apply``,
    ``!!python/object/new``, ``!!python/name`` and ``!!python/module`` tags import
    and instantiate the fully-qualified name carried in the tag *suffix*, which is
    a generic arbitrary-code / arbitrary-import vector (e.g.
    ``!!python/object/apply:os.system``). The SDK only emits a handful of these
    for plain enums / classes that do not use the ``register_class`` data tags, so
    we keep the exact serialized format but only allow suffixes on
    :data:`SAFE_DESERIALIZATION_ALLOWLIST`.

    Args:
        wrapped: The original ruamel multi-constructor (the unbound function from
            the constructor-class registry, invoked as ``wrapped(self, suffix, node)``).
        prefix: The ``!!python/...`` tag prefix being guarded (for error messages).

    Returns:
        A multi-constructor with the same ``(self, suffix, node)`` signature that
        raises :class:`ConstructorError` for any non-allow-listed suffix.
    """

    def _constructor(constructor, suffix, node):
        if suffix not in SAFE_DESERIALIZATION_ALLOWLIST:
            raise ConstructorError(
                None,
                None,
                f"refusing to deserialize {prefix}{suffix}; this name is not part of "
                "the safe deserialization allow-list",
                None,
            )
        return wrapped(constructor, suffix, node)

    return _constructor


# Create YAML handler and register all custom types
class QiliYAML(YAML):
    """
    Custom YAML handler for QiliSDK.
    """

    def __init__(self, **kwargs: list[str] | str | None) -> None:
        """
        Initialize the YAML handler with custom settings.
        """
        super().__init__(**kwargs)

    #: Loaders that must mirror the constructor side of every `register_class`
    #: call (so the SDK's `!ClassName` data tags also load on `safe_yaml`).
    #: Populated below, after the safe instance exists.
    _mirror_loaders: ClassVar[list[YAML]] = []

    def register_class(self, cls=None, *, shared: bool = False):
        """
        Register a class with the YAML handler, assigning it a unique tag.

        The class is also registered on every loader in ``_mirror_loaders`` so a
        single ``@yaml.register_class`` keeps the safe-by-default loader in sync
        with the SDK's own ``!ClassName`` data tags (these reconstruct via
        ``__new__``/attrs and are safe to expose on the safe loader).
        """
        if cls is None:

            def decorator(target_cls):
                return self.register_class(target_cls, shared=shared)

            return decorator
        if not cls.__dict__.get("yaml_tag", None):
            cls.yaml_tag = f"!{cls.__module__.split('.')[0]}.{cls.__name__}" if shared else f"!{cls.__name__}"
        result = super().register_class(cls)
        # Mirror the exact constructor ruamel just registered (which is either
        # `cls.from_yaml` or an internal `construct_yaml_object` closure) onto the
        # safe loader(s), so `!ClassName` data tags load there too. ruamel's
        # `add_constructor` is a classmethod that mutates a *shared* class-level
        # registry, so the safe loaders use their own per-instance
        # `yaml_constructors` dict; write the mirrored entry straight into it.
        constructor = self.constructor.yaml_constructors.get(cls.yaml_tag)
        if constructor is not None:
            for loader in self._mirror_loaders:
                loader.constructor.yaml_constructors[cls.yaml_tag] = constructor
        return result


yaml = QiliYAML(typ="unsafe")

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

# Built-in deque
yaml.representer.add_representer(deque, deque_representer)
yaml.constructor.add_constructor("!deque", deque_constructor)


# --------------------------------------------------------------------------- #
# Safe loader used by the public deserialize API (safe-by-default)
# --------------------------------------------------------------------------- #
# Reuses the pure-data constructors verbatim, gates the import-driven tags behind
# the allow-list, and rejects the dill-backed code tags. It carries no
# representers (it is never used for dumping). The SDK's own `!ClassName` tags are
# mirrored onto it automatically via `QiliYAML.register_class` (see below).
#
# IMPORTANT: ruamel's `add_constructor` is a classmethod that mutates a registry
# shared by every instance of the constructor class, so a second `QiliYAML`
# instance would NOT be isolated from `yaml`. We therefore give the safe loader
# its own per-instance `yaml_constructors`/`yaml_multi_constructors` dicts (seeded
# from the trusted loader's current registry) and only ever mutate those dicts.
safe_yaml = QiliYAML(typ="unsafe")
safe_yaml.constructor.yaml_constructors = dict(yaml.constructor.yaml_constructors)
safe_yaml.constructor.yaml_multi_constructors = dict(yaml.constructor.yaml_multi_constructors)

# Override the import-driven tags with allow-list-gated constructors.
safe_yaml.constructor.yaml_constructors["!defaultdict"] = safe_defaultdict_constructor
safe_yaml.constructor.yaml_constructors["!type"] = safe_type_constructor
safe_yaml.constructor.yaml_constructors["!PydanticModel"] = safe_pydantic_model_constructor

# Override the code-bearing tags: reject instead of running dill.
safe_yaml.constructor.yaml_constructors["!function"] = _reject_code_constructor("!function")
safe_yaml.constructor.yaml_constructors["!lambda"] = _reject_code_constructor("!lambda")

# Gate ruamel's generic `!!python/...` object/import tags behind the allow-list.
# The copied dict shares the class-level multi-constructor bound methods; replace
# each with an allow-list-checked wrapper bound to the safe constructor.
for _py_prefix in (
    "tag:yaml.org,2002:python/object:",
    "tag:yaml.org,2002:python/object/apply:",
    "tag:yaml.org,2002:python/object/new:",
    "tag:yaml.org,2002:python/name:",
    "tag:yaml.org,2002:python/module:",
):
    _wrapped = safe_yaml.constructor.yaml_multi_constructors.get(_py_prefix)
    if _wrapped is not None:
        # `_wrapped` is the unbound function from the class registry; ruamel calls
        # multi-constructors as `constructor(self, tag_suffix, node)`, so the
        # wrapper keeps that same 3-arg signature.
        safe_yaml.constructor.yaml_multi_constructors[_py_prefix] = _allowlisted_python_multi_constructor(
            _wrapped, _py_prefix.removeprefix("tag:yaml.org,2002:")
        )

# From now on, every `@yaml.register_class` also lands its `!ClassName`
# constructor on `safe_yaml`. (Downstream modules import `yaml` and run their
# decorators only after this module has finished executing, so the mirror list is
# already populated when they register.)
QiliYAML._mirror_loaders.append(safe_yaml)  # noqa: SLF001
