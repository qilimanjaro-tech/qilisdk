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

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Final, Literal, TypeAlias, cast

import numpy as np

_HASH_DIGEST_SIZE: Final[int] = 8
_CHUNK_SIZE_BYTES: Final[int] = 8


def _digest_to_hash_int(digest: bytes) -> int:
    value = int.from_bytes(digest, byteorder="big", signed=True)
    # -1 is reserved internally by Python's hashing C-API.
    return -2 if value == -1 else value


def _length_prefix(payload: bytes) -> bytes:
    return len(payload).to_bytes(_CHUNK_SIZE_BYTES, byteorder="big", signed=False)


def _encode_scalar(value: object) -> bytes:
    if value is None:
        return b"none"
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, np.generic):
        return _encode_scalar(value.item())
    if isinstance(value, (bool, float, int)):
        return b"real:" + _encode_real_number(value)
    if isinstance(value, complex):
        if value.imag == 0:
            return b"real:" + _encode_real_number(value.real)
        real_payload = _encode_real_number(value.real)
        imag_payload = _encode_real_number(value.imag)
        return b"complex:" + _length_prefix(real_payload) + real_payload + _length_prefix(imag_payload) + imag_payload
    return b""


def _encode_real_number(value: float | bool) -> bytes:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return f"{value}/1".encode("utf-8")

    if math.isnan(value):
        return b"nan"
    if math.isinf(value):
        return b"+inf" if value > 0 else b"-inf"

    numerator, denominator = value.as_integer_ratio()
    return f"{numerator}/{denominator}".encode("utf-8")


#: Which branch of :func:`_encode_object` a value takes, decided once per class
_EncodeKind: TypeAlias = Literal["scalar", "ndarray", "mapping", "set", "sequence", "other"]

_KIND_CACHE: Final[dict[type, _EncodeKind]] = {}
_CLASS_NAME_CACHE: Final[dict[type, bytes]] = {}


def _encode_kind(cls: type) -> _EncodeKind:
    """Return which branch of :func:`_encode_object` a class takes.

    The answer depends only on the class, but asking it means a chain of ``isinstance`` checks,
    four of them against ``collections.abc`` ABCs, which dominates hashing of deeply nested
    objects. Caching it per class turns the whole chain into a single dict lookup. The order below
    must match the order of the branches it stands in for.

    Returns:
        _EncodeKind: the branch this class belongs to.
    """
    kind = _KIND_CACHE.get(cls)
    if kind is None:
        if cls is type(None) or issubclass(cls, (str, bytes, np.generic, bool, float, int, complex)):
            kind = "scalar"
        elif issubclass(cls, np.ndarray):
            kind = "ndarray"
        elif issubclass(cls, Mapping):
            kind = "mapping"
        elif issubclass(cls, AbstractSet):
            kind = "set"
        elif issubclass(cls, Sequence):
            kind = "sequence"
        else:
            kind = "other"
        _KIND_CACHE[cls] = kind
    return kind


def _class_name(cls: type) -> bytes:
    """Return the encoded ``module.qualname`` of a class.

    Returns:
        bytes: the utf-8 encoded fully qualified class name.
    """
    name = _CLASS_NAME_CACHE.get(cls)
    if name is None:
        name = f"{cls.__module__}.{cls.__qualname__}".encode("utf-8")
        _CLASS_NAME_CACHE[cls] = name
    return name


def _encode_object_via_custom_hash(value: object) -> bytes:
    object_hash_method = value.__class__.__hash__
    if object_hash_method in {None, object.__hash__}:
        return b""

    object_hash = object_hash_method(value)  # ty:ignore[too-many-positional-arguments]
    return b"object-hash:" + _class_name(value.__class__) + b":" + str(object_hash).encode("utf-8")


def _encode_object(value: object) -> bytes:
    kind = _encode_kind(value.__class__)

    if kind == "scalar":
        return _encode_scalar(value)

    if kind == "ndarray":
        value = cast("np.ndarray", value)
        dtype_payload = str(value.dtype).encode("utf-8")
        shape_payload = ",".join(str(dim) for dim in value.shape).encode("utf-8")
        data_payload = value.tobytes()
        return (
            b"ndarray:"
            + _length_prefix(dtype_payload)
            + dtype_payload
            + _length_prefix(shape_payload)
            + shape_payload
            + data_payload
        )

    if kind == "mapping":
        value = cast("Mapping", value)
        encoded_items = [_encode_object(key) + _encode_object(item) for key, item in value.items()]
        encoded_items.sort()
        return b"mapping:" + b"".join(_length_prefix(item) + item for item in encoded_items)

    if kind == "set":
        value = cast("AbstractSet", value)
        encoded_items = [_encode_object(item) for item in value]
        encoded_items.sort()
        return b"set:" + b"".join(_length_prefix(item) + item for item in encoded_items)

    if kind == "sequence":
        value = cast("Sequence", value)
        encoded_items = [_encode_object(item) for item in value]
        return b"sequence:" + b"".join(_length_prefix(item) + item for item in encoded_items)

    custom_hash_payload = _encode_object_via_custom_hash(value)
    if custom_hash_payload:
        return custom_hash_payload

    if hasattr(value, "__dict__"):
        state_payload = _encode_object(vars(value))
        return b"object-state:" + _class_name(value.__class__) + b":" + state_payload

    class_name = f"{value.__class__.__module__}.{value.__class__.__qualname__}".encode("utf-8")
    return b"object-repr:" + class_name + b":" + repr(value).encode("utf-8")


def hash(*objects: object) -> int:
    """Stable blake2b-based hash for arbitrary Python objects used by qilisdk models.

    Returns:
        int: blake2b-based hash integer compatible with Python's ``__hash__``.
    """
    hasher = hashlib.blake2b(digest_size=_HASH_DIGEST_SIZE)
    for value in objects:
        payload = _encode_object(value)
        hasher.update(_length_prefix(payload))
        hasher.update(payload)
    return _digest_to_hash_int(hasher.digest())
