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

# ruff: noqa: ANN401

from io import StringIO
from pathlib import Path
from typing import Any, TypeVar, overload

from qilisdk.yaml import safe_yaml, yaml

T = TypeVar("T")


class SerializationError(Exception):
    """Custom exception for serialization errors."""


class DeserializationError(Exception):
    """Custom exception for deserialization errors."""


def serialize(obj: Any) -> str:
    """Serialize an object to a YAML string.

    Args:
        obj (Any): The object to serialize.

    Raises:
        SerializationError: If serialization fails.

    Returns:
        str: The serialized YAML string.
    """
    try:
        with StringIO() as stream:
            yaml.dump(obj, stream)
            return stream.getvalue()
    except Exception as e:
        raise SerializationError(f"Failed to serialize object {e}") from e


def serialize_to(obj: Any, file: str) -> None:
    """Serialize an object to a YAML file.

    Args:
        obj (Any): The object to serialize.
        file (str): The file path where the YAML data will be written.

    Raises:
        SerializationError: If serialization to file fails.
    """
    try:
        yaml.dump(obj, Path(file))
    except Exception as e:
        raise SerializationError(f"Failed to serialize object {e} to file {file}") from e


@overload
def deserialize(string: str, *, trust_code: bool = ...) -> Any: ...


@overload
def deserialize(string: str, cls: type[T], *, trust_code: bool = ...) -> T: ...


def deserialize(string: str, cls: type[T] | None = None, *, trust_code: bool = False) -> Any | T:
    """Deserialize a YAML string to an object.

    By default this uses a safe, data-only loader: documents carrying the
    code-bearing ``!function``/``!lambda`` tags, or ``!type``/``!PydanticModel``/
    ``!defaultdict`` tags pointing at a name that is not on the deserialization
    allow-list, are rejected with a :class:`DeserializationError` instead of
    executing arbitrary code at parse time.

    Args:
        string (str): The YAML string to deserialize.
        cls (type[T], optional): The class type to cast the deserialized object to. Defaults to None.
        trust_code (bool, optional): When ``True``, use the unrestricted loader
            that reconstructs ``!function``/``!lambda`` via ``dill`` and imports
            arbitrary ``!type``/``!PydanticModel``/``!defaultdict`` names. This
            executes arbitrary code and MUST only ever be used on fully trusted
            input; never wire it to network- or user-supplied data. Defaults to ``False``.

    Raises:
        DeserializationError: If deserialization fails, a code-bearing/non-allow-listed
            tag is rejected, or the resulting object is not of the specified type.

    Returns:
        Any | T: The deserialized object, optionally cast to the specified class type.
    """
    loader = yaml if trust_code else safe_yaml
    try:
        with StringIO(string) as stream:
            result = loader.load(stream)
    except Exception as e:
        raise DeserializationError(f"Failed to deserialize YAML string: {e}") from e
    if cls is not None and not isinstance(result, cls):
        raise DeserializationError(f"Deserialized object is not of type {cls.__name__}")
    return result


@overload
def deserialize_from(file: str, *, trust_code: bool = ...) -> Any: ...


@overload
def deserialize_from(file: str, cls: type[T], *, trust_code: bool = ...) -> T: ...


def deserialize_from(file: str, cls: type[T] | None = None, *, trust_code: bool = False) -> Any | T:
    """Deserialize a YAML file to an object.

    By default this uses the same safe, data-only loader as :func:`deserialize`:
    code-bearing or non-allow-listed tags are rejected rather than executed.

    Args:
        file (str): The file path of the YAML file to deserialize.
        cls (type[T], optional): The class type to cast the deserialized object to. Defaults to None.
        trust_code (bool, optional): When ``True``, use the unrestricted loader
            that executes arbitrary code via ``dill``/``__import__``. MUST only be
            used on fully trusted files; never wire it to network- or
            user-supplied paths. Defaults to ``False``.

    Raises:
        DeserializationError: If deserialization fails, a code-bearing/non-allow-listed
            tag is rejected, or the resulting object is not of the specified type.

    Returns:
        Any | T: The deserialized object, optionally cast to the specified class type.
    """
    loader = yaml if trust_code else safe_yaml
    try:
        result = loader.load(Path(file))
    except Exception as e:
        raise DeserializationError(f"Failed to deserialize YAML string {e} from file {file}") from e
    if cls is not None and not isinstance(result, cls):
        raise DeserializationError(f"Deserialized object is not of type {cls.__name__}")
    return result
