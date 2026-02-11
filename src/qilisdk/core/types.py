# Copyright 2026 Qilimanjaro Quantum Tech
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

from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Self, TypeAlias

if TYPE_CHECKING:
    from ruamel.yaml.nodes import ScalarNode
    from ruamel.yaml.representer import RoundTripRepresenter


Number: TypeAlias = int | float | complex
RealNumber: TypeAlias = int | float


class YamledEnum(str, Enum):
    yaml_tag: ClassVar[str]

    @classmethod
    def to_yaml(cls, representer: RoundTripRepresenter, node: type[Self]) -> ScalarNode:
        """
        Method to be called automatically during YAML serialization.

        Returns:
            ScalarNode: The YAML scalar node representing the class.
        """
        return representer.represent_scalar(cls.yaml_tag, f"{node.value}")

    @classmethod
    def from_yaml(cls, _, node: ScalarNode) -> Self:
        """
        Method to be called automatically during YAML deserialization.

        Returns:
            Self: The class instance created from the YAML node value.
        """
        return cls(node.value)
