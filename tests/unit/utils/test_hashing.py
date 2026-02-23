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

import numpy as np

from qilisdk.utils.hashing import hash as qili_hash


class _CustomHash:
    def __init__(self, value: int):
        self.value = value

    def __hash__(self) -> int:
        return self.value


class _StateObject:
    __hash__ = None

    def __init__(self, left: int, right: str):
        self.left = left
        self.right = right


def test_qili_hash_numeric_equivalence():
    assert qili_hash(1) == qili_hash(1.0)


def test_qili_hash_boolean_numeric_equivalence():
    assert qili_hash(True) == qili_hash(1)
    assert qili_hash(False) == qili_hash(0.0)


def test_qili_hash_complex_real_equivalence():
    assert qili_hash(3 + 0j) == qili_hash(3)
    assert qili_hash(3 + 1j) != qili_hash(3 + 2j)


def test_qili_hash_special_float_values():
    assert qili_hash(float("nan")) == qili_hash(np.nan)
    assert qili_hash(float("inf")) == qili_hash(np.inf)
    assert qili_hash(float("-inf")) == qili_hash(-np.inf)
    assert qili_hash(float("inf")) != qili_hash(float("-inf"))


def test_qili_hash_mapping_order_independent():
    assert qili_hash({"a": 1, "b": 2}) == qili_hash({"b": 2, "a": 1})


def test_qili_hash_set_order_independent():
    assert qili_hash({1, "x", 2.0}) == qili_hash({"x", 2.0, 1})


def test_qili_hash_sequence_order_dependent():
    assert qili_hash([1, 2]) != qili_hash([2, 1])


def test_qili_hash_multiple_arguments_order_dependent():
    assert qili_hash("a", 1) == qili_hash("a", 1)
    assert qili_hash("a", 1) != qili_hash(1, "a")


def test_qili_hash_ndarray_support():
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    b = np.array([[1, 2], [3, 4]], dtype=np.float64)
    c = np.array([[1, 2], [4, 3]], dtype=np.float64)

    assert qili_hash(a) == qili_hash(b)
    assert qili_hash(a) != qili_hash(c)


def test_qili_hash_uses_custom_hash_method_for_objects():
    a = _CustomHash(7)
    b = _CustomHash(7)
    c = _CustomHash(8)

    assert qili_hash(a) == qili_hash(b)
    assert qili_hash(a) != qili_hash(c)


def test_qili_hash_uses_object_state_when_object_is_unhashable():
    a = _StateObject(left=1, right="x")
    b = _StateObject(left=1, right="x")
    c = _StateObject(left=2, right="x")

    assert qili_hash(a) == qili_hash(b)
    assert qili_hash(a) != qili_hash(c)
