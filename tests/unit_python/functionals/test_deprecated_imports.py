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

import importlib
import sys

import pytest

import qilisdk.functionals
from qilisdk import algorithms


@pytest.mark.parametrize("name", ["QuantumReservoir", "VariationalProgram", "VariationalProgramResult"])
def test_moved_algorithm_imports_from_functionals_warn(name):
    with pytest.warns(DeprecationWarning, match=f"Importing {name} from qilisdk.functionals is deprecated"):
        moved = getattr(qilisdk.functionals, name)

    assert moved is getattr(algorithms, name)


@pytest.mark.parametrize("module", ["quantum_reservoirs", "variational_program", "variational_program_result"])
def test_moved_algorithm_modules_warn(module):
    sys.modules.pop(f"qilisdk.functionals.{module}", None)

    with pytest.warns(DeprecationWarning, match=f"qilisdk.functionals.{module} is deprecated"):
        importlib.import_module(f"qilisdk.functionals.{module}")


def test_unknown_functionals_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute 'NotAThing'"):
        _ = qilisdk.functionals.NotAThing
