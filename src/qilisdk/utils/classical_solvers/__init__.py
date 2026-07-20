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

import sys

from qilisdk._optionals import (
    DependencyGroup,
    ImportedFeature,
    OptionalFeature,
    RequirementMode,
    Symbol,
    import_optional_dependencies,
)

from .base_solver import ClassicalSolver
from .brute_force_solver import BruteForceSolver
from .scipy_solver import ScipySolver

__all__ = ["BruteForceSolver", "ClassicalSolver", "ScipySolver"]

# ScipSolver relies on the optional ``pyscipopt`` dependency
OPTIONAL_FEATURES: list[OptionalFeature] = [
    OptionalFeature(
        name="scip",
        mode=RequirementMode.ALL,
        dependency_groups=[DependencyGroup(dists=["pyscipopt"], extra="scip")],
        symbols=[Symbol(path="qilisdk.utils.classical_solvers.scip_solver", name="ScipSolver")],
    ),
]
current_module = sys.modules[__name__]
for feature in OPTIONAL_FEATURES:
    imported_feature: ImportedFeature = import_optional_dependencies(feature)
    for symbol_name, symbol_obj in imported_feature.symbols.items():
        setattr(current_module, symbol_name, symbol_obj)
        __all__ += [symbol_name]  # noqa: PLE0604
