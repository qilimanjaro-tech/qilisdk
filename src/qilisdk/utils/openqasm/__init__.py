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

__all__ = []

OPENQASM2_PATH = "qilisdk.utils.openqasm.openqasm2"
OPENQASM3_PATH = "qilisdk.utils.openqasm.openqasm3"

OPTIONAL_FEATURES: list[OptionalFeature] = [
    OptionalFeature(
        name="openqasm",
        mode=RequirementMode.ALL,
        dependency_groups=[DependencyGroup(dists=["openqasm3"], extra="openqasm")],
        symbols=[
            Symbol(path=OPENQASM2_PATH, name="to_qasm2"),
            Symbol(path=OPENQASM2_PATH, name="to_qasm2_file"),
            Symbol(path=OPENQASM2_PATH, name="from_qasm2"),
            Symbol(path=OPENQASM2_PATH, name="from_qasm2_file"),
            Symbol(path=OPENQASM3_PATH, name="to_qasm3"),
            Symbol(path=OPENQASM3_PATH, name="to_qasm3_file"),
            Symbol(path=OPENQASM3_PATH, name="from_qasm3"),
            Symbol(path=OPENQASM3_PATH, name="from_qasm3_file"),
        ],
    ),
]

current_module = sys.modules[__name__]

# Dynamically import (or stub) each feature's symbols and attach them
for feature in OPTIONAL_FEATURES:
    imported_feature: ImportedFeature = import_optional_dependencies(feature)
    for symbol_name, symbol_obj in imported_feature.symbols.items():
        setattr(current_module, symbol_name, symbol_obj)
        __all__ += [symbol_name]  # noqa: PLE0604
