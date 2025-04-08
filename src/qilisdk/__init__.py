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

import sys
from typing import List

from qilisdk.common import Algorithm, Model, Optimizer, Result

from ._optionals import ImportedFeature, OptionalFeature, Symbol, import_optional_dependencies

# Put your always-available, core symbols here
__all__ = ["Algorithm", "Model", "Optimizer", "Result"]

# Define your optional features
OPTIONAL_FEATURES: List[OptionalFeature] = [
    OptionalFeature(
        name="cuda-backend",
        dependencies=["cudaq"],
        symbols=[Symbol(path="qilisdk.extras.cuda_backend", name="CudaBackend")],
    ),
    OptionalFeature(
        name="qaas",
        dependencies=["httpx", "keyring", "pydantic", "pydantic-settings"],
        symbols=[Symbol(path="qilisdk.extras.qaas.qaas_backend", name="QaaSBackend")],
    ),
    # Add more OptionalFeature() entries for other extras if needed
]

current_module = sys.modules[__name__]

# Dynamically import (or stub) each feature's symbols and attach them
for feature in OPTIONAL_FEATURES:
    imported_feature: ImportedFeature = import_optional_dependencies(feature)
    for symbol_name, symbol_obj in imported_feature.symbols.items():
        setattr(current_module, symbol_name, symbol_obj)
        __all__ += [symbol_name]  # noqa: PLE0604
