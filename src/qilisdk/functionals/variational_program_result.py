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
"""Deprecated alias of :mod:`qilisdk.algorithms.variational_program_result`."""

import warnings

from qilisdk.algorithms.variational_program_result import VariationalProgramResult

warnings.warn(
    "qilisdk.functionals.variational_program_result is deprecated, import from qilisdk.algorithms.variational_program_result instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["VariationalProgramResult"]
