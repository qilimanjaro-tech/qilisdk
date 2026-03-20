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

"""Public API for quantum functionals and their results.

This package exposes the core functional abstractions used to describe
executable quantum routines (digital propagation, analog evolution,
quantum reservoirs, variational programs) together with their result
containers.
"""

from .analog_evolution import AnalogEvolution
from .digital_propagation import DigitalPropagation
from .functional_result import FunctionalResult
from .quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from .variational_program import VariationalProgram
from .variational_program_result import VariationalProgramResult

__all__ = [
    "AnalogEvolution",
    "DigitalPropagation",
    "FunctionalResult",
    "QuantumReservoir",
    "ReservoirInput",
    "ReservoirLayer",
    "VariationalProgram",
    "VariationalProgramResult",
]
