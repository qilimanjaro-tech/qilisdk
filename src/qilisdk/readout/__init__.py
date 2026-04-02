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

"""Readout configuration and result types for quantum backend execution.

The primary interface for specifying readout is :class:`Readout`.  Build a specification by
chaining ``with_*`` methods and pass it to :meth:`~qilisdk.backends.Backend.execute`::

    from qilisdk.analog import Z
    from qilisdk.readout import Readout

    spec = Readout().with_expectation(observables=[Z(0)])
    result = backend.execute(functional, readout=spec)
    ev = result.expectation_values  # list[float]

Readout specification (primary API):
    :class:`Readout` -- builder for type-safe readout declarations.

Readout method classes (constructed internally by :class:`Readout`):
    :class:`ReadoutMethod` -- abstract base.
    :class:`SamplingReadout` -- measure in the computational basis.
    :class:`ExpectationReadout` -- compute observable expectation values.
    :class:`StateTomographyReadout` -- return the full quantum state.

Result containers (see :mod:`qilisdk.readout.readout_result`):
    :class:`ReadoutResult` -- abstract single-result base.
    :class:`SamplingReadoutResult` -- bitstring counts and probabilities.
    :class:`ExpectationReadoutResult` -- expectation values.
    :class:`StateTomographyReadoutResult` -- quantum state and probabilities.
    :class:`ReadoutCompositeResults` -- aggregated container for one execution step.
"""

from .readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout
from .readout_result import (
    E,
    ExpectationReadoutResult,
    ReadoutCompositeResults,
    ReadoutResult,
    S,
    SamplingReadoutResult,
    StateTomographyReadoutResult,
    T,
    has_expectation_values,
    has_sampling,
    has_state_tomography,
)
from .readout_spec import Readout

__all__ = [
    "E",
    "ExpectationReadout",
    "ExpectationReadoutResult",
    "ReadoutCompositeResults",
    "ReadoutMethod",
    "ReadoutResult",
    "Readout",
    "S",
    "SamplingReadout",
    "SamplingReadoutResult",
    "StateTomographyReadout",
    "StateTomographyReadoutResult",
    "T",
    "has_expectation_values",
    "has_sampling",
    "has_state_tomography",
]
