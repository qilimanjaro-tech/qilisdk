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

This package exposes the readout method classes used to describe *how* results
should be extracted from a quantum backend, together with the result containers
that hold the extracted data.

Readout methods (see :mod:`qilisdk.readout.readout`):
    :class:`ReadoutMethod` -- abstract base with factory helpers.
    :class:`SamplingReadout` -- measure in the computational basis.
    :class:`ExpectationReadout` -- compute observable expectation values.
    :class:`StateTomographyReadout` -- return the full quantum state.

Result containers (see :mod:`qilisdk.readout.readout_result`):
    :class:`ReadoutResult` -- abstract single-result base.
    :class:`SamplingReadoutResult` -- bitstring counts and probabilities.
    :class:`ExpectationReadoutResult` -- expectation values.
    :class:`StateTomographyReadoutResult` -- quantum state and probabilities.
"""

from .readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout
from .readout_result import (
    ExpectationReadoutResult,
    ReadoutCompositeResults,
    ReadoutResult,
    SamplingReadoutResult,
    StateTomographyReadoutResult,
)

__all__ = [
    "ExpectationReadout",
    "ExpectationReadoutResult",
    "ReadoutCompositeResults",
    "ReadoutMethod",
    "ReadoutResult",
    "SamplingReadout",
    "SamplingReadoutResult",
    "StateTomographyReadout",
    "StateTomographyReadoutResult",
]
