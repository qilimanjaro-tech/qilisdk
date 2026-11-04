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

"""Helpers shared by the timing suite.

Each limit in this suite is roughly ten times what the operation takes on one thread of a
developer machine, so the tests catch order-of-magnitude slowdowns rather than small fluctuations.
"""

import time
from collections.abc import Callable
from typing import Any

from qilisdk.analog.hamiltonian import X, Z
from qilisdk.analog.schedule import Schedule

REPEATS = 3


def best_time(operation: Callable[[], Any]) -> float:
    """Run a callable a few times and return its fastest wall-clock time.

    The fastest run is the least affected by other load on the machine, and the first run also
    absorbs one-off costs such as lazy imports.
    """
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - start)
    return min(timings)


def linear_ramp_schedule(nqubits: int, dt: float) -> Schedule:
    """Return a schedule that ramps linearly from an X driver to a ZZ chain over ten time units."""
    return Schedule(
        hamiltonians={
            "driver": sum(X(i) for i in range(nqubits)),
            "problem": sum(Z(i) * Z(i + 1) for i in range(nqubits - 1)),
        },
        coefficients={"driver": {(0, 10): lambda t: 1 - t / 10}, "problem": {(0, 10): lambda t: t / 10}},
        dt=dt,
    )
