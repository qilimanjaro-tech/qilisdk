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

"""Shared fixtures for the timing suite.

Each limit in this suite is roughly ten times what the operation takes on a developer machine,
so the tests catch order-of-magnitude slowdowns rather than small fluctuations. Run the suite
without ``-n``, since parallel workers compete for the CPU and skew the measurements.
"""

import time
from collections.abc import Callable
from typing import Any

import pytest

REPEATS = 3


@pytest.fixture
def best_time() -> Callable[[Callable[[], Any]], float]:
    """Return a function that runs a callable a few times and gives its fastest wall-clock time.

    The fastest run is the least affected by other load on the machine, and the first run also
    absorbs one-off costs such as lazy imports.
    """

    def measure(operation: Callable[[], Any]) -> float:
        timings = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            operation()
            timings.append(time.perf_counter() - start)
        return min(timings)

    return measure
