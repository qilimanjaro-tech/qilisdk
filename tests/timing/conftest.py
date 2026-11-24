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

"""Configuration for the timing suite.

OpenMP (used by QTensor, Eigen and QiliSim) and OpenBLAS (used by numpy and scipy) read their
thread counts when they are first loaded, so they are limited to one thread here, before any test
module imports them. This makes the timings independent of how many cores the machine has, which
only works if the suite is run on its own, as in ``pytest tests/timing``. Run it without ``-n``,
too, since parallel workers compete for the CPU and skew the measurements.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
