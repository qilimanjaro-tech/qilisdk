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
"""Equation-based dataset generators for time-series and reservoir computing.

All datasets derive from :class:`Dataset` and expose a common
:meth:`Dataset.generate` method returning a :class:`DatasetSample`. Data is
synthesised on the fly from the governing equations rather than loaded from
disk, so any number of points can be produced reproducibly.

Example:
    >>> from qilisdk.ml.datasets import MackeyGlass
    >>> inputs, targets = MackeyGlass(tau=17.0).generate(2000)
    >>> inputs.shape, targets.shape
    ((2000, 1), (2000, 1))
"""

from qilisdk.ml.datasets.dataset import Dataset, DatasetSample
from qilisdk.ml.datasets.henon import HenonMap
from qilisdk.ml.datasets.logistic_map import LogisticMap
from qilisdk.ml.datasets.lorenz import Lorenz
from qilisdk.ml.datasets.mackey_glass import MackeyGlass
from qilisdk.ml.datasets.narma import NARMA
from qilisdk.ml.datasets.santa_fe_laser import SantaFeLaser

__all__ = [
    "NARMA",
    "Dataset",
    "DatasetSample",
    "HenonMap",
    "LogisticMap",
    "Lorenz",
    "MackeyGlass",
    "SantaFeLaser",
]
