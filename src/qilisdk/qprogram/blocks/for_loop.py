# Copyright 2023 Qilimanjaro Quantum Tech
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


from qilisdk.qprogram.variables import QProgramVariable
from qilisdk.yaml import yaml

from .block import Block


@yaml.register_class
class ForLoop(Block):
    def __init__(self, variable: QProgramVariable, start: float, stop: float, step: float) -> None:
        super().__init__()
        self.variable: QProgramVariable = variable
        self.start: float = start
        self.stop: float = stop
        self.step: float = step
