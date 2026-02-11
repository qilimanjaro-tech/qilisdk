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
from .variables import QProgramVariable

"""VariableAllocated Exception class"""


class VariableAllocated(Exception):
    """Exception raised when trying to allocate a variable in a loop that is already been allocated by another loop.

    Args:
        message (str): Optional message to be displayed
    """

    def __init__(self, variable: QProgramVariable) -> None:
        super().__init__(f"Cannot allocate variable {variable}. Variable is already allocated.")
