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

from qilisdk.qprogram.element import Element
from qilisdk.qprogram.variables import QProgramVariable


class Operation(Element):
    def get_variables(self) -> set[QProgramVariable]:
        """Get a set of the variables used in operation, if any.

        Returns:
            set[Variable]: The set of variables used in operation.
        """
        return {attribute for attribute in self.__dict__.values() if isinstance(attribute, QProgramVariable)}
