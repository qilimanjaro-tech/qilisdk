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


class OutOfBoundsException(Exception):
    """Raised when a variable value falls outside its configured bounds."""


class NotSupportedOperation(Exception):
    """Raised when a requested operation is not supported by the backend."""


class InvalidBoundsError(Exception):
    """Raised when lower/upper bounds are inconsistent or invalid."""


class EvaluationError(Exception):
    """Raised when a symbolic expression cannot be evaluated."""
