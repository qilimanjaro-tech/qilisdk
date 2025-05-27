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

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from qilisdk.analog import Hamiltonian, QuantumObject, Schedule, TimeEvolution
from qilisdk.analog.hamiltonian import PauliOperator
from qilisdk.common.optimizer import Optimizer
from qilisdk.digital import VQE, Circuit
from qilisdk.yaml import yaml

from .qaas_analog_result import QaaSAnalogResult
from .qaas_digital_result import QaaSDigitalResult
from .qaas_time_evolution_result import QaaSTimeEvolutionResult
from .qaas_vqe_result import QaaSVQEResult


class QaaSModel(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, arbitrary_types_allowed=True)


class LoginPayload(BaseModel): ...


class Token(QaaSModel):
    """
    Represents the structure of the login response:
    {
        "accessToken": "...",
        "expiresIn": 123456789,
        "issuedAt": "123456789",
        "refreshToken": "...",
        "tokenType": "bearer"
    }
    """

    access_token: str = Field(alias="accessToken")
    expires_in: int = Field(alias="expiresIn")
    issued_at: str = Field(alias="issuedAt")
    refresh_token: str = Field(alias="refreshToken")
    token_type: str = Field(alias="tokenType")


class DeviceStatus(str, Enum):
    """Device status typing for posting"""

    ONLINE = "online"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class DeviceType(str, Enum):
    """Device type"""

    QUANTUM_DIGITAL = "quantum_device"
    QUANTUM_ANALOG = "quantum_analog_device"
    SIMULATOR = "simulator_device"


@yaml.register_class
class Device(QaaSModel):
    id: int = Field(...)
    name: str = Field(...)
    status: DeviceStatus = Field(...)
    type: DeviceType = Field(...)


class ExecutePayloadType(str, Enum):
    DIGITAL = "digital"
    ANALOG = "analog"
    VQE = "vqe"
    TIME_EVOLUTION = "time_evolution"


@yaml.register_class
class DigitalPayload(QaaSModel):
    circuit: Circuit = Field(...)
    nshots: int = Field(...)


@yaml.register_class
class AnalogPayload(QaaSModel):
    schedule: Schedule = Field(...)
    initial_state: QuantumObject = Field(...)
    observables: list[PauliOperator | Hamiltonian] = Field(...)
    store_intermediate_results: bool = Field(...)


@yaml.register_class
class VQEPayload(QaaSModel):
    vqe: VQE = Field(...)
    optimizer: Optimizer = Field(...)
    nshots: int = Field(...)
    store_intermediate_results: bool = Field(...)


@yaml.register_class
class TimeEvolutionPayload(QaaSModel):
    time_evolution: TimeEvolution = Field()
    store_intermediate_results: bool = Field()


@yaml.register_class
class ExecutePayload(QaaSModel):
    type: ExecutePayloadType = Field(...)
    device_id: int = Field(...)
    digital_payload: DigitalPayload | None = None
    analog_payload: AnalogPayload | None = None
    vqe_payload: VQEPayload | None = None
    time_evolution_payload: TimeEvolutionPayload | None = None


@yaml.register_class
class ExecuteResponse(QaaSModel):
    type: ExecutePayloadType = Field(...)
    digital_result: QaaSDigitalResult | None = None
    analog_result: QaaSAnalogResult | None = None
    vqe_result: QaaSVQEResult | None = None
    time_evolution_result: QaaSTimeEvolutionResult | None = None
