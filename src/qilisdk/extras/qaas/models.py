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
# ruff: noqa: ANN001, ANN202, PLR6301
from email.utils import parsedate_to_datetime
from enum import Enum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer, field_validator

from qilisdk.analog import Hamiltonian, QuantumObject, Schedule, TimeEvolution
from qilisdk.analog.analog_result import AnalogResult
from qilisdk.analog.hamiltonian import PauliOperator
from qilisdk.common.optimizer import Optimizer
from qilisdk.digital import VQE, Circuit
from qilisdk.digital.digital_result import DigitalResult
from qilisdk.digital.vqe import VQEResult
from qilisdk.utils.serialization import deserialize, serialize


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

    QPU_ANALOG = "qpu.analog"
    QPU_DIGITAL = "qpu.digital"
    SIMULATOR = "simulator"


class Device(QaaSModel):
    # TODO (vyron): Remove `id: int` when `code` is implemented server-side.
    id: int = Field(...)
    code: str = Field(...)
    name: str = Field(...)
    type: DeviceType = Field(...)
    status: DeviceStatus = Field(...)
    pending_jobs: int = Field(alias="number_pending_jobs")
    static_features: dict
    dynamic_features: dict


class ExecuteType(str, Enum):
    DIGITAL = "digital"
    ANALOG = "analog"
    VQE = "vqe"
    TIME_EVOLUTION = "time_evolution"


class DigitalPayload(QaaSModel):
    circuit: Circuit = Field(...)
    nshots: int = Field(...)

    @field_serializer("circuit")
    def _serialize_circuit(self, circuit: Circuit, _info):
        return serialize(circuit)

    @field_validator("circuit", mode="before")
    def _load_circuit(cls, v):
        if isinstance(v, str):
            return deserialize(v, Circuit)
        return v


class AnalogPayload(QaaSModel):
    schedule: Schedule = Field(...)
    initial_state: QuantumObject = Field(...)
    observables: list[PauliOperator | Hamiltonian] = Field(...)
    store_intermediate_results: bool = Field(...)

    @field_serializer("schedule")
    def _serialize_schedule(self, schedule: Schedule, _info):
        return serialize(schedule)

    @field_validator("schedule", mode="before")
    def _validate_schedule(cls, v):
        if isinstance(v, str):
            return deserialize(v, Schedule)
        return v

    @field_serializer("initial_state")
    def _serialize_initial_state(self, initial_state: QuantumObject, _info):
        return serialize(initial_state)

    @field_validator("initial_state", mode="before")
    def _validate_initial_state(cls, v):
        if isinstance(v, str):
            return deserialize(v, QuantumObject)
        return v

    @field_serializer("observables")
    def _serialize_observables(self, observables: list[PauliOperator | Hamiltonian], _info):
        return [serialize(obs) for obs in observables]

    @field_validator("observables", mode="before")
    def _validate_observables(cls, v):
        if isinstance(v, list) and all(isinstance(item, str) for item in v):
            return [deserialize(item) for item in v]
        return v


class VQEPayload(QaaSModel):
    vqe: VQE = Field(...)
    optimizer: Optimizer = Field(...)
    nshots: int = Field(...)
    store_intermediate_results: bool = Field(...)

    @field_serializer("vqe")
    def _serialize_vqe(self, vqe: VQE, _info):
        return serialize(vqe)

    @field_validator("vqe", mode="before")
    def _load_vqe(cls, v):
        if isinstance(v, str):
            return deserialize(v, VQE)
        return v

    @field_serializer("optimizer")
    def _serialize_optimizer(self, optimizer: Optimizer, _info):
        return serialize(optimizer)

    @field_validator("optimizer", mode="before")
    def _load_optimizer(cls, v):
        if isinstance(v, str):
            return deserialize(v, Optimizer)
        return v


class TimeEvolutionPayload(QaaSModel):
    time_evolution: TimeEvolution = Field()
    store_intermediate_results: bool = Field()

    @field_serializer("time_evolution")
    def _serialize_time_evolution(self, time_evolution: TimeEvolution, _info):
        return serialize(time_evolution)

    @field_validator("time_evolution", mode="before")
    def _load_time_evolution(cls, v):
        if isinstance(v, str):
            return deserialize(v, TimeEvolution)
        return v


class ExecutePayload(QaaSModel):
    type: ExecuteType = Field(...)
    digital_payload: DigitalPayload | None = None
    analog_payload: AnalogPayload | None = None
    vqe_payload: VQEPayload | None = None
    time_evolution_payload: TimeEvolutionPayload | None = None


class ExecuteResult(QaaSModel):
    type: ExecuteType = Field(...)
    digital_result: DigitalResult | None = None
    analog_result: AnalogResult | None = None
    vqe_result: VQEResult | None = None
    time_evolution_result: AnalogResult | None = None

    @field_serializer("digital_result")
    def _serialize_digital_result(self, digital_result: DigitalResult, _info):
        return serialize(digital_result)

    @field_validator("digital_result", mode="before")
    def _load_digital_result(cls, v):
        if isinstance(v, str):
            return deserialize(v, DigitalResult)
        return v

    @field_serializer("vqe_result")
    def _serialize_vqe_result(self, vqe_result: VQEResult, _info):
        return serialize(vqe_result)

    @field_validator("digital_result", mode="before")
    def _load_vqe_result(cls, v):
        if isinstance(v, str):
            return deserialize(v, VQEResult)
        return v

    @field_serializer("analog_result")
    def _serialize_analog_result(self, analog_result: AnalogResult, _info):
        return serialize(analog_result)

    @field_validator("analog_result", mode="before")
    def _load_analog_result(cls, v):
        if isinstance(v, str):
            return deserialize(v, AnalogResult)
        return v

    @field_serializer("time_evolution_result")
    def _serialize_time_evolution_result(self, time_evolution_result: AnalogResult, _info):
        return serialize(time_evolution_result)

    @field_validator("time_evolution_result", mode="before")
    def _load_time_evolution_result(cls, v):
        if isinstance(v, str):
            return deserialize(v, AnalogResult)
        return v


class JobStatus(str, Enum):
    NOT_SENT = "not_sent"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    QUEUED = "queued"
    CANCELLED = "cancelled"


class Job(QaaSModel):
    id: int = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    device_id: int = Field(...)
    status: JobStatus = Field(...)
    created_at: AwareDatetime = Field(...)
    modified_at: AwareDatetime | None = None
    # payload: ExecutePayload | None = None
    result: ExecuteResult | None = None
    logs: str | None = None
    error: str | None = None
    error_logs: str | None = None

    @field_validator("created_at", mode="before")
    def _parse_created_at(cls, v):
        if isinstance(v, str):
            # parse "Fri, 04 Jul 2025 12:36:40 GMT"
            return parsedate_to_datetime(v)
        return v

    @field_validator("modified_at", mode="before")
    def _parse_modified_at(cls, v):
        if isinstance(v, str):
            # parse "Fri, 04 Jul 2025 12:36:40 GMT"
            return parsedate_to_datetime(v)
        return v
