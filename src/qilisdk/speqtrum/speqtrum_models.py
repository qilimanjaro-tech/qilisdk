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
from typing import Any, Callable, Generic, TypeVar, cast, overload

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer, field_validator

from qilisdk.experiments import (
    RabiExperiment,
    RabiExperimentResult,
    T1Experiment,
    T1ExperimentResult,
    T2Experiment,
    T2ExperimentResult,
    TwoTonesExperiment,
    TwoTonesExperimentResult,
)
from qilisdk.functionals import (
    Sampling,
    SamplingResult,
    TimeEvolution,
    TimeEvolutionResult,
    VariationalProgram,
    VariationalProgramResult,
)
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.utils.serialization import deserialize, serialize


class SpeQtrumModel(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, arbitrary_types_allowed=True)


class LoginPayload(BaseModel): ...


class Token(SpeQtrumModel):
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
    issued_at: int = Field(alias="issuedAt")
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


class Device(SpeQtrumModel):
    code: str = Field(...)
    nqubits: int = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    type: DeviceType = Field(...)
    status: DeviceStatus = Field(...)


class ExecuteType(str, Enum):
    SAMPLING = "sampling"
    TIME_EVOLUTION = "time_evolution"
    VARIATIONAL_PROGRAM = "variational_program"
    RABI_EXPERIMENT = "rabi_experiment"
    T1_EXPERIMENT = "t1_experiment"
    T2_EXPERIMENT = "t2_experiment"
    TWO_TONES_EXPERIMENT = "two_tones_experiment"


class SamplingPayload(SpeQtrumModel):
    sampling: Sampling = Field(...)

    @field_serializer("sampling")
    def _serialize_sampling(self, sampling: Sampling, _info):
        return serialize(sampling)

    @field_validator("sampling", mode="before")
    def _load_sampling(cls, v):
        if isinstance(v, str):
            return deserialize(v, Sampling)
        return v


class TimeEvolutionPayload(SpeQtrumModel):
    time_evolution: TimeEvolution = Field(...)

    @field_serializer("time_evolution")
    def _serialize_time_evolution(self, time_evolution: TimeEvolution, _info):
        return serialize(time_evolution)

    @field_validator("time_evolution", mode="before")
    def _load_time_evolution(cls, v):
        if isinstance(v, str):
            return deserialize(v, TimeEvolution)
        return v


class VariationalProgramPayload(SpeQtrumModel):
    variational_program: VariationalProgram = Field(...)

    @field_serializer("variational_program")
    def _serialize_variational_program(self, variational_program: VariationalProgram, _info):
        return serialize(variational_program)

    @field_validator("variational_program", mode="before")
    def _load_variational_program(cls, v):
        if isinstance(v, str):
            return deserialize(v, VariationalProgram)
        return v


class RabiExperimentPayload(SpeQtrumModel):
    rabi_experiment: RabiExperiment = Field(...)

    @field_serializer("rabi_experiment")
    def _serialize_rabi_experiment(self, rabi_experiment: RabiExperiment, _info):
        return serialize(rabi_experiment)

    @field_validator("rabi_experiment", mode="before")
    def _load_rabi_experiment(cls, v):
        if isinstance(v, str):
            return deserialize(v, RabiExperiment)
        return v


class T1ExperimentPayload(SpeQtrumModel):
    t1_experiment: T1Experiment = Field(...)

    @field_serializer("t1_experiment")
    def _serialize_t1_experiment(self, t1_experiment: T1Experiment, _info):
        return serialize(t1_experiment)

    @field_validator("t1_experiment", mode="before")
    def _load_t1_experiment(cls, v):
        if isinstance(v, str):
            return deserialize(v, T1Experiment)
        return v


class T2ExperimentPayload(SpeQtrumModel):
    t2_experiment: T2Experiment = Field(...)

    @field_serializer("t2_experiment")
    def _serialize_t2_experiment(self, t2_experiment: T2Experiment, _info):
        return serialize(t2_experiment)

    @field_validator("t2_experiment", mode="before")
    def _load_t2_experiment(cls, v):
        if isinstance(v, str):
            return deserialize(v, T2Experiment)
        return v


class TwoTonesExperimentPayload(SpeQtrumModel):
    two_tones_experiment: TwoTonesExperiment = Field(...)

    @field_serializer("two_tones_experiment")
    def _serialize_two_tones_experiment(self, two_tones_experiment: TwoTonesExperiment, _info):
        return serialize(two_tones_experiment)

    @field_validator("two_tones_experiment", mode="before")
    def _load_two_tones_experiment(cls, v):
        if isinstance(v, str):
            return deserialize(v, TwoTonesExperiment)
        return v


class ExecutePayload(SpeQtrumModel):
    type: ExecuteType = Field(...)
    sampling_payload: SamplingPayload | None = None
    time_evolution_payload: TimeEvolutionPayload | None = None
    variational_program_payload: VariationalProgramPayload | None = None
    rabi_experiment_payload: RabiExperimentPayload | None = None
    t1_experiment_payload: T1ExperimentPayload | None = None
    t2_experiment_payload: T2ExperimentPayload | None = None
    two_tones_experiment_payload: TwoTonesExperimentPayload | None = None


class ExecuteResult(SpeQtrumModel):
    type: ExecuteType = Field(...)
    sampling_result: SamplingResult | None = None
    time_evolution_result: TimeEvolutionResult | None = None
    variational_program_result: VariationalProgramResult | None = None
    rabi_experiment_result: RabiExperimentResult | None = None
    t1_experiment_result: T1ExperimentResult | None = None
    t2_experiment_result: T2ExperimentResult | None = None
    two_tones_experiment_result: TwoTonesExperimentResult | None = None

    @field_serializer("sampling_result")
    def _serialize_sampling_result(self, sampling_result: SamplingResult, _info):
        return serialize(sampling_result) if sampling_result is not None else None

    @field_validator("sampling_result", mode="before")
    def _load_sampling_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, SamplingResult)
        return v

    @field_serializer("time_evolution_result")
    def _serialize_time_evolution_result(self, time_evolution_result: TimeEvolutionResult, _info):
        return serialize(time_evolution_result) if time_evolution_result is not None else None

    @field_validator("time_evolution_result", mode="before")
    def _load_time_evolution_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, TimeEvolutionResult)
        return v

    @field_serializer("variational_program_result")
    def _serialize_variational_program_result(self, variational_program_result: VariationalProgramResult, _info):
        return serialize(variational_program_result) if variational_program_result is not None else None

    @field_validator("variational_program_result", mode="before")
    def _load_variational_program_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, VariationalProgramResult)
        return v

    @field_serializer("rabi_experiment_result")
    def _serialize_rabi_experiment_result(self, rabi_experiment_result: RabiExperimentResult, _info):
        return serialize(rabi_experiment_result) if rabi_experiment_result is not None else None

    @field_validator("rabi_experiment_result", mode="before")
    def _load_rabi_experiment_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, RabiExperimentResult)
        return v

    @field_serializer("t1_experiment_result")
    def _serialize_t1_experiment_result(self, t1_experiment_result: T1ExperimentResult, _info):
        return serialize(t1_experiment_result) if t1_experiment_result is not None else None

    @field_validator("t1_experiment_result", mode="before")
    def _load_t1_experiment_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, T1ExperimentResult)
        return v

    @field_serializer("t2_experiment_result")
    def _serialize_t2_experiment_resultt(self, t2_experiment_result: T2ExperimentResult, _info):
        return serialize(t2_experiment_result) if t2_experiment_result is not None else None

    @field_validator("t2_experiment_result", mode="before")
    def _load_t2_experiment_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, T2ExperimentResult)
        return v

    @field_serializer("two_tones_experiment_result")
    def _serialize_two_tones_experiment_result(self, two_tones_experiment_result: TwoTonesExperimentResult, _info):
        return serialize(two_tones_experiment_result) if two_tones_experiment_result is not None else None

    @field_validator("two_tones_experiment_result", mode="before")
    def _load_two_tones_experiment_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, TwoTonesExperimentResult)
        return v


TFunctionalResult_co = TypeVar("TFunctionalResult_co", bound=FunctionalResult, covariant=True)
TVariationalInnerResult = TypeVar("TVariationalInnerResult", bound=FunctionalResult)


ResultExtractor = Callable[[ExecuteResult], TFunctionalResult_co]


# these helpers live outside the models so they can be referenced by default values
def _require_sampling_result(result: ExecuteResult) -> SamplingResult:
    if result.sampling_result is None:
        raise RuntimeError("SpeQtrum did not return a sampling_result for a sampling execution.")
    return result.sampling_result


def _require_time_evolution_result(result: ExecuteResult) -> TimeEvolutionResult:
    if result.time_evolution_result is None:
        raise RuntimeError("SpeQtrum did not return a time_evolution_result for a time evolution execution.")
    return result.time_evolution_result


def _require_variational_program_result(result: ExecuteResult) -> VariationalProgramResult:
    if result.variational_program_result is None:
        raise RuntimeError("SpeQtrum did not return a variational_program_result for a variational program execution.")
    return result.variational_program_result


def _require_rabi_experiment_result(result: ExecuteResult) -> RabiExperimentResult:
    if result.rabi_experiment_result is None:
        raise RuntimeError("SpeQtrum did not return a rabi_experiment_result for a Rabi experiment execution.")
    return result.rabi_experiment_result


def _require_t1_experiment_result(result: ExecuteResult) -> T1ExperimentResult:
    if result.t1_experiment_result is None:
        raise RuntimeError("SpeQtrum did not return a t1_experiment_result for a T1 experiment execution.")
    return result.t1_experiment_result


def _require_t2_experiment_result(result: ExecuteResult) -> T2ExperimentResult:
    if result.t2_experiment_result is None:
        raise RuntimeError("SpeQtrum did not return a t2_experiment_result for a T2 experiment execution.")
    return result.t2_experiment_result


def _require_two_tones_experiment_result(result: ExecuteResult) -> TwoTonesExperimentResult:
    if result.two_tones_experiment_result is None:
        raise RuntimeError(
            "SpeQtrum did not return a two_tones_experiment_result for a Two-Tones experiment execution."
        )
    return result.two_tones_experiment_result


def _require_variational_program_result_typed(
    inner_result_type: type[TVariationalInnerResult],
) -> ResultExtractor[VariationalProgramResult[TVariationalInnerResult]]:
    def _extractor(result: ExecuteResult) -> VariationalProgramResult[TVariationalInnerResult]:
        variational_result = _require_variational_program_result(result)
        optimal_results = variational_result.optimal_execution_results
        if not isinstance(optimal_results, inner_result_type):
            raise RuntimeError(
                "SpeQtrum returned a variational program result whose optimal execution result "
                f"({type(optimal_results).__qualname__}) does not match the expected "
                f"{inner_result_type.__qualname__}."
            )
        return cast("VariationalProgramResult[TVariationalInnerResult]", variational_result)

    return _extractor


class JobHandle(SpeQtrumModel, Generic[TFunctionalResult_co]):
    """Strongly typed reference to a submitted SpeQtrum job."""

    id: int
    execute_type: ExecuteType
    extractor: ResultExtractor[TFunctionalResult_co] = Field(repr=False, exclude=True)

    @classmethod
    def sampling(cls, job_id: int) -> "JobHandle[SamplingResult]":
        return cls(id=job_id, execute_type=ExecuteType.SAMPLING, extractor=_require_sampling_result)  # type: ignore[return-value, arg-type]

    @classmethod
    def time_evolution(cls, job_id: int) -> "JobHandle[TimeEvolutionResult]":
        return cls(id=job_id, execute_type=ExecuteType.TIME_EVOLUTION, extractor=_require_time_evolution_result)  # type: ignore[return-value, arg-type]

    @overload
    @classmethod
    def variational_program(cls, job_id: int) -> "JobHandle[VariationalProgramResult]": ...

    @overload
    @classmethod
    def variational_program(
        cls, job_id: int, *, result_type: type[TVariationalInnerResult]
    ) -> "JobHandle[VariationalProgramResult[TVariationalInnerResult]]": ...

    @classmethod
    def variational_program(
        cls, job_id: int, *, result_type: type[TVariationalInnerResult] | None = None
    ) -> "JobHandle[Any]":
        """Create a variational-program handle for an existing job identifier.

        Args:
            job_id: Numeric identifier returned by the SpeQtrum service.
            result_type: Optional functional result type expected within the
                variational program payload. When provided the returned handle
                enforces that the optimiser output matches this type.

        Returns:
            JobHandle: A handle whose ``get_results`` invocation yields a
            ``VariationalProgramResult`` preserving the requested inner result
            type when supplied.
        """
        if result_type is None:
            handle = cls(
                id=job_id,
                execute_type=ExecuteType.VARIATIONAL_PROGRAM,
                extractor=_require_variational_program_result,  # type: ignore[arg-type]
            )
            return cast("JobHandle[VariationalProgramResult]", handle)

        extractor = _require_variational_program_result_typed(result_type)
        handle = cls(id=job_id, execute_type=ExecuteType.VARIATIONAL_PROGRAM, extractor=extractor)  # type: ignore[arg-type]
        return cast("JobHandle[VariationalProgramResult[TVariationalInnerResult]]", handle)

    @classmethod
    def rabi_experiment(cls, job_id: int) -> "JobHandle[RabiExperimentResult]":
        return cls(id=job_id, execute_type=ExecuteType.RABI_EXPERIMENT, extractor=_require_rabi_experiment_result)  # type: ignore[return-value, arg-type]

    @classmethod
    def t1_experiment(cls, job_id: int) -> "JobHandle[T1ExperimentResult]":
        return cls(id=job_id, execute_type=ExecuteType.T1_EXPERIMENT, extractor=_require_t1_experiment_result)  # type: ignore[return-value, arg-type]

    @classmethod
    def t2_experiment(cls, job_id: int) -> "JobHandle[T2ExperimentResult]":
        return cls(id=job_id, execute_type=ExecuteType.T2_EXPERIMENT, extractor=_require_t2_experiment_result)  # type: ignore[return-value, arg-type]

    @classmethod
    def two_tones_experiment(cls, job_id: int) -> "JobHandle[TwoTonesExperimentResult]":
        return cls(
            id=job_id,
            execute_type=ExecuteType.TWO_TONES_EXPERIMENT,
            extractor=_require_two_tones_experiment_result,  # type: ignore[return-value, arg-type]
        )

    def bind(self, detail: "JobDetail") -> "TypedJobDetail[TFunctionalResult_co]":
        """Attach this handle's typing information to a concrete job detail.

        Args:
            detail: Un-typed job detail payload returned by the SpeQtrum API.

        Returns:
            TypedJobDetail: Wrapper exposing ``get_results`` with the typing
            captured when the handle was created.
        """
        return TypedJobDetail.model_validate(
            {
                **detail.model_dump(),
                "expected_type": self.execute_type,
                "extractor": self.extractor,
            }
        )


class JobStatus(str, Enum):
    "Job has not been submitted to the Lab api"

    PENDING = "pending"
    "Job has been queued but not yet validated"
    VALIDATING = "validating"
    "Job has been validated and is queued for execution"
    QUEUED = "queued"
    "Job is being executed on the device"
    RUNNING = "running"
    "Job finished successfully"
    COMPLETED = "completed"
    "Job failed due to an error"
    ERROR = "error"
    "Job was cancelled by the user or system"
    CANCELLED = "cancelled"
    "Job failed due to timeout"
    TIMEOUT = "timeout"


class JobType(str, Enum):
    DIGITAL = "digital"
    PULSE = "pulse"
    ANALOG = "analog"
    VARIATIONAL = "variational"


class JobId(SpeQtrumModel):
    """Handle/reference you normally get back immediately after `POST /execute`."""

    id: int = Field(...)


class JobInfo(JobId):
    """
    Light-weight representation suitable for 'list jobs' and polling
    when you do *not* need logs or results.
    """

    name: str = Field(...)
    description: str = Field(...)
    device_id: int = Field(...)
    status: JobStatus = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None

    @field_validator("created_at", mode="before")
    def _parse_created_at(cls, v):
        return parsedate_to_datetime(v) if isinstance(v, str) else v

    @field_validator("updated_at", mode="before")
    def _parse_updated_at(cls, v):
        return parsedate_to_datetime(v) if isinstance(v, str) else v

    @field_validator("completed_at", mode="before")
    def _parse_completed_at(cls, v):
        return parsedate_to_datetime(v) if isinstance(v, str) else v


class JobDetail(JobInfo):
    """
    Full representation returned by `GET /jobs/{id}` when payload/result/logs
    are requested.
    """

    payload: ExecutePayload | None = None
    result: ExecuteResult | None = None
    jobType: JobType | None = None
    logs: str | None = None
    error: str | None = None
    error_logs: str | None = None


class TypedJobDetail(JobDetail, Generic[TFunctionalResult_co]):
    """`JobDetail` subclass that exposes a strongly typed `get_results` method."""

    expected_type: ExecuteType = Field(repr=False)
    extractor: ResultExtractor[TFunctionalResult_co] = Field(repr=False, exclude=True)

    def get_results(self) -> TFunctionalResult_co:
        """Return the strongly typed execution result.

        Returns:
            ResultT_co: Result payload associated with the completed job,
            respecting the type information carried by the originating
            ``JobHandle``.

        Raises:
            RuntimeError: If SpeQtrum has not populated the result payload or
                the execute type disagrees with the handle.
        """
        if self.result is None:
            raise RuntimeError("The job completed without a result payload; inspect `error` or `logs` for details.")

        if self.result.type != self.expected_type:
            raise RuntimeError(
                f"Expected a result of type '{self.expected_type.value}' but received '{self.result.type.value}'."
            )

        return self.extractor(self.result)
