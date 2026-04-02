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
from __future__ import annotations

from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar, cast, overload

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer, field_validator

from qilisdk.core.result import Result
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
    AnalogEvolution,
    DigitalPropagation,
    FunctionalResult,
    QuantumReservoir,
    VariationalProgram,
    VariationalProgramResult,
)
from qilisdk.readout import ReadoutSpec
from qilisdk.utils.serialization import deserialize, serialize


class SpeQtrumModel(BaseModel):
    """Base Pydantic model for all SpeQtrum API data structures.

    Configures alias resolution and arbitrary-type support used throughout the
    SpeQtrum payload and response models.
    """

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, arbitrary_types_allowed=True)


class LoginPayload(BaseModel):
    """Placeholder model for the login request payload."""


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
    """Enumeration of possible device statuses reported by the SpeQtrum API."""

    ONLINE = "online"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class DeviceType(str, Enum):
    """Enumeration of hardware device types available in SpeQtrum."""

    QPU_ANALOG = "qpu.analog"
    QPU_DIGITAL = "qpu.digital"
    SIMULATOR = "simulator"


class Device(SpeQtrumModel):
    """Description of a quantum device registered in SpeQtrum."""

    code: str = Field(...)
    nqubits: int = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    type: DeviceType = Field(...)
    status: DeviceStatus = Field(...)


class ExecuteType(str, Enum):
    """Discriminator for the type of functional or experiment being executed."""

    DIGITAL_PROPAGATION = "digital_propagation"
    ANALOG_EVOLUTION = "analog_evolution"
    QUANTUM_RESERVOIR = "quantum_reservoir"
    VARIATIONAL_PROGRAM = "variational_program"
    RABI_EXPERIMENT = "rabi_experiment"
    T1_EXPERIMENT = "t1_experiment"
    T2_EXPERIMENT = "t2_experiment"
    TWO_TONES_EXPERIMENT = "two_tones_experiment"


class DigitalPropagationPayload(SpeQtrumModel):
    """Payload model wrapping a ``DigitalPropagation`` and its readout methods for API submission."""

    digital_propagation: DigitalPropagation = Field(...)
    readout: ReadoutSpec = Field(...)

    @field_serializer("digital_propagation")
    def _serialize_sampling(self, digital_propagation: DigitalPropagation, _info):
        return serialize(digital_propagation)

    @field_validator("digital_propagation", mode="before")
    def _load_sampling(cls, v):
        if isinstance(v, str):
            return deserialize(v, DigitalPropagation)
        return v

    @field_serializer("readout")
    def _serialize_readout(self, readout: ReadoutSpec, _info):
        return serialize(readout)

    @field_validator("readout", mode="before")
    def _load_readout(cls, v):
        if isinstance(v, str):
            return deserialize(v, ReadoutSpec)
        return v


class AnalogEvolutionPayload(SpeQtrumModel):
    """Payload model wrapping an ``AnalogEvolution`` and its readout methods for API submission."""

    analog_evolution: AnalogEvolution = Field(...)
    readout: ReadoutSpec = Field(...)

    @field_serializer("analog_evolution")
    def _serialize_time_evolution(self, analog_evolution: AnalogEvolution, _info):
        return serialize(analog_evolution)

    @field_validator("analog_evolution", mode="before")
    def _load_time_evolution(cls, v):
        if isinstance(v, str):
            return deserialize(v, AnalogEvolution)
        return v

    @field_serializer("readout")
    def _serialize_readout(self, readout: ReadoutSpec, _info):
        return serialize(readout)

    @field_validator("readout", mode="before")
    def _load_readout(cls, v):
        if isinstance(v, str):
            return deserialize(v, ReadoutSpec)
        return v


class QuantumReservoirPayload(SpeQtrumModel):
    """Payload model wrapping a ``QuantumReservoir`` and its readout methods for API submission."""

    quantum_reservoir: QuantumReservoir = Field(...)
    readout: ReadoutSpec = Field(...)

    @field_serializer("quantum_reservoir")
    def _serialize_time_evolution(self, quantum_reservoir: QuantumReservoir, _info):
        return serialize(quantum_reservoir)

    @field_validator("quantum_reservoir", mode="before")
    def _load_time_evolution(cls, v):
        if isinstance(v, str):
            return deserialize(v, AnalogEvolution)
        return v

    @field_serializer("readout")
    def _serialize_readout(self, readout: ReadoutSpec, _info):
        return serialize(readout)

    @field_validator("readout", mode="before")
    def _load_readout(cls, v):
        if isinstance(v, str):
            return deserialize(v, ReadoutSpec)
        return v


class VariationalProgramPayload(SpeQtrumModel):
    """Payload model wrapping a ``VariationalProgram`` and its readout methods for API submission."""

    variational_program: VariationalProgram = Field(...)
    readout: ReadoutSpec = Field(...)

    @field_serializer("variational_program")
    def _serialize_variational_program(self, variational_program: VariationalProgram, _info):
        return serialize(variational_program)

    @field_validator("variational_program", mode="before")
    def _load_variational_program(cls, v):
        if isinstance(v, str):
            return deserialize(v, VariationalProgram)
        return v

    @field_serializer("readout")
    def _serialize_readout(self, readout: ReadoutSpec, _info):
        return serialize(readout)

    @field_validator("readout", mode="before")
    def _load_readout(cls, v):
        if isinstance(v, str):
            return deserialize(v, ReadoutSpec)
        return v


class RabiExperimentPayload(SpeQtrumModel):
    """Payload model wrapping a ``RabiExperiment`` for API submission."""

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
    """Payload model wrapping a ``T1Experiment`` for API submission."""

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
    """Payload model wrapping a ``T2Experiment`` for API submission."""

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
    """Payload model wrapping a ``TwoTonesExperiment`` for API submission."""

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
    """Top-level execution payload sent to the SpeQtrum ``/execute`` endpoint.

    Exactly one of the optional payload fields should be populated, matching
    the discriminator stored in ``type``.
    """

    type: ExecuteType = Field(...)
    digital_propagation_payload: DigitalPropagationPayload | None = None
    analog_evolution_payload: AnalogEvolutionPayload | None = None
    quantum_reservoir_payload: QuantumReservoirPayload | None = None
    variational_program_payload: VariationalProgramPayload | None = None
    rabi_experiment_payload: RabiExperimentPayload | None = None
    t1_experiment_payload: T1ExperimentPayload | None = None
    t2_experiment_payload: T2ExperimentPayload | None = None
    two_tones_experiment_payload: TwoTonesExperimentPayload | None = None


class ExecuteResult(SpeQtrumModel):
    """Deserialized execution result returned by the SpeQtrum API.

    The ``type`` discriminator indicates which result field is populated.
    Use the corresponding accessor (e.g. ``functional_result``,
    ``variational_program_result``) to retrieve the typed payload.
    """

    type: ExecuteType = Field(...)
    functional_result: FunctionalResult | None = None
    variational_program_result: VariationalProgramResult | None = None
    rabi_experiment_result: RabiExperimentResult | None = None
    t1_experiment_result: T1ExperimentResult | None = None
    t2_experiment_result: T2ExperimentResult | None = None
    two_tones_experiment_result: TwoTonesExperimentResult | None = None

    @field_serializer("functional_result")
    def _serialize_sampling_result(self, functional_result: FunctionalResult, _info):
        return serialize(functional_result) if functional_result is not None else None

    @field_validator("functional_result", mode="before")
    def _load_sampling_result(cls, v):
        if isinstance(v, str) and v.startswith("!"):
            return deserialize(v, FunctionalResult)
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
    def _serialize_t2_experiment_result(self, t2_experiment_result: T2ExperimentResult, _info):
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


TFunctionalResult_co = TypeVar("TFunctionalResult_co", bound=Result, covariant=True)
TVariationalInnerResult = TypeVar("TVariationalInnerResult", bound=FunctionalResult)


ResultExtractor = Callable[[ExecuteResult], TFunctionalResult_co]
"""Type alias for a callable that extracts a typed result from an ``ExecuteResult``."""


# these helpers live outside the models so they can be referenced by default values
def _require_functional_result(result: ExecuteResult) -> FunctionalResult:
    """Extract and return the ``FunctionalResult`` from *result*.

    Args:
        result (ExecuteResult): The execution result to inspect.

    Returns:
        FunctionalResult: The contained functional result.

    Raises:
        RuntimeError: If the ``functional_result`` field is ``None``.
    """
    if result.functional_result is None:
        raise RuntimeError("SpeQtrum did not return a functional_result for the execution.")
    return result.functional_result


def _require_variational_program_result(result: ExecuteResult) -> VariationalProgramResult:
    """Extract and return the ``VariationalProgramResult`` from *result*.

    Args:
        result (ExecuteResult): The execution result to inspect.

    Returns:
        VariationalProgramResult: The contained variational program result.

    Raises:
        RuntimeError: If the ``variational_program_result`` field is ``None``.
    """
    if result.variational_program_result is None:
        raise RuntimeError("SpeQtrum did not return a variational_program_result for a variational program execution.")
    return result.variational_program_result


def _require_rabi_experiment_result(result: ExecuteResult) -> RabiExperimentResult:
    """Extract and return the ``RabiExperimentResult`` from *result*.

    Args:
        result (ExecuteResult): The execution result to inspect.

    Returns:
        RabiExperimentResult: The contained Rabi experiment result.

    Raises:
        RuntimeError: If the ``rabi_experiment_result`` field is ``None``.
    """
    if result.rabi_experiment_result is None:
        raise RuntimeError("SpeQtrum did not return a rabi_experiment_result for a Rabi experiment execution.")
    return result.rabi_experiment_result


def _require_t1_experiment_result(result: ExecuteResult) -> T1ExperimentResult:
    """Extract and return the ``T1ExperimentResult`` from *result*.

    Args:
        result (ExecuteResult): The execution result to inspect.

    Returns:
        T1ExperimentResult: The contained T1 experiment result.

    Raises:
        RuntimeError: If the ``t1_experiment_result`` field is ``None``.
    """
    if result.t1_experiment_result is None:
        raise RuntimeError("SpeQtrum did not return a t1_experiment_result for a T1 experiment execution.")
    return result.t1_experiment_result


def _require_t2_experiment_result(result: ExecuteResult) -> T2ExperimentResult:
    """Extract and return the ``T2ExperimentResult`` from *result*.

    Args:
        result (ExecuteResult): The execution result to inspect.

    Returns:
        T2ExperimentResult: The contained T2 experiment result.

    Raises:
        RuntimeError: If the ``t2_experiment_result`` field is ``None``.
    """
    if result.t2_experiment_result is None:
        raise RuntimeError("SpeQtrum did not return a t2_experiment_result for a T2 experiment execution.")
    return result.t2_experiment_result


def _require_two_tones_experiment_result(result: ExecuteResult) -> TwoTonesExperimentResult:
    """Extract and return the ``TwoTonesExperimentResult`` from *result*.

    Args:
        result (ExecuteResult): The execution result to inspect.

    Returns:
        TwoTonesExperimentResult: The contained Two-Tones experiment result.

    Raises:
        RuntimeError: If the ``two_tones_experiment_result`` field is ``None``.
    """
    if result.two_tones_experiment_result is None:
        raise RuntimeError(
            "SpeQtrum did not return a two_tones_experiment_result for a Two-Tones experiment execution."
        )
    return result.two_tones_experiment_result


def _require_variational_program_result_typed(
    inner_result_type: type[TVariationalInnerResult],
) -> ResultExtractor[VariationalProgramResult[TVariationalInnerResult]]:
    """Build a ``ResultExtractor`` that validates the inner result type of a variational program.

    Args:
        inner_result_type (type[TVariationalInnerResult]): Expected type of
            the ``optimal_execution_results`` within the
            ``VariationalProgramResult``.

    Returns:
        ResultExtractor[VariationalProgramResult[TVariationalInnerResult]]:
        An extractor callable that raises ``RuntimeError`` when the inner
        result type does not match.
    """

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
    def functional(cls: type[JobHandle[FunctionalResult]], job_id: int) -> JobHandle[FunctionalResult]:
        """Create a handle for a ``DigitalPropagation`` or ``AnalogEvolution`` job.

        Args:
            job_id (int): Numeric identifier returned by the SpeQtrum service.

        Returns:
            JobHandle[FunctionalResult]: A handle whose result type is
            ``FunctionalResult``.
        """
        return cls(id=job_id, execute_type=ExecuteType.DIGITAL_PROPAGATION, extractor=_require_functional_result)

    @overload
    @classmethod
    def variational_program(cls, job_id: int) -> JobHandle[VariationalProgramResult]: ...

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
                extractor=_require_variational_program_result,
            )
            return cast("JobHandle[VariationalProgramResult]", handle)

        extractor = _require_variational_program_result_typed(result_type)
        handle = cls(id=job_id, execute_type=ExecuteType.VARIATIONAL_PROGRAM, extractor=extractor)
        return cast("JobHandle[VariationalProgramResult[TVariationalInnerResult]]", handle)

    @classmethod
    def rabi_experiment(cls: type[JobHandle[RabiExperimentResult]], job_id: int) -> JobHandle[RabiExperimentResult]:
        """Create a handle for a Rabi experiment job.

        Args:
            job_id (int): Numeric identifier returned by the SpeQtrum service.

        Returns:
            JobHandle[RabiExperimentResult]: A handle whose result type is
            ``RabiExperimentResult``.
        """
        return cls(id=job_id, execute_type=ExecuteType.RABI_EXPERIMENT, extractor=_require_rabi_experiment_result)

    @classmethod
    def t1_experiment(cls: type[JobHandle[T1ExperimentResult]], job_id: int) -> JobHandle[T1ExperimentResult]:
        """Create a handle for a T1 experiment job.

        Args:
            job_id (int): Numeric identifier returned by the SpeQtrum service.

        Returns:
            JobHandle[T1ExperimentResult]: A handle whose result type is
            ``T1ExperimentResult``.
        """
        return cls(id=job_id, execute_type=ExecuteType.T1_EXPERIMENT, extractor=_require_t1_experiment_result)

    @classmethod
    def t2_experiment(cls: type[JobHandle[T2ExperimentResult]], job_id: int) -> JobHandle[T2ExperimentResult]:
        """Create a handle for a T2 experiment job.

        Args:
            job_id (int): Numeric identifier returned by the SpeQtrum service.

        Returns:
            JobHandle[T2ExperimentResult]: A handle whose result type is
            ``T2ExperimentResult``.
        """
        return cls(id=job_id, execute_type=ExecuteType.T2_EXPERIMENT, extractor=_require_t2_experiment_result)

    @classmethod
    def two_tones_experiment(
        cls: type[JobHandle[TwoTonesExperimentResult]], job_id: int
    ) -> JobHandle[TwoTonesExperimentResult]:
        """Create a handle for a Two-Tones experiment job.

        Args:
            job_id (int): Numeric identifier returned by the SpeQtrum service.

        Returns:
            JobHandle[TwoTonesExperimentResult]: A handle whose result type is
            ``TwoTonesExperimentResult``.
        """
        return cls(
            id=job_id,
            execute_type=ExecuteType.TWO_TONES_EXPERIMENT,
            extractor=_require_two_tones_experiment_result,
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
    """Enumeration of possible job lifecycle states."""

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
    """Enumeration of job categories used by the SpeQtrum scheduler."""

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
