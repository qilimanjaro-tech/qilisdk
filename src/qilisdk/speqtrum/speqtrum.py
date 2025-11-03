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
from __future__ import annotations

import base64
import binascii
import json
import time
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Generator, TypeVar, cast, overload

import httpx
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from qilisdk.functionals import (
    Sampling,
    SamplingResult,
    TimeEvolution,
    TimeEvolutionResult,
    VariationalProgram,
    VariationalProgramResult,
)
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.settings import get_settings
from qilisdk.speqtrum.experiments import (
    RabiExperiment,
    RabiExperimentResult,
    T1Experiment,
    T1ExperimentResult,
)

from .keyring import delete_credentials, load_credentials, store_credentials
from .speqtrum_models import (
    Device,
    ExecutePayload,
    ExecuteType,
    JobDetail,
    JobHandle,
    JobId,
    JobInfo,
    JobStatus,
    JobType,
    RabiExperimentPayload,
    SamplingPayload,
    T1ExperimentPayload,
    TimeEvolutionPayload,
    Token,
    TypedJobDetail,
    VariationalProgramPayload,
)

if TYPE_CHECKING:
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional


ResultT = TypeVar("ResultT", bound=FunctionalResult)


def _safe_json_loads(value: str, *, context: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode JSON for {}: {}", context, exc)
        return None


def _safe_b64_decode(value: str, *, context: str) -> str | None:
    try:
        decoded_bytes = base64.b64decode(value)
    except (binascii.Error, ValueError) as exc:
        logger.warning("Failed to base64 decode {}: {}", context, exc)
        return None
    try:
        return decoded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("Failed to UTF-8 decode {}: {}", context, exc)
        return None


def _safe_b64_json(value: str, *, context: str) -> Any | None:
    decoded_text = _safe_b64_decode(value, context=context)
    if decoded_text is None:
        return None
    return _safe_json_loads(decoded_text, context=context)


class _BearerAuth(httpx.Auth):
    """Bearer token auth handler with automatic refresh support."""
    requires_response_body = True

    def __init__(self, client: SpeQtrum) -> None:
        self._client = client

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._client.token.access_token}"
        response = yield request

        if response.status_code == httpx.codes.UNAUTHORIZED:
            settings = get_settings()
            refresh_request = httpx.Request(
                "POST",
                settings.speqtrum_api_url + "/authorisation-tokens/refresh",
                headers={"Authorization": f"Bearer {self._client.token.refresh_token}"},
            )
            refresh_response = yield refresh_request

            try:
                refresh_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Token refresh failed with status {} {}",
                    exc.response.status_code,
                    exc.response.reason_phrase,
                )
                raise

            try:
                payload = refresh_response.json()
            except json.JSONDecodeError as exc:
                logger.error("Token refresh returned invalid JSON: {}", exc)
                raise RuntimeError("SpeQtrum token refresh failed: invalid JSON payload") from exc

            try:
                token = Token(**payload)
            except (TypeError, ValidationError) as exc:
                logger.error("Token refresh returned malformed payload: {}", exc)
                raise RuntimeError("SpeQtrum token refresh failed: malformed token payload") from exc

            self._client.token = token
            store_credentials(self._client.username, self._client.token)
            request.headers["Authorization"] = f"Bearer {self._client.token.access_token}"
            yield request


class SpeQtrum:
    """Synchronous client for the Qilimanjaro SpeQtrum API."""

    def __init__(self) -> None:
        logger.debug("Initializing SpeQtrum client")
        credentials = load_credentials()
        if credentials is None:
            logger.error("No credentials found. Call `SpeQtrum.login()` before instantiation.")
            raise RuntimeError("Missing credentials - invoke SpeQtrum.login() first.")
        self.username, self.token = credentials
        logger.success("SpeQtrum client initialised for user '{}'", self.username)

    @classmethod
    def _get_headers(cls) -> dict:
        from qilisdk import __version__  # noqa: PLC0415

        return {"User-Agent": f"qilisdk/{__version__}"}

    def _create_client(self) -> httpx.Client:
        """Return a freshly configured HTTP client for SpeQtrum interactions."""
        settings = get_settings()
        return httpx.Client(
            base_url=settings.speqtrum_api_url,
            headers=self._get_headers(),
            auth=_BearerAuth(self),
        )

    @classmethod
    def login(
        cls,
        username: str | None = None,
        apikey: str | None = None,
    ) -> bool:
        """Authenticate and cache credentials in the system keyring.

        Args:
            username: SpeQtrum account user name. If ``None``, the value is read
                from the environment.
            apikey: SpeQtrum API key. If ``None``, the value is read from the
                environment.

        Returns:
            ``True`` if authentication succeeds, otherwise ``False``.

        Note:
            The resulting tokens are stored securely in the OS keyring so that future
            :class:`SpeQtrum` constructions require no explicit credentials.
        """
        # Use provided parameters or fall back to environment variables via Settings()
        settings = get_settings()
        username = username or settings.speqtrum_username
        apikey = apikey or settings.speqtrum_apikey

        if not username or not apikey:
            logger.warning("Login called without credentials - aborting")
            return False

        # Send login request to QaaS
        logger.debug("Attempting login for user '{}'", username)
        try:
            assertion = {
                "username": username,
                "api_key": apikey,
                "audience": settings.speqtrum_audience,
                "iat": int(datetime.now(timezone.utc).timestamp()),
            }
            encoded_assertion = urlsafe_b64encode(json.dumps(assertion, indent=2).encode("utf-8")).decode("utf-8")
            with httpx.Client(
                base_url=settings.speqtrum_api_url,
                headers=cls._get_headers(),
                timeout=10.0,
            ) as client:
                response = client.post(
                    "/authorisation-tokens",
                    json={
                        "grantType": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": encoded_assertion,
                        "scope": "user profile",
                    },
                )
                response.raise_for_status()
                token = Token(**response.json())
        except httpx.HTTPStatusError as exc:
            logger.error("Login failed - server returned {} {}", exc.response.status_code, exc.response.reason_phrase)
            return False
        except httpx.RequestError:
            logger.exception("Network error while logging in to QaaS")
            return False

        store_credentials(username=username, token=token)
        logger.success("Login successful for user '{}'", username)
        return True

    @classmethod
    def logout(cls) -> None:
        """Delete cached credentials from the keyring."""
        delete_credentials()
        logger.info("Cached credentials removed - user logged out")

    def list_devices(self, where: Callable[[Device], bool] | None = None) -> list[Device]:
        """Return all visible devices, optionally filtered.

        Args:
            where: A predicate that retains a device when it evaluates to
                ``True``. Pass ``None`` to disable filtering.

        Returns:
            A list of :class:`~qilisdk.models.Device` objects.
        """
        logger.debug("Fetching device list from server…")
        with self._create_client() as client:
            response = client.get("/devices")
            response.raise_for_status()

            devices = TypeAdapter(list[Device]).validate_python(response.json()["items"])

        logger.success("{} devices retrieved", len(devices))
        return [d for d in devices if where(d)] if where else devices

    def list_jobs(self, where: Callable[[JobInfo], bool] | None = None) -> list[JobInfo]:
        """Return lightweight job summaries.

        Args:
            where: Optional predicate applied client-side. A
                :class:`~qilisdk.models.JobInfo` remains in the list if the
                predicate returns ``True``. ``None`` disables filtering.

        Returns:
            A list of :class:`~qilisdk.models.JobInfo` objects.
        """
        logger.debug("Fetching job list…")
        with self._create_client() as client:
            response = client.get("/jobs")
            response.raise_for_status()

            jobs = TypeAdapter(list[JobInfo]).validate_python(response.json()["items"])

        logger.success("{} jobs retrieved", len(jobs))
        return [j for j in jobs if where(j)] if where else jobs

    @overload
    def get_job(self, job: JobHandle[ResultT]) -> TypedJobDetail[ResultT]: ...

    @overload
    def get_job(self, job: int) -> JobDetail: ...

    def get_job(self, job: int | JobHandle[Any]) -> JobDetail | TypedJobDetail[Any]:
        """Fetch the complete record of *job*.

        Args:
            job: Either the integer identifier or a previously returned `JobHandle`.

        Returns:
            A :class:`~qilisdk.models.JobDetail` snapshot. When a handle is supplied the
            result is wrapped in :class:`~qilisdk.models.TypedJobDetail` to expose typed accessors.
        """
        job_id = job.id if isinstance(job, JobHandle) else job
        logger.debug("Retrieving job {} details", job_id)
        with self._create_client() as client:
            response = client.get(
                f"/jobs/{job_id}",
                params={
                    "payload": True,
                    "result": True,
                    "logs": True,
                    "error_logs": True,
                    "error": True,
                },
            )
            response.raise_for_status()
            data = response.json()

        raw_payload = data.get("payload")
        if isinstance(raw_payload, str):
            data["payload"] = _safe_json_loads(raw_payload, context=f"job {job_id} payload")

        raw_result = data.get("result")
        if isinstance(raw_result, str):
            data["result"] = _safe_b64_json(raw_result, context=f"job {job_id} result")

        raw_error = data.get("error")
        if isinstance(raw_error, str):
            data["error"] = _safe_b64_decode(raw_error, context=f"job {job_id} error")

        raw_logs = data.get("logs")
        if isinstance(raw_logs, str):
            data["logs"] = _safe_b64_decode(raw_logs, context=f"job {job_id} logs")

        raw_error_logs = data.get("error_logs")
        if isinstance(raw_error_logs, str):
            data["error_logs"] = _safe_b64_decode(raw_error_logs, context=f"job {job_id} error logs")

        job_detail = TypeAdapter(JobDetail).validate_python(data)
        logger.debug("Job {} details retrieved (status {})", job_id, job_detail.status.value)
        if isinstance(job, JobHandle):
            return job.bind(job_detail)
        return job_detail

    @overload
    def wait_for_job(
        self,
        job: JobHandle[ResultT],
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> TypedJobDetail[ResultT]: ...

    @overload
    def wait_for_job(
        self,
        job: int,
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> JobDetail: ...

    def wait_for_job(
        self,
        job: int | JobHandle[Any],
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> JobDetail | TypedJobDetail[Any]:
        """Block until the job referenced by *job* reaches a terminal state.

        Args:
            job: Either the integer job identifier or a previously returned `JobHandle`.
            poll_interval: Seconds between successive polls. Defaults to ``5``.
            timeout: Maximum wait time in seconds. ``None`` waits indefinitely.

        Returns:
            Final :class:`~qilisdk.models.JobDetail` snapshot, optionally wrapped with type-safe accessors.

        Raises:
            TimeoutError: If *timeout* elapses before the job finishes.
        """
        job_id = job.id if isinstance(job, JobHandle) else job
        logger.info("Waiting for job {} (poll={}s, timeout={}s)…", job_id, poll_interval, timeout)
        start_t = time.monotonic()
        terminal_states = {
            JobStatus.COMPLETED,
            JobStatus.ERROR,
            JobStatus.CANCELLED,
        }

        # poll until we hit a terminal state or timeout
        while True:
            current = self.get_job(job_id)

            if current.status in terminal_states:
                logger.success("Job {} reached terminal state {}", job_id, current.status.value)
                if isinstance(job, JobHandle):
                    return job.bind(current)
                return current

            if timeout is not None and (time.monotonic() - start_t) >= timeout:
                logger.error(
                    "Timeout while waiting for job {} after {}s (last status {})",
                    job_id,
                    timeout,
                    current.status.value,
                )
                raise TimeoutError(
                    f"Timed out after {timeout}s while waiting for job {job_id} (last status {current.status.value!r})"
                )

            logger.debug("Job {} still {}, sleeping {}s", job_id, current.status.value, poll_interval)
            time.sleep(poll_interval)

    @overload
    def submit(self, functional: Sampling, device: str, job_name: str | None = None) -> JobHandle[SamplingResult]: ...

    @overload
    def submit(
        self, functional: TimeEvolution, device: str, job_name: str | None = None
    ) -> JobHandle[TimeEvolutionResult]: ...

    @overload
    def submit(
        self, functional: VariationalProgram[Sampling], device: str, job_name: str | None = None
    ) -> JobHandle[VariationalProgramResult[SamplingResult]]: ...

    @overload
    def submit(
        self, functional: VariationalProgram[TimeEvolution], device: str, job_name: str | None = None
    ) -> JobHandle[VariationalProgramResult[TimeEvolutionResult]]: ...

    @overload
    def submit(
        self, functional: VariationalProgram[PrimitiveFunctional[ResultT]], device: str, job_name: str | None = None
    ) -> JobHandle[VariationalProgramResult[ResultT]]: ...

    @overload
    def submit(
        self, functional: RabiExperiment, device: str, job_name: str | None = None
    ) -> JobHandle[RabiExperimentResult]: ...

    @overload
    def submit(
        self, functional: T1Experiment, device: str, job_name: str | None = None
    ) -> JobHandle[T1ExperimentResult]: ...

    def submit(self, functional: Functional, device: str, job_name: str | None = None) -> JobHandle[FunctionalResult]:
        """
        Submit a quantum functional for execution on the selected device.

        The concrete subclass of
        :class:`~qilisdk.functionals.functional.Functional` provided in
        *functional* determines which private ``_execute_*`` routine is
        invoked. Supported types are:

        * :class:`~qilisdk.functionals.sampling.Sampling`
        * :class:`~qilisdk.functionals.time_evolution.TimeEvolution`
        * :class:`~qilisdk.functionals.variational_program.VariationalProgram`
        * :class:`~qilisdk.speqtrum.experiments.experiment_functional.RabiExperiment`
        * :class:`~qilisdk.speqtrum.experiments.experiment_functional.T1Experiment`

        A backend device must be selected beforehand with
        :py:meth:`set_device`.

        Args:
            functional: A fully configured functional instance (e.g.,
                ``Sampling`` or ``TimeEvolution``) that defines the quantum
                workload to be executed.
            device: Device code returned by :py:meth:`list_devices`.
            job_name (optional): The name of the job, this can help you identify different jobs easier. Default: None.

        Returns:
            JobHandle: A typed handle carrying the numeric job identifier and result type metadata.

        Raises:
            NotImplementedError: If *functional* is not of a supported type.
        """
        if isinstance(functional, VariationalProgram):
            inner = functional.functional
            if isinstance(inner, Sampling):
                return self._submit_variational_program(cast("VariationalProgram[Sampling]", functional), device, job_name)
            if isinstance(inner, TimeEvolution):
                return self._submit_variational_program(
                    cast("VariationalProgram[TimeEvolution]", functional), device, job_name
                )

            # Fallback to untyped handle for custom primitives.
            job_handle = self._submit_variational_program(cast("VariationalProgram[Any]", functional), device, job_name)
            return cast("JobHandle[FunctionalResult]", job_handle)

        if isinstance(functional, Sampling):
            return self._submit_sampling(functional, device, job_name)

        if isinstance(functional, TimeEvolution):
            return self._submit_time_evolution(functional, device, job_name)

        if isinstance(functional, RabiExperiment):
            return self._submit_rabi_program(functional, device, job_name)

        if isinstance(functional, T1Experiment):
            return self._submit_t1_program(functional, device, job_name)

        logger.error("Unsupported functional type: {}", type(functional).__qualname__)
        raise NotImplementedError(
            f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
        )

    def _submit_sampling(
        self, sampling: Sampling, device: str, job_name: str | None = None
    ) -> JobHandle[SamplingResult]:
        payload = ExecutePayload(
            type=ExecuteType.SAMPLING,
            sampling_payload=SamplingPayload(sampling=sampling),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.DIGITAL,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing Sampling on device {}", device)
        with self._create_client() as client:
            response = client.post("/execute", json=json)
            response.raise_for_status()
            job = JobId(**response.json())
        logger.info("Sampling job submitted: {}", job.id)
        return JobHandle.sampling(job.id)

    def _submit_rabi_program(
        self, rabi_experiment: RabiExperiment, device: str, job_name: str | None = None
    ) -> JobHandle[RabiExperimentResult]:
        payload = ExecutePayload(
            type=ExecuteType.RABI_EXPERIMENT,
            rabi_experiment_payload=RabiExperimentPayload(rabi_experiment=rabi_experiment),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.PULSE,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing Rabi experiment on device {}", device)
        with self._create_client() as client:
            response = client.post("/execute", json=json)
            response.raise_for_status()
            job = JobId(**response.json())
        logger.info("Rabi experiment job submitted: {}", job.id)
        return JobHandle.rabi_experiment(job.id)

    def _submit_t1_program(
        self, t1_experiment: T1Experiment, device: str, job_name: str | None = None
    ) -> JobHandle[T1ExperimentResult]:
        payload = ExecutePayload(
            type=ExecuteType.T1_EXPERIMENT,
            t1_experiment_payload=T1ExperimentPayload(t1_experiment=t1_experiment),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.PULSE,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing T1 experiment on device {}", device)
        with self._create_client() as client:
            response = client.post("/execute", json=json)
            response.raise_for_status()
            job = JobId(**response.json())
        logger.info("T1 experiment job submitted: {}", job.id)
        return JobHandle.t1_experiment(job.id)

    def _submit_time_evolution(
        self, time_evolution: TimeEvolution, device: str, job_name: str | None = None
    ) -> JobHandle[TimeEvolutionResult]:
        payload = ExecutePayload(
            type=ExecuteType.TIME_EVOLUTION,
            time_evolution_payload=TimeEvolutionPayload(time_evolution=time_evolution),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.ANALOG,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing time evolution on device {}", device)
        with self._create_client() as client:
            response = client.post("/execute", json=json)
            response.raise_for_status()
            job = JobId(**response.json())
        logger.info("Time Evolution job submitted: {}", job.id)
        return JobHandle.time_evolution(job.id)

    @overload
    def _submit_variational_program(
        self, variational_program: VariationalProgram[Sampling], device: str, job_name: str | None = None
    ) -> JobHandle[VariationalProgramResult[SamplingResult]]: ...

    @overload
    def _submit_variational_program(
        self, variational_program: VariationalProgram[TimeEvolution], device: str, job_name: str | None = None
    ) -> JobHandle[VariationalProgramResult[TimeEvolutionResult]]: ...

    @overload
    def _submit_variational_program(
        self, variational_program: VariationalProgram[Any], device: str, job_name: str | None = None
    ) -> JobHandle[VariationalProgramResult]: ...

    def _submit_variational_program(
        self, variational_program: VariationalProgram[Any], device: str, job_name: str | None = None
    ) -> JobHandle[VariationalProgramResult]:
        payload = ExecutePayload(
            type=ExecuteType.VARIATIONAL_PROGRAM,
            variational_program_payload=VariationalProgramPayload(variational_program=variational_program),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.VARIATIONAL,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing variational program on device {}", device)
        with self._create_client() as client:
            response = client.post("/execute", json=json)
            response.raise_for_status()
            job = JobId(**response.json())
        logger.info("Variational program job submitted: {}", job.id)
        inner = variational_program.functional
        if isinstance(inner, Sampling):
            return JobHandle.variational_program(job.id, result_type=SamplingResult)
        if isinstance(inner, TimeEvolution):
            return JobHandle.variational_program(job.id, result_type=TimeEvolutionResult)
        return JobHandle.variational_program(job.id)
