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
from typing import TYPE_CHECKING, Any, Callable, Generator, TypeAlias, TypeVar, cast, overload

import httpx
from loguru import logger
from pydantic import TypeAdapter, ValidationError

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
from qilisdk.settings import get_settings

from .keyring import delete_credentials, load_credentials, store_credentials
from .speqtrum_models import (
    AnalogEvolutionPayload,
    Device,
    DigitalPropagationPayload,
    ExecutePayload,
    ExecuteType,
    JobDetail,
    JobHandle,
    JobId,
    JobInfo,
    JobStatus,
    JobType,
    QuantumReservoirPayload,
    RabiExperimentPayload,
    T1ExperimentPayload,
    T2ExperimentPayload,
    Token,
    TwoTonesExperimentPayload,
    TypedJobDetail,
    VariationalProgramPayload,
)

if TYPE_CHECKING:
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional
    from qilisdk.readout import E, Readout, S, T


TFunctionalResult = TypeVar("TFunctionalResult", bound=FunctionalResult)
JSONValue: TypeAlias = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None

_SKIP_ENSURE_OK_EXTENSION = "qilisdk.skip_ensure_ok"
_CONTEXT_EXTENSION = "qilisdk.request_context"
_EXECUTE_URL = "/execute"


class SpeQtrumAPIError(httpx.HTTPStatusError):
    """Raised when the SpeQtrum API responds with a non-success HTTP status.

    Wraps :class:`httpx.HTTPStatusError` with a human-readable error message
    extracted from the response body when possible.
    """

    def __init__(self, message: str, *, request: httpx.Request, response: httpx.Response) -> None:
        """Initialise the error.

        Args:
            message (str): Human-readable description of the failure.
            request (httpx.Request): The outgoing HTTP request that triggered
                the error.
            response (httpx.Response): The HTTP response carrying the error
                status.
        """
        super().__init__(message, request=request, response=response)


def _safe_json_loads(value: str, *, context: str) -> JSONValue | None:
    """Attempt to parse *value* as JSON, returning ``None`` on failure.

    Args:
        value (str): Raw JSON string.
        context (str): Label used in warning messages on parse failure.

    Returns:
        JSONValue | None: Parsed JSON value, or ``None`` if decoding fails.
    """
    try:
        result = json.loads(value)
        return cast("JSONValue", result)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode JSON for {}: {}", context, exc)
        return None


def _safe_b64_decode(value: str, *, context: str) -> str | None:
    """Base64-decode *value* and return the UTF-8 string, or ``None`` on failure.

    Args:
        value (str): Base64-encoded string.
        context (str): Label used in warning messages on decode failure.

    Returns:
        str | None: Decoded UTF-8 string, or ``None`` if any decoding step fails.
    """
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


def _safe_b64_json(value: str, *, context: str) -> JSONValue | None:
    """Base64-decode *value*, then JSON-parse the result.

    Args:
        value (str): Base64-encoded JSON string.
        context (str): Label used in warning messages on failure.

    Returns:
        JSONValue | None: Parsed JSON value, or ``None`` if any step fails.
    """
    decoded_text = _safe_b64_decode(value, context=context)
    if decoded_text is None:
        return None
    return _safe_json_loads(decoded_text, context=context)


def _request_extensions(*, context: str | None = None, skip_ensure_ok: bool = False) -> dict[str, Any] | None:
    """Build an ``httpx`` request extensions dictionary.

    Args:
        context (str | None): Human-readable label attached to the request for
            error reporting. ``None`` omits the key.
        skip_ensure_ok (bool): When ``True``, the ``_ensure_ok`` event hook
            will not raise on non-success status codes.

    Returns:
        dict[str, Any] | None: Extensions mapping, or ``None`` when no flags
        are set.
    """
    extensions: dict[str, Any] = {}
    if context:
        extensions[_CONTEXT_EXTENSION] = context
    if skip_ensure_ok:
        extensions[_SKIP_ENSURE_OK_EXTENSION] = True
    return extensions or None


def _response_context(response: httpx.Response) -> str:
    """Extract a human-readable context label from a response's originating request.

    Args:
        response (httpx.Response): The HTTP response to inspect.

    Returns:
        str: The context string stored in the request extensions, or a
        fallback ``"METHOD URL"`` string.
    """
    request = response.request
    if request is None:
        return "SpeQtrum API call"
    context = request.extensions.get(_CONTEXT_EXTENSION)
    if isinstance(context, str) and context.strip():
        return context
    return f"{request.method} {request.url}"


def _stringify_payload(payload: JSONValue | None) -> str | None:
    """Convert a parsed JSON error payload into a one-line human-readable string.

    Looks for common error-message keys (``message``, ``detail``, ``error``,
    etc.) and falls back to ``json.dumps`` when none are found.

    Args:
        payload (JSONValue | None): Parsed JSON body of the error response.

    Returns:
        str | None: A concise error description, or ``None`` when *payload*
        is ``None``.
    """
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in ("message", "detail", "error", "error_description", "title"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                code = payload.get("code") or payload.get("error_code") or payload.get("errorCode")
                code_suffix = f" (code={code})" if isinstance(code, (str, int)) else ""
                return value.strip() + code_suffix
        try:
            return json.dumps(payload, sort_keys=True)
        except (TypeError, ValueError):
            return str(payload)
    if isinstance(payload, list):
        preview = ", ".join(str(item) for item in payload[:3])
        return preview or str(payload)
    if isinstance(payload, (str, int, float, bool)):
        return str(payload)
    return None


def _summarize_error_payload(response: httpx.Response) -> str:
    """Produce a single-line summary of the error body carried by *response*.

    Args:
        response (httpx.Response): The non-success HTTP response.

    Returns:
        str: A best-effort textual summary of the error payload.
    """
    context = _response_context(response)
    try:
        body_text = response.text or ""
    except httpx.ResponseNotRead:
        try:
            response.read()  # ensure body is buffered so we can reuse it later
            body_text = response.text or ""
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read response body for {}: {}", context, exc)
            body_text = ""
    payload = _safe_json_loads(body_text, context=f"{context} error body") if body_text else None
    detail = _stringify_payload(payload)
    if detail:
        return detail
    if body_text.strip():
        return body_text.strip()
    return "no response body"


def _ensure_ok(response: httpx.Response) -> None:
    """``httpx`` event hook that raises :class:`SpeQtrumAPIError` on non-success status codes.

    Skips the check when the request carries the
    ``_SKIP_ENSURE_OK_EXTENSION`` flag or when the status is
    ``401 Unauthorized`` (handled separately by token refresh logic).

    Args:
        response (httpx.Response): The HTTP response to validate.

    Raises:
        SpeQtrumAPIError: If the response status indicates an error.
    """
    request = response.request
    if request is not None and request.extensions.get(_SKIP_ENSURE_OK_EXTENSION):
        return
    if response.status_code == httpx.codes.UNAUTHORIZED:
        return
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        context = _response_context(response)
        detail = _summarize_error_payload(response)
        logger.error(
            "{} failed with status {} {}: {}",
            context,
            response.status_code,
            response.reason_phrase,
            detail,
        )
        message = f"{context} failed with status {response.status_code}: {detail}"
        raise SpeQtrumAPIError(message, request=response.request, response=response) from None


class _BearerAuth(httpx.Auth):
    """Bearer-token authentication handler with automatic token refresh.

    Implements the ``httpx`` auth-flow protocol. On a ``401 Unauthorized``
    response the handler transparently refreshes the access token using the
    stored refresh token and retries the original request.
    """

    requires_response_body = True

    def __init__(self, client: SpeQtrum) -> None:
        """Initialise the auth handler.

        Args:
            client (SpeQtrum): The parent ``SpeQtrum`` client whose token
                will be read and updated.
        """
        self._client = client

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Yield authenticated requests, refreshing the token on ``401``.

        Args:
            request (httpx.Request): The outgoing request to authenticate.

        Yields:
            httpx.Request: Requests decorated with an ``Authorization`` header.

        Raises:
            RuntimeError: if Speqtrum token refresh fails.
            HTTPStatusError: if the refresh token fails.
        """
        request.headers["Authorization"] = f"Bearer {self._client.token.access_token}"
        response = yield request

        if response.status_code == httpx.codes.UNAUTHORIZED:
            settings = get_settings()
            refresh_request = httpx.Request(
                "POST",
                settings.speqtrum_api_url + "/authorisation-tokens/refresh",
                headers={"Authorization": f"Bearer {self._client.token.refresh_token}"},
                extensions=_request_extensions(context="Refreshing SpeQtrum token", skip_ensure_ok=True),
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

            if not isinstance(payload, dict):
                logger.error("Token refresh returned non-object payload: {}", type(payload).__name__)
                raise RuntimeError("SpeQtrum token refresh failed: malformed token payload")

            payload.pop("refreshToken", None)

            try:
                token = Token(**payload, refreshToken=self._client.token.refresh_token)
            except (TypeError, ValidationError) as exc:
                logger.error("Token refresh returned malformed payload: {}", exc)
                raise RuntimeError("SpeQtrum token refresh failed: malformed token payload") from exc

            self._client.token = token
            store_credentials(self._client.username, self._client.token)
            request.headers["Authorization"] = f"Bearer {self._client.token.access_token}"
            yield request


class SpeQtrum:
    """Synchronous client for the Qilimanjaro SpeQtrum API.

    Provides methods for authentication, device discovery, job submission,
    and result retrieval. Credentials are loaded from the system keyring;
    call :meth:`login` first if no credentials have been cached.

    Supported functional types for submission include
    :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation`,
    :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`,
    :class:`~qilisdk.functionals.variational_program.VariationalProgram`,
    :class:`~qilisdk.functionals.quantum_reservoir.QuantumReservoir`, and
    various experiment types.
    """

    def __init__(self) -> None:
        """Initialise the ``SpeQtrum`` client using cached keyring credentials.

        Raises:
            RuntimeError: If no credentials are found in the keyring. Call
                :meth:`login` before instantiation.
        """
        logger.debug("Initializing SpeQtrum client")
        credentials = load_credentials()
        if credentials is None:
            logger.error("No credentials found. Call `SpeQtrum.login()` before instantiation.")
            raise RuntimeError("Missing credentials - invoke SpeQtrum.login() first.")
        self.username, self.token = credentials
        logger.success("SpeQtrum client initialised for user '{}'", self.username)

    @classmethod
    def _get_headers(cls) -> dict:
        """Return default HTTP headers including the SDK ``User-Agent``.

        Returns:
            dict: Headers dictionary with at least the ``User-Agent`` key.
        """
        from qilisdk import __version__  # noqa: PLC0415

        return {"User-Agent": f"qilisdk/{__version__}"}

    def _create_client(self) -> httpx.Client:
        """Return a freshly configured HTTP client for SpeQtrum interactions."""
        settings = get_settings()
        return httpx.Client(
            base_url=settings.speqtrum_api_url,
            headers=self._get_headers(),
            auth=_BearerAuth(self),
            event_hooks={"response": [_ensure_ok]},
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
        apikey = apikey or (
            settings.speqtrum_apikey.get_secret_value() if settings.speqtrum_apikey is not None else None
        )

        if not username or not apikey:
            logger.error("Login called without credentials.")
            return False

        # Send login request to QaaS
        logger.debug("Attempting login for user '{}'", username)
        assertion = {
            "username": username,
            "api_key": apikey,
            "audience": settings.speqtrum_audience,
            "iat": int(datetime.now(timezone.utc).timestamp()),
        }
        encoded_assertion = urlsafe_b64encode(json.dumps(assertion, indent=2).encode("utf-8")).decode("utf-8")
        try:
            with httpx.Client(
                base_url=settings.speqtrum_api_url,
                headers=cls._get_headers(),
                timeout=10.0,
                event_hooks={"response": [_ensure_ok]},
            ) as client:
                response = client.post(
                    "/authorisation-tokens",
                    json={
                        "grantType": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": encoded_assertion,
                        "scope": "user profile",
                    },
                    extensions=_request_extensions(context="Authenticating user"),
                )
                response.raise_for_status()
                token = Token(**response.json())
        except httpx.HTTPError:
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
            response = client.get("/devices", extensions=_request_extensions(context="Fetching device list"))
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
            response = client.get("/jobs", extensions=_request_extensions(context="Fetching job list"))
        jobs = TypeAdapter(list[JobInfo]).validate_python(response.json()["items"])
        logger.success("{} jobs retrieved", len(jobs))
        return [j for j in jobs if where(j)] if where else jobs

    @overload
    def get_job(self, job: JobHandle[TFunctionalResult]) -> TypedJobDetail[TFunctionalResult]: ...

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
                extensions=_request_extensions(context=f"Fetching job {job_id}"),
            )
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
        job: JobHandle[TFunctionalResult],
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> TypedJobDetail[TFunctionalResult]: ...

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
    def submit(
        self,
        functional: PrimitiveFunctional,
        device: str,
        readout: Readout[S, E, T],
        job_name: str | None = None,
    ) -> JobHandle[FunctionalResult[S, E, T]]: ...

    @overload
    def submit(
        self,
        functional: VariationalProgram,
        device: str,
        readout: Readout[S, E, T],
        job_name: str | None = None,
    ) -> JobHandle[VariationalProgramResult]: ...

    @overload
    def submit(
        self, functional: RabiExperiment, device: str, job_name: str | None = None
    ) -> JobHandle[RabiExperimentResult]: ...

    @overload
    def submit(
        self, functional: T1Experiment, device: str, job_name: str | None = None
    ) -> JobHandle[T1ExperimentResult]: ...

    def submit(
        self,
        functional: Functional,
        device: str,
        readout: Readout[S, E, T] | None = None,  # type: ignore[type-arg]
        job_name: str | None = None,
    ) -> JobHandle:
        """
        Submit a quantum functional for execution on the selected device.

        Supported types:

        * :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation`
        * :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`
        * :class:`~qilisdk.functionals.variational_program.VariationalProgram`
        * :class:`~qilisdk.experiments.experiment_functional.RabiExperiment`
        * :class:`~qilisdk.experiments.experiment_functional.T1Experiment`
        * :class:`~qilisdk.experiments.experiment_functional.T2Experiment`
        * :class:`~qilisdk.experiments.experiment_functional.TwoTonesExperiment`

        Args:
            functional: A fully configured functional instance that defines the quantum workload.
            device: Device code returned by :py:meth:`list_devices`.
            readout: Readout method(s) specifying how results should be measured.
                Required for ``DigitalPropagation``, ``AnalogEvolution``, and ``VariationalProgram``.
            job_name (optional): The name of the job. Default: None.

        Returns:
            JobHandle: A typed handle carrying the numeric job identifier and result type metadata.

        Raises:
            NotImplementedError: If *functional* is not of a supported type.
            ValueError: If *readout* is required but not provided, or contains invalid methods.
        """

        # Experiments without readout

        if isinstance(functional, RabiExperiment):
            return self._submit_rabi(functional, device, job_name)

        if isinstance(functional, T1Experiment):
            return self._submit_t1(functional, device, job_name)

        if isinstance(functional, T2Experiment):
            return self._submit_t2(functional, device, job_name)

        if isinstance(functional, TwoTonesExperiment):
            return self._submit_two_tones(functional, device, job_name)

        # Functionals with readout

        if readout is None:
            raise ValueError("Readout can't be none when submitting a functional")

        if isinstance(functional, VariationalProgram):
            return self._submit_variational_program(functional, device, readout, job_name)

        if isinstance(functional, DigitalPropagation):
            return self._submit_digital_propagation(functional, device, readout, job_name)

        if isinstance(functional, AnalogEvolution):
            return self._submit_analog_evolution(functional, device, readout, job_name)

        if isinstance(functional, QuantumReservoir):
            return self._submit_quantum_reservoir_functional(functional, device, readout, job_name)

        logger.error("Unsupported functional type: {}", type(functional).__qualname__)
        raise NotImplementedError(f"{type(self).__qualname__} does not support {type(functional).__qualname__}")

    def _submit_digital_propagation(
        self, functional: DigitalPropagation, device: str, readout: Readout[S, E, T], job_name: str | None = None
    ) -> JobHandle[FunctionalResult]:
        """Submit a ``DigitalPropagation`` functional to the SpeQtrum API.

        Args:
            functional (DigitalPropagation): The digital propagation to execute.
            device (str): Target device code.
            readout (Readout[S, E, T]): Readout methods for measurement.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[FunctionalResult]: A handle for tracking the submitted job.
        """
        payload = ExecutePayload(
            type=ExecuteType.DIGITAL_PROPAGATION,
            digital_propagation_payload=DigitalPropagationPayload(digital_propagation=functional, readout=readout),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.DIGITAL,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing DigitalPropagation on device {}", device)
        with self._create_client() as client:
            response = client.post(
                _EXECUTE_URL, json=json, extensions=_request_extensions(context="Executing DigitalPropagation")
            )
        job = JobId(**response.json())
        logger.info("DigitalPropagation job submitted: {}", job.id)
        return JobHandle.functional(job.id)

    def _submit_rabi(
        self, rabi_experiment: RabiExperiment, device: str, job_name: str | None = None
    ) -> JobHandle[RabiExperimentResult]:
        """Submit a Rabi experiment to the SpeQtrum API.

        Args:
            rabi_experiment (RabiExperiment): The experiment to execute.
            device (str): Target device code.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[RabiExperimentResult]: A handle for tracking the submitted job.
        """
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
            response = client.post(
                _EXECUTE_URL,
                json=json,
                extensions=_request_extensions(context="Executing Rabi experiment"),
            )
        job = JobId(**response.json())
        logger.info("Rabi experiment job submitted: {}", job.id)
        return JobHandle.rabi_experiment(job.id)

    def _submit_t1(
        self, t1_experiment: T1Experiment, device: str, job_name: str | None = None
    ) -> JobHandle[T1ExperimentResult]:
        """Submit a T1 experiment to the SpeQtrum API.

        Args:
            t1_experiment (T1Experiment): The experiment to execute.
            device (str): Target device code.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[T1ExperimentResult]: A handle for tracking the submitted job.
        """
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
            response = client.post(
                _EXECUTE_URL,
                json=json,
                extensions=_request_extensions(context="Executing T1 experiment"),
            )
        job = JobId(**response.json())
        logger.info("T1 experiment job submitted: {}", job.id)
        return JobHandle.t1_experiment(job.id)

    def _submit_t2(
        self, t2_experiment: T2Experiment, device: str, job_name: str | None = None
    ) -> JobHandle[T2ExperimentResult]:
        """Submit a T2 experiment to the SpeQtrum API.

        Args:
            t2_experiment (T2Experiment): The experiment to execute.
            device (str): Target device code.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[T2ExperimentResult]: A handle for tracking the submitted job.
        """
        payload = ExecutePayload(
            type=ExecuteType.T2_EXPERIMENT,
            t2_experiment_payload=T2ExperimentPayload(t2_experiment=t2_experiment),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.PULSE,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing T2 experiment on device {}", device)
        with self._create_client() as client:
            response = client.post(
                _EXECUTE_URL,
                json=json,
                extensions=_request_extensions(context="Executing T2 experiment"),
            )
        job = JobId(**response.json())
        logger.info("T2 experiment job submitted: {}", job.id)
        return JobHandle.t2_experiment(job.id)

    def _submit_two_tones(
        self, two_tones_experiment: TwoTonesExperiment, device: str, job_name: str | None = None
    ) -> JobHandle[TwoTonesExperimentResult]:
        """Submit a Two-Tones experiment to the SpeQtrum API.

        Args:
            two_tones_experiment (TwoTonesExperiment): The experiment to execute.
            device (str): Target device code.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[TwoTonesExperimentResult]: A handle for tracking the
            submitted job.
        """
        payload = ExecutePayload(
            type=ExecuteType.TWO_TONES_EXPERIMENT,
            two_tones_experiment_payload=TwoTonesExperimentPayload(two_tones_experiment=two_tones_experiment),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.PULSE,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing Two-Tones experiment on device {}", device)
        with self._create_client() as client:
            response = client.post(
                _EXECUTE_URL,
                json=json,
                extensions=_request_extensions(context="Executing Two-Tones experiment"),
            )
        job = JobId(**response.json())
        logger.info("Two-Tones experiment job submitted: {}", job.id)
        return JobHandle.two_tones_experiment(job.id)

    def _submit_analog_evolution(
        self, functional: AnalogEvolution, device: str, readout: Readout[S, E, T], job_name: str | None = None
    ) -> JobHandle[FunctionalResult]:
        """Submit an ``AnalogEvolution`` functional to the SpeQtrum API.

        Args:
            functional (AnalogEvolution): The analog evolution to execute.
            device (str): Target device code.
            readout (Readout[S, E, T]): Readout methods for measurement.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[FunctionalResult]: A handle for tracking the submitted job.
        """
        payload = ExecutePayload(
            type=ExecuteType.ANALOG_EVOLUTION,
            analog_evolution_payload=AnalogEvolutionPayload(analog_evolution=functional, readout=readout),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.ANALOG,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing AnalogEvolution on device {}", device)
        with self._create_client() as client:
            response = client.post(
                _EXECUTE_URL,
                json=json,
                extensions=_request_extensions(context="Executing AnalogEvolution"),
            )
        job = JobId(**response.json())
        logger.info("AnalogEvolution job submitted: {}", job.id)
        return JobHandle.functional(job.id)

    def _submit_quantum_reservoir_functional(
        self, functional: QuantumReservoir, device: str, readout: Readout[S, E, T], job_name: str | None = None
    ) -> JobHandle[FunctionalResult]:
        """Submit a ``QuantumReservoir`` functional to the SpeQtrum API.

        Args:
            functional (QuantumReservoir): The quantum reservoir to execute.
            device (str): Target device code.
            readout (Readout[S, E, T]): Readout methods for measurement.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[FunctionalResult]: A handle for tracking the submitted job.
        """
        payload = ExecutePayload(
            type=ExecuteType.ANALOG_EVOLUTION,
            quantum_reservoir_payload=QuantumReservoirPayload(quantum_reservoir=functional, readout=readout),
        )
        json = {
            "device_code": device,
            "payload": payload.model_dump_json(),
            "job_type": JobType.ANALOG,
            "meta": {},
        }
        if job_name:
            json["name"] = job_name
        logger.debug("Executing AnalogEvolution on device {}", device)
        with self._create_client() as client:
            response = client.post(
                _EXECUTE_URL,
                json=json,
                extensions=_request_extensions(context="Executing AnalogEvolution"),
            )
        job = JobId(**response.json())
        logger.info("AnalogEvolution job submitted: {}", job.id)
        return JobHandle.functional(job.id)

    def _submit_variational_program(
        self,
        variational_program: VariationalProgram,
        device: str,
        readout: Readout[S, E, T],
        job_name: str | None = None,
    ) -> JobHandle[VariationalProgramResult[S, E, T]]:
        """Submit a ``VariationalProgram`` to the SpeQtrum API.

        Args:
            variational_program (VariationalProgram): The variational program
                to execute.
            device (str): Target device code.
            readout (Readout[S, E, T]): Readout methods for measurement.
            job_name (str | None): Optional human-readable job name.

        Returns:
            JobHandle[VariationalProgramResult]: A handle for tracking the
            submitted job.
        """
        payload = ExecutePayload(
            type=ExecuteType.VARIATIONAL_PROGRAM,
            variational_program_payload=VariationalProgramPayload(
                variational_program=variational_program, readout=readout
            ),
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
            response = client.post(
                _EXECUTE_URL,
                json=json,
                extensions=_request_extensions(context="Executing variational program"),
            )
        job = JobId(**response.json())
        logger.info("Variational program job submitted: {}", job.id)
        return JobHandle.variational_program(job.id)

    def __repr__(self) -> str:
        """Return a string representation of the client.

        Returns:
            str: A string of the form ``SpeQtrum(username=...)``.
        """
        return f"{type(self).__name__}(username={self.username})"
