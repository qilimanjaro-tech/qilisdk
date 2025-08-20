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
import json
import time
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, cast

import httpx
from loguru import logger
from pydantic import TypeAdapter

from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.settings import get_settings

from .keyring import delete_credentials, load_credentials, store_credentials
from .speqtrum_models import (
    Device,
    ExecutePayload,
    ExecuteType,
    JobDetail,
    JobId,
    JobInfo,
    JobStatus,
    SamplingPayload,
    TimeEvolutionPayload,
    Token,
    VariationalProgramPayload,
)

if TYPE_CHECKING:
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional


class SpeQtrum:
    """Synchronous client for the Qilimanjaro SpeQtrum API."""

    def __init__(self) -> None:
        logger.debug("Initializing QaaS client")
        credentials = load_credentials()
        if credentials is None:
            logger.error("No QaaS credentials found. Call `.login()` or set env vars before instantiation.")
            raise RuntimeError("Missing QaaS credentials - invoke SpeQtrum.login() first.")
        self._username, self._token = credentials
        self._selected_device: int | None = None
        self._handlers: dict[type[Functional], Callable[[Functional], int]] = {
            Sampling: lambda f: self._submit_sampling(cast("Sampling", f)),
            TimeEvolution: lambda f: self._submit_time_evolution(cast("TimeEvolution", f)),
            VariationalProgram: lambda f: self._submit_variational_program(cast("VariationalProgram", f)),
        }
        self._settings = get_settings()
        logger.success("QaaS client initialised for user '{}'", self._username)

    @classmethod
    def _get_headers(cls) -> dict:
        from qilisdk import __version__  # noqa: PLC0415

        return {"User-Agent": f"qilisdk/{__version__}"}

    def _get_authorized_headers(self) -> dict:
        return {**self._get_headers(), "Authorization": f"Bearer {self._token.access_token}"}

    # TODO (vyron): Change this to `code: str` when implemented server-side.
    @property
    def selected_device(self) -> int | None:
        """ID of the currently selected device.

        Returns:
            The device identifier or ``None`` if no device is selected.
        """
        return self._selected_device

    def set_device(self, id: int) -> None:
        """Select the backend device for subsequent executions.

        Args:
            id: Device identifier returned by :py:meth:`list_devices`.
        """
        self._selected_device = id

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
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    settings.speqtrum_api_url + "/authorisation-tokens",
                    json={
                        "grantType": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": encoded_assertion,
                        "scope": "user profile",
                    },
                    headers=cls._get_headers(),
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
        with httpx.Client() as client:
            response = client.get(self._settings.speqtrum_api_url + "/devices", headers=self._get_authorized_headers())
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
        with httpx.Client() as client:
            response = client.get(self._settings.speqtrum_api_url + "/jobs", headers=self._get_authorized_headers())
            response.raise_for_status()

            jobs = TypeAdapter(list[JobInfo]).validate_python(response.json()["items"])

        logger.success("{} jobs retrieved", len(jobs))
        return [j for j in jobs if where(j)] if where else jobs

    def get_job_details(self, id: int) -> JobDetail:
        """Fetch the complete record of *id*.

        Args:
            id: Identifier of the job.

        Returns:
            A :class:`~qilisdk.models.JobDetail` instance containing payload,
            result, logs and error information.
        """
        logger.debug("Retrieving job {} details", id)
        with httpx.Client() as client:
            response = client.get(
                f"{self._settings.speqtrum_api_url}/jobs/{id}",
                headers=self._get_authorized_headers(),
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

        raw_payload = data["payload"]
        if raw_payload is not None:
            data["payload"] = json.loads(raw_payload)

        raw_result = data.get("result")
        if raw_result is not None:
            decoded_result: bytes = base64.b64decode(raw_result)
            text_result = decoded_result.decode("utf-8")
            data["result"] = json.loads(text_result)

        raw_error = data.get("error")
        if raw_error is not None:
            decoded_error: bytes = base64.b64decode(raw_error)
            data["error"] = decoded_error.decode("utf-8")

        raw_logs = data.get("logs")
        if raw_logs is not None:
            decoded_logs: bytes = base64.b64decode(raw_logs)
            data["logs"] = decoded_logs.decode("utf-8")

        job_detail = TypeAdapter(JobDetail).validate_python(data)
        logger.debug("Job {} details retrieved (status {})", id, job_detail.status.value)
        return job_detail

    def wait_for_job(
        self,
        id: int,
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> JobDetail:
        """Block until *id* reaches a terminal state.

        Args:
            id: Job identifier.
            poll_interval: Seconds between successive polls. Defaults to ``5``.
            timeout: Maximum wait time in seconds. ``None`` waits indefinitely.

        Returns:
            Final :class:`~qilisdk.models.JobDetail` snapshot.

        Raises:
            TimeoutError: If *timeout* elapses before the job finishes.
        """
        logger.info("Waiting for job {} (poll={}s, timeout={}s)…", id, poll_interval, timeout)
        start_t = time.monotonic()
        terminal_states = {
            JobStatus.COMPLETED,
            JobStatus.ERROR,
            JobStatus.CANCELLED,
        }

        # poll until we hit a terminal state or timeout
        while True:
            current = self.get_job_details(id)

            if current.status in terminal_states:
                logger.success("Job {} reached terminal state {}", id, current.status.value)
                return current

            if timeout is not None and (time.monotonic() - start_t) >= timeout:
                logger.error(
                    "Timeout while waiting for job {} after {}s (last status {})", id, timeout, current.status.value
                )
                raise TimeoutError(
                    f"Timed out after {timeout}s while waiting for job {id} (last status {current.status.value!r})"
                )

            logger.debug("Job {} still {}, sleeping {}s", id, current.status.value, poll_interval)
            time.sleep(poll_interval)

    def _ensure_device_selected(self) -> int:
        if self._selected_device is None:
            logger.error("Attempted execution without selecting a device")
            raise ValueError("Device not selected.")
        return self._selected_device

    def submit(self, functional: PrimitiveFunctional) -> int:
        """
        Submit a quantum functional for execution on the selected device.

        The concrete subclass of
        :class:`~qilisdk.functionals.functional.Functional` provided in
        *functional* determines which private ``_execute_*`` routine is
        invoked. Supported types are:

        * :class:`~qilisdk.functionals.sampling.Sampling`
        * :class:`~qilisdk.functionals.time_evolution.TimeEvolution`

        A backend device must be selected beforehand with
        :py:meth:`set_device`.

        Args:
            functional: A fully configured functional instance (e.g.,
                ``Sampling`` or ``TimeEvolution``) that defines the quantum
                workload to be executed.

        Returns:
            int: The numeric identifier of the created job on SpeQtrum.

        Raises:
            NotImplementedError: If *functional* is not of a supported type.
        """
        try:
            handler = self._handlers[type(functional)]
        except KeyError as exc:
            logger.error("Unsupported functional type: {}", type(functional).__qualname__)
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
            ) from exc

        logger.info("Submitting {}", type(functional).__qualname__)
        job_id = handler(functional)
        logger.success("Submission complete - job {}", job_id)
        return job_id

    def _submit_sampling(self, sampling: Sampling) -> int:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.SAMPLING,
            sampling_payload=SamplingPayload(sampling=sampling),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                self._settings.speqtrum_api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
        return job.id

    def _submit_time_evolution(self, time_evolution: TimeEvolution) -> int:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.TIME_EVOLUTION,
            time_evolution_payload=TimeEvolutionPayload(time_evolution=time_evolution),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        logger.debug("Executing time evolution on device {}", device)
        with httpx.Client() as client:
            response = client.post(
                self._settings.speqtrum_api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
        logger.info("Time evolution job submitted: {}", job.id)
        return job.id

    def _submit_variational_program(
        self,
        variational_program: VariationalProgram,
    ) -> int:
        """Run a Variational Program on the selected device.

        Args:
            variational_program: Problem definition containing Hamiltonian and ansatz.
            optimizer: Classical optimizer that updates the variational
                parameters between circuit evaluations.
            nshots: Number of shots per circuit evaluation. Defaults to
                ``1000``.
            store_intermediate_results: Whether to keep intermediate energies
                and parameter vectors for later analysis.

        Returns:
            The numeric identifier of the created job.
        """
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.VARIATIONAL_PROGRAM,
            variational_program_payload=VariationalProgramPayload(
                variational_program=variational_program,
            ),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                self._settings.speqtrum_api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id
