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
import logging
import time
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import httpx
from pydantic import TypeAdapter

from qilisdk.common.result import Result
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.settings import get_settings

from .keyring import delete_credentials, load_credentials, store_credentials
from .qaas_models import (
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
    VQEPayload,
)

if TYPE_CHECKING:
    from qilisdk.digital.vqe import VQE
    from qilisdk.functionals.functional import Functional
    from qilisdk.optimizers.optimizer import Optimizer

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
)

TResult = TypeVar("TResult", bound=Result)


class QaaS:
    """Synchronous client for the Qilimanjaro QaaS REST API."""

    def __init__(self) -> None:
        credentials = load_credentials()
        if credentials is None:
            raise RuntimeError(
                "No valid QaaS credentials found in keyring."
                "Please call QaaSBackend.login(username, apikey) or ensure environment variables are set."
            )
        self._username, self._token = credentials
        self._selected_device: int | None = None
        self._handlers: dict[type[Functional[Any]], Callable[[Functional[Any]], int]] = {
            Sampling: lambda f: self._execute_sampling(cast("Sampling", f)),
            TimeEvolution: lambda f: self._execute_time_evolution(cast("TimeEvolution", f)),
        }
        self._settings = get_settings()

    def _get_headers(self) -> dict:  # noqa: PLR6301
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
            username: QaaS account user name. If ``None``, the value is read
                from the environment.
            apikey: QaaS API key. If ``None``, the value is read from the
                environment.

        Returns:
            ``True`` if authentication succeeds, otherwise ``False``.

        Note:
            The resulting tokens are stored in the OS keyring so that future
            :class:`QaaSBackend` constructions require no explicit credentials.
        """
        # Use provided parameters or fall back to environment variables via Settings()
        settings = get_settings()
        username = username or settings.qaas_username
        apikey = apikey or settings.qaas_apikey

        if not username or not apikey:
            return False

        # Send login request to QaaS
        try:
            assertion = {
                "username": username,
                "api_key": apikey,
                "audience": settings.qaas_audience,
                "iat": int(datetime.now(timezone.utc).timestamp()),
            }
            encoded_assertion = urlsafe_b64encode(json.dumps(assertion, indent=2).encode("utf-8")).decode("utf-8")
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    settings.qaas_api_url + "/authorisation-tokens",
                    json={
                        "grantType": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": encoded_assertion,
                        "scope": "user profile",
                    },
                    headers={"User-Agent": "qilisdk/0.1.4"},
                )
                response.raise_for_status()
                token = Token(**response.json())
        except httpx.RequestError:
            return False

        store_credentials(username=username, token=token)
        return True

    @classmethod
    def logout(cls) -> None:
        """Delete cached credentials from the keyring."""
        delete_credentials()

    def list_devices(self, where: Callable[[Device], bool] | None = None) -> list[Device]:
        """Return all visible devices, optionally filtered.

        Args:
            where: A predicate that retains a device when it evaluates to
                ``True``. Pass ``None`` to disable filtering.

        Returns:
            A list of :class:`~qilisdk.models.Device` objects.
        """
        with httpx.Client() as client:
            response = client.get(self._settings.qaas_api_url + "/devices", headers=self._get_authorized_headers())
            response.raise_for_status()

            devices = TypeAdapter(list[Device]).validate_python(response.json()["items"])

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
        with httpx.Client() as client:
            response = client.get(self._settings.qaas_api_url + "/jobs", headers=self._get_authorized_headers())
            response.raise_for_status()

            jobs = TypeAdapter(list[JobInfo]).validate_python(response.json()["items"])

        return [j for j in jobs if where(j)] if where else jobs

    def get_job_details(self, id: int) -> JobDetail:
        """Fetch the complete record of *id*.

        Args:
            id: Identifier of the job.

        Returns:
            A :class:`~qilisdk.models.JobDetail` instance containing payload,
            result, logs and error information.
        """
        with httpx.Client() as client:
            response = client.get(
                f"{self._settings.qaas_api_url}/jobs/{id}",
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

        return TypeAdapter(JobDetail).validate_python(data)

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
                return current

            if timeout is not None and (time.monotonic() - start_t) >= timeout:
                raise TimeoutError(
                    f"Timed out after {timeout}s while waiting for job {id} (last status {current.status.value!r})"
                )

            time.sleep(poll_interval)

    def _ensure_device_selected(self) -> int:
        if self._selected_device is None:
            raise ValueError("Device not selected.")
        return self._selected_device

    def submit(self, functional: Functional) -> int:
        try:
            handler = self._handlers[type(functional)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
            ) from exc

        return handler(functional)

    def _execute_sampling(self, sampling: Sampling) -> int:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.SAMPLING,
            sampling_payload=SamplingPayload(sampling=sampling),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                self._settings.qaas_api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id

    def _execute_time_evolution(self, time_evolution: TimeEvolution) -> int:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.TIME_EVOLUTION,
            time_evolution_payload=TimeEvolutionPayload(time_evolution=time_evolution),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                self._settings.qaas_api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id

    def submit_vqe(
        self, vqe: VQE, optimizer: Optimizer, nshots: int = 1000, store_intermediate_results: bool = False
    ) -> int:
        """Run a Variational Quantum Eigensolver on the selected device.

        Args:
            vqe: Problem definition containing Hamiltonian and ansatz.
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
            type=ExecuteType.VQE,
            vqe_payload=VQEPayload(
                vqe=vqe, optimizer=optimizer, nshots=nshots, store_intermediate_results=store_intermediate_results
            ),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                self._settings.qaas_api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id
