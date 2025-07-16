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
from os import environ
from typing import TYPE_CHECKING

import httpx
from pydantic import TypeAdapter, ValidationError

from qilisdk.analog.analog_backend import AnalogBackend
from qilisdk.digital.digital_backend import DigitalBackend

from .keyring import delete_credentials, load_credentials, store_credentials
from .models import (
    AnalogPayload,
    Device,
    DigitalPayload,
    ExecutePayload,
    ExecuteType,
    JobDetail,
    JobId,
    JobInfo,
    JobStatus,
    TimeEvolutionPayload,
    Token,
    VQEPayload,
)
from .qaas_settings import QaaSSettings

if TYPE_CHECKING:
    from qilisdk.analog import Hamiltonian, QuantumObject, Schedule, TimeEvolution
    from qilisdk.analog.hamiltonian import PauliOperator
    from qilisdk.common.optimizer import Optimizer
    from qilisdk.digital import VQE, Circuit

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
)


class QaaSBackend(DigitalBackend, AnalogBackend):
    """
    Manages communication with a hypothetical QaaS service via synchronous HTTP calls.

    Credentials to log in can come from:
      a) method parameters,
      b) environment (via Pydantic),
      c) keyring (fallback).
    """

    _api_url: str = environ.get("QILISDK_QAAS_API_URL", "https://qilimanjaro.ddns.net/public-api/api/v1")
    _audience: str = environ.get("QILISDK_QAAS_AUDIENCE", "urn:qilimanjaro.tech:public-api:beren")

    def __init__(self) -> None:
        """
        Normally, you won't call __init__() directly.
        Instead, use QaaSBackend.login(...) to create a logged-in instance.
        """  # noqa: DOC501
        credentials = load_credentials()
        if credentials is None:
            raise RuntimeError(
                "No valid QaaS credentials found in keyring."
                "Please call QaaSBackend.login(username, apikey) or ensure environment variables are set."
            )
        self._username, self._token = credentials
        self._selected_device: int | None = None

    def _get_headers(self) -> dict:  # noqa: PLR6301
        from qilisdk import __version__  # noqa: PLC0415

        return {"User-Agent": f"qilisdk/{__version__}"}

    def _get_authorized_headers(self) -> dict:
        return {**self._get_headers(), "Authorization": f"Bearer {self._token.access_token}"}

    # TODO (vyron): Change this to `code: str` when implemented server-side.
    @property
    def selected_device(self) -> int | None:
        return self._selected_device

    def set_device(self, id: int) -> None:
        self._selected_device = id

    @classmethod
    def login(
        cls,
        username: str | None = None,
        apikey: str | None = None,
    ) -> bool:
        # Use provided parameters or fall back to environment variables via Settings()
        if not username or not apikey:
            try:
                # Load environment variables into the settings object.
                settings = QaaSSettings()  # type: ignore[call-arg]
                username = username or settings.username
                apikey = apikey or settings.apikey
            except ValidationError:
                # Environment credentials could not be validated.
                # Optionally, log error details here.
                return False

        if not username or not apikey:
            # Insufficient credentials provided.
            return False

        # Send login request to QaaS
        try:
            assertion = {
                "username": username,
                "api_key": apikey,
                "audience": QaaSBackend._audience,
                "iat": int(datetime.now(timezone.utc).timestamp()),
            }
            encoded_assertion = urlsafe_b64encode(json.dumps(assertion, indent=2).encode("utf-8")).decode("utf-8")
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    QaaSBackend._api_url + "/authorisation-tokens",
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
            # Log error message
            return False

        store_credentials(username=username, token=token)
        return True

    @classmethod
    def logout(cls) -> None:
        delete_credentials()

    def list_devices(self) -> list[Device]:
        with httpx.Client() as client:
            response = client.get(QaaSBackend._api_url + "/devices", headers=self._get_authorized_headers())
            response.raise_for_status()

            devices_list_adapter = TypeAdapter(list[Device])
            devices = devices_list_adapter.validate_python(response.json()["items"])

            return devices

    def list_jobs(self) -> list[JobInfo]:
        with httpx.Client() as client:
            response = client.get(QaaSBackend._api_url + "/jobs", headers=self._get_authorized_headers())
            response.raise_for_status()

            jobs_list_adapter = TypeAdapter(list[JobInfo])
            jobs = jobs_list_adapter.validate_python(response.json()["items"])

            return jobs

    def get_job_details(self, id: int) -> JobDetail:
        with httpx.Client() as client:
            response = client.get(
                f"{QaaSBackend._api_url}/jobs/{id}",
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

        data["payload"] = json.loads(data["payload"])

        decoded_result: bytes = base64.b64decode(data.get("result"))
        text_result = decoded_result.decode("utf-8")
        data["result"] = json.loads(text_result)

        decoded_error: bytes = base64.b64decode(data.get("error"))
        data["error"] = decoded_error.decode("utf-8")

        decoded_logs: bytes = base64.b64decode(data.get("logs"))
        data["logs"] = decoded_logs.decode("utf-8")

        return TypeAdapter(JobDetail).validate_python(data)

    def wait_for_job(
        self,
        id: int,
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> JobDetail:
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
                    f"Timed out after {timeout}s while waiting for job {id} "
                    f"(last status {current.status.value!r})"
                )

            time.sleep(poll_interval)

    def _ensure_device_selected(self) -> int:
        if self._selected_device is None:
            raise ValueError("Device not selected.")
        return self._selected_device

    def execute(self, circuit: Circuit, nshots: int = 1000) -> int:  # type: ignore[override]
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.DIGITAL,
            digital_payload=DigitalPayload(circuit=circuit, nshots=nshots),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                QaaSBackend._api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id

    def evolve(  # type: ignore[override]
        self,
        schedule: Schedule,
        initial_state: QuantumObject,
        observables: list[PauliOperator | Hamiltonian],
        store_intermediate_results: bool = False,
    ) -> int:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.ANALOG,
            analog_payload=AnalogPayload(
                schedule=schedule,
                initial_state=initial_state,
                observables=observables,
                store_intermediate_results=store_intermediate_results,
            ),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                QaaSBackend._api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id

    def run_vqe(
        self, vqe: VQE, optimizer: Optimizer, nshots: int = 1000, store_intermediate_results: bool = False
    ) -> int:
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
                QaaSBackend._api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id

    def run_time_evolution(self, time_evolution: TimeEvolution, store_intermediate_results: bool = False) -> int:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecuteType.TIME_EVOLUTION,
            time_evolution_payload=TimeEvolutionPayload(
                time_evolution=time_evolution, store_intermediate_results=store_intermediate_results
            ),
        )
        json = {"device_id": device, "payload": payload.model_dump_json(), "meta": {}}
        with httpx.Client() as client:
            response = client.post(
                QaaSBackend._api_url + "/execute",
                headers=self._get_authorized_headers(),
                json=json,
            )
            response.raise_for_status()
            job = JobId(**response.json())
            return job.id
