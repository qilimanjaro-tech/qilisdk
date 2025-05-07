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

import json
import logging
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from os import environ
from typing import TYPE_CHECKING, cast

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
    ExecutePayloadType,
    ExecuteResponse,
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

    from .qaas_analog_result import QaaSAnalogResult
    from .qaas_digital_result import QaaSDigitalResult
    from .qaas_time_evolution_result import QaaSTimeEvolutionResult
    from .qaas_vqe_result import QaaSVQEResult

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

    _api_url: str = environ.get("PUBLIC_API_URL", "https://qilimanjaroqaas.ddns.net:8080/api/v1")

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
        self._selected_device: Device | None = None

    @property
    def selected_device(self) -> Device | None:
        return self._selected_device

    def set_device(self, device: Device) -> None:
        self._selected_device = device

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
                "user_id": None,
                "audience": QaaSBackend._api_url,
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
                    headers={"X-Client-Version": "0.23.2"},
                )
                response.raise_for_status()
                # Suppose QaaS returns {"token": "..."} in JSON
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
        with httpx.Client(timeout=20.0) as client:
            response = client.get(
                QaaSBackend._api_url + "/devices",
                headers={"X-Client-Version": "0.23.2", "Authorization": f"Bearer {self._token.access_token}"},
            )
            response.raise_for_status()

            devices_list_adapter = TypeAdapter(list[Device])
            devices = devices_list_adapter.validate_python(response.json()["items"])

            # Previous two lines are the same as doing:
            # response_json = response.json()
            # devices = [Device(**item) for item in response_json["items"]]

            return devices

    def _ensure_device_selected(self) -> Device:
        if self._selected_device is None:
            raise ValueError("Device not selected.")
        return self._selected_device

    def execute(self, circuit: Circuit, nshots: int = 1000) -> QaaSDigitalResult:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecutePayloadType.DIGITAL,
            device_id=device.id,
            digital_payload=DigitalPayload(circuit=circuit, nshots=nshots),
        )
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                QaaSBackend._api_url + "/execute",
                headers={"X-Client-Version": "0.23.2", "Authorization": f"Bearer {self._token.access_token}"},
                json=payload.model_dump_json(),
            )
            response.raise_for_status()
            execute_response = ExecuteResponse(**response.json())
            return cast("QaaSDigitalResult", execute_response.digital_result)

    def evolve(
        self,
        schedule: Schedule,
        initial_state: QuantumObject,
        observables: list[PauliOperator | Hamiltonian],
        store_intermediate_results: bool = False,
    ) -> QaaSAnalogResult:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecutePayloadType.ANALOG,
            device_id=device.id,
            analog_payload=AnalogPayload(
                schedule=schedule,
                initial_state=initial_state,
                observables=observables,
                store_intermediate_results=store_intermediate_results,
            ),
        )
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                QaaSBackend._api_url + "/execute",
                headers={"X-Client-Version": "0.23.2", "Authorization": f"Bearer {self._token.access_token}"},
                json=payload.model_dump_json(),
            )
            response.raise_for_status()
            execute_response = ExecuteResponse(**response.json())
            return cast("QaaSAnalogResult", execute_response.analog_result)

    def run_vqe(
        self, vqe: VQE, optimizer: Optimizer, nshots: int = 1000, store_intermediate_results: bool = False
    ) -> QaaSVQEResult:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecutePayloadType.VQE,
            device_id=device.id,
            vqe_payload=VQEPayload(
                vqe=vqe, optimizer=optimizer, nshots=nshots, store_intermediate_results=store_intermediate_results
            ),
        )
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                QaaSBackend._api_url + "/execute",
                headers={"X-Client-Version": "0.23.2", "Authorization": f"Bearer {self._token.access_token}"},
                json=payload.model_dump_json(),
            )
            response.raise_for_status()
            execute_response = ExecuteResponse(**response.json())
            return cast("QaaSVQEResult", execute_response.vqe_result)

    def run_time_evolution(
        self, time_evolution: TimeEvolution, store_intermediate_results: bool = False
    ) -> QaaSTimeEvolutionResult:
        device = self._ensure_device_selected()
        payload = ExecutePayload(
            type=ExecutePayloadType.TIME_EVOLUTION,
            device_id=device.id,
            time_evolution_payload=TimeEvolutionPayload(
                time_evolution=time_evolution, store_intermediate_results=store_intermediate_results
            ),
        )
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                QaaSBackend._api_url + "/execute",
                headers={"X-Client-Version": "0.23.2", "Authorization": f"Bearer {self._token.access_token}"},
                json=payload.model_dump_json(),
            )
            response.raise_for_status()
            execute_response = ExecuteResponse(**response.json())
            return cast("QaaSTimeEvolutionResult", execute_response.time_evolution_result)
