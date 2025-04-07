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
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError

from qilisdk.analog.analog_backend import AnalogBackend
from qilisdk.analog.analog_result import AnalogResult
from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.analog.schedule import Schedule
from qilisdk.common.algorithm import Algorithm
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.digital_backend import DigitalBackend
from qilisdk.digital.digital_result import DigitalResult

from .keyring import load_credentials_from_keyring, store_credentials_in_keyring
from .models import Token
from .settings import Settings

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
)


def base64url_encode(payload: dict | bytes | str) -> str:
    """Encode a given payload to base64 string

    Args:
        payload ( dict | bytes | str): data to be encoded

    Returns:
        str: base64 encoded data
    """
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    if not isinstance(payload, bytes):
        payload = payload.encode("utf-8")
    return urlsafe_b64encode(payload).decode("utf-8")


def base64_decode(encoded_data: str) -> str:
    """Decodes a base64 encoded string

    Args:
        encoded_data (str): a base64 encoded string

    Returns:
        Any: The data decoded
    """
    return urlsafe_b64decode(encoded_data).decode("utf-8")


class QaaSBackend(DigitalBackend, AnalogBackend):
    """
    Manages communication with a hypothetical QaaS service via synchronous HTTP calls.

    Credentials to log in can come from:
      a) method parameters,
      b) environment (via Pydantic),
      c) keyring (fallback).
    """

    _api_url: str = "https://qilimanjaroqaas.ddns.net:8080/api/v1"
    _authorization_request_url: str = "https://qilimanjaroqaas.ddns.net:8080/api/v1/authorisation-tokens"
    _authorization_refresh_url: str = "https://qilimanjaroqaas.ddns.net:8080/api/v1/authorisation-tokens/refresh"

    def __init__(self, username: str, token: Token) -> None:
        """
        Normally, you won't call __init__() directly.
        Instead, use QaaSBackend.login(...) to create a logged-in instance.
        """
        self._username: str = username
        self._token: Token = token

    @classmethod
    def login(
        cls,
        username: str | None = None,
        apikey: str | None = None,
        store_in_keyring: bool = True,
    ) -> "QaaSBackend":
        # 1) Check if credentials were explicitly provided
        if username and apikey:
            final_username = username
            final_apikey = apikey
            if store_in_keyring:
                store_credentials_in_keyring(username, apikey)
        else:
            # 2) Try environment or .env
            try:
                settings = Settings()
                final_username = settings.username
                final_apikey = settings.apikey
            except ValidationError:
                # 3) Fallback: keyring
                creds = load_credentials_from_keyring()
                if not creds:
                    raise RuntimeError("No valid QaaS credentials found (args, env, or keyring).")
                final_username, final_apikey = creds

        # 4) Send login request to QaaS
        # Use a short-lived client just for this login call:
        try:
            with httpx.Client(timeout=10.0) as client:
                assertion = {
                    "username": final_username,
                    "api_key": final_apikey,
                    "user_id": None,
                    "audience": QaaSBackend._api_url,
                    "iat": int(datetime.now(timezone.utc).timestamp()),
                }
                encoded_assertion = base64url_encode(json.dumps(assertion, indent=2))
                response = client.post(
                    QaaSBackend._authorization_request_url,
                    json={
                        "grantType": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": encoded_assertion,
                        "scope": "user profile",
                    },
                    headers={"X-Client-Version": "0.23.2"},
                )
                response.raise_for_status()
                # Suppose QaaS returns {"token": "..."} in JSON
                token_data = Token(**response.json())
        except httpx.RequestError as exc:
            raise RuntimeError(f"Login request failed: {exc}") from exc

        # Create a new QaaSBackend instance with the validated credentials & token
        # print(f"Logged in successfully as {final_username}")
        return cls(username=final_username, token=token_data)

    def evolve(
        self,
        schedule: Schedule,
        initial_state: QuantumObject,
        observables: list[PauliOperator | Hamiltonian],
        store_intermediate_results: bool = False,
    ) -> AnalogResult:
        raise NotImplementedError

    def execute(self, circuit: Circuit, nshots: int = 1000) -> DigitalResult:
        raise NotImplementedError

    @classmethod
    def session(
        cls,
        username: str | None = None,
        apikey: str | None = None,
        store_in_keyring: bool = True,
    ) -> QaaSSession:
        backend = cls.login(username=username, apikey=apikey, store_in_keyring=store_in_keyring)
        return QaaSSession(backend)

    def start_session(self) -> str:
        try:
            with httpx.Client(timeout=10.0) as client:
                url = f"{self._api_url}/session/start"
                response = client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._token.token}"}
                )
                response.raise_for_status()
                data = response.json()
                session_id = data["session_id"]
                return session_id
        except httpx.RequestError:
            raise

    def execute_in_session(self, session_id: str, algorithm: str) -> dict:
        try:
            with httpx.Client(timeout=10.0) as client:
                url = f"{self._api_url}/session/execute"
                response = client.post(
                    url,
                    json={"session_id": session_id, "algorithm": algorithm},
                    headers={"Authorization": f"Bearer {self._token.token}"}
                )
                response.raise_for_status()
                return response.json()
        except httpx.RequestError:
            raise

    def close_session(self, session_id: str) -> None:
        try:
            with httpx.Client(timeout=10.0) as client:
                url = f"{self._api_url}/session/close"
                response = client.post(
                    url,
                    json={"session_id": session_id},
                    headers={"Authorization": f"Bearer {self._token.token}"}
                )
                response.raise_for_status()
        except httpx.RequestError:
            raise


    # def execute(self, circuit_data: dict[str, Any]) -> dict[str, Any]:
    #     """
    #     Sends a circuit definition to the QaaS API for execution.
    #     Must have a valid token from login.
    #     """
    #     if not self._token:
    #         raise RuntimeError("Not logged in. Call QaaSBackend.login() first.")

    #     try:
    #         response = self._client.post(
    #             f"{self.base_url}/execute",
    #             json={"circuit": circuit_data},
    #             headers={"Authorization": f"Bearer {self._token}"},
    #         )
    #         response.raise_for_status()
    #         return response.json()
    #     except httpx.RequestError as exc:
    #         print(f"Execution error: {exc}")
    #         return {}


class QaaSSession:
    """
    A separate session class that wraps a QaaSBackend instance and provides a context manager
    interface.
    """
    def __init__(self, backend: QaaSBackend) -> None:
        self._backend = backend
        self._session_id: str | None = None

    def __enter__(self) -> "QaaSSession":
        self._session_id = self._backend.start_session()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        if self._session_id is not None:
            self._backend.close_session(self._session_id)
            self._session_id = None

    def execute(self, algorithm: Algorithm):
        if self._session_id is None:
            raise RuntimeError("Session not started.")
        return self._backend.execute_in_session(self._session_id, algorithm)
