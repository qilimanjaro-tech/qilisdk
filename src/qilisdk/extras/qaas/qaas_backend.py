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
from typing import TYPE_CHECKING

import httpx
from pydantic import ValidationError

from qilisdk.analog.analog_backend import AnalogBackend
from qilisdk.digital.digital_backend import DigitalBackend

from .keyring import delete_credentials, load_credentials, store_credentials
from .models import Token
from .qaas_settings import QaaSSettings

if TYPE_CHECKING:
    from qilisdk.analog.analog_result import AnalogResult
    from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
    from qilisdk.analog.quantum_objects import QuantumObject
    from qilisdk.analog.schedule import Schedule
    from qilisdk.common.algorithm import Algorithm
    from qilisdk.digital.circuit import Circuit

    from .qaas_digital_result import QaaSDigitalResult

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

    _api_url: str = "https://qilimanjaroqaas.ddns.net:8080/api/v1"
    _authorization_request_url: str = "https://qilimanjaroqaas.ddns.net:8080/api/v1/authorisation-tokens"
    _authorization_refresh_url: str = "https://qilimanjaroqaas.ddns.net:8080/api/v1/authorisation-tokens/refresh"

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

    @classmethod
    def login(
        cls,
        username: str | None = None,
        apikey: str | None = None,
    ) -> "QaaSBackend":
        # Use provided parameters or fall back to environment variables via Settings()
        if not username or not apikey:
            try:
                settings = QaaSSettings()  # Load environment variables into the settings object.
                username = username or settings.username
                apikey = apikey or settings.apikey
            except ValidationError:
                # Environment credentials could not be validated.
                # Optionally, log error details here.
                return False

        if not username or not apikey:
            # Insufficient credentials provided.
            return False

        # 4) Send login request to QaaS
        # Use a short-lived client just for this login call:
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
                token = Token(**response.json())
        except httpx.RequestError:
            # Log error message
            return False

        store_credentials(username=username, token=token)
        return True

    @classmethod
    def logout() -> None:
        delete_credentials()

    def execute(self, circuit: Circuit, nshots: int = 1000) -> QaaSDigitalResult:
        raise NotImplementedError

    def evolve(
        self,
        schedule: Schedule,
        initial_state: QuantumObject,
        observables: list[PauliOperator | Hamiltonian],
        store_intermediate_results: bool = False,
    ) -> AnalogResult:
        raise NotImplementedError

    # Algorithms may need to call execute or evolve multiple times. This is why we should also allow users
    # to run them as a "block" within our QaaS.

    # 1st Proposal. If algorithms will use the same job execution flow, we can run them directly.

    def run(self, algorithm: Algorithm) -> None:
        raise NotImplementedError

    # 2nd Proposal (if needed). Execute algorithms through a dedicated Session.

    @classmethod
    def session(
        cls,
        username: str | None = None,
        apikey: str | None = None,
        store_in_keyring: bool = True,
    ) -> QaaSSession:
        backend = cls()
        return QaaSSession(backend)

    def start_session(self) -> str:
        # Do something to initiate a session and get back the session_id
        ...

    def run_in_session(self, session_id: str, algorithm: str) -> dict:
        # Run an algorithm within the session
        ...

    def close_session(self, session_id: str) -> None:
        # Do something to close the session
        ...


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

    def run(self, algorithm: Algorithm) -> None:
        if self._session_id is None:
            raise RuntimeError("Session not started.")
        return self._backend.run_in_session(self._session_id, algorithm)
