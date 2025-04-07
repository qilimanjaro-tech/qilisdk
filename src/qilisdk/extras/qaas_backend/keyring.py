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

import keyring

SERVICE_NAME = "QaaSKeyring"


def store_credentials(username: str, auth_key: str) -> None:
    """Store username and auth_key in the OS keyring."""
    keyring.set_password(SERVICE_NAME, "username", username)
    keyring.set_password(SERVICE_NAME, "apikey", auth_key)


def delete_credentials() -> None:
    """Store username and auth_key in the OS keyring."""
    keyring.delete_password(SERVICE_NAME, "username")
    keyring.delete_password(SERVICE_NAME, "apikey")


def load_credentials() -> tuple[str, str] | None:
    """Load username and auth_key from keyring.

    Raises:
        Exception: _description_

    Returns:
        tuple[str, str] | None: Username and API key
    """
    try:
        username = keyring.get_password(SERVICE_NAME, "username")
        auth_key = keyring.get_password(SERVICE_NAME, "apikey")
        if username and auth_key:
            return username, auth_key
    except keyring.errors.KeyringError as e:
        raise Exception(f"[Keyring] Unable to load credentials: {e}")
    return None
