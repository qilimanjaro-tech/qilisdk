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
from pydantic import ValidationError

from .models import Token

KEYRING_IDENTIFIER = "QaaSKeyring"


def store_credentials(username: str, token: Token) -> None:
    """
    Store the username and token in the keyring.
    """
    keyring.set_password(KEYRING_IDENTIFIER, "username", username)
    keyring.set_password(KEYRING_IDENTIFIER, "token", token.model_dump_json())


def delete_credentials() -> None:
    """
    Delete username and token from the keyring.
    """
    keyring.delete_password(KEYRING_IDENTIFIER, "username")
    keyring.delete_password(KEYRING_IDENTIFIER, "token")


def load_credentials() -> tuple[str, Token] | None:
    """
    Attempt to load the stored username and token from the keyring.

    Returns:
        A tuple (username, Token) if both exist and can be parsed; otherwise, None.
    """
    username = keyring.get_password(KEYRING_IDENTIFIER, "username")
    token_json = keyring.get_password(KEYRING_IDENTIFIER, "token")
    if username is None or token_json is None:
        return None
    try:
        token = Token.model_validate_json(token_json)
        return username, token
    except ValidationError:
        return None
