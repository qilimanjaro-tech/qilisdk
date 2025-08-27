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

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_logging_config_path() -> Path:
    return Path(__file__).with_name("logging_config.yaml").resolve()


class Precision(str, Enum):
    COMPLEX_64 = "COMPLEX_64"
    COMPLEX_128 = "COMPLEX_128"


class QiliSDKSettings(BaseSettings):
    """
    Environment-based configuration settings for QiliSDK.

    These settings are automatically loaded from environment variables
    prefixed with `QILISDK_`, or from a local `.env` file if present.
    """

    model_config = SettingsConfigDict(env_prefix="qilisdk_", env_file=".env", env_file_encoding="utf-8")

    arithmetic_precision: Precision = Field(
        default=Precision.COMPLEX_128, description="[env: QILISDK_ARITHMETIC_PRECISION]"
    )
    logging_config_path: Path = Field(
        default_factory=default_logging_config_path,
        description="YAML file used for logging configuration. [env: QILISDK_LOGGING_CONFIG_PATH]",
    )
    speqtrum_username: str | None = Field(
        default=None,
        description="SpeQtrum username used for authentication. [env: QILISDK_SPEQTRUM_USERNAME]",
    )
    speqtrum_apikey: str | None = Field(
        default=None,
        description="SpeQtrum API key associated with the user account. [env: QILISDK_SPEQTRUM_APIKEY]",
    )
    speqtrum_api_url: str = Field(
        default="https://qilimanjaro.ddns.net/public-api/api/v1",
        description="Base URL of the SpeQtrum API endpoint. [env: QILISDK_SPEQTRUM_API_URL]",
    )
    speqtrum_audience: str = Field(
        default="urn:qilimanjaro.tech:public-api:beren",
        description="Audience claim expected in the JWT used for authentication. [env: QILISDK_SPEQTRUM_AUDIENCE]",
    )


@lru_cache(maxsize=1)
def get_settings() -> QiliSDKSettings:
    """
    Returns a singleton instance of QiliSDKSettings.

    This function caches the parsed environment-based settings to avoid
    redundant re-parsing across the application lifecycle.

    Returns:
        QiliSDKSettings: The cached configuration object populated from environment variables.
    """
    return QiliSDKSettings()
