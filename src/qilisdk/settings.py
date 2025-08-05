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

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_logging_config_path() -> Path:
    return Path(__file__).with_name("logging_config.yaml").resolve()


class QiliSDKSettings(BaseSettings):
    """
    Environment-based configuration settings for QiliSDK.

    These settings are automatically loaded from environment variables
    prefixed with `QILISDK_`, or from a local `.env` file if present.
    """

    model_config = SettingsConfigDict(env_prefix="qilisdk_", env_file=".env", env_file_encoding="utf-8")

    logging_config_path: Path = Field(
        default_factory=default_logging_config_path,
        description="YAML file used for logging configuration. [env: QILISDK_LOGGING_CONFIG_PATH]",
    )
    qaas_username: str | None = Field(
        default=None,
        description="QaaS username used for authentication. [env: QILISDK_QAAS_USERNAME]",
    )
    qaas_apikey: str | None = Field(
        default=None,
        description="QaaS API key associated with the user account. [env: QILISDK_QAAS_APIKEY]",
    )
    qaas_api_url: str = Field(
        default="https://qilimanjaro.ddns.net/public-api/api/v1",
        description="Base URL of the QaaS API endpoint. [env: QILISDK_QAAS_API_URL]",
    )
    qaas_audience: str = Field(
        default="urn:qilimanjaro.tech:public-api:beren",
        description="Audience claim expected in the JWT used for authentication. [env: QILISDK_QAAS_AUDIENCE]",
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
