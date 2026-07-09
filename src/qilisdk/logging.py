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

"""config.py"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from ruamel.yaml import YAML

from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from types import FrameType

# Different logging levels:
# - TRACE: Very detailed information, probably should be printed to a file and parsed through
# - DEBUG: Detailed information meant for developers
# - INFO: Detailed information meant for users
# - SUCCESS: Indicate successful operations, such as execution
# - WARNING: Indicate potential issues
# - ERROR: Indicate errors
# - CRITICAL: Indicate critical errors

# Consistent-width icons for every level
LEVEL_ICONS: dict[str, str] = {
    "TRACE": "🔬",
    "DEBUG": "🐞",
    "INFO": "💡",
    "SUCCESS": "✅",
    "WARNING": "🚧",
    "ERROR": "❌",
    "CRITICAL": "💀",
}


class SinkConfig(BaseModel):
    """
    Configuration for a single Loguru sink.
    """

    sink: str | Path
    level: str = "INFO"
    format: str | None = None
    filter: str | dict[str, str] | None = None
    colorize: bool = False
    enqueue: bool = False
    rotation: str | None = None
    serialize: bool = False


class InterceptLibraryConfig(BaseModel):
    """
    Configuration for intercepting a stdlib logging library.
    """

    name: str
    level: str = "ERROR"


class LoggingSettings(BaseSettings):
    """
    Pydantic settings for Loguru configuration loaded from YAML or JSON.
    """

    sinks: list[SinkConfig] = []
    intercept_libraries: list[InterceptLibraryConfig] = []

    @classmethod
    def load(cls, path: str | Path) -> LoggingSettings:
        path = Path(path)
        yaml = YAML(typ="safe")
        data = yaml.load(path)
        return cls(**data)


class InterceptHandler(logging.Handler):
    """
    Redirect stdlib 'logging' records to Loguru, optionally filtering by name_prefix.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: PLR6301
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str | None = None) -> None:
    """
    Load settings (path overridden by environment variable) and configure Loguru + stdlib logging intercept.

    Args:
        level (str | None): If provided, override the level of every configured sink with this
            value (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"). All other sink options
            (format, filter, colorize, ...) are preserved. If None, each sink uses its configured level.
    """
    # Determine config path
    config_path = Path(get_settings().logging_config_path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    settings = LoggingSettings.load(config_path)

    # 0) Normalize level icons to consistent-width emoji so message columns align.
    for level_name, icon in LEVEL_ICONS.items():
        logger.level(level_name, icon=icon)

    # 1) Remove all pre-configured Loguru handlers
    logger.remove()

    # 2) Add configured sinks
    for sink_conf in settings.sinks:
        params = sink_conf.model_dump()
        sink_target = params.pop("sink")

        # Resolve stderr/stdout
        if isinstance(sink_target, str) and sink_target.lower() == "stderr":
            sink_target = sys.stderr
        elif isinstance(sink_target, str) and sink_target.lower() == "stdout":
            sink_target = sys.stdout

        # Apply a global level override, keeping every other sink option intact.
        if level is not None:
            params["level"] = level

        clean_params = {k: v for k, v in params.items() if v is not None}

        logger.add(sink_target, **clean_params)

    # 3) Quiet down noisy libraries before intercepting
    for intercept_library in settings.intercept_libraries:
        logging.getLogger(intercept_library.name).setLevel(intercept_library.level)

    # 4) Intercept stdlib -> Loguru
    handler = InterceptHandler()
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.NOTSET)

    for logger_name in list(logging.root.manager.loggerDict.keys()):
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).propagate = True


def set_logging_level(level: str) -> None:
    """
    Change the logging level while preserving the format, filter and colorization
    defined in the logging configuration file.

    Args:
        level (str): The logging level to set (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    configure_logging(level=level)
