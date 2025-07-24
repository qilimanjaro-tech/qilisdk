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

import logging
from sys import stderr
from typing import TYPE_CHECKING, Any, Mapping, TypedDict

from loguru import logger

if TYPE_CHECKING:
    from types import FrameType


class InterceptHandler(logging.Handler):
    """
    Redirect stdlib 'logging' records to Loguru.
    Optionally ignore records whose logger name doesn't start with `name_prefix`.
    """

    def __init__(self, *, name_prefix: str | None = None) -> None:
        super().__init__()
        self.name_prefix = name_prefix

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        # Optional filter: keep only your library's records
        if self.name_prefix and not record.name.startswith(self.name_prefix):
            return

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        # Walk back to the frame where the logging call was made so Loguru shows the right caller
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoguruRecord(TypedDict, total=False):
    name: str
    extra: Mapping[str, Any]


logger.remove()


def only_qilisdk(record: LoguruRecord) -> bool:
    return record["name"].startswith("qilisdk") or record["extra"].get("component") is not None


logger.add(
    stderr,
    level="INFO",
    format="<fg #7f1cdb>QiliSDK</> | <green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <lvl>{level}</> | <lvl>{message}</>",
    filter=only_qilisdk,
    colorize=True,
    enqueue=False,
)
logger.add(
    "app.jsonl",
    level="DEBUG",
    serialize=True,
    rotation="10 MB")

# 4) Quiet noisy libs *before* intercepting
for name in ("httpx", "httpcore", "keyring"):
    logging.getLogger(name).setLevel(logging.WARNING)

# 5) Intercept stdlib -> Loguru
logging.root.handlers = [InterceptHandler()]
logging.root.setLevel(logging.NOTSET)
for n in list(logging.root.manager.loggerDict.keys()):
    logging.getLogger(n).handlers = []
    logging.getLogger(n).propagate = True
