# Copyright 2023 Qilimanjaro Quantum Tech
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
import os

from qilisdk import __version__

# Logging level from 0 (all) to 4 (errors) (see https://docs.python.org/3/library/logging.html#logging-levels)
LIBRARY_LOG_LEVEL = int(os.environ.get("LIBRARY_LOG_LEVEL", 20))  # noqa: PLW1508


class CustomHandler(logging.StreamHandler):
    """Custom handler for logging algorithm."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: PLR6301
        """Format the record with specific format.
        Returns:
            str: the formatted log.
        """

        fmt = f"[qilisdk] [{__version__}|%(levelname)s|%(asctime)s]: %(message)s"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


logger = logging.getLogger(__name__)
logger.setLevel(LIBRARY_LOG_LEVEL)
logger.addHandler(CustomHandler())
