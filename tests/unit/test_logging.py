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

import logging
import sys
from types import SimpleNamespace

import pytest
from loguru_caplog import loguru_caplog as caplog  # noqa: F401

from qilisdk import _logging
from qilisdk._logging import InterceptHandler
from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import LT, BinaryVariable, Domain, OneHot, Variable


def test_log_output(caplog):  # noqa: F811
    N = 2
    b = [BinaryVariable(f"b({i})") for i in range(N)]
    x = Variable("x", Domain.POSITIVE_INTEGER, bounds=(0, 10), encoding=OneHot)

    m = Model("test")

    m.set_objective(x + 1, sense=ObjectiveSense.MAXIMIZE)

    m.add_constraint("con2", LT(b[1], 2))

    m.to_qubo()

    test_message = " because it is always feasible."
    assert test_message in caplog.text


class FakeSink:
    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self):
        return dict(self._data)


def test_configure_logging_loads_resolved_path(monkeypatch):
    rel_path = "./nonexistent.yaml"

    monkeypatch.setattr(
        _logging,
        "get_settings",
        lambda: SimpleNamespace(logging_config_path=rel_path),
    )

    with pytest.raises(FileNotFoundError, match=r"nonexistent.yaml"):
        _logging.configure_logging()


def test_configure_logging_adds_sinks(monkeypatch):
    sink_out = FakeSink(
        sink="stdout",
        level="DEBUG",
        format=None,
    )
    sink_err = FakeSink(
        sink="stderr",
        level="INFO",
        format=None,
    )

    settings = SimpleNamespace(
        sinks=[sink_out, sink_err],
        intercept_libraries=[],
    )

    monkeypatch.setattr(
        _logging.LoggingSettings,
        "load",
        staticmethod(lambda _: settings),
    )

    added = []

    def fake_add(target, **kwargs):
        added.append({"target": target, "kwargs": kwargs})

    monkeypatch.setattr(_logging.logger, "add", fake_add)

    _logging.configure_logging()

    assert len(added) == 2
    assert added[0]["target"] == sys.stdout
    assert added[0]["kwargs"]["level"] == "DEBUG"
    assert added[1]["target"] == sys.stderr
    assert added[1]["kwargs"]["level"] == "INFO"


def test_emit_falls_back_to_levelno(monkeypatch):
    handler = InterceptHandler()

    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=20,
        msg="warning",
        args=(),
        exc_info=None,
    )

    def raise_value_error(name):
        raise ValueError

    monkeypatch.setattr(
        "qilisdk._logging.logger.level",
        raise_value_error,
    )

    captured = {}

    monkeypatch.setattr(
        "qilisdk._logging.logger.opt",
        lambda **kwargs: SimpleNamespace(log=lambda level, msg: captured.update({"level": level, "message": msg})),
    )

    handler.emit(record)

    assert captured["level"] == logging.WARNING


class FakeFrame:
    def __init__(self, filename, back=None):
        self.f_code = type("Code", (), {"co_filename": filename})
        self.f_back = back


def test_emit_executes_while_loop(monkeypatch):
    # this test ensures that the while loop in emit() is executed at least once

    handler = InterceptHandler()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="trigger while",
        args=(),
        exc_info=None,
    )

    # Fake frame chain:
    # first frame -> logging file
    # second frame -> user file
    fake_user_frame = FakeFrame("user.py", None)
    fake_logging_frame = FakeFrame(logging.__file__, fake_user_frame)

    monkeypatch.setattr(
        logging,
        "currentframe",
        lambda: fake_logging_frame,
    )

    monkeypatch.setattr(
        "qilisdk._logging.logger.level",
        lambda name: type("L", (), {"name": name}),
    )

    monkeypatch.setattr(
        "qilisdk._logging.logger.opt",
        lambda **kwargs: type("Opt", (), {"log": lambda *_: None})(),
    )

    handler.emit(record)
