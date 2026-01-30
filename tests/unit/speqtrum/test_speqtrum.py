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

"""
Unit-tests for the SpeQtrum synchronous client.

All tests are *function* based (no classes) so they integrate with plain
pytest discovery.
"""

from __future__ import annotations

import base64
import collections
import json
import types
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from qilisdk.experiments.experiment_functional import RabiExperiment, T1Experiment, T2Experiment, TwoTonesExperiment
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.variational_program import VariationalProgram

pytest.importorskip("httpx", reason="SpeQtrum tests require the 'speqtrum' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "keyring",
    reason="SpeQtrum tests require the 'speqtrum' optional dependency",
    exc_type=ImportError,
)

import httpx

import qilisdk.speqtrum.speqtrum as speqtrum
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.speqtrum.speqtrum_models import ExecuteResult


# ────────────────────────────────────────────────────────────────────────────────
# helper objects used by multiple tests
# ────────────────────────────────────────────────────────────────────────────────
class DummyResponse:
    """Mimics an httpx.Response for the calls we use."""

    def __init__(self, payload: dict | None = None):
        self._payload = payload or {"id": 42}

    # SpeQtrum never checks the status code directly - it just calls this method
    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyClient:
    """
    Lightweight stand-in for ``httpx.Client`` that can be parametrised
    per-test for ``get`` / ``post`` return values.
    """

    def __init__(self, *, get_payload: dict | None = None, post_payload: dict | None = None, **_):
        self._get_payload = get_payload or {}
        self._post_payload = post_payload or {}

    # make it usable as a context-manager
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    # the real client's signature is much richer, but SpeQtrum only uses a subset
    def get(self, *_, **__):
        return DummyResponse(self._get_payload)

    def post(self, *_, **__):
        return DummyResponse(self._post_payload)


# --------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------


def test_init_raises_when_no_credentials(monkeypatch):
    """Constructing SpeQtrum without cached credentials must fail."""
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: None)

    with pytest.raises(RuntimeError):
        speqtrum.SpeQtrum()


def test_init_succeeds_with_stub_credentials(monkeypatch):
    """When credentials exist, the client initialises and stores them."""

    tok = SimpleNamespace(access_token="xyz")

    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("alice", tok))
    q = speqtrum.SpeQtrum()

    assert q.username == "alice"
    assert q.token is tok


class FakeSampling(Sampling):
    def __init__(self): ...


class FakeTimeEvolution(TimeEvolution):
    def __init__(self): ...


class FakeRabiExperiment(RabiExperiment):
    def __init__(self): ...


class FakeT1Experiment(T1Experiment):
    def __init__(self): ...


class FakeT2Experiment(T2Experiment):
    def __init__(self): ...


class FakeTwoTonesExperiment(TwoTonesExperiment):
    def __init__(self): ...


class FakeVariationalProgram(VariationalProgram):
    def __init__(self, functional):
        self._functional = functional


def test_submit_dispatches_to_sampling_handler(monkeypatch):
    monkeypatch.setattr(speqtrum, "Sampling", FakeSampling)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(post_payload={"id": 99}),
        raising=True,
    )
    q = speqtrum.SpeQtrum()
    handle = q.submit(FakeSampling(), device="some_device", job_name="my_job")
    assert handle.id == 99


def test_submit_dispatches_to_variational_program_handler(monkeypatch):
    monkeypatch.setattr(speqtrum, "Sampling", FakeSampling)
    monkeypatch.setattr(speqtrum, "TimeEvolution", FakeTimeEvolution)
    monkeypatch.setattr(speqtrum, "RabiExperiment", FakeRabiExperiment)
    monkeypatch.setattr(speqtrum, "VariationalProgram", FakeVariationalProgram)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(post_payload={"id": 88}),
        raising=True,
    )
    q = speqtrum.SpeQtrum()
    handle = q.submit(FakeVariationalProgram(FakeSampling()), device="some_device", job_name="my_vp_job")
    q.submit(FakeVariationalProgram(FakeTimeEvolution()), device="some_device", job_name="my_vp_job")
    q.submit(FakeVariationalProgram(FakeRabiExperiment()), device="some_device", job_name="my_vp_job")
    assert handle.id == 88


def test_submit_dispatches_to_time_evolution_handler(monkeypatch):
    monkeypatch.setattr(speqtrum, "TimeEvolution", FakeTimeEvolution)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(post_payload={"id": 77}),
        raising=True,
    )
    q = speqtrum.SpeQtrum()
    handle = q.submit(FakeTimeEvolution(), device="some_device", job_name="te_job")
    assert handle.id == 77


def test_submit_dispatches_to_rabi_handler(monkeypatch):
    monkeypatch.setattr(speqtrum, "RabiExperiment", FakeRabiExperiment)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(post_payload={"id": 66}),
        raising=True,
    )
    q = speqtrum.SpeQtrum()
    handle = q.submit(FakeRabiExperiment(), device="some_device", job_name="rabi_job")
    assert handle.id == 66


def test_submit_dispatches_to_t1_experiment_handler(monkeypatch):
    monkeypatch.setattr(speqtrum, "T1Experiment", FakeT1Experiment)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(post_payload={"id": 55}),
        raising=True,
    )
    q = speqtrum.SpeQtrum()
    handle = q.submit(FakeT1Experiment(), device="some_device", job_name="t1_job")
    assert handle.id == 55


def test_submit_dispatches_to_t2_experiment_handler(monkeypatch):
    monkeypatch.setattr(speqtrum, "T2Experiment", FakeT2Experiment)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(post_payload={"id": 44}),
        raising=True,
    )
    q = speqtrum.SpeQtrum()
    handle = q.submit(FakeT2Experiment(), device="some_device", job_name="t2_job")
    assert handle.id == 44


# for two tones


def test_submit_dispatches_to_two_tone_handler(monkeypatch):
    monkeypatch.setattr(speqtrum, "TwoTonesExperiment", FakeTwoTonesExperiment)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(post_payload={"id": 33}),
        raising=True,
    )
    q = speqtrum.SpeQtrum()
    handle = q.submit(FakeTwoTonesExperiment(), device="some_device", job_name="two_tone_job")
    assert handle.id == 33


def test_submit_unknown_functional_raises(monkeypatch):
    """Passing an unsupported functional type should raise `NotImplementedError`."""
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    client = speqtrum.SpeQtrum()

    class SomethingElse:
        pass

    with pytest.raises(NotImplementedError):
        client.submit(SomethingElse(), device="some_device")


def test_wait_for_job_completes(monkeypatch):
    """The polling loop must exit once the job status becomes TERMINAL."""

    # ------------------------------------------------------------------
    # fake status enum that provides `.value` for timeout message
    # ------------------------------------------------------------------
    class DummyStatus(Enum):
        RUNNING = "RUNNING"
        COMPLETED = "COMPLETED"
        ERROR = "ERROR"
        CANCELLED = "CANCELLED"

    monkeypatch.setattr(speqtrum, "JobStatus", DummyStatus)

    # ------------------------------------------------------------------
    # credentials and client set-up
    # ------------------------------------------------------------------
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    q = speqtrum.SpeQtrum()

    # ------------------------------------------------------------------
    # sequence of fake JobDetail objects
    # ------------------------------------------------------------------
    class FakeJob:
        def __init__(self, status):
            self.status = status

    calls = {"n": 0}

    def fake_get_job(self, _id):
        calls["n"] += 1
        return FakeJob(DummyStatus.RUNNING if calls["n"] == 1 else DummyStatus.COMPLETED)

    monkeypatch.setattr(speqtrum.SpeQtrum, "get_job", fake_get_job, raising=True)
    monkeypatch.setattr(speqtrum.time, "sleep", lambda *_: None)  # skip real sleeping

    # run - should finish on 2nd iteration
    result = q.wait_for_job(123, poll_interval=0.0, timeout=5)
    assert result.status is DummyStatus.COMPLETED
    assert calls["n"] >= 2


def test_wait_for_job_with_handle_returns_typed_detail(monkeypatch):
    """Passing a `JobHandle` should return a `TypedJobDetail` with typed result access."""

    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    q = speqtrum.SpeQtrum()

    handle = speqtrum.JobHandle.sampling(7)
    sampling_result = SamplingResult(nshots=2, samples={"00": 2})
    detail = speqtrum.JobDetail(
        id=7,
        name="job",
        description="desc",
        device_id=1,
        status=speqtrum.JobStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        updated_at=None,
        completed_at=datetime.now(timezone.utc),
        payload=None,
        result=ExecuteResult(type=speqtrum.ExecuteType.SAMPLING, sampling_result=sampling_result),
    )

    monkeypatch.setattr(speqtrum.SpeQtrum, "get_job", lambda self, _: detail, raising=True)

    typed_detail = q.wait_for_job(handle, poll_interval=0.0)
    assert isinstance(typed_detail, speqtrum.TypedJobDetail)
    assert typed_detail.id == 7
    typed_result = typed_detail.get_results()
    assert isinstance(typed_result, SamplingResult)
    assert typed_result.samples == sampling_result.samples
    assert typed_result.nshots == sampling_result.nshots


def test_ensure_ok_raises_with_api_payload():
    """_ensure_ok must surface API-provided error messages in the exception."""

    request = httpx.Request("GET", "https://speqtrum.example/devices")
    response = httpx.Response(400, request=request, json={"message": "Bad request", "code": "E_BAD"})

    with pytest.raises(speqtrum.SpeQtrumAPIError) as excinfo:
        speqtrum._ensure_ok(response)

    assert "Bad request" in str(excinfo.value)
    assert "E_BAD" in str(excinfo.value)

    response = httpx.Response(httpx.codes.UNAUTHORIZED, request=request)
    speqtrum._ensure_ok(response)

    request.extensions[speqtrum._SKIP_ENSURE_OK_EXTENSION] = True
    response = httpx.Response(400, request=request)
    speqtrum._ensure_ok(response)

    response = httpx.Response(400, request=request)
    speqtrum._ensure_ok(response)


def test_create_client_registers_response_hook(monkeypatch) -> None:
    """SpeQtrum HTTP client must install _ensure_ok as a response hook."""

    captured: dict[str, Any] = {}

    class RecordingClient:
        def __init__(self, *_, **kwargs):
            captured["event_hooks"] = kwargs.get("event_hooks")

        def __enter__(self):
            return self

        def __exit__(self, *_): ...

    token = SimpleNamespace(access_token="tok", refresh_token="rtok")
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("user", token))
    monkeypatch.setattr(speqtrum.httpx, "Client", RecordingClient)

    client = speqtrum.SpeQtrum()
    http_client = client._create_client()

    assert captured["event_hooks"]["response"]
    assert speqtrum._ensure_ok in captured["event_hooks"]["response"]

    # the helper returns a client instance; ensure the stub can be closed cleanly
    assert http_client.__enter__() is http_client  # noqa: PLC2801


def test_variational_program_handle_preserves_inner_result(monkeypatch):
    """A variational program handle should surface the inner functional result type."""

    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    q = speqtrum.SpeQtrum()

    handle = speqtrum.JobHandle.variational_program(21, result_type=SamplingResult)
    sampling_result = SamplingResult(nshots=3, samples={"01": 1, "10": 2})
    optimizer_result = OptimizerResult(optimal_cost=0.5, optimal_parameters=[0.1, 0.2, 0.3])
    variational_result = VariationalProgramResult(optimizer_result=optimizer_result, result=sampling_result)
    detail = speqtrum.JobDetail(
        id=21,
        name="vp",
        description="variational",
        device_id=2,
        status=speqtrum.JobStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        updated_at=None,
        completed_at=datetime.now(timezone.utc),
        payload=None,
        result=ExecuteResult(
            type=speqtrum.ExecuteType.VARIATIONAL_PROGRAM,
            variational_program_result=variational_result,
        ),
    )

    monkeypatch.setattr(speqtrum.SpeQtrum, "get_job", lambda self, _: detail, raising=True)

    typed_detail = q.wait_for_job(handle, poll_interval=0.0)
    assert isinstance(typed_detail, speqtrum.TypedJobDetail)
    typed_result = typed_detail.get_results()
    assert isinstance(typed_result, VariationalProgramResult)
    optimal_results = typed_result.optimal_execution_results
    assert isinstance(optimal_results, SamplingResult)


def test_variational_program_handle_with_wrong_type_raises(monkeypatch):
    """Providing an incorrect inner result type should raise at extraction time."""

    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    q = speqtrum.SpeQtrum()

    handle = speqtrum.JobHandle.variational_program(22, result_type=TimeEvolutionResult)
    sampling_result = SamplingResult(nshots=1, samples={"00": 1})
    optimizer_result = OptimizerResult(optimal_cost=0.2, optimal_parameters=[0.0])
    variational_result = VariationalProgramResult(optimizer_result=optimizer_result, result=sampling_result)
    detail = speqtrum.JobDetail(
        id=22,
        name="vp",
        description="variational",
        device_id=2,
        status=speqtrum.JobStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        updated_at=None,
        completed_at=datetime.now(timezone.utc),
        payload=None,
        result=ExecuteResult(
            type=speqtrum.ExecuteType.VARIATIONAL_PROGRAM,
            variational_program_result=variational_result,
        ),
    )

    monkeypatch.setattr(speqtrum.SpeQtrum, "get_job", lambda self, _: detail, raising=True)

    typed_detail = q.wait_for_job(handle, poll_interval=0.0)
    with pytest.raises(RuntimeError):
        typed_detail.get_results()


def test_wait_for_job_times_out(monkeypatch):
    """When the timeout is reached the helper must raise `TimeoutError`."""

    class DummyStatus(Enum):
        RUNNING = "RUNNING"
        COMPLETED = "COMPLETED"
        ERROR = "ERROR"
        CANCELLED = "CANCELLED"

    monkeypatch.setattr(speqtrum, "JobStatus", DummyStatus)
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    q = speqtrum.SpeQtrum()

    # always RUNNING → never terminal
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "get_job",
        lambda *_: types.SimpleNamespace(status=DummyStatus.RUNNING),
        raising=True,
    )
    monkeypatch.setattr(speqtrum.time, "sleep", lambda *_: None)

    # deterministically advance monotonic clock
    t = {"v": 0.0}

    def fake_monotonic():
        t["v"] += 0.1
        return t["v"]

    monkeypatch.setattr(speqtrum.time, "monotonic", fake_monotonic)

    with pytest.raises(TimeoutError):
        q.wait_for_job(1, poll_interval=0.0, timeout=0.0)


def test_login_success(monkeypatch) -> None:
    """`login` returns True and stores the credentials on success."""
    # replace the outbound HTTP client with a stub
    monkeypatch.setattr(
        speqtrum.httpx,
        "Client",
        lambda **_: DummyClient(post_payload={"access_token": "abc", "expires_at": 0, "refresh_token": "r"}),
    )

    # spy on store_credentials so we can assert it was called
    stored: list[tuple[str, object]] = []

    def fake_store(username: str, token):
        stored.append((username, token))

    monkeypatch.setattr(speqtrum, "store_credentials", fake_store)

    # any object with an __init__(**kwargs) works here - we never use the token later
    class FakeToken:
        def __init__(self, **_): ...

    monkeypatch.setattr(speqtrum, "Token", FakeToken)

    assert speqtrum.SpeQtrum.login("bob", "APIKEY") is True
    assert stored
    assert stored[0][0] == "bob"


def test_login_http_error(monkeypatch) -> None:
    """`login` returns True and stores the credentials on success."""
    # replace the outbound HTTP client with a stub
    monkeypatch.setattr(
        speqtrum.httpx,
        "Client",
        lambda **_: DummyClient(post_payload={"access_token": "abc", "expires_at": 0, "refresh_token": "r"}),
    )

    monkeypatch.setattr(
        speqtrum,
        "get_settings",
        lambda: SimpleNamespace(speqtrum_api_url="https://mock.api", speqtrum_audience="audience"),
    )

    # httpsx.Client.post will raise HTTPStatusError
    def raise_http_error(*args, **kwargs):
        request = httpx.Request("POST", "https://mock.api/authorisation-tokens")
        response = httpx.Response(status_code=401, request=request)
        raise httpx.HTTPStatusError("Unauthorized", request=request, response=response)

    monkeypatch.setattr(speqtrum.httpx, "Client", raise_http_error)

    def fake_store(username: str, token): ...

    monkeypatch.setattr(speqtrum, "store_credentials", fake_store)

    # any object with an __init__(**kwargs) works here - we never use the token later
    class FakeToken:
        def __init__(self, **_): ...

    monkeypatch.setattr(speqtrum, "Token", FakeToken)

    assert speqtrum.SpeQtrum.login("bob", "APIKEY") is False


def test_login_requires_credentials(monkeypatch):
    """If neither parameters nor environment variables are supplied, login fails."""

    # Patch `get_settings` to return an object with missing credentials
    class DummySettings:
        speqtrum_username = None
        speqtrum_apikey = None

    monkeypatch.setattr(speqtrum, "get_settings", DummySettings)

    assert speqtrum.SpeQtrum.login(username=None, apikey=None) is False


def test_logout_deletes_credentials(monkeypatch):
    """`logout` must call the keyring deletion helper exactly once."""
    called = {"n": 0}

    def fake_delete():
        called["n"] += 1

    monkeypatch.setattr(speqtrum, "delete_credentials", fake_delete)
    speqtrum.SpeQtrum.logout()

    assert called["n"] == 1


def test_list_devices_filters_client_side(monkeypatch):
    """Client should apply the *where* predicate after pulling raw data."""
    monkeypatch.setattr(
        speqtrum.httpx,
        "Client",
        lambda **_: DummyClient(get_payload={"items": [{"id": 1}, {"id": 2}]}),
    )
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(speqtrum.TypeAdapter, "validate_python", lambda self, data: data)  # identity - skip pydantic

    client = speqtrum.SpeQtrum()
    all_devices = client.list_devices()
    only_two = client.list_devices(lambda d: d["id"] == 2)

    assert [d["id"] for d in all_devices] == [1, 2]
    assert only_two == [{"id": 2}]


def test_bearer_auth_refreshes_after_unauthorized(monkeypatch):
    """Unauthorized responses should trigger a refresh token request."""

    token = SimpleNamespace(access_token="old_access", refresh_token="old_refresh")
    client = SimpleNamespace(token=token, username="alice")
    auth = speqtrum._BearerAuth(client)

    monkeypatch.setattr(speqtrum, "get_settings", lambda: SimpleNamespace(speqtrum_api_url="https://mock.api"))

    stored = {}

    def fake_store(username: str, refreshed_token):
        stored["args"] = (username, refreshed_token)

    monkeypatch.setattr(speqtrum, "store_credentials", fake_store)

    request = httpx.Request("GET", "https://mock.api/jobs")
    flow = auth.auth_flow(request)

    first_request = next(flow)
    assert first_request.headers["Authorization"] == "Bearer old_access"

    unauthorized = httpx.Response(status_code=401, request=first_request)
    refresh_request = flow.send(unauthorized)
    assert refresh_request.method == "POST"
    assert str(refresh_request.url) == "https://mock.api/authorisation-tokens/refresh"
    assert refresh_request.headers["Authorization"] == "Bearer old_refresh"

    refresh_payload = {
        "accessToken": "new_access",
        "expiresIn": 123,
        "issuedAt": 100,
        "refreshToken": "new_refresh",
        "tokenType": "bearer",
    }
    retry_request = flow.send(httpx.Response(status_code=200, json=refresh_payload, request=refresh_request))
    assert retry_request is request
    assert retry_request.headers["Authorization"] == "Bearer new_access"

    assert "args" in stored
    stored_username, stored_token = stored["args"]
    assert stored_username == "alice"
    assert stored_token.access_token == "new_access"

    with pytest.raises(StopIteration):
        flow.send(httpx.Response(status_code=200, request=retry_request))


def test_list_jobs(monkeypatch):
    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(get_payload={"items": [{"id": 1}, {"id": 2}, {"id": 3}]}),
        raising=True,
    )
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(speqtrum.TypeAdapter, "validate_python", lambda self, data: data)  # identity - skip pydantic
    client = speqtrum.SpeQtrum()
    jobs = client.list_jobs()
    assert [job["id"] for job in jobs] == [1, 2, 3]


def test_get_job(monkeypatch):
    class DummyPayload2(collections.UserDict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def get(self, key, default=None):
            return getattr(self, key, default)

        def model_dump(self):
            return dict(self)

    json_payload = {
        "id": 42,
        "name": "test_job",
        "description": "A test job",
        "device_id": 1,
        "status": speqtrum.JobStatus.COMPLETED,
        "error_logs": "file.log",
        "payload": "bad payload",
        "result": "bad result",
        "jobType": "digital",
        "logs": "none",
        "error": "none",
        "created_at": datetime.now(timezone.utc),
    }
    payload = DummyPayload2(**json_payload)

    monkeypatch.setattr(
        speqtrum.SpeQtrum,
        "_create_client",
        lambda self: DummyClient(get_payload=payload),
        raising=True,
    )
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    monkeypatch.setattr(speqtrum.TypeAdapter, "validate_python", lambda self, data: data)  # identity - skip pydantic
    client = speqtrum.SpeQtrum()
    job = client.get_job(42)
    assert job["id"] == 42
    assert job["name"] == "test_job"
    assert isinstance(job["status"], speqtrum.JobStatus)
    job = client.get_job(speqtrum.JobHandle.sampling(42))
    assert isinstance(job, speqtrum.TypedJobDetail)


def run_auth(monkeypatch):
    token = SimpleNamespace(access_token="old_access", refresh_token="old_refresh")
    client = SimpleNamespace(token=token)
    auth = speqtrum._BearerAuth(client)
    monkeypatch.setattr(
        speqtrum,
        "get_settings",
        lambda: SimpleNamespace(speqtrum_api_url="https://mock.api"),
    )

    request = httpx.Request("GET", "https://mock.api/jobs")
    flow = auth.auth_flow(request)

    yielded_request = next(flow)
    assert yielded_request.headers["Authorization"] == "Bearer old_access"

    response = httpx.Response(status_code=httpx.codes.UNAUTHORIZED, request=yielded_request)
    refresh_request = flow.send(response)

    assert refresh_request.method == "POST"
    assert refresh_request.headers["Authorization"] == "Bearer old_refresh"

    flow.send(httpx.Response(status_code=200, request=refresh_request))


def test_auth_flow_json_error(monkeypatch):
    class DummyResponse(httpx.Response):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def json(self):
            raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(
        speqtrum.httpx,
        "Response",
        DummyResponse,
    )
    with pytest.raises(RuntimeError):
        run_auth(monkeypatch)


def test_auth_flow_http_error(monkeypatch):
    class DummyResponse(httpx.Response):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def raise_for_status(self):
            raise httpx.HTTPStatusError("Unauthorized", request=self.request, response=self)

    monkeypatch.setattr(
        speqtrum.httpx,
        "Response",
        DummyResponse,
    )

    with pytest.raises(httpx.HTTPStatusError):
        run_auth(monkeypatch)


def test_auth_flow_invalid_json(monkeypatch):
    class DummyResponse(httpx.Response):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def json(self):
            return "not json"

    monkeypatch.setattr(
        speqtrum.httpx,
        "Response",
        DummyResponse,
    )

    with pytest.raises(RuntimeError):
        run_auth(monkeypatch)


def test_auth_flow_token_error(monkeypatch):
    class DummyResponse(httpx.Response):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def json(self):
            return {"access_token": 3}

    monkeypatch.setattr(
        speqtrum.httpx,
        "Response",
        DummyResponse,
    )

    with pytest.raises(RuntimeError):
        run_auth(monkeypatch)


def test_summarize_error(monkeypatch) -> None:
    # need to simulate a httpx.Response that raises ResponseNotRead on .text access
    class DummyResponse(httpx.Response):
        _error_on_read: bool = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._read_called = False
            self.request = httpx.Request("GET", "https://example.com")

        @property
        def text(self):
            if not self._read_called:
                raise httpx.ResponseNotRead
            return super().text

        def read(self):
            self._read_called = True
            return super().read()

    # valid JSON body
    json_body = {"message": "Error occurred", "code": "E123"}
    json_body_str = json.dumps(json_body)
    response1 = DummyResponse(status_code=400, content=json_body_str.encode("utf-8"))
    summary1 = speqtrum._summarize_error_payload(response1)
    assert "Error occurred" in summary1
    assert "E123" in summary1

    # invalid JSON body, but non-empty text
    response2 = DummyResponse(status_code=400, content=b"Some error text")
    summary2 = speqtrum._summarize_error_payload(response2)
    assert summary2 == "Some error text"

    # empty body
    response3 = DummyResponse(status_code=400, content=b"")
    summary3 = speqtrum._summarize_error_payload(response3)
    assert summary3 == "no response body"

    # monekypatch the read
    def fake_read(self):
        raise ValueError("Simulated read error")

    # simulate error on reading
    response3 = DummyResponse(status_code=400, content=b"")
    monkeypatch.setattr(DummyResponse, "read", fake_read)
    summary3 = speqtrum._summarize_error_payload(response3)
    assert summary3 == "no response body"


def test_none_payload():
    assert speqtrum._stringify_payload(None) is None


def test_dict_with_message_and_code():
    payload = {"message": "Hello", "code": 400}
    assert speqtrum._stringify_payload(payload) == "Hello (code=400)"


def test_dict_with_message_and_string_code():
    payload = {"message": "Error occurred", "error_code": "E123"}
    assert speqtrum._stringify_payload(payload) == "Error occurred (code=E123)"


def test_dict_with_alternate_message_keys():
    payload = {"detail": "Details here"}
    assert speqtrum._stringify_payload(payload) == "Details here"


def test_dict_ignores_empty_or_whitespace_strings():
    payload = {"message": "   ", "title": "Valid title"}
    assert speqtrum._stringify_payload(payload) == "Valid title"


def test_dict_without_message_like_keys_serializes_to_json():
    payload = {"b": 2, "a": 1}
    result = speqtrum._stringify_payload(payload)
    assert result == json.dumps(payload, sort_keys=True)


def test_dict_with_unserializable_value_falls_back_to_str():
    payload = {"a": {1, 2, 3}}
    result = speqtrum._stringify_payload(payload)
    assert result == str(payload)


def test_dict_with_non_string_message_value_ignored():
    payload = {"message": 123, "detail": "ok"}
    assert speqtrum._stringify_payload(payload) == "ok"


def test_list_payload_with_items():
    payload = [1, "two", 3, 4]
    assert speqtrum._stringify_payload(payload) == "1, two, 3"


def test_list_payload_with_fewer_than_three_items():
    payload = ["a"]
    assert speqtrum._stringify_payload(payload) == "a"


def test_empty_list_payload():
    payload = []
    assert speqtrum._stringify_payload(payload) == "[]"


@pytest.mark.parametrize(
    # ()"payload,expected",
    ("payload", "expected"),
    [
        ("hello", "hello"),
        (123, "123"),
        (45.6, "45.6"),
        (True, "True"),
        (False, "False"),
    ],
)
def test_primitive_payloads(payload, expected):
    assert speqtrum._stringify_payload(payload) == expected


def test_unsupported_type_returns_none():
    class CustomObject:
        pass

    payload = CustomObject()
    assert speqtrum._stringify_payload(payload) is None


def test_response_context_with_no_request(monkeypatch):
    class FakeResponse(httpx.Response):
        _val = None

        @property
        def request(self):
            return self._val

        @request.setter
        def request(self, value): ...

    response = FakeResponse(200)
    context = speqtrum._response_context(response)
    assert context == "SpeQtrum API call"


def test_response_context_with_no_extension():
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(200, request=request)
    context = speqtrum._response_context(response)
    assert context == "GET https://example.com"


def test_response_context_with_extension():
    request = httpx.Request("POST", "https://example.com")
    request.extensions[speqtrum._CONTEXT_EXTENSION] = "Custom context"
    response = httpx.Response(200, request=request)
    context = speqtrum._response_context(response)
    assert context == "Custom context"


def test_safe_b64_json_with_valid_base64_json():
    original_data = {"key": "value", "number": 42}
    json_str = json.dumps(original_data)
    b64_str = base64.urlsafe_b64encode(json_str.encode("utf-8")).decode("utf-8").rstrip("=")
    result = speqtrum._safe_b64_json(b64_str, context="test context")
    assert result == original_data
