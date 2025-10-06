"""
Unit-tests for the SpeQtrum synchronous client.

All tests are *function* based (no classes) so they integrate with plain
pytest discovery.
"""

from __future__ import annotations

import types
from enum import Enum
from types import SimpleNamespace

import pytest

import qilisdk.speqtrum.speqtrum as speqtrum


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


# ────────────────────────────────────────────────────────────────────────────────
# tests
# ────────────────────────────────────────────────────────────────────────────────
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

    assert q._username == "alice"
    assert q._token is tok


def test_submit_dispatches_to_sampling_handler(monkeypatch):
    """`submit` must route to _execute_sampling when given a Sampling instance."""

    # Stub the Sampling symbol **before** constructing SpeQtrum so that __init__
    # collects the correct type inside _handlers.
    class FakeSampling:
        pass

    monkeypatch.setattr(speqtrum, "Sampling", FakeSampling)

    # simple credentials stub
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))

    # Replace the real network-hitting method with something predictable.
    monkeypatch.setattr(speqtrum.SpeQtrum, "_submit_sampling", lambda self, f, device_id: 99, raising=True)

    q = speqtrum.SpeQtrum()
    assert q.submit(FakeSampling(), 0) == 99


def test_submit_unknown_functional_raises(monkeypatch):
    """Passing an unsupported functional type should raise `NotImplementedError`."""
    monkeypatch.setattr(speqtrum, "load_credentials", lambda: ("u", SimpleNamespace(access_token="t")))
    client = speqtrum.SpeQtrum()

    class SomethingElse:
        pass

    with pytest.raises(NotImplementedError):
        client.submit(SomethingElse(), device_id=0)


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

    def fake_get_job_details(self, _id):
        calls["n"] += 1
        return FakeJob(DummyStatus.RUNNING if calls["n"] == 1 else DummyStatus.COMPLETED)

    monkeypatch.setattr(speqtrum.SpeQtrum, "get_job_details", fake_get_job_details, raising=True)
    monkeypatch.setattr(speqtrum.time, "sleep", lambda *_: None)  # skip real sleeping

    # run - should finish on 2nd iteration
    result = q.wait_for_job(123, poll_interval=0.0, timeout=5)
    assert result.status is DummyStatus.COMPLETED
    assert calls["n"] >= 2


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
        "get_job_details",
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


def test_login_success(monkeypatch):
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
        def __init__(self, **_):
            pass

    monkeypatch.setattr(speqtrum, "Token", FakeToken)

    assert speqtrum.SpeQtrum.login("bob", "APIKEY") is True
    assert stored
    assert stored[0][0] == "bob"


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
