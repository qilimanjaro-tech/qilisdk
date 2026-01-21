# replace these with the real import paths in your project:
import pytest

pytest.importorskip(
    "keyring",
    reason="SpeQtrum tests require the 'speqtrum' optional dependency",
    exc_type=ImportError,
)

import keyring
from pydantic import ValidationError

from qilisdk.speqtrum.keyring import KEYRING_IDENTIFIER, delete_credentials, load_credentials, store_credentials
from qilisdk.speqtrum.speqtrum_models import Token


class FakeToken:
    """A minimal stand-in for Token instances."""

    def __init__(self, json_str: str):
        self._json = json_str

    def model_dump_json(self) -> str:
        return self._json


def test_store_credentials_calls_keyring_set_password(monkeypatch):
    calls = []

    def fake_set_password(service, key, password):
        calls.append((service, key, password))

    monkeypatch.setattr(keyring, "set_password", fake_set_password)

    fake_token = FakeToken('{"foo":"bar"}')
    store_credentials("alice", fake_token)

    assert calls == [
        (KEYRING_IDENTIFIER, "username", "alice"),
        (KEYRING_IDENTIFIER, "token", '{"foo":"bar"}'),
    ]


def test_delete_credentials_calls_keyring_delete_password(monkeypatch):
    calls = []

    def fake_delete_password(service, key):
        calls.append((service, key))

    monkeypatch.setattr(keyring, "delete_password", fake_delete_password)

    delete_credentials()

    assert calls == [
        (KEYRING_IDENTIFIER, "username"),
        (KEYRING_IDENTIFIER, "token"),
    ]


def test_bad_delete(monkeypatch):
    # generate an identifer that is unlikely to exist
    KEYRING_IDENTIFIER = "dskhfkjhkjhjkrhrkrhjkrhkrjhvhcxkjvhcxkjvh213123"

    # make sure it doesn't exist
    assert keyring.get_password(KEYRING_IDENTIFIER, "username") is None
    assert keyring.get_password(KEYRING_IDENTIFIER, "token") is None

    delete_credentials()


def test_load_credentials_returns_none_if_either_missing(monkeypatch):
    # username missing
    monkeypatch.setattr(keyring, "get_password", lambda svc, key: None)
    assert load_credentials() is None

    # token missing
    def get_pw(svc, key):
        return "alice" if key == "username" else None

    monkeypatch.setattr(keyring, "get_password", get_pw)
    assert load_credentials() is None


def test_load_credentials_returns_none_if_token_json_invalid(monkeypatch):
    # both present
    monkeypatch.setattr(keyring, "get_password", lambda svc, key: "alice" if key == "username" else "bad-json")

    # force Token.model_validate_json to raise ValidationError
    def fake_validate(json_str):
        raise ValidationError.from_exception_data("Token", [])

    monkeypatch.setattr(Token, "model_validate_json", staticmethod(fake_validate))

    assert load_credentials() is None


def test_load_credentials_returns_tuple_on_success(monkeypatch):
    # both present
    monkeypatch.setattr(keyring, "get_password", lambda svc, key: "bob" if key == "username" else '{"x":1}')

    fake_token = FakeToken('{"x":1}')

    # make model_validate_json return our fake_token
    def fake_validate(json_str):
        assert json_str == '{"x":1}'
        return fake_token

    monkeypatch.setattr(Token, "model_validate_json", staticmethod(fake_validate))

    result = load_credentials()
    assert isinstance(result, tuple)
    assert result == ("bob", fake_token)
