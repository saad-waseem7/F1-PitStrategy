from __future__ import annotations

import pytest
import requests

from src.data import ingest
from src.data.ingest import OpenF1Client


class FakeResponse:
    def __init__(self, status: int = 200, payload=None, headers=None, bad_json: bool = False) -> None:
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("truncated")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class ScriptedSession:
    """Returns the scripted responses in order; an Exception instance is raised instead."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(ingest.time, "sleep", recorded.append)
    return recorded


def make_client(script: list, max_retries: int = 5, empty_retries: int = 2) -> tuple[OpenF1Client, ScriptedSession]:
    client = OpenF1Client("https://x/v1", delay_s=0.0, max_retries=max_retries, timeout_s=5, retry_delay_s=3.0, empty_retries=empty_retries)
    session = ScriptedSession(script)
    client._session = session
    return client, session


ROWS = [{"a": 1}, {"a": 2}]


def test_rate_limit_is_retried_after_3_seconds(sleeps):
    client, session = make_client([FakeResponse(429), FakeResponse(429), FakeResponse(200, ROWS)])
    df = client.get("laps", session_key=1)
    assert len(df) == 2 and session.calls == 3
    assert sleeps == [0.0, 3.0, 3.0]  # initial spacing, then a fixed 3 s per retry


def test_retry_after_header_is_honoured_when_longer(sleeps):
    client, _ = make_client([FakeResponse(429, headers={"Retry-After": "10"}), FakeResponse(200, ROWS)])
    client.get("laps")
    assert sleeps[-1] == 10.0


def test_network_error_and_server_error_are_retried(sleeps):
    client, session = make_client([requests.ConnectionError("reset"), FakeResponse(503), FakeResponse(200, ROWS)])
    assert len(client.get("laps")) == 2
    assert session.calls == 3


def test_truncated_or_malformed_body_is_retried(sleeps):
    client, _ = make_client([FakeResponse(200, bad_json=True), FakeResponse(200, {"detail": "slow down"}), FakeResponse(200, ROWS)])
    assert len(client.get("laps")) == 2


def test_gives_up_loudly_after_max_retries(sleeps):
    client, session = make_client([FakeResponse(429)] * 3, max_retries=3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.get("laps")
    assert session.calls == 3


def test_404_means_no_data_and_is_not_retried(sleeps):
    client, session = make_client([FakeResponse(404)])
    assert client.get("race_control").empty
    assert session.calls == 1


def test_client_errors_other_than_404_raise_immediately(sleeps):
    client, session = make_client([FakeResponse(422)])
    with pytest.raises(requests.HTTPError):
        client.get("laps")
    assert session.calls == 1


def test_get_nonempty_retries_empty_result(sleeps):
    client, session = make_client([FakeResponse(404), FakeResponse(200, ROWS)])
    assert len(client.get_nonempty("laps")) == 2
    assert session.calls == 2 and 3.0 in sleeps


def test_get_nonempty_accepts_persistent_empty(sleeps):
    client, session = make_client([FakeResponse(404)] * 3, empty_retries=2)
    assert client.get_nonempty("laps").empty
    assert session.calls == 3