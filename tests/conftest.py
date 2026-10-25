import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def ctgov_page1() -> dict:
    return _load("ctgov_page1.json")


@pytest.fixture
def ctgov_page2() -> dict:
    return _load("ctgov_page2.json")


@pytest.fixture
def tucatinib_study(ctgov_page1) -> dict:
    return ctgov_page1["studies"][0]


@pytest.fixture
def openfda_count() -> dict:
    return _load("openfda_count.json")


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Stands in for requests.Session: replays payloads in order and records each call."""

    def __init__(self, payloads: list[dict], status_code: int = 200):
        self._payloads = list(payloads)
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self._payloads.pop(0), self.status_code)


@pytest.fixture
def fake_session():
    """Factory fixture: fake_session([page1, page2]) -> FakeSession."""
    return FakeSession
