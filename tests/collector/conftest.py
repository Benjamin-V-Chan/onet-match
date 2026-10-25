"""Shared test fixtures: a fake urllib opener for offline O*NET client tests."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest


class FakeResponse(io.BytesIO):
    """A minimal file-like HTTP response usable as a context manager."""

    def __init__(self, payload: Any, status: int = 200):
        super().__init__(json.dumps(payload).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def getcode(self):
        return self.status


class FakeOpener:
    """Queues responses/exceptions and returns them in order from ``open``.

    Each queued item is either a payload (dict/list -> 200 response) or an
    exception instance to raise. Records every requested URL in ``calls``.
    """

    def __init__(self, queue: list[Any]):
        self._queue = list(queue)
        self.calls: list[str] = []

    def open(self, req, timeout=None):  # noqa: A003 - mimic urllib API
        self.calls.append(req.full_url if hasattr(req, "full_url") else req.get_full_url())
        if not self._queue:
            raise AssertionError("FakeOpener ran out of queued responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def http_error(code: int, headers: dict | None = None, body: Any = None) -> urllib.error.HTTPError:
    fp = io.BytesIO(json.dumps(body).encode()) if body is not None else None
    return urllib.error.HTTPError("https://example.test", code, "err", headers or {}, fp)


@pytest.fixture
def make_client():
    """Factory building an OnetClient wired to a FakeOpener with no real sleeps."""
    from onet_data_collector.client import OnetClient

    def _factory(queue: list[Any], **kwargs):
        opener = FakeOpener(queue)
        client = OnetClient(
            "user", "pass", opener=opener, sleep=lambda _s: None, **kwargs
        )
        return client, opener

    return _factory
