"""Tests for the OnetClient HTTP layer."""

from __future__ import annotations

import pytest

from onet_data_collector.exceptions import (
    OnetAuthError,
    OnetConfigError,
    OnetHTTPError,
    OnetValidationError,
)
from tests.collector.conftest import http_error


def test_requires_credentials():
    from onet_data_collector.client import OnetClient

    with pytest.raises(OnetConfigError):
        OnetClient("", "")


def test_request_success(make_client):
    client, opener = make_client([{"api_version": "1.9"}])
    assert client.request("about") == {"api_version": "1.9"}
    assert opener.calls[0].endswith("/ws/about")


def test_query_params_encoded(make_client):
    client, opener = make_client([{"occupation": []}])
    client.request("online/search", ("keyword", "data science"), ("end", 50))
    assert "keyword=data+science" in opener.calls[0]
    assert "end=50" in opener.calls[0]


def test_auth_error_not_retried(make_client):
    client, opener = make_client([http_error(401)])
    with pytest.raises(OnetAuthError):
        client.request("about")
    assert len(opener.calls) == 1


def test_validation_error_carries_payload(make_client):
    client, _ = make_client([http_error(422, body={"error": "bad keyword"})])
    with pytest.raises(OnetValidationError) as exc:
        client.request("online/search", ("keyword", ""))
    assert exc.value.payload["error"] == "bad keyword"


def test_retries_then_succeeds_on_500(make_client):
    client, opener = make_client([http_error(500), http_error(503), {"ok": True}])
    assert client.request("about") == {"ok": True}
    assert len(opener.calls) == 3


def test_gives_up_after_max_retries(make_client):
    client, opener = make_client([http_error(500)] * 5, max_retries=2)
    with pytest.raises(OnetHTTPError):
        client.request("about")
    assert len(opener.calls) == 3  # initial + 2 retries


def test_call_returns_error_dict_instead_of_raising(make_client):
    client, _ = make_client([http_error(500)], max_retries=0)
    result = client.call("about")
    assert "error" in result


def test_paginate_walks_all_pages(make_client):
    page1 = {"total": 3, "start": 1, "end": 2, "occupation": [{"code": "a"}, {"code": "b"}]}
    page2 = {"total": 3, "start": 3, "end": 3, "occupation": [{"code": "c"}]}
    client, opener = make_client([page1, page2])
    items = client.paginate("online/search", ("keyword", "x"), item_key="occupation", page_size=2)
    assert [i["code"] for i in items] == ["a", "b", "c"]
    assert len(opener.calls) == 2


def test_paginate_respects_max_items(make_client):
    page1 = {"total": 10, "start": 1, "end": 2, "occupation": [{"code": "a"}, {"code": "b"}]}
    client, opener = make_client([page1])
    items = client.paginate(
        "online/search", ("keyword", "x"), item_key="occupation", page_size=2, max_items=1
    )
    assert len(items) == 1


def test_paginate_handles_single_dict_item(make_client):
    page = {"total": 1, "start": 1, "end": 1, "occupation": {"code": "solo"}}
    client, _ = make_client([page])
    items = client.paginate("online/search", ("keyword", "x"), item_key="occupation")
    assert items == [{"code": "solo"}]
