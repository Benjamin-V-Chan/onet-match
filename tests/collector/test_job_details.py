"""Tests for fetch_job_details CSV reading and raw JSON output."""

from __future__ import annotations

import json

import pytest

from onet_data_collector.job_details import fetch_job_details
from tests.collector.conftest import http_error


def _write_csv(tmp_path, rows, header="Job Code"):
    p = tmp_path / "codes.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def test_fetch_dedupes_and_writes(tmp_path, make_client):
    csv_path = _write_csv(tmp_path, ["15-2051.00", "15-2051.00", "15-1252.00"])
    client, opener = make_client([{"occupation": {"code": "15-2051.00"}}, {"occupation": {"code": "15-1252.00"}}])
    out = tmp_path / "details.json"
    result = fetch_job_details("u", "p", csv_path, str(out), client=client)

    assert len(result) == 2  # deduped
    assert len(opener.calls) == 2
    assert json.loads(out.read_text())[0]["occupation"]["code"] == "15-2051.00"


def test_fetch_skips_errors(tmp_path, make_client):
    csv_path = _write_csv(tmp_path, ["good", "bad"])
    client, _ = make_client([{"occupation": {"code": "good"}}, http_error(500)], max_retries=0)
    out = tmp_path / "details.json"
    result = fetch_job_details("u", "p", csv_path, str(out), client=client, skip_errors=True)
    assert len(result) == 1


def test_fetch_missing_column_raises(tmp_path, make_client):
    csv_path = _write_csv(tmp_path, ["x"], header="WrongColumn")
    client, _ = make_client([])
    with pytest.raises(KeyError):
        fetch_job_details("u", "p", csv_path, str(tmp_path / "o.json"), client=client)


def test_fetch_missing_file_raises(tmp_path, make_client):
    client, _ = make_client([])
    with pytest.raises(FileNotFoundError):
        fetch_job_details("u", "p", str(tmp_path / "nope.csv"), str(tmp_path / "o.json"), client=client)
