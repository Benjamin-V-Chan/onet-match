"""Tests for the bulk database loader (offline, tiny synthetic zip)."""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import pytest

from onet_data_collector.bulk import (
    BulkRelease,
    cached_releases,
    download_bulk,
    latest_release,
    load_bulk,
    release_tag,
)
from onet_data_collector.cli import main
from onet_data_collector.exceptions import OnetConfigError, OnetHTTPError
from tests.collector.conftest import http_error


class BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class BytesOpener:
    """Returns queued byte payloads (or raises queued exceptions)."""

    def __init__(self, queue):
        self._queue = list(queue)
        self.calls = []

    def open(self, req, timeout=None):  # noqa: A003
        self.calls.append(req.full_url)
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return BytesResponse(item)


def _zip(files: dict[str, str], prefix: str = "db_31_0_text/") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in files.items():
            zf.writestr(prefix + name, body)
    return buf.getvalue()


NEW_LAYOUT = {
    "Occupation Data.txt": "O*NET-SOC Code\tTitle\tDescription\n15-2051.00\tData Scientists\tAnalyze data.\n",
    "Job Titles.txt": "O*NET-SOC Code\tJob Title\tShort Title\tSource(s)\n15-2051.00\tML Engineer\tn/a\t08\n",
    "Essential Skills.txt": "O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value\n"
                            "15-2051.00\t2.A.1.a\tReading Comprehension\tIM\t4.0\n",
    "Transferable Skills.txt": "O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value\n"
                               "15-2051.00\t2.B.1.a\tSocial Perceptiveness\tIM\t3.0\n",
    "Read Me.txt": "readme",
}

FIXED_NOW = lambda: dt.datetime(2026, 10, 23, tzinfo=dt.timezone.utc)  # noqa: E731


def test_release_tag_validates():
    assert release_tag("31.0") == "31_0"
    with pytest.raises(OnetConfigError):
        release_tag("latest")


def test_latest_release_parses_page_and_falls_back():
    page = b'<a href="/dl_files/database/db_30_1_excel.zip">x</a><a href="db_31_0_mysql.zip">'
    assert latest_release(opener=BytesOpener([page])) == "31.0"
    assert latest_release(opener=BytesOpener([b"nothing here"])) == "31.0"  # fallback
    assert latest_release(opener=BytesOpener([http_error(500)])) == "31.0"


def test_download_extracts_and_writes_manifest(tmp_path):
    opener = BytesOpener([_zip(NEW_LAYOUT)])
    rel = download_bulk("31.0", cache_dir=tmp_path, opener=opener, now=FIXED_NOW)
    assert opener.calls == ["https://www.onetcenter.org/dl_files/database/db_31_0_text.zip"]
    assert (tmp_path / "db_31_0" / "Occupation Data.txt").is_file()
    m = json.loads((tmp_path / "db_31_0" / "manifest.json").read_text())
    assert m["release"] == "31.0"
    assert m["downloaded_at"].startswith("2026-10-23")
    assert len(m["sha256"]) == 64
    assert "Job Titles.txt" in m["files"]
    assert rel.release == "31.0"


def test_download_reuses_cache_unless_forced(tmp_path):
    download_bulk("31.0", cache_dir=tmp_path, opener=BytesOpener([_zip(NEW_LAYOUT)]))
    empty = BytesOpener([])
    download_bulk("31.0", cache_dir=tmp_path, opener=empty)  # no network call
    assert empty.calls == []
    forced = BytesOpener([_zip(NEW_LAYOUT)])
    download_bulk("31.0", cache_dir=tmp_path, opener=forced, force=True)
    assert len(forced.calls) == 1


def test_download_rejects_non_zip(tmp_path):
    with pytest.raises(OnetHTTPError):
        download_bulk("31.0", cache_dir=tmp_path, opener=BytesOpener([b"<html>not a zip"]))


def test_tables_normalise_columns_and_virtual_skills(tmp_path):
    rel = download_bulk("31.0", cache_dir=tmp_path, opener=BytesOpener([_zip(NEW_LAYOUT)]))
    titles = rel.table("job_titles")
    assert list(titles.columns) == ["soc_code", "job_title", "short_title", "sources"]
    skills = rel.table("skills")
    assert set(skills["element_name"]) == {"Reading Comprehension", "Social Perceptiveness"}
    assert "occupation_data" in rel.available()
    with pytest.raises(FileNotFoundError):
        rel.table("task_ratings")
    with pytest.raises(KeyError):
        rel.table("nope")


def test_legacy_file_names_are_supported(tmp_path):
    legacy = {
        "Alternate Titles.txt": "O*NET-SOC Code\tAlternate Title\tShort Title\tSource(s)\n11-1011.00\tCEO\tn/a\t10\n",
        "Skills.txt": "O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value\n11-1011.00\t2.A.1.a\tReading\tIM\t4\n",
    }
    d = tmp_path / "db_29_0"
    d.mkdir()
    for name, body in legacy.items():
        (d / name).write_text(body)
    (d / "manifest.json").write_text(json.dumps({"release": "29.0"}))
    rel = load_bulk("29.0", cache_dir=tmp_path)
    assert rel.table("job_titles")["job_title"].tolist() == ["CEO"]
    assert rel.table("skills")["element_name"].tolist() == ["Reading"]


def test_load_bulk_picks_newest_and_errors_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bulk(cache_dir=tmp_path)
    for rel in ("29.3", "31.0", "30.1"):
        d = tmp_path / f"db_{rel.replace('.', '_')}"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"release": rel}))
    assert cached_releases(tmp_path) == ["29.3", "30.1", "31.0"]
    assert isinstance(load_bulk(cache_dir=tmp_path), BulkRelease)
    assert load_bulk(cache_dir=tmp_path).release == "31.0"


def test_cli_bulk_info(tmp_path, capsys):
    download_bulk("31.0", cache_dir=tmp_path, opener=BytesOpener([_zip(NEW_LAYOUT)]))
    assert main(["bulk", "info", "--cache-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "O*NET 31.0" in out and "sha256" in out
