"""Tests for keyword_search built on a fake client."""

from __future__ import annotations

from onet_data_collector.keyword_search import COLUMNS, keyword_search


def test_keyword_search_shapes_rows(make_client):
    page = {
        "total": 2,
        "start": 1,
        "end": 2,
        "occupation": [
            {"code": "15-2051.00", "title": "Data Scientists", "relevance_score": 100},
            {"code": "15-1252.00", "title": "Software Developers", "relevance_score": 90},
        ],
    }
    client, _ = make_client([page])
    df = keyword_search("u", "p", "data science", client=client)

    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert df.loc[0, "SOC Code"] == "15"
    assert df.loc[0, "Job Code"] == "15-2051.00"
    assert (df["Keyword"] == "data science").all()


def test_keyword_search_empty(make_client):
    client, _ = make_client([{"total": 0, "start": 0, "end": 0, "occupation": []}])
    df = keyword_search("u", "p", "zzzz", client=client)
    assert df.empty
    assert list(df.columns) == COLUMNS
