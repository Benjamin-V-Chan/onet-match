"""Tests for the robust flattening in condensed_job_details."""

from __future__ import annotations

import json

import pandas as pd

from onet_data_collector.condensed_job_details import condense_job_details


def _write(tmp_path, data):
    p = tmp_path / "details.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_condense_full_record(tmp_path):
    data = [
        {
            "occupation": {
                "code": "15-2051.00",
                "title": "Data Scientists",
                "description": "Analyze data.",
                "tags": {"bright_outlook": True, "green": False},
            },
            "tasks": {"task": [{"statement": "Build models"}, {"statement": "Clean data"}]},
            "technology_skills": {"category": [{"title": {"name": "Python"}}]},
            "skills": {"element": [{"name": "Programming"}, {"name": "Mathematics"}]},
            "job_zone": {"title": "Job Zone Five"},
            "related_occupations": {"occupation": [{"title": "Statisticians"}]},
        }
    ]
    out = tmp_path / "condensed.csv"
    df = condense_job_details(_write(tmp_path, data), str(out))

    row = df.iloc[0]
    assert row["occupation_code"] == "15-2051.00"
    assert row["bright_outlook"] is True or row["bright_outlook"] == True  # noqa: E712
    assert row["tasks"] == "Build models; Clean data"
    assert row["technology_skills"] == "Python"
    assert row["skills"] == "Programming; Mathematics"
    assert row["job_zone"] == "Job Zone Five"
    assert row["related_occupations"] == "Statisticians"
    assert out.is_file()


def test_condense_handles_single_dict_not_list(tmp_path):
    # O*NET sometimes returns a single object instead of a list.
    data = [
        {
            "occupation": {"code": "x", "title": "T"},
            "skills": {"element": {"name": "Solo Skill"}},
        }
    ]
    df = condense_job_details(_write(tmp_path, data), str(tmp_path / "o.csv"))
    assert df.iloc[0]["skills"] == "Solo Skill"


def test_condense_handles_missing_fields(tmp_path):
    data = [{"occupation": {"code": "x", "title": "T"}}]
    df = condense_job_details(_write(tmp_path, data), str(tmp_path / "o.csv"))
    row = df.iloc[0]
    assert row["skills"] == ""
    assert row["tasks"] == ""
    assert row["bright_outlook"] == False  # noqa: E712


def test_condense_empty_list(tmp_path):
    df = condense_job_details(_write(tmp_path, []), str(tmp_path / "o.csv"))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
