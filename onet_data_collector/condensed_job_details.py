"""Flatten raw O*NET occupation-detail JSON into a tidy CSV.

Raw O*NET detail documents are deeply nested and shape-inconsistent (a field can
be absent, a single object, or a list). :func:`condense_job_details` normalises
each occupation into a single flat row keyed by ``occupation_code``, joining
multi-valued fields with ``"; "`` so the result loads cleanly into pandas or a
spreadsheet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ._logging import get_logger
from .utils import as_list, dig

log = get_logger("condense")

JOIN = "; "


def _join(items: Any, *keys: str) -> str:
    """Join a collection's per-item text values with ``"; "``.

    For each item, the first present key in ``keys`` (supporting dotted paths
    like ``"title.name"``) is used. Plain string items are used as-is.
    """
    values: list[str] = []
    for item in as_list(items):
        value = None
        if isinstance(item, str):
            value = item
        else:
            for key in keys:
                value = dig(item, *key.split(".")) if "." in key else _get(item, key)
                if value:
                    break
        if value:
            values.append(str(value).strip())
    return JOIN.join(values)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else None


def _condense_one(job: dict[str, Any]) -> dict[str, Any]:
    occupation = job.get("occupation", {}) or {}
    tags = occupation.get("tags", {}) or {}
    return {
        "occupation_code": occupation.get("code", ""),
        "occupation_title": occupation.get("title", ""),
        "description": occupation.get("description", ""),
        "bright_outlook": bool(tags.get("bright_outlook", False)),
        "green": bool(tags.get("green", False)),
        "tasks": _join(dig(job, "tasks", "task"), "statement", "name"),
        "technology_skills": _join(dig(job, "technology_skills", "category"), "title.name", "title"),
        "tools_used": _join(dig(job, "tools_used", "category"), "title.name", "title"),
        "knowledge": _join(dig(job, "knowledge", "element"), "name"),
        "skills": _join(dig(job, "skills", "element"), "name"),
        "abilities": _join(dig(job, "abilities", "element"), "name"),
        "work_activities": _join(dig(job, "work_activities", "element"), "name"),
        "detailed_work_activities": _join(
            dig(job, "detailed_work_activities", "activity"), "name", "title"
        ),
        "work_context": _join(dig(job, "work_context", "element"), "name"),
        "job_zone": dig(job, "job_zone", "title", default=""),
        "education": _join(dig(job, "education", "level_required", "category"), "name"),
        "interests": _join(dig(job, "interests", "element"), "name"),
        "work_styles": _join(dig(job, "work_styles", "element"), "name"),
        "work_values": _join(dig(job, "work_values", "element"), "name"),
        "related_occupations": _join(dig(job, "related_occupations", "occupation"), "title"),
        "additional_information": _join(dig(job, "additional_information", "source"), "name"),
    }


def condense_job_details(input_json_path: str, output_csv_path: str) -> pd.DataFrame:
    """Flatten raw detail JSON into a CSV and return the resulting DataFrame.

    Args:
        input_json_path: Path to the raw JSON produced by
            :func:`onet_data_collector.job_details.fetch_job_details`.
        output_csv_path: Where to write the flattened CSV.

    Returns:
        The condensed DataFrame (one row per occupation).
    """
    path = Path(input_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    job_details = json.loads(path.read_text(encoding="utf-8"))
    rows = [_condense_one(job) for job in as_list(job_details)]

    df = pd.DataFrame(rows)
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Condensed %d occupations to %s", len(df), output_csv_path)
    return df
