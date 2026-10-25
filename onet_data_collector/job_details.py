"""Full occupation-detail retrieval.

Reads a CSV of occupation codes (produced by :mod:`keyword_search`), pulls the
full ``online/occupations/{code}/details`` document for each unique code, and
writes the raw JSON to disk. The raw layer is preserved verbatim so downstream
flattening can be re-run without re-hitting the API.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ._logging import get_logger
from .client import OnetClient
from .exceptions import OnetHTTPError

log = get_logger("job_details")


def _read_job_codes(input_csv_path: str, code_column: str) -> list[str]:
    path = Path(input_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    codes: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None or code_column not in reader.fieldnames:
            raise KeyError(
                f"Column '{code_column}' not found in {input_csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            code = (row.get(code_column) or "").strip()
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def fetch_job_details(
    username: str,
    password: str,
    input_csv_path: str,
    output_json_path: str,
    *,
    code_column: str = "Job Code",
    skip_errors: bool = True,
    client: OnetClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch full occupation details for every code in ``input_csv_path``.

    Args:
        input_csv_path: CSV containing a column of O*NET-SOC codes.
        output_json_path: Where to write the list of raw detail documents.
        code_column: Name of the column holding job codes.
        skip_errors: If ``True``, log and skip codes that fail instead of
            aborting the whole run.
        client: Optional pre-built :class:`OnetClient` to reuse.

    Returns:
        The list of raw detail documents that were successfully fetched.
    """
    onet = client or OnetClient(username, password)
    codes = _read_job_codes(input_csv_path, code_column)
    log.info("Fetching details for %d unique occupation codes.", len(codes))

    details: list[dict[str, Any]] = []
    for i, job_code in enumerate(codes, start=1):
        log.info("[%d/%d] Fetching details for %s", i, len(codes), job_code)
        try:
            details.append(onet.request(f"online/occupations/{job_code}/details"))
        except OnetHTTPError as exc:
            if skip_errors:
                log.warning("Skipping %s: %s", job_code, exc)
                continue
            raise

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    log.info("Wrote %d detail records to %s", len(details), output_json_path)
    return details
