"""Keyword-driven occupation discovery.

Given one or more search terms, query the O*NET ``online/search`` endpoint and
return a tidy :class:`pandas.DataFrame` of matching occupations. This is the
entry point of the pipeline: it turns human language ("data scientist",
"nursing") into formal O*NET occupation codes for downstream detail pulls.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ._logging import get_logger
from .client import OnetClient

log = get_logger("keyword_search")

# Column order for the returned DataFrame (stable across runs).
COLUMNS = ["Keyword", "Job Code", "Job Title", "SOC Code", "Relevance Score"]


def _rows_for_keyword(onet: OnetClient, keyword: str, max_results: int | None) -> list[dict]:
    log.info("Searching O*NET for keyword: %s", keyword)
    occupations = onet.paginate(
        "online/search",
        ("keyword", keyword),
        item_key="occupation",
        page_size=50,
        max_items=max_results,
    )
    rows = []
    for occ in occupations:
        job_code = occ.get("code", "")
        rows.append(
            {
                "Keyword": keyword,
                "Job Code": job_code,
                "Job Title": occ.get("title", ""),
                # SOC major group = first two digits of the O*NET-SOC code.
                "SOC Code": job_code.split("-")[0] if job_code else "",
                "Relevance Score": occ.get("relevance_score"),
            }
        )
    log.info("Found %d occupations for '%s'.", len(rows), keyword)
    return rows


def keyword_search(
    username: str,
    password: str,
    keyword: str,
    *,
    max_results: int | None = None,
    client: OnetClient | None = None,
) -> pd.DataFrame:
    """Search O*NET for a single keyword and return matching occupations.

    Args:
        username: O*NET Web Services username.
        password: O*NET Web Services password.
        keyword: The search term.
        max_results: Optional cap on the number of occupations returned.
            ``None`` collects every result across all pages.
        client: An existing :class:`OnetClient` to reuse (avoids re-auth when
            searching many keywords). If ``None`` one is created.

    Returns:
        A DataFrame with columns: ``Keyword``, ``Job Code``, ``Job Title``,
        ``SOC Code``, ``Relevance Score``.
    """
    onet = client or OnetClient(username, password)
    if client is None:
        about = onet.about()
        log.info("Connected to O*NET Web Services version %s", about.get("api_version", "?"))
    rows = _rows_for_keyword(onet, keyword, max_results)
    return pd.DataFrame(rows, columns=COLUMNS)


def keyword_search_many(
    username: str,
    password: str,
    keywords: Iterable[str],
    *,
    max_results: int | None = None,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Search many keywords with a single authenticated client.

    Args:
        keywords: Iterable of search terms.
        max_results: Optional per-keyword cap.
        drop_duplicates: If ``True``, keep only the first (keyword, job code)
            pair for each job code across the whole result set.

    Returns:
        A concatenated DataFrame across all keywords.
    """
    onet = OnetClient(username, password)
    about = onet.about()
    log.info("Connected to O*NET Web Services version %s", about.get("api_version", "?"))

    frames = [
        keyword_search(username, password, kw, max_results=max_results, client=onet)
        for kw in keywords
    ]
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=COLUMNS)
    )
    if drop_duplicates:
        combined = combined.drop_duplicates(subset="Job Code").reset_index(drop=True)
    return combined
