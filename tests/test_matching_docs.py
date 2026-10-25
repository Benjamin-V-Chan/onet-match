"""Docs that describe code must stay in sync with it."""

from __future__ import annotations

from pathlib import Path

from onet_analysis.matching import synth as S

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_synthetic_prompt_doc_matches_code():
    doc = (DOCS / "SYNTHETIC_DATA.md").read_text(encoding="utf-8")
    assert f"version `{S.PROMPT_VERSION}`" in doc
    assert S.SYSTEM_PROMPT in doc
    for itype, desc in S.INPUT_TYPES.items():
        assert f'"{itype}": {desc}' in doc
    rules = S.render_prompt("{title}", "d", [], [], 1).split("Rules:\n", 1)[1]
    for line in rules.splitlines():
        if line.startswith("- Do not use") or line.startswith("- Return"):
            continue  # contain the rendered title / n
        assert line in doc


def test_api_doc_lists_every_route():
    from onet_analysis.matching.api import create_app

    doc = (DOCS / "API.md").read_text(encoding="utf-8")
    routes = {r.path for r in create_app(engine=object()).routes if r.path.startswith("/") and
              not r.path.startswith(("/docs", "/openapi", "/redoc"))}
    for path in routes:
        assert f"`{path.split()[0]}" in doc or path in doc, path
