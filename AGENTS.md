# Notes for coding agents

## Using onet-semantic-search from another project

- `pip install "onet-semantic-search[matching] @ git+https://github.com/Benjamin-V-Chan/onet-semantic-search"`
- `from onet_match import match`. Then `match(text, top_k=5)` returns a
  `MatchResult` with `.matches` (a list of `OccupationMatch`) and `.profile`.
  `.to_dict()` gives JSON-ready output.
- The first call downloads the model bundle (about 0.5 GB) from the Hugging
  Face Hub. Set `ONET_MATCH_BUNDLE` to a local bundle directory to stay
  offline.
- Load the engine once and reuse it: `default_engine()` caches it, and each
  query takes about 50 ms on CPU or MPS after loading.
- Service integration: `onet-match serve` exposes the HTTP contract in
  `docs/API.md`.

## Working in this repo

- Setup: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest -q`. They run fully offline, using tiny
  local models and a mocked LLM.
- Packages:
  - `onet_match/`: public import alias.
  - `onet_analysis/matching/`: catalog, datasets, retrieval, training,
    evaluation, serving, API and report.
  - `onet_data_collector/`: O*NET bulk download and Web Services client.
  - `onet_analysis/` (top level): classic EDA, clustering and keyword
    attribution.
- CLI entry point: `onet_analysis/cli.py` (`onet-match`, alias `onet-analyze`).
  The collector's is `onet_data_collector/cli.py` (`onet-collect`).
- Results: every number in the README or docs must come from a saved run in
  `outputs/runs/<run>/metrics.json`. `onet-match matching report` regenerates
  `docs/RESULTS.md`, so don't edit it by hand.
- `docs/API.md` and `docs/SYNTHETIC_DATA.md` are checked against the code by
  `tests/test_matching_docs.py`.
- Gitignored: `data/` (built by `onet-match matching build-data`), `models/`
  and `notes/`.
