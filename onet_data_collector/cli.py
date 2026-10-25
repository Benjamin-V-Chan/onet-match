"""Command-line interface for onet_data_collector.

Exposes the pipeline as the ``onet-collect`` console script:

    onet-collect search "data scientist" "nursing" -o data/raw/keywords.csv
    onet-collect details data/raw/keywords.csv -o data/raw/job_details.json
    onet-collect condense data/raw/job_details.json -o data/processed/condensed.csv
    onet-collect pipeline "data scientist" "nursing" --out-dir data
    onet-collect bulk download                      # full O*NET database (no credentials)
    onet-collect bulk info

Credentials come from ``--username/--password``, the ``ONET_USERNAME`` /
``ONET_PASSWORD`` environment variables, or a ``.env`` file. Missing values are
prompted for interactively when run in a terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from ._logging import configure, get_logger
from .bulk import cached_releases, download_bulk, load_bulk
from .condensed_job_details import condense_job_details
from .config import resolve_credentials
from .exceptions import OnetError
from .job_details import fetch_job_details
from .keyword_search import keyword_search_many

log = get_logger("cli")


def _add_credential_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--username", help="O*NET username (or set ONET_USERNAME).")
    parser.add_argument("--password", help="O*NET password (or set ONET_PASSWORD).")
    parser.add_argument("--env-file", default=".env", help="Path to a .env file (default: .env).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onet-collect", description=__doc__.split("\n", 1)[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p_search = sub.add_parser("search", help="Keyword-search O*NET for occupations.")
    p_search.add_argument("keywords", nargs="+", help="One or more search terms.")
    p_search.add_argument("-o", "--output", default="data/raw/keyword_search_results.csv")
    p_search.add_argument("--max-results", type=int, default=None, help="Per-keyword cap.")
    p_search.add_argument("--dedupe", action="store_true", help="Drop duplicate job codes.")
    _add_credential_args(p_search)

    # details
    p_details = sub.add_parser("details", help="Fetch full occupation detail JSON.")
    p_details.add_argument("input_csv", help="CSV of occupation codes from `search`.")
    p_details.add_argument("-o", "--output", default="data/raw/job_details.json")
    p_details.add_argument("--code-column", default="Job Code")
    _add_credential_args(p_details)

    # condense
    p_condense = sub.add_parser("condense", help="Flatten raw detail JSON to CSV.")
    p_condense.add_argument("input_json", help="Raw JSON from `details`.")
    p_condense.add_argument("-o", "--output", default="data/processed/condensed_job_details.csv")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Run search -> details -> condense end to end.")
    p_pipe.add_argument("keywords", nargs="+", help="One or more search terms.")
    p_pipe.add_argument("--out-dir", default="data", help="Base output directory.")
    p_pipe.add_argument("--max-results", type=int, default=None)
    _add_credential_args(p_pipe)

    # bulk database download (no credentials needed)
    p_bulk = sub.add_parser("bulk", help="Download / inspect the full O*NET database release.")
    bulk_sub = p_bulk.add_subparsers(dest="bulk_command", required=True)
    p_dl = bulk_sub.add_parser("download", help="Download and cache a release.")
    p_dl.add_argument("--release", default=None, help="e.g. 31.0 (default: latest).")
    p_dl.add_argument("--cache-dir", default=None, help="Cache root (default: $ONET_BULK_DIR or ~/.cache).")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if cached.")
    p_info = bulk_sub.add_parser("info", help="Show cached releases and their manifest.")
    p_info.add_argument("--release", default=None)
    p_info.add_argument("--cache-dir", default=None)

    return parser


def _creds(args: argparse.Namespace) -> tuple[str, str]:
    return resolve_credentials(
        getattr(args, "username", None),
        getattr(args, "password", None),
        dotenv_path=getattr(args, "env_file", ".env"),
        allow_prompt=True,
    )


def _cmd_search(args: argparse.Namespace) -> None:
    username, password = _creds(args)
    df = keyword_search_many(
        username, password, args.keywords,
        max_results=args.max_results, drop_duplicates=args.dedupe,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Wrote %d rows to %s", len(df), args.output)


def _cmd_details(args: argparse.Namespace) -> None:
    username, password = _creds(args)
    fetch_job_details(username, password, args.input_csv, args.output, code_column=args.code_column)


def _cmd_condense(args: argparse.Namespace) -> None:
    condense_job_details(args.input_json, args.output)


def _cmd_pipeline(args: argparse.Namespace) -> None:
    username, password = _creds(args)
    out = Path(args.out_dir)
    keywords_csv = out / "raw" / "keyword_search_results.csv"
    details_json = out / "raw" / "job_details.json"
    condensed_csv = out / "processed" / "condensed_job_details.csv"

    df = keyword_search_many(username, password, args.keywords, max_results=args.max_results)
    keywords_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(keywords_csv, index=False)
    log.info("Stage 1/3 complete: %d occupations -> %s", len(df), keywords_csv)

    fetch_job_details(username, password, str(keywords_csv), str(details_json))
    log.info("Stage 2/3 complete: raw details -> %s", details_json)

    condense_job_details(str(details_json), str(condensed_csv))
    log.info("Stage 3/3 complete: condensed table -> %s", condensed_csv)


def _cmd_bulk(args: argparse.Namespace) -> None:
    if args.bulk_command == "download":
        release = download_bulk(args.release, cache_dir=args.cache_dir, force=args.force)
    else:
        log.info("Cached releases: %s", ", ".join(cached_releases(args.cache_dir)) or "(none)")
        release = load_bulk(args.release, cache_dir=args.cache_dir)
    m = release.manifest
    print(f"O*NET {m['release']} at {release.path}")
    print(f"  source:     {m['source_url']}")
    print(f"  downloaded: {m['downloaded_at']}")
    print(f"  sha256:     {m['sha256']}")
    print(f"  tables:     {', '.join(release.available())}")


_COMMANDS = {
    "search": _cmd_search,
    "details": _cmd_details,
    "condense": _cmd_condense,
    "pipeline": _cmd_pipeline,
    "bulk": _cmd_bulk,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure("DEBUG" if args.verbose else "INFO")
    try:
        _COMMANDS[args.command](args)
    except OnetError as exc:
        log.error("%s", exc)
        return 1
    except (FileNotFoundError, KeyError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
