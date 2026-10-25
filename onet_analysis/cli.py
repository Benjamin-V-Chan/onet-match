"""Command-line interface (``onet-match``; ``onet-analyze`` is an alias).

    onet-analyze collect "healthcare" "technology"        # run the collector ETL
    onet-analyze eda                                       # coverage + distribution plots
    onet-analyze keywords                                  # keyword -> field attribution
    onet-analyze cluster --clusters 6                      # role clustering
    onet-analyze all "healthcare" "technology"            # collect + every analysis
    onet-analyze match "ran the robotics club's CAD team"  # free text -> O*NET occupations
    onet-analyze serve                                     # HTTP API (FastAPI)
    onet-analyze matching build-data | eval | train-biencoder | train-reranker | bundle | ...

Paths default to ``<cwd>/data`` and ``<cwd>/results`` and can be relocated with
``--base``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .clustering import cluster_occupations, plot_clusters
from .eda import run_eda
from .keyword_attribution import analyze_keywords, plot_attribution
from .loading import load_csv
from .paths import default_paths

log = logging.getLogger("onet_analysis.cli")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep third-party libraries from flooding the console at INFO.
    for noisy in ("matplotlib", "httpx", "httpx2", "urllib3", "huggingface_hub", "sentence_transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onet-match", description=__doc__.split("\n", 1)[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--base", default=".", help="Base directory for data/ and results/.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Run the collector ETL (search/details/condense).")
    p_collect.add_argument("keywords", nargs="*", help="Search terms (defaults to a starter set).")
    p_collect.add_argument("--username")
    p_collect.add_argument("--password")
    p_collect.add_argument("--max-results", type=int, default=None)

    sub.add_parser("eda", help="Coverage and distribution analysis of keyword results.")
    sub.add_parser("keywords", help="Keyword-to-field attribution over condensed details.")

    p_cluster = sub.add_parser("cluster", help="Cluster occupations from their text.")
    p_cluster.add_argument("--clusters", type=int, default=5)

    p_all = sub.add_parser("all", help="Collect then run every analysis.")
    p_all.add_argument("keywords", nargs="*")
    p_all.add_argument("--username")
    p_all.add_argument("--password")
    p_all.add_argument("--max-results", type=int, default=None)
    p_all.add_argument("--clusters", type=int, default=5)

    # Semantic matching ------------------------------------------------------
    p_match = sub.add_parser("match", help="Match free text to the closest O*NET occupations.")
    p_match.add_argument("text", nargs="+", help="Any text about work (title, posting, experience, ...).")
    p_match.add_argument("--top-k", type=int, default=5)
    p_match.add_argument("--bundle", default=None, help="Bundle dir (default: $ONET_MATCH_BUNDLE or models/bundle).")
    p_match.add_argument("--json", action="store_true", help="Print the full JSON result.")

    p_serve = sub.add_parser("serve", help="Run the matching HTTP API.")
    p_serve.add_argument("--bundle", default=None)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_m = sub.add_parser("matching", help="Build data, train, evaluate and package matching models.")
    msub = p_m.add_subparsers(dest="matching_command", required=True)

    p = msub.add_parser("build-data", help="Download O*NET bulk data; build catalog + labeled pairs.")
    p.add_argument("--release", default=None, help="O*NET release, e.g. 31.0 (default: latest).")
    p.add_argument("--seed", type=int, default=13)

    p = msub.add_parser("eval", help="Benchmark a model spec on the fixed eval sets.")
    p.add_argument("spec", help="tfidf | bm25 | dense=<model> [+rerank=<model>]")
    p.add_argument("--run-name", default=None)
    p.add_argument("--fields", default="title+desc+tasks",
                   choices=["title", "title+desc", "title+desc+tasks", "title+desc+tasks+skills"])
    p.add_argument("--no-latency", action="store_true")

    p = msub.add_parser("train-biencoder", help="Contrastive fine-tuning (MNRL, optional hard negatives).")
    p.add_argument("--name", required=True, help="Output dir name under models/.")
    p.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--sources", default="title,task,dwa,composite",
                   help="Comma-separated training sources (title,task,dwa,composite,synthetic).")
    p.add_argument("--no-hard-negatives", action="store_true")
    p.add_argument("--fraction", type=float, default=1.0, help="Subsample training pairs (learning curves).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=-1, help="Stop early (timing pilots).")
    p.add_argument("--eval", action="store_true", help="Evaluate after training (run name = --name).")

    p = msub.add_parser("train-reranker", help="Fine-tune a cross-encoder over bi-encoder candidates.")
    p.add_argument("--name", required=True)
    p.add_argument("--biencoder", required=True, help="Path to the fine-tuned bi-encoder.")
    p.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    p.add_argument("--sources", default="title,task,dwa,composite")
    p.add_argument("--queries", type=int, default=30000)
    p.add_argument("--negatives", type=int, default=4, help="Hard negatives per query.")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval", action="store_true")

    p = msub.add_parser("tune-fusion", help="Choose the reranker blend weight on validation data.")
    p.add_argument("spec")
    p.add_argument("--eval-as", default=None, metavar="RUN", help="Then evaluate the tuned spec as this run.")

    p = msub.add_parser("bundle", help="Package a model spec for serving.")
    p.add_argument("spec")
    p.add_argument("--run", required=True, help="Benchmark run whose fitted temperature to use.")
    p.add_argument("--out", default=None, help="Output dir (default: models/bundle).")

    p = msub.add_parser("synth", help="Generate LLM paraphrased messy inputs (needs an API key env var).")
    p.add_argument("--role", choices=["train", "eval"], required=True)
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai-compatible", "mock"])
    p.add_argument("--model", required=True)
    p.add_argument("--pilot", type=int, default=None, help="Only N random occupations (pilot run).")
    p.add_argument("--per-type", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint URL.")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (non-Claude).")
    p.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens (non-Claude).")

    p = msub.add_parser("lookup", help="Keyword search of O*NET titles.")
    p.add_argument("query", nargs="+")
    msub.add_parser("report", help="Render docs/RESULTS.md from outputs/runs.")

    return parser


def _run_collect(args, paths) -> None:
    from .collect import DEFAULT_KEYWORDS, collect_all

    keywords = args.keywords or DEFAULT_KEYWORDS
    collect_all(keywords, paths, username=args.username, password=args.password,
                max_results=args.max_results)


def _run_eda(paths) -> None:
    data = load_csv(paths.keyword_search_csv)
    result = run_eda(data, paths.results, paths.processed)
    log.info("SOC coverage:\n%s", result.soc_code_coverage.to_string(index=False))
    _write_summary(paths, result)


def _run_keywords(paths) -> None:
    keyword_df = load_csv(paths.keyword_search_csv)
    condensed_df = load_csv(paths.condensed_csv)
    result = analyze_keywords(keyword_df, condensed_df)
    paths.data_results.mkdir(parents=True, exist_ok=True)
    result.to_csv(paths.keyword_analysis_csv)
    plot_attribution(result, paths.data_results)
    log.info("Keyword attribution written to %s", paths.keyword_analysis_csv)


def _run_cluster(paths, n_clusters) -> None:
    condensed_df = load_csv(paths.condensed_csv, required=False)
    df = condensed_df if condensed_df is not None else load_csv(paths.keyword_search_csv)
    result = cluster_occupations(df, n_clusters=n_clusters)
    assignments_path = paths.processed / "cluster_assignments.csv"
    paths.processed.mkdir(parents=True, exist_ok=True)
    result.assignments.to_csv(assignments_path, index=False)
    plot_clusters(result, paths.results / "occupation_clusters.png")
    log.info("Clustering done (k=%d, silhouette=%s).", result.n_clusters, result.silhouette)
    for cid, terms in sorted(result.top_terms.items()):
        log.info("Cluster %d: %s", cid, ", ".join(terms) or "(no distinctive terms)")


def _write_summary(paths, eda_result) -> None:
    summary = paths.results / "keyword_search_evaluation_summary.txt"
    lines = [
        "Keyword Search Evaluation Summary",
        "=" * 33,
        "",
        "SOC Code Coverage:",
        eda_result.soc_code_coverage.to_string(index=False),
        "",
        f"Plots written: {', '.join(p.name for p in eda_result.plots)}",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")


def _print_match(result) -> None:
    print(f"Query: {result.query}")
    for m in result.matches:
        print(f"  {m.rank}. {m.title} ({m.soc})  confidence={m.confidence:.2f}  [{m.major_group_title}]")
        for t in m.top_tasks[:2]:
            print(f"       - {t}")
    prof = result.profile
    print("  Skills:    " + ", ".join(d["name"] for d in prof["skills"][:6]))
    print("  Knowledge: " + ", ".join(d["name"] for d in prof["knowledge"][:6]))
    print(f"  Interests: {prof['riasec_code']} (" + ", ".join(
        f"{d['name']} {d['weight']:.1f}" for d in prof["interests"][:3]) + ")")


def _run_match(args) -> None:
    import json

    from .matching.engine import default_engine

    result = default_engine(args.bundle).match(" ".join(args.text), top_k=args.top_k)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_match(result)


def _run_serve(args) -> None:
    import os

    import uvicorn

    if args.bundle:
        os.environ["ONET_MATCH_BUNDLE"] = args.bundle
    from .matching.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


def _run_matching(args, paths) -> None:
    import json

    from .matching import pipeline as P

    cmd = args.matching_command
    if cmd == "build-data":
        print(json.dumps(P.build_data(paths, release=args.release, seed=args.seed), indent=2))
    elif cmd == "eval":
        P.evaluate(paths, args.spec, run_name=args.run_name, fields=args.fields, latency=not args.no_latency)
    elif cmd == "train-biencoder":
        out = P.train_biencoder(paths, base=args.base_model, name=args.name, sources=args.sources.split(","),
                                hard_negatives=not args.no_hard_negatives, fraction=args.fraction,
                                epochs=args.epochs, batch_size=args.batch_size, max_steps=args.max_steps)
        if args.eval:
            P.evaluate(paths, f"dense={out}", run_name=args.name)
    elif cmd == "train-reranker":
        out = P.train_reranker(paths, biencoder=args.biencoder, base=args.base_model, name=args.name,
                               sources=args.sources.split(","), n_queries=args.queries, n_negatives=args.negatives,
                               epochs=args.epochs,
                               batch_size=args.batch_size, max_steps=args.max_steps)
        if args.eval:
            P.evaluate(paths, f"dense={args.biencoder}+rerank={out}", run_name=args.name)
    elif cmd == "tune-fusion":
        fusion = P.tune_fusion(paths, args.spec)
        print(json.dumps({k: v for k, v in fusion.items() if k != "grid"}, indent=2))
        if args.eval_as:
            P.evaluate(paths, fusion["tuned_spec"], run_name=args.eval_as)
    elif cmd == "bundle":
        out = P.bundle(paths, args.spec, run_name=args.run, out=args.out and __import__("pathlib").Path(args.out))
        log.info("Bundle written to %s", out)
    elif cmd == "synth":
        report = P.synthesize(paths, role=args.role, provider=args.provider, model=args.model, pilot=args.pilot,
                              n_per_type=args.per_type, workers=args.workers, base_url=args.base_url,
                              api_key_env=args.api_key_env, price_in=args.price_in, price_out=args.price_out)
        print(json.dumps(report, indent=2))
    elif cmd == "lookup":
        print(P.lookup(paths, " ".join(args.query)).to_string(index=False))
    elif cmd == "report":
        from .matching.report import render_report

        log.info("Report written to %s", render_report(paths))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    paths = default_paths(args.base)
    try:
        if args.command == "collect":
            _run_collect(args, paths)
        elif args.command == "eda":
            _run_eda(paths)
        elif args.command == "keywords":
            _run_keywords(paths)
        elif args.command == "cluster":
            _run_cluster(paths, args.clusters)
        elif args.command == "all":
            _run_collect(args, paths)
            _run_eda(paths)
            _run_keywords(paths)
            _run_cluster(paths, args.clusters)
        elif args.command == "match":
            _run_match(args)
        elif args.command == "serve":
            _run_serve(args)
        elif args.command == "matching":
            _run_matching(args, paths)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
