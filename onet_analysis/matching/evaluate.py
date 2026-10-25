"""Benchmark harness: identical eval sets, metrics, calibration and latency for every model.

Each call to :func:`run_eval` writes ``outputs/runs/<run_name>/``:

* ``config.json``  — model/ranker config, data provenance, temperature
* ``metrics.json`` — per eval set: overall + per input type (top-1/top-5/MRR@k,
  major-group top-1, ECE, reliability bins), validation metrics, latency
* ``predictions_<set>.parquet`` — per-query top-1/top-5, correctness, confidence

All numbers in the write-up are rendered from these files
(:mod:`onet_analysis.matching.report`).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .metrics import fit_temperature, gold_hits, per_query_frame, summarize
from .retrieval import Ranker

log = logging.getLogger("onet_analysis.matching.evaluate")

REAL_SOURCES = ("title", "task", "dwa", "composite")
LATENCY_QUERIES = 200


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - not a git checkout
        return None


def build_eval_sets(pairs: pd.DataFrame, extra: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """The fixed benchmark: real held-out test rows (+ optional synthetic set).

    ``title_hard`` is reported as a slice of ``real_test`` (same predictions).
    """
    sets = {"real_test": pairs[(pairs["split"] == "test") & pairs["source"].isin(REAL_SOURCES)]}
    for name, df in (extra or {}).items():
        if df is not None and len(df):
            sets[name] = df
    return {k: v.reset_index(drop=True) for k, v in sets.items()}


def _set_metrics(frame: pd.DataFrame) -> dict:
    out = {"overall": summarize(frame),
           "by_input_type": {t: summarize(g) for t, g in frame.groupby("input_type")}}
    if "hard" in frame and frame["hard"].any():
        out["title_hard"] = summarize(frame[frame["hard"].astype(bool)])
    return out


def measure_latency(ranker: Ranker, queries: list[str], n: int = LATENCY_QUERIES, seed: int = 0) -> dict:
    """Single-query wall-clock latency (ms) — the interactive serving case."""
    rng = np.random.default_rng(seed)
    sample = [queries[i] for i in rng.choice(len(queries), size=min(n, len(queries)), replace=False)]
    ranker.rank(sample[:3])  # warm-up (lazy init, kernel compilation)
    times = []
    for q in sample:
        t0 = time.perf_counter()
        ranker.rank([q])
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"n": len(t), "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95))}


def run_eval(
    run_name: str,
    ranker: Ranker,
    socs: list[str],
    pairs: pd.DataFrame,
    out_root: str | Path,
    *,
    extra_sets: dict[str, pd.DataFrame] | None = None,
    config: dict | None = None,
    val_limit: int | None = None,
    latency: bool = True,
    ranker_for_val: Callable[[], Ranker] | None = None,
) -> dict:
    """Evaluate ``ranker`` on the fixed benchmark and persist the run.

    Args:
        run_name: Directory name under ``out_root``.
        ranker: Model to evaluate (its documents must be in ``socs`` order).
        socs: Occupation codes aligned with ``ranker.docs``.
        pairs: Real-data pairs with splits (val is used to fit the temperature).
        out_root: Usually ``outputs/runs``.
        extra_sets: Additional eval sets, e.g. ``{"synthetic_test": df, "gold": df}``.
        config: Extra metadata stored in ``config.json``.
        val_limit: Optional cap on validation rows (stratified sample) for speed.
        latency: Measure single-query latency.
    """
    out_dir = Path(out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    val = pairs[(pairs["split"] == "val") & pairs["source"].isin(REAL_SOURCES)]
    if val_limit and len(val) > val_limit:
        val = val.groupby("source").sample(frac=val_limit / len(val), random_state=0)
    val = val.reset_index(drop=True)

    t0 = time.perf_counter()
    val_rank = ranker.rank(val["text"].tolist())
    hits = gold_hits(val_rank, socs, [g.split(";") for g in val["gold"]])
    temperature = fit_temperature(val_rank.scores, hits)
    val_frame = per_query_frame(val_rank, socs, val, temperature)
    log.info("[%s] val: top1=%.3f  T=%.4g  (%.1fs)", run_name, val_frame["hit1"].mean(), temperature,
             time.perf_counter() - t0)

    metrics: dict = {"run": run_name, "temperature": temperature, "val": _set_metrics(val_frame), "sets": {}}
    sets = build_eval_sets(pairs, extra_sets)
    timing = {}
    for name, data in sets.items():
        t0 = time.perf_counter()
        ranking = ranker.rank(data["text"].tolist())
        elapsed = time.perf_counter() - t0
        timing[name] = {"queries": len(data), "seconds": elapsed, "queries_per_second": len(data) / elapsed}
        frame = per_query_frame(ranking, socs, data, temperature)
        frame.to_parquet(out_dir / f"predictions_{name}.parquet", index=False)
        metrics["sets"][name] = _set_metrics(frame)
        o = metrics["sets"][name]["overall"]
        log.info("[%s] %s: n=%d top1=%.3f top5=%.3f mrr=%.3f ece=%.3f", run_name, name, o["n"],
                 o["top1"], o["top5"], o["mrr"], o["ece"])
    metrics["throughput"] = timing
    if latency:
        metrics["latency"] = measure_latency(ranker, sets["real_test"]["text"].tolist())
        log.info("[%s] latency p50=%.1fms p95=%.1fms", run_name, metrics["latency"]["p50_ms"],
                 metrics["latency"]["p95_ms"])

    meta = {
        "run": run_name,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "ranker": ranker.config(),
        "temperature": temperature,
        "val_rows": len(val),
        "git_rev": _git_rev(Path(__file__).resolve().parent),
        "platform": f"{platform.platform()} / {platform.processor() or platform.machine()}",
        **(config or {}),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_run(run_dir: str | Path) -> tuple[dict, dict]:
    run_dir = Path(run_dir)
    return (json.loads((run_dir / "config.json").read_text()),
            json.loads((run_dir / "metrics.json").read_text()))
