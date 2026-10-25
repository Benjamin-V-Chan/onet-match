"""Ranking metrics and confidence calibration.

Confidence = softmax over the top-k candidate scores with a single temperature
``T`` fitted on the validation split (temperature scaling, Guo et al. 2017).
Because gold labels can be sets, the fitted objective is the negative log of
the total probability assigned to gold candidates. The reported confidence of a
match is its softmax probability; ECE and the reliability curve are computed on
the top-1 confidence vs. top-1 correctness.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .catalog import major_group
from .retrieval import Ranking, log_space, softmax

ECE_BINS = 10


def gold_hits(ranking: Ranking, socs: Sequence[str], golds: Sequence[Sequence[str]]) -> np.ndarray:
    """Boolean (n, k): candidate j of query i is a gold occupation."""
    soc_arr = np.asarray(socs)
    cand = soc_arr[ranking.indices]
    return np.array([[c in set(g) for c in row] for row, g in zip(cand, golds)], dtype=bool)


def fit_temperature(scores: np.ndarray, hits: np.ndarray) -> float:
    """Grid-search the temperature minimising gold NLL (queries with a gold in top-k).

    The grid is relative to the score spread, so it works for cosine
    similarities (~0.01 apart) and cross-encoder logits (~several units) alike.
    """
    mask = hits.any(axis=1)
    if not mask.any():
        return 1.0
    s, h = scores[mask], hits[mask]
    spread = float(np.median(s[:, 0] - s[:, -1])) or 1.0
    best_t, best_nll = 1.0, np.inf
    for t in log_space(spread / 200, spread * 20, 400):
        p = softmax(s, t)
        nll = -np.log(np.clip((p * h).sum(axis=1), 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> tuple[float, list[dict]]:
    """Expected calibration error and per-bin (confidence, accuracy, count)."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece, n = [], 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        acc, c = float(correct[m].mean()), float(conf[m].mean())
        ece += m.sum() / n * abs(acc - c)
        bins.append({"lo": float(lo), "hi": float(hi), "confidence": c, "accuracy": acc, "count": int(m.sum())})
    return float(ece), bins


def per_query_frame(
    ranking: Ranking, socs: Sequence[str], data: pd.DataFrame, temperature: float
) -> pd.DataFrame:
    """One row per query: correctness flags, reciprocal rank, top-1 confidence."""
    golds = [str(g).split(";") for g in data["gold"]]
    hits = gold_hits(ranking, socs, golds)
    first = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
    soc_arr = np.asarray(socs)
    top1 = soc_arr[ranking.indices[:, 0]]
    probs = softmax(ranking.scores, temperature)
    out = data[["id", "text", "source", "input_type", "gold"]].copy().reset_index(drop=True)
    if "hard" in data:
        out["hard"] = data["hard"].to_numpy()
    out["top1"] = top1
    out["top5"] = [";".join(soc_arr[row[:5]]) for row in ranking.indices]
    out["hit1"] = first == 1
    out["hit5"] = (first >= 1) & (first <= 5)
    out["rr"] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    out["major_hit1"] = [major_group(t) in {major_group(g) for g in gs} for t, gs in zip(top1, golds)]
    out["confidence"] = probs[:, 0]
    return out


def summarize(frame: pd.DataFrame) -> dict:
    """Aggregate metrics for a set of per-query rows."""
    if frame.empty:
        return {"n": 0}
    ece, bins = reliability(frame["confidence"].to_numpy(), frame["hit1"].to_numpy())
    return {
        "n": int(len(frame)),
        "top1": float(frame["hit1"].mean()),
        "top5": float(frame["hit5"].mean()),
        "mrr": float(frame["rr"].mean()),
        "major_top1": float(frame["major_hit1"].mean()),
        "mean_confidence": float(frame["confidence"].mean()),
        "ece": ece,
        "reliability": bins,
    }
