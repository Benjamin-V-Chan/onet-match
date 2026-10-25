"""Labeled (text -> O*NET-SOC) datasets built from real O*NET data, plus splits.

Every row has a *set* of gold codes (``gold``, ``;``-joined): 6k+ title strings
(and many generic tasks / work activities) legitimately belong to more than one
occupation, so a prediction is correct when it hits any gold code.

Sources and the input type they stand in for:

==============  ===============  ============================================
source          input_type       content
==============  ===============  ============================================
``title``       ``job_title``    O*NET Job Titles + Sample of Reported Titles
``task``        ``task``         Task Statements
``dwa``         ``activity``     Detailed Work Activities (via Tasks to DWAs)
``composite``   ``description``  2-4 tasks of one occupation joined together
==============  ===============  ============================================

Splits (``train`` / ``val`` / ``test``) are assigned per *unique normalised
text*, stratified by its primary occupation: each occupation contributes ~10%
of its titles and of its tasks to ``val`` and ~10% to ``test``. Composites are
sampled only from tasks in their own split, and evaluation representations
exclude held-out tasks (:func:`heldout_task_ids`), so no test text is ever
visible to a model as training data or as part of an occupation document.

The **hard** title set is the subset of test titles that share no content token
with the official title of any of their gold occupations (e.g. "Sr. Backend
Eng II" vs. "Software Developers").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import OccupationCatalog
from .text import normalize, shares_no_tokens

log = logging.getLogger("onet_analysis.matching.datasets")

SOURCE_INPUT_TYPE = {
    "title": "job_title",
    "task": "task",
    "dwa": "activity",
    "composite": "description",
}
SPLIT_FRACTIONS = {"val": 0.10, "test": 0.10}
COMPOSITES_PER_OCC = {"train": 10, "val": 2, "test": 2}
MAX_DWA_LABELS_PER_TRAIN_PAIR = 10
COLUMNS = ["id", "text", "source", "input_type", "gold", "primary_soc", "split", "task_id"]


def gold_list(gold: str) -> list[str]:
    return [g for g in str(gold).split(";") if g]


def _group_texts(df: pd.DataFrame, targets: set[str]) -> pd.DataFrame:
    """Collapse (soc, text) rows to one row per normalised text and its list of correct SOC codes."""
    df = df[df["soc"].isin(targets)].copy()
    df["norm"] = df["text"].map(normalize)
    df = df[df["norm"].str.len() > 0]
    grouped = df.groupby("norm").agg(
        text=("text", "first"),
        gold=("soc", lambda s: ";".join(sorted(set(s)))),
        task_id=("task_id", "first") if "task_id" in df else ("text", lambda s: ""),
    ).reset_index()
    grouped["primary_soc"] = grouped["gold"].str.split(";").str[0]
    return grouped


def _stratified_split(groups: pd.DataFrame, rng: np.random.Generator, stratify: bool = True) -> pd.Series:
    """Assign train/val/test per row, stratified by ``primary_soc``."""
    split = pd.Series("train", index=groups.index)
    buckets = groups.groupby("primary_soc").groups.values() if stratify else [groups.index]
    for idx in buckets:
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * SPLIT_FRACTIONS["test"]))
        n_val = int(round(n * SPLIT_FRACTIONS["val"]))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def build_real_pairs(
    catalog: OccupationCatalog,
    job_titles: pd.DataFrame,
    reported_titles: pd.DataFrame,
    *,
    seed: int = 13,
) -> pd.DataFrame:
    """Build titles / tasks / DWAs / composites with splits.

    Args:
        catalog: The occupation catalog (defines the target occupations).
        job_titles: Bulk ``job_titles`` table (``soc_code``, ``job_title``).
        reported_titles: Bulk ``reported_titles`` table.
        seed: RNG seed (splits and composites are fully deterministic).
    """
    rng = np.random.default_rng(seed)
    targets = set(catalog.socs)
    official = {normalize(t) for t in catalog.occupations["title"]}
    official_by_soc = catalog.titles()
    frames = []

    # Titles ---------------------------------------------------------------
    titles = pd.concat([
        job_titles[["soc_code", "job_title"]],
        reported_titles[["soc_code", "job_title"]],
    ]).rename(columns={"soc_code": "soc", "job_title": "text"})
    tg = _group_texts(titles, targets)
    n_before = len(tg)
    tg = tg[~tg["norm"].isin(official)].reset_index(drop=True)  # trivially matched
    log.info("Titles: %d unique (%d identical to an official title dropped).", len(tg), n_before - len(tg))
    tg["split"] = _stratified_split(tg, rng)
    tg["source"], tg["task_id"] = "title", ""
    frames.append(tg)

    # Tasks ----------------------------------------------------------------
    tasks = catalog.tasks[["soc", "task_id", "text"]]
    kg = _group_texts(tasks, targets)
    kg["split"] = _stratified_split(kg, rng)
    kg["source"] = "task"
    frames.append(kg)
    # Every task row (not just the first per text) inherits its text's split.
    task_split = tasks.assign(norm=tasks["text"].map(normalize)).merge(
        kg[["norm", "split"]], on="norm")[["soc", "task_id", "text", "split"]]

    # Detailed work activities (generic; random split across DWAs) ---------
    dg = _group_texts(catalog.dwas[["soc", "text"]], targets)
    dg["split"] = _stratified_split(dg, rng, stratify=False)
    dg["source"], dg["task_id"] = "dwa", ""
    frames.append(dg)

    # Composite pseudo-descriptions (tasks from one occupation, same split) --
    comp = []
    for (soc, split), grp in task_split.groupby(["soc", "split"]):
        pool = grp["text"].tolist()
        if len(pool) < 2:
            continue
        seen = set()
        for _ in range(COMPOSITES_PER_OCC[split]):
            k = int(rng.integers(2, min(4, len(pool)) + 1))
            picks = tuple(sorted(rng.choice(len(pool), size=k, replace=False)))
            if picks in seen:
                continue
            seen.add(picks)
            order = list(picks)
            rng.shuffle(order)
            text = " ".join(pool[i] for i in order)
            comp.append({"text": text, "gold": soc, "primary_soc": soc, "split": split,
                         "source": "composite", "task_id": ""})
    frames.append(pd.DataFrame(comp))

    out = pd.concat(frames, ignore_index=True)
    out["input_type"] = out["source"].map(SOURCE_INPUT_TYPE)
    out["id"] = [f"{s}-{i:06d}" for i, s in enumerate(out["source"])]
    out = out[COLUMNS]
    out["hard"] = (out["source"] == "title") & out.apply(
        lambda r: all(shares_no_tokens(r["text"], official_by_soc[g]) for g in gold_list(r["gold"])),
        axis=1,
    )
    return out


def heldout_task_ids(pairs: pd.DataFrame, catalog: OccupationCatalog) -> set[str]:
    """Task IDs whose text is in val/test; excluded from eval representations."""
    held = pairs[(pairs["source"] == "task") & (pairs["split"] != "train")]
    norms = set(held["text"].map(normalize))
    t = catalog.tasks
    return set(t.loc[t["text"].map(normalize).isin(norms), "task_id"])


def training_pairs(
    pairs: pd.DataFrame,
    *,
    sources: tuple[str, ...] = ("title", "task", "dwa", "composite"),
    seed: int = 13,
) -> pd.DataFrame:
    """Explode train rows into (anchor, soc) pairs for contrastive training.

    DWAs can belong to 100+ occupations; at most
    :data:`MAX_DWA_LABELS_PER_TRAIN_PAIR` random labels are kept per DWA so
    generic activities don't dominate the training mix.
    """
    rng = np.random.default_rng(seed)
    rows = []
    train = pairs[(pairs["split"] == "train") & pairs["source"].isin(sources)]
    for r in train.itertuples(index=False):
        golds = gold_list(r.gold)
        if r.source == "dwa" and len(golds) > MAX_DWA_LABELS_PER_TRAIN_PAIR:
            golds = list(rng.choice(golds, size=MAX_DWA_LABELS_PER_TRAIN_PAIR, replace=False))
        for g in golds:
            rows.append({"anchor": r.text, "soc": g, "source": r.source, "gold": r.gold})
    return pd.DataFrame(rows)


def summarize(pairs: pd.DataFrame) -> dict:
    """Counts per source x split (+ hard titles), for logging and the write-up."""
    counts = pairs.groupby(["source", "split"]).size().unstack(fill_value=0)
    summary = {src: {k: int(v) for k, v in row.items()} for src, row in counts.iterrows()}
    summary["title_hard"] = {
        s: int(((pairs["hard"]) & (pairs["split"] == s)).sum()) for s in ("train", "val", "test")
    }
    summary["multi_label_rows"] = int(pairs["gold"].str.contains(";").sum())
    return summary


def save_pairs(pairs: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(path, index=False)
    (path.with_suffix(".summary.json")).write_text(json.dumps(summarize(pairs), indent=2))
    return path


def load_pairs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No dataset at {path}. Run `onet-analyze matching build-data`.")
    return pd.read_parquet(path)
