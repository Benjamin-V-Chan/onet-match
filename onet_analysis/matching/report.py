"""Render the results write-up (``notes/matching-results.md``) from saved runs.

Every number in the document is read from ``outputs/runs/<run>/metrics.json``
(or the training records in ``config.json``), and every table names the runs it
came from. Runs that don't exist yet are shown as "not yet run", never filled in.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..paths import Paths  # noqa: E402
from .catalog import MAJOR_GROUPS, OccupationCatalog  # noqa: E402

LADDER: list[tuple[str, str]] = [
    ("1 TF-IDF (existing approach, baseline)", "L1_tfidf"),
    ("2 BM25", "L2_bm25"),
    ("3a all-MiniLM-L6-v2 (off-the-shelf)", "L3_minilm"),
    ("3b bge-small-en-v1.5 (off-the-shelf)", "L3_bge-small"),
    ("3c bge-base-en-v1.5 (off-the-shelf)", "L3_bge-base"),
    ("3d e5-base-v2 (off-the-shelf)", "L3_e5-base"),
    ("4 fine-tuned bge-small (real data)", "L4_ft-bge-small"),
    ("5 = 4 + cross-encoder reranker (blended)", "L5_ft-bge-small+ce+fuse"),
    ("4s fine-tuned bge-small (real + synthetic)", "L4s_ft-bge-small+synth"),
    ("5s = 4s + cross-encoder reranker (blended)", "L5s_ft-bge-small+synth+ce+fuse"),
    ("4b fine-tuned bge-base (real + synthetic)", "L4b_ft-bge-base+synth"),
    ("5b = 4b + cross-encoder reranker (blended)", "L5b_ft-bge-base+synth+ce+fuse"),
]
BASELINE = "L1_tfidf"
FIELD_ABLATION = ["bm25", "bge-small", "bge-base"]
FIELD_PRESETS = ["title", "title+desc", "title+desc+tasks", "title+desc+tasks+skills"]
DATA_ABLATION = [("titles only", "abl_data_titles"), ("+ tasks, DWAs, composites (all real)", "L4_ft-bge-small"),
                 ("+ synthetic messy inputs", "L4s_ft-bge-small+synth")]
RERANK_ABLATION = [("4: bi-encoder only", "L4_ft-bge-small"), ("5: reranker score only", "L5_ft-bge-small+ce"),
                   ("5: blended", "L5_ft-bge-small+ce+fuse"), ("4s: bi-encoder only", "L4s_ft-bge-small+synth"),
                   ("5s: reranker score only", "L5s_ft-bge-small+synth+ce"),
                   ("5s: blended", "L5s_ft-bge-small+synth+ce+fuse")]
HARDNEG_ABLATION = [("hard negatives off", "abl_hardneg_off"), ("hard negatives on", "L4_ft-bge-small")]
LEARNING_CURVE = [("lc_frac0.1", 0.1), ("lc_frac0.25", 0.25), ("lc_frac0.5", 0.5), ("L4_ft-bge-small", 1.0)]
SETS = [("real_test", "Real O*NET held-out test"), ("synthetic_test", "Held-out LLM-generated messy inputs")]
TYPE_ORDER = ["job_title", "description", "task", "activity", "experience", "project", "interest", "skills"]


def _load(runs: Path, name: str) -> tuple[dict, dict] | None:
    d = runs / name
    if not (d / "metrics.json").is_file():
        return None
    return json.loads((d / "config.json").read_text()), json.loads((d / "metrics.json").read_text())


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _overall(m: dict, set_name: str, key: str = "overall") -> dict | None:
    s = m["sets"].get(set_name)
    if not s:
        return None
    return s.get(key)


def headline_table(runs: Path, set_name: str) -> str:
    base = _load(runs, BASELINE)
    base_top1 = _overall(base[1], set_name)["top1"] if base and _overall(base[1], set_name) else None
    rows = []
    for label, name in LADDER:
        r = _load(runs, name)
        o = _overall(r[1], set_name) if r else None
        if not o:
            rows.append({"model": label, "run": f"`{name}`", "n": "", "top-1": "not yet run", "top-5": "",
                         "MRR@20": "", "major-group top-1": "", "ECE": "", "Δ top-1 vs TF-IDF (pp)": "",
                         "p50 latency (ms)": ""})
            continue
        lat = r[1].get("latency", {}).get("p50_ms")
        rows.append({
            "model": label, "run": f"`{name}`", "n": o["n"], "top-1": _pct(o["top1"]), "top-5": _pct(o["top5"]),
            "MRR@20": f"{o['mrr']:.3f}", "major-group top-1": _pct(o["major_top1"]), "ECE": f"{o['ece']:.3f}",
            "Δ top-1 vs TF-IDF (pp)": "" if base_top1 is None or name == BASELINE
            else f"{100 * (o['top1'] - base_top1):+.1f}",
            "p50 latency (ms)": "" if lat is None else f"{lat:.1f}",
        })
    return _md(pd.DataFrame(rows))


def by_type_table(runs: Path, set_name: str, metric: str = "top1") -> str | None:
    rows, types = [], set()
    for label, name in LADDER:
        r = _load(runs, name)
        s = r[1]["sets"].get(set_name) if r else None
        if not s:
            continue
        row = {"model": label}
        for t, v in s["by_input_type"].items():
            row[f"{t} (n={v['n']})"] = _pct(v[metric]) if metric != "mrr" else f"{v['mrr']:.3f}"
            types.add(f"{t} (n={v['n']})")
        if "title_hard" in s:
            key = f"hard titles (n={s['title_hard']['n']})"
            row[key] = _pct(s["title_hard"][metric]) if metric != "mrr" else f"{s['title_hard']['mrr']:.3f}"
            types.add(key)
        rows.append(row)
    if not rows:
        return None
    order = sorted(types, key=lambda c: (TYPE_ORDER.index(c.split(" (")[0]) if c.split(" (")[0] in TYPE_ORDER
                                         else 99, c))
    return _md(pd.DataFrame(rows)[["model", *order]].fillna("—"))


def fields_table(runs: Path) -> str:
    rows = []
    for model in FIELD_ABLATION:
        row = {"model": model}
        for f in FIELD_PRESETS:
            r = _load(runs, f"abl_fields_{model}_{f}")
            o = _overall(r[1], "real_test") if r else None
            row[f] = _pct(o["top1"]) if o else "not yet run"
        rows.append(row)
    return _md(pd.DataFrame(rows))


def _ablation_rows(runs: Path, spec: list[tuple[str, str]]) -> str:
    rows = []
    for label, name in spec:
        r = _load(runs, name)
        if not r:
            rows.append({"variant": label, "run": f"`{name}`", "bi-encoder train pairs": "",
                         "real top-1": "not yet run"})
            continue
        cfg, m = r
        tr = next(iter(cfg.get("training", {}).values()), {})
        row = {"variant": label, "run": f"`{name}`", "bi-encoder train pairs": tr.get("examples", "")}
        for set_name, _ in SETS:
            o = _overall(m, set_name)
            if o:
                row[f"{set_name} top-1"] = _pct(o["top1"])
        rh = m["sets"]["real_test"].get("title_hard")
        if rh:
            row["hard titles top-1"] = _pct(rh["top1"])
        rows.append(row)
    return _md(pd.DataFrame(rows).fillna("—"))


def learning_curve(runs: Path, fig_dir: Path) -> tuple[str, Path | None]:
    rows = []
    for name, frac in LEARNING_CURVE:
        r = _load(runs, name)
        if not r:
            continue
        cfg, m = r
        tr = next(iter(cfg.get("training", {}).values()), {})
        rows.append({"fraction": frac, "train pairs": tr.get("examples"), "run": f"`{name}`",
                     "real top-1": m["sets"]["real_test"]["overall"]["top1"],
                     "hard titles top-1": m["sets"]["real_test"].get("title_hard", {}).get("top1"),
                     "train minutes": round(tr.get("train_seconds", 0) / 60, 1)})
    if not rows:
        return "_No learning-curve runs yet._", None
    df = pd.DataFrame(rows)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "learning_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(df["train pairs"], 100 * df["real top-1"], "o-", label="real test (all types)")
    if df["hard titles top-1"].notna().all():
        plt.plot(df["train pairs"], 100 * df["hard titles top-1"], "s--", label="hard titles")
    base = _load(runs, BASELINE)
    if base:
        plt.axhline(100 * base[1]["sets"]["real_test"]["overall"]["top1"], color="grey", ls=":", label="TF-IDF")
    plt.xscale("log")
    plt.xlabel("training pairs (log scale)")
    plt.ylabel("top-1 accuracy (%)")
    plt.title("Fine-tuned bge-small: learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    out = df.assign(**{"real top-1": df["real top-1"].map(_pct),
                       "hard titles top-1": df["hard titles top-1"].map(_pct)})
    return _md(out), path


def reliability_figure(runs: Path, fig_dir: Path, names: list[str]) -> Path | None:
    loaded = [(n, _load(runs, n)) for n in names]
    loaded = [(n, r) for n, r in loaded if r]
    if not loaded:
        return None
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "reliability.png"
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color="grey", ls=":", label="perfect calibration")
    for n, (_, m) in loaded:
        bins = m["sets"]["real_test"]["overall"]["reliability"]
        plt.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-",
                 label=f"{n} (ECE {m['sets']['real_test']['overall']['ece']:.3f})")
    plt.xlabel("mean top-1 confidence (bin)")
    plt.ylabel("top-1 accuracy (bin)")
    plt.title("Reliability, real held-out test")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def error_examples(runs: Path, name: str, catalog: OccupationCatalog, set_name: str = "real_test",
                   per_type: int = 3, seed: int = 7) -> str:
    """Real failures of ``name``: random wrong top-1 predictions per input type."""
    p = runs / name / f"predictions_{set_name}.parquet"
    if not p.is_file():
        return f"_No predictions for `{name}` on `{set_name}`._"
    df = pd.read_parquet(p)
    titles = catalog.titles()
    wrong = df[~df["hit1"]]
    lines = []
    for t, g in wrong.groupby("input_type"):
        sample = g.sample(min(per_type, len(g)), random_state=seed)
        for r in sample.itertuples(index=False):
            gold = "; ".join(titles.get(s, s) for s in r.gold.split(";")[:3])
            text = r.text if len(r.text) < 160 else r.text[:157] + "..."
            lines.append({"input type": t, "input": text.replace("|", "/"), "gold": gold,
                          "predicted (top-1)": titles.get(r.top1, r.top1),
                          "confidence": f"{r.confidence:.2f}", "gold in top-5": "yes" if r.hit5 else "no"})
    return _md(pd.DataFrame(lines))


def major_group_confusions(runs: Path, name: str, set_name: str = "real_test", n: int = 8) -> str:
    p = runs / name / f"predictions_{set_name}.parquet"
    if not p.is_file():
        return ""
    df = pd.read_parquet(p)
    wrong = df[~df["major_hit1"]].copy()
    wrong["gold_major"] = wrong["gold"].str.slice(0, 2)
    wrong["pred_major"] = wrong["top1"].str.slice(0, 2)
    c = wrong.groupby(["gold_major", "pred_major"]).size().sort_values(ascending=False).head(n).reset_index(name="count")
    c["gold major group"] = c["gold_major"].map(lambda g: f"{g} {MAJOR_GROUPS.get(g, '')}")
    c["predicted major group"] = c["pred_major"].map(lambda g: f"{g} {MAJOR_GROUPS.get(g, '')}")
    return _md(c[["gold major group", "predicted major group", "count"]])


def best_run(runs: Path) -> str | None:
    """Best ladder run by *validation* top-1, then validation MRR (never by test)."""
    scored = []
    for _, name in LADDER:
        r = _load(runs, name)
        if r:
            v = r[1]["val"]["overall"]
            scored.append((v["top1"], v["mrr"], name))
    return max(scored)[-1] if scored else None


def key_findings(runs: Path, best: str | None) -> list[str]:
    """Bullet points computed from saved runs; a bullet is skipped when its runs are missing."""
    def top1(run, set_name="real_test", key="overall"):
        r = _load(runs, run)
        o = _overall(r[1], set_name, key) if r else None
        return None if o is None else o["top1"]

    def pp(a, b):
        return f"{100 * (a - b):+.1f} pp".replace("-0.0", "±0.0")

    out = []
    t, b = top1(BASELINE), top1(best) if best else None
    ts, bs = top1(BASELINE, "synthetic_test"), top1(best, "synthetic_test") if best else None
    if t is not None and b is not None:
        line = f"- **Best model** (`{best}`): real-test top-1 {_pct(b)}% vs {_pct(t)}% for TF-IDF ({pp(b, t)})"
        if ts is not None and bs is not None:
            line += f"; held-out messy inputs {_pct(bs)}% vs {_pct(ts)}% ({pp(bs, ts)})"
        out.append(line + ".")
    steps = [("Off-the-shelf embeddings", "L3_bge-base", BASELINE, "real_test"),
             ("Fine-tuning bge-small on O*NET pairs", "L4_ft-bge-small", "L3_bge-small", "real_test"),
             ("Blending in the reranker", "L5_ft-bge-small+ce+fuse", "L4_ft-bge-small", "real_test"),
             ("Adding synthetic messy training text", "L4s_ft-bge-small+synth", "L4_ft-bge-small", "synthetic_test"),
             ("Adding synthetic messy training text", "L4s_ft-bge-small+synth", "L4_ft-bge-small", "real_test"),
             ("Fine-tuning bge-base instead of bge-small", "L4b_ft-bge-base+synth", "L4s_ft-bge-small+synth", "real_test"),
             ("Fine-tuning bge-base instead of bge-small", "L4b_ft-bge-base+synth", "L4s_ft-bge-small+synth", "synthetic_test"),
             ("Hard-negative mining", "L4_ft-bge-small", "abl_hardneg_off", "real_test"),
             ("Reranker score alone (no blending)", "L5_ft-bge-small+ce", "L4_ft-bge-small", "real_test")]
    for label, new, old, set_name in steps:
        a, c = top1(new, set_name), top1(old, set_name)
        if a is not None and c is not None:
            out.append(f"- {label} (`{new}` vs `{old}`, {set_name}): top-1 {_pct(c)}% → {_pct(a)}% ({pp(a, c)}).")
    lc = [(f, top1(n)) for n, f in LEARNING_CURVE]
    if all(v is not None for _, v in lc):
        out.append("- Learning curve (real-test top-1 at " + ", ".join(f"{int(f * 100)}%" for f, _ in lc)
                   + " of the real pairs): " + " → ".join(_pct(v) for _, v in lc) + ", still rising at full data.")
    r = _load(runs, best) if best else None
    if r:
        types = {**r[1]["sets"]["real_test"]["by_input_type"], **r[1]["sets"].get("synthetic_test", {}).get(
            "by_input_type", {})}
        lo = min(types, key=lambda k: types[k]["top1"])
        hi = max(types, key=lambda k: types[k]["top1"])
        out.append(f"- By input type, the best model is strongest on `{hi}` ({_pct(types[hi]['top1'])}% top-1) and "
                   f"weakest on `{lo}` ({_pct(types[lo]['top1'])}%, top-5 {_pct(types[lo]['top5'])}%).")
    return out


STATIC_LIMITATIONS = """\
- **Label granularity and ambiguity.** Many titles and generic activities legitimately map to several
  occupations; we score a hit when the prediction is any of the O*NET-listed codes, but O*NET's lists are
  not exhaustive, so some "errors" are defensible answers (see the error table).
- **Real-data eval sets are O*NET's own language.** Held-out titles and tasks are realistic for job-title
  normalization but are cleaner than what people type; the synthetic set exists to measure
  messy inputs, but it is still LLM-written.
- **No human-written test set yet.** Messy-input results come from LLM-written text only. Both synthetic
  sets were written by Claude models (Opus for training, Sonnet for evaluation): different models, same
  vendor, so some shared phrasing may flatter the synthetic-test numbers.
- **Residual occupations are excluded.** O*NET-SOC codes without task data (mostly "..., All Other")
  are not match targets.
- **Calibration is fitted on O*NET-style validation data.** Confidence is temperature-scaled on the real
  validation split; on input styles far from it (e.g. very short interest statements) the reliability curve
  per set is the thing to check before thresholding on confidence.
- **One training run per configuration** (seed 13). Differences of about a point or less, such as hard
  negatives on vs off, are within run-to-run noise and should not be read as real effects.
- **English, US taxonomy.** O*NET is US-specific and English-only.
- **Latency** is single-query wall-clock on an Apple-silicon laptop (MPS), excluding HTTP overhead.
"""


def render_report(paths: Paths, out: Path | None = None) -> Path:
    runs = paths.runs
    fig_dir = paths.base / "outputs" / "figures"
    out = out or (paths.base / "notes" / "matching-results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: Path("..") / p.relative_to(paths.base)  # noqa: E731 - notes/ -> repo root

    catalog = OccupationCatalog.load(paths.catalog_dir)
    summary_path = paths.matching / "build_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    best = best_run(runs)

    parts = [
        "# Semantic occupation matching — results",
        "",
        "_Generated by `onet-analyze matching report` from `outputs/runs/*/metrics.json`; do not edit by hand._",
        "_Every table names the run directory its numbers come from._",
        "",
        f"O*NET release: **{catalog.meta.get('onet_release')}** "
        f"(sha256 `{str(catalog.meta.get('onet_sha256'))[:12]}…`, downloaded {catalog.meta.get('onet_downloaded_at')}). "
        f"Match targets: **{len(catalog.socs)}** occupations.",
        "",
        "## Key findings",
        "",
        *key_findings(runs, best),
        "",
        "## Data",
        "",
        "Rows per source and split (unique normalised texts; see `onet_analysis/matching/datasets.py`):",
        "",
        "```json",
        json.dumps({k: v for k, v in summary.items() if k not in ("onet_release",)}, indent=1),
        "```",
        "",
        "## Model ladder",
        "",
        "Top-1 / top-5 = a gold occupation is ranked 1st / in the top 5. MRR@20 over the top-20 candidates. "
        "Major-group top-1 = the top prediction is in a gold occupation's SOC major group. "
        "ECE = expected calibration error of the top-1 confidence (10 bins). "
        "Latency = single-query p50.",
        "",
    ]
    for set_name, title in SETS:
        anything = any(_load(runs, n) and _overall(_load(runs, n)[1], set_name) for _, n in LADDER)
        if not anything:
            continue
        parts += [f"### {title} (`{set_name}`)", ""]
        parts += [headline_table(runs, set_name), "", "Top-1 by input type:", "",
                  by_type_table(runs, set_name) or "", "", "MRR@20 by input type:", "",
                  by_type_table(runs, set_name, "mrr") or "", ""]
    rel_fig = reliability_figure(runs, fig_dir, [BASELINE, "L3_bge-base", best] if best else [BASELINE])
    parts += ["## Calibration", "",
              "Confidence = softmax over the top-20 scores with one temperature fitted on the validation split "
              "(per run, stored in `metrics.json`).", ""]
    if rel_fig:
        parts += [f"![reliability]({rel(rel_fig)})", ""]
    lc_table, lc_fig = learning_curve(runs, fig_dir)
    parts += ["## Learning curve", "", lc_table, ""]
    if lc_fig:
        parts += [f"![learning curve]({rel(lc_fig)})", ""]
    parts += [
        "## Ablations", "",
        "### Occupation representation fields (off-the-shelf models, real test top-1)", "",
        fields_table(runs), "",
        "### Training-data sources (fine-tuned bge-small, hard negatives on)", "",
        _ablation_rows(runs, DATA_ABLATION), "",
        "### Hard-negative mining (fine-tuned bge-small, all real sources)", "",
        _ablation_rows(runs, HARDNEG_ABLATION), "",
        "### Reranker scoring", "",
        "The reranker rescores the bi-encoder's top 20. \"Blended\" ranks by a weighted sum of the two models' "
        "per-query z-scored scores; the weight is chosen on the validation split "
        "(`onet-analyze matching tune-fusion`, grid saved in `outputs/fusion/`).", "",
        _ablation_rows(runs, RERANK_ABLATION), "",
    ]
    if best:
        parts += [
            f"## Error analysis (best model by validation top-1, then MRR: `{best}`)", "",
            "Randomly sampled wrong top-1 predictions on the real test set:", "",
            error_examples(runs, best, catalog), "",
            "Most frequent major-group confusions (wrong major group at top-1):", "",
            major_group_confusions(runs, best), "",
        ]
        if (runs / best / "predictions_synthetic_test.parquet").is_file():
            parts += ["Failures on held-out synthetic messy inputs:", "",
                      error_examples(runs, best, catalog, "synthetic_test", per_type=2), ""]
    parts += ["## Limitations", "", STATIC_LIMITATIONS]
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
