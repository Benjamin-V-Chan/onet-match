"""High-level steps behind ``onet-analyze matching ...`` (data -> train -> eval -> bundle).

Each function reads/writes the canonical layout in :class:`onet_analysis.paths.Paths`
so that every number in the write-up can be regenerated with the commands in
the README.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..paths import Paths
from . import datasets as D
from . import synth as S
from .catalog import FIELD_PRESETS, OccupationCatalog
from .engine import DEFAULT_K, build_ranker, parse_spec, write_bundle
from .evaluate import load_run, run_eval

log = logging.getLogger("onet_analysis.matching.pipeline")

REAL_SOURCES = ("title", "task", "dwa", "composite")


# --------------------------------------------------------------------------- data
def build_data(paths: Paths, *, release: str | None = None, seed: int = 13) -> dict:
    """Download/reuse the O*NET bulk release, build the catalog and the real pairs."""
    from onet_data_collector.bulk import download_bulk

    bulk = download_bulk(release)
    catalog = OccupationCatalog.from_bulk(bulk)
    catalog.save(paths.catalog_dir)
    pairs = D.build_real_pairs(catalog, bulk.table("job_titles"), bulk.table("reported_titles"), seed=seed)
    D.save_pairs(pairs, paths.pairs)
    summary = D.summarize(pairs)
    summary["onet_release"] = bulk.release
    summary["n_occupations"] = len(catalog.socs)
    (paths.matching / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_inputs(paths: Paths) -> tuple[OccupationCatalog, pd.DataFrame]:
    return OccupationCatalog.load(paths.catalog_dir), D.load_pairs(paths.pairs)


def eval_docs(catalog: OccupationCatalog, pairs: pd.DataFrame, fields: str = "title+desc+tasks",
              n_tasks: int = 10) -> list[str]:
    """Occupation documents with every val/test task removed (leakage-safe)."""
    return catalog.representations(FIELD_PRESETS[fields], n_tasks=n_tasks,
                                   exclude_task_ids=D.heldout_task_ids(pairs, catalog))


def synthetic_path(paths: Paths, role: str) -> Path:
    return paths.matching / f"synthetic_{role}.parquet"


def raw_synth_path(paths: Paths, role: str) -> Path:
    return paths.matching / "synth" / f"raw_{role}.jsonl"


def extra_eval_sets(paths: Paths, catalog: OccupationCatalog) -> dict[str, pd.DataFrame]:
    sets = {}
    sp = synthetic_path(paths, "eval")
    if sp.is_file():
        sets["synthetic_test"] = pd.read_parquet(sp)
    return sets


# --------------------------------------------------------------------------- evaluation
def normalize_spec(paths: Paths, spec: str) -> tuple[str, str]:
    """(canonical spec with local model paths relative to the project base, spec with absolute paths).

    The canonical form is what runs record and bundles are checked against, so
    ``dense=models/x`` and ``dense=/abs/project/models/x`` are the same model.
    """
    s = parse_spec(spec)
    canon, absolute = dict(s), dict(s)
    for key in ("model", "rerank"):
        v = s[key]
        if v and not Path(v).exists() and (paths.base / v).exists():
            v = str(paths.base / v)
        if v and Path(v).exists():
            full = Path(v).resolve()
            absolute[key] = str(full)
            try:
                canon[key] = str(full.relative_to(paths.base.resolve()))
            except ValueError:
                canon[key] = str(full)

    def fmt(d: dict) -> str:
        out = d["retriever"] + (f"={d['model']}" if d["model"] else "")
        out += f"+rerank={d['rerank']}" if d["rerank"] else ""
        return out + (f"+fuse={d['fuse']:g}" if d.get("fuse") is not None else "")

    return fmt(canon), fmt(absolute)


def default_run_name(spec: str, fields: str) -> str:
    s = parse_spec(spec)
    parts = [s["retriever"]]
    if s["model"]:
        parts.append(Path(s["model"]).name)
    if s["rerank"]:
        parts.append("rerank-" + Path(s["rerank"]).name)
    if fields != "title+desc+tasks":
        parts.append(fields)
    return "_".join(parts)


def evaluate(paths: Paths, spec: str, *, run_name: str | None = None, fields: str = "title+desc+tasks",
             latency: bool = True, k: int = DEFAULT_K, extra_config: dict | None = None) -> dict:
    catalog, pairs = load_inputs(paths)
    docs = eval_docs(catalog, pairs, fields)
    spec, abs_spec = normalize_spec(paths, spec)
    ranker = build_ranker(abs_spec, docs, k=k)
    name = run_name or default_run_name(spec, fields)
    cfg = {"spec": spec, "fields": fields, "onet_release": catalog.meta.get("onet_release"),
           "dataset_summary": D.summarize(pairs), **(extra_config or {})}
    for p in (parse_spec(abs_spec)["model"], parse_spec(abs_spec)["rerank"]):
        if p and (Path(p) / "training.json").is_file():
            cfg.setdefault("training", {})[Path(p).name] = json.loads((Path(p) / "training.json").read_text())
    return run_eval(name, ranker, catalog.socs, pairs, paths.runs, extra_sets=extra_eval_sets(paths, catalog),
                    config=cfg, latency=latency)


def tune_fusion(paths: Paths, spec: str, *, fields: str = "title+desc+tasks", k: int = DEFAULT_K,
                step: float = 0.05) -> dict:
    """Pick the reranker/retriever blend weight by validation top-1; saves outputs/fusion/<reranker>.json."""
    from .metrics import gold_hits
    from .retrieval import Ranking, fuse_scores

    catalog, pairs = load_inputs(paths)
    spec, abs_spec = normalize_spec(paths, spec)
    s = parse_spec(abs_spec)
    if not s["rerank"]:
        raise ValueError("tune-fusion needs a spec with +rerank=<model>")
    ranker = build_ranker({**s, "fuse": None}, eval_docs(catalog, pairs, fields), k=k)
    val = pairs[(pairs["split"] == "val") & pairs["source"].isin(REAL_SOURCES)].reset_index(drop=True)
    texts = val["text"].tolist()
    parts = [ranker.candidates(texts[i:i + 512]) for i in range(0, len(texts), 512)]
    top, bi = np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])
    ce = ranker.reranker.rerank(texts, top, ranker.docs)
    hits = gold_hits(Ranking(top, bi), catalog.socs, [g.split(";") for g in val["gold"]])
    grid = []
    for w in np.round(np.arange(0, 1 + 1e-9, step), 4):
        best = np.argmax(fuse_scores(bi, ce, float(w)), axis=1)
        grid.append({"weight": float(w), "val_top1": float(hits[np.arange(len(hits)), best].mean())})
    chosen = max(grid, key=lambda g: (g["val_top1"], -abs(g["weight"] - 0.5)))
    base = parse_spec(spec)
    tuned = normalize_spec(paths, f"{spec.split('+fuse=')[0]}+fuse={chosen['weight']:g}")[0]
    report = {"spec": spec, "tuned_spec": tuned, "val_rows": len(val), "chosen": chosen, "grid": grid}
    out = paths.base / "outputs" / "fusion" / f"{Path(base['rerank']).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- training
def _training_frame(paths: Paths, pairs: pd.DataFrame, sources: Sequence[str]) -> pd.DataFrame:
    frame = pairs
    if "synthetic" in sources:
        sp = synthetic_path(paths, "train")
        if not sp.is_file():
            raise FileNotFoundError(f"{sp} not found — run `onet-analyze matching synth --role train` first.")
        frame = pd.concat([pairs, pd.read_parquet(sp)], ignore_index=True)
    return frame


def train_biencoder(
    paths: Paths,
    *,
    base: str = "BAAI/bge-small-en-v1.5",
    name: str,
    sources: Sequence[str] = REAL_SOURCES,
    hard_negatives: bool = True,
    fraction: float = 1.0,
    epochs: float = 1.0,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_steps: int = -1,
    seed: int = 13,
    bf16: bool = True,
) -> Path:
    from . import train as T
    from .retrieval import DenseRetriever

    catalog, pairs = load_inputs(paths)
    docs = eval_docs(catalog, pairs)
    tp = D.training_pairs(_training_frame(paths, pairs, sources), sources=tuple(sources), seed=seed)
    if fraction < 1.0:
        tp = tp.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    base_ret = DenseRetriever(base)
    neighbors = None
    if hard_negatives:
        neighbors = T.occupation_neighbors(base_ret.fit(docs).doc_embeddings)
    examples = T.build_biencoder_examples(tp, catalog.socs, docs, neighbors=neighbors, seed=seed)
    out = paths.models / name
    record = T.train_biencoder(base, examples, out, epochs=epochs, batch_size=batch_size,
                               learning_rate=learning_rate, max_steps=max_steps, seed=seed, bf16=bf16,
                               model=base_ret.model)
    record.update(sources=list(sources), hard_negatives=hard_negatives, fraction=fraction,
                  pairs_by_source=tp["source"].value_counts().to_dict())
    (out / "training.json").write_text(json.dumps(record, indent=2))
    return out


def train_reranker(
    paths: Paths,
    *,
    biencoder: str,
    base: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    name: str,
    sources: Sequence[str] = REAL_SOURCES,
    n_queries: int = 30000,
    n_negatives: int = 4,
    epochs: float = 1.0,
    batch_size: int = 32,
    max_steps: int = -1,
    seed: int = 13,
    bf16: bool = True,
) -> Path:
    from . import train as T

    catalog, pairs = load_inputs(paths)
    docs = eval_docs(catalog, pairs)
    frame = _training_frame(paths, pairs, sources)
    train = frame[(frame["split"] == "train") & frame["source"].isin(sources)]
    if len(train) > n_queries:
        train = train.groupby("source").sample(frac=n_queries / len(train), random_state=seed)
    ranker = build_ranker(f"dense={biencoder}", docs, k=DEFAULT_K)
    ranking = ranker.rank(train["text"].tolist())
    examples = T.build_reranker_examples(train["text"].tolist(), train["gold"].tolist(), ranking.indices,
                                         catalog.socs, docs, n_negatives=n_negatives, seed=seed)
    out = paths.models / name
    record = T.train_reranker(base, examples, out, epochs=epochs, batch_size=batch_size,
                              max_steps=max_steps, seed=seed, bf16=bf16)
    record.update(biencoder=str(biencoder), sources=list(sources),
                  queries_by_source=train["source"].value_counts().to_dict())
    (out / "training.json").write_text(json.dumps(record, indent=2))
    return out


# --------------------------------------------------------------------------- bundle
def bundle(paths: Paths, spec: str, *, run_name: str, out: Path | None = None,
           fields: str = "title+desc+tasks") -> Path:
    """Package ``spec`` for serving, using the temperature fitted in ``run_name``."""
    cfg, metrics = load_run(paths.runs / run_name)
    spec, abs_spec = normalize_spec(paths, spec)
    if cfg.get("spec") != spec:
        raise ValueError(f"Run {run_name} evaluated {cfg.get('spec')!r}, not {spec!r}.")
    if cfg.get("fields", "title+desc+tasks") != fields:
        raise ValueError(f"Run {run_name} used fields {cfg.get('fields')!r}, not {fields!r}.")
    catalog = OccupationCatalog.load(paths.catalog_dir)
    return write_bundle(out or paths.bundle, abs_spec, catalog, temperature=metrics["temperature"],
                        fields=FIELD_PRESETS[fields], name=run_name, source_run=run_name, display_spec=spec)


# --------------------------------------------------------------------------- synthetic data
def synthesize(
    paths: Paths,
    *,
    role: str,
    provider: str,
    model: str,
    pilot: int | None = None,
    n_per_type: int = 3,
    workers: int = 8,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    price_in: float | None = None,
    price_out: float | None = None,
    seed: int = 13,
) -> dict:
    """Generate synthetic inputs. ``pilot=N`` samples N occupations into a separate log."""
    catalog, pairs = load_inputs(paths)
    kwargs = {"base_url": base_url, "api_key_env": api_key_env} if provider.startswith("openai") else {}
    prov = S.make_provider(provider, model, **kwargs)
    socs = None
    raw = raw_synth_path(paths, role)
    if pilot:
        rng = np.random.default_rng(seed)
        socs = list(rng.choice(catalog.socs, size=pilot, replace=False))
        raw = paths.matching / "synth" / f"pilot_{role}_{model.replace('/', '_')}.jsonl"
    S.generate(catalog, pairs, prov, raw, role=role, socs=socs, n_per_type=n_per_type, workers=workers,
               seed=seed)
    df = S.load_generated(raw, catalog, role=role, model=model)
    usage = S.usage_report(raw, price_in=price_in, price_out=price_out)
    report = {"role": role, "provider": provider, "model": model, "prompt_version": S.PROMPT_VERSION,
              "raw_log": str(raw), "snippets": len(df), "usage": usage,
              "by_input_type": df["input_type"].value_counts().to_dict() if len(df) else {},
              "title_leak_rate": float(df["title_leak"].mean()) if len(df) else None,
              "mean_words": float(df["text"].str.split().str.len().mean()) if len(df) else None}
    if pilot:
        key = f"{role}:{model}"
        u = usage.get(key, {})
        if u.get("usd") is not None and u["calls"] - u["failed"] > 0:
            per_call = u["usd"] / (u["calls"] - u["failed"])
            report["estimated_full_run_usd"] = per_call * len(catalog.socs)
        out = paths.matching / "synth" / f"pilot_{role}_{model.replace('/', '_')}.parquet"
        df.to_parquet(out, index=False)
        report["pilot_parquet"] = str(out)
        (paths.base / "outputs" / "synth").mkdir(parents=True, exist_ok=True)
        (paths.base / "outputs" / "synth" / f"pilot_{role}_{model.replace('/', '_')}.json").write_text(
            json.dumps(report, indent=2))
    else:
        df.to_parquet(synthetic_path(paths, role), index=False)
        (paths.base / "outputs" / "synth").mkdir(parents=True, exist_ok=True)
        (paths.base / "outputs" / "synth" / f"{role}_{model.replace('/', '_')}.json").write_text(
            json.dumps(report, indent=2))
    return report


def lookup(paths: Paths, query: str, n: int = 10) -> pd.DataFrame:
    """Keyword search over official + alternate titles."""
    from onet_data_collector.bulk import load_bulk

    from .retrieval import BM25Retriever

    catalog = OccupationCatalog.load(paths.catalog_dir)
    titles = load_bulk(catalog.meta.get("onet_release")).table("job_titles")
    alt = titles.groupby("soc_code")["job_title"].apply(lambda s: " ; ".join(s)).to_dict()
    docs = [f"{t} ; {alt.get(s, '')}" for s, t in zip(catalog.socs, catalog.occupations["title"])]
    scores = BM25Retriever().fit(docs).score([query])[0]
    top = np.argsort(-scores)[:n]
    return pd.DataFrame({"soc": np.asarray(catalog.socs)[top],
                         "title": catalog.occupations["title"].to_numpy()[top],
                         "score": scores[top].round(2)})
