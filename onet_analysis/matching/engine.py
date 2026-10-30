"""Serving: load a model bundle and match free text to occupations.

A **bundle** is a self-contained directory produced by
``onet-analyze matching bundle`` — the occupation catalog, the model(s), cached
occupation embeddings and the calibration temperature — so serving needs
neither the O*NET release nor the training data::

    bundle/
      bundle.json          spec, temperature, provenance (source run, O*NET release)
      catalog/             occupation tables (parquet)
      biencoder/           sentence-transformers model (dense bundles)
      reranker/            cross-encoder (optional)
      doc_embeddings.npy   occupation embeddings (dense bundles)

Bundle resolution for :func:`default_engine`: ``$ONET_MATCH_BUNDLE``, then
``./models/bundle``, then the pretrained bundle on the Hugging Face Hub
(``$ONET_MATCH_HF_REPO``, default :data:`DEFAULT_HF_REPO`), downloaded once into
the Hugging Face cache. ``hf:<owner>/<repo>`` selects a Hub bundle explicitly.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .catalog import DEFAULT_FIELDS, MAJOR_GROUPS, RIASEC, OccupationCatalog
from .retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Ranker,
    TfidfRetriever,
    softmax,
)

log = logging.getLogger("onet_analysis.matching.engine")

BUNDLE_FORMAT = 1
DEFAULT_K = 20
MAX_QUERY_CHARS = 8000


# --------------------------------------------------------------------------- spec
def parse_spec(spec: str) -> dict:
    """``tfidf`` | ``bm25`` | ``dense=<model>`` with optional ``+rerank=<model>[+fuse=<weight>]``.

    Examples: ``bm25``, ``dense=BAAI/bge-small-en-v1.5``,
    ``dense=models/biencoder-full+rerank=models/reranker-full+fuse=0.3``.
    """
    spec, _, fuse = spec.partition("+fuse=")
    first, _, rest = spec.partition("+rerank=")
    kind, _, model = first.partition("=")
    if kind not in ("tfidf", "bm25", "dense"):
        raise ValueError(f"Unknown retriever {kind!r} in spec {spec!r}.")
    if kind == "dense" and not model:
        raise ValueError("dense retriever needs a model: dense=<name-or-path>")
    if fuse and not rest:
        raise ValueError("+fuse=<weight> needs a +rerank=<model>")
    return {"retriever": kind, "model": model or None, "rerank": rest or None,
            "fuse": float(fuse) if fuse else None}


def build_ranker(spec: str | dict, docs: list[str], *, k: int = DEFAULT_K, device: str | None = None,
                 query_prompt: str | None = None, doc_embeddings: np.ndarray | None = None) -> Ranker:
    s = parse_spec(spec) if isinstance(spec, str) else spec
    if s["retriever"] == "tfidf":
        retriever = TfidfRetriever().fit(docs)
    elif s["retriever"] == "bm25":
        retriever = BM25Retriever().fit(docs)
    else:
        retriever = DenseRetriever(s["model"], device=device, query_prompt=query_prompt,
                                   doc_prompt=s.get("doc_prompt"))
        if doc_embeddings is not None:
            retriever.doc_embeddings = doc_embeddings
        else:
            retriever.fit(docs)
    reranker = CrossEncoderReranker(s["rerank"], device=device) if s.get("rerank") else None
    return Ranker(retriever, docs, reranker=reranker, k=k, fuse=s.get("fuse"))


# --------------------------------------------------------------------------- results
@dataclass
class OccupationMatch:
    rank: int
    soc: str
    title: str
    confidence: float
    score: float
    major_group: str
    major_group_title: str
    job_zone: int | None
    riasec_code: str
    description: str
    top_tasks: list[str]
    top_skills: list[dict]
    top_knowledge: list[dict]


@dataclass
class MatchResult:
    query: str
    matches: list[OccupationMatch]
    profile: dict
    model: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _elements(df: pd.DataFrame, n: int) -> list[dict]:
    df = df.sort_values("importance", ascending=False).head(n)
    return [{"name": r.name, "importance": round(float(r.importance), 2),
             "level": None if pd.isna(r.level) else round(float(r.level), 2)}
            for r in df.itertuples(index=False)]


def aggregate_profile(catalog: OccupationCatalog, socs: Sequence[str], weights: Sequence[float],
                      n: int = 10) -> dict:
    """Skills / knowledge / RIASEC interests across matched occupations.

    Each element's weight is the confidence-weighted mean of its O*NET
    importance rating (1-5) across the matched occupations; interests use the
    O*NET occupational-interest score (1-7). Occupations without ratings for a
    domain simply don't contribute to it.
    """
    w = pd.Series(np.asarray(weights, dtype=float), index=list(socs))
    if w.sum() <= 0:
        w[:] = 1.0
    w = w / w.sum()

    def weighted(df: pd.DataFrame, value: str) -> list[dict]:
        sub = df[df["soc"].isin(w.index)].copy()
        if sub.empty:
            return []
        sub["w"] = sub["soc"].map(w)
        # Renormalise over occupations that actually have ratings for this domain.
        cover = sub.groupby("soc")["w"].first().sum()
        agg = (sub.assign(wv=sub["w"] * sub[value]).groupby("name")["wv"].sum() / cover)
        agg = agg.sort_values(ascending=False).head(n)
        return [{"name": k, "weight": round(float(v), 3)} for k, v in agg.items()]

    interests = weighted(catalog.interests, "score")
    order = {name: i for i, name in enumerate(RIASEC)}
    interests = sorted(interests, key=lambda d: (-d["weight"], order.get(d["name"], 9)))
    return {
        "skills": weighted(catalog.skills, "importance"),
        "knowledge": weighted(catalog.knowledge, "importance"),
        "interests": interests,
        "riasec_code": "".join(d["name"][0] for d in interests[:3]),
        "weighting": "confidence-weighted mean of O*NET importance (skills/knowledge, 1-5) "
                     "and occupational-interest scores (RIASEC, 1-7)",
    }


# --------------------------------------------------------------------------- engine
class MatchEngine:
    """Match free text (titles, descriptions, experience, interests, skill lists) to occupations."""

    def __init__(self, catalog: OccupationCatalog, ranker: Ranker, temperature: float, *,
                 info: dict | None = None):
        self.catalog = catalog
        self.ranker = ranker
        self.temperature = temperature
        self.info = info or {}
        self._socs = np.asarray(catalog.socs)
        self._occ = catalog.occupations.set_index("soc")
        self._lock = threading.Lock()  # torch models are not re-entrant across threads

    # ------------------------------------------------------------ construction
    @classmethod
    def from_bundle(cls, directory: str | Path, *, device: str | None = None) -> "MatchEngine":
        d = Path(directory)
        if not (d / "bundle.json").is_file():
            raise FileNotFoundError(
                f"No match bundle at {d}. Build one with `onet-analyze matching bundle <spec>` "
                "or set ONET_MATCH_BUNDLE."
            )
        cfg = json.loads((d / "bundle.json").read_text())
        catalog = OccupationCatalog.load(d / "catalog")
        docs = catalog.representations(cfg["fields"], n_tasks=cfg["n_tasks"])
        spec = dict(cfg["spec"])
        if spec.get("model_dir"):
            spec["model"] = str(d / spec["model_dir"])
        if spec.get("rerank_dir"):
            spec["rerank"] = str(d / spec["rerank_dir"])
        emb_path = d / "doc_embeddings.npy"
        ranker = build_ranker(spec, docs, k=cfg.get("k", DEFAULT_K), device=device,
                              query_prompt=spec.get("query_prompt"),
                              doc_embeddings=np.load(emb_path) if emb_path.is_file() else None)
        return cls(catalog, ranker, cfg["temperature"], info=cfg)

    # ------------------------------------------------------------------ query
    def match(self, text: str, top_k: int = 5) -> MatchResult:
        return self.match_many([text], top_k=top_k)[0]

    def match_many(self, texts: Sequence[str], top_k: int = 5) -> list[MatchResult]:
        if not texts:
            return []
        clean = [str(t)[:MAX_QUERY_CHARS].strip() for t in texts]
        if any(not t for t in clean):
            raise ValueError("Input text must be non-empty.")
        top_k = max(1, min(int(top_k), self.ranker.k))
        with self._lock:
            ranking = self.ranker.rank(clean)
        probs = softmax(ranking.scores, self.temperature)
        out = []
        for q, idx, scores, p in zip(texts, ranking.indices, ranking.scores, probs):
            socs = list(self._socs[idx[:top_k]])
            matches = [self._occupation_match(r + 1, s, float(p[r]), float(scores[r]))
                       for r, s in enumerate(socs)]
            profile = aggregate_profile(self.catalog, socs, p[:top_k])
            out.append(MatchResult(q, matches, profile, model=self.model_info()))
        return out

    def _occupation_match(self, rank: int, soc: str, conf: float, score: float) -> OccupationMatch:
        o = self._occ.loc[soc]
        c = self.catalog
        return OccupationMatch(
            rank=rank, soc=soc, title=o["title"], confidence=round(conf, 4), score=round(score, 4),
            major_group=o["major_group"], major_group_title=MAJOR_GROUPS.get(o["major_group"], ""),
            job_zone=None if pd.isna(o["job_zone"]) else int(o["job_zone"]),
            riasec_code=o["riasec_code"], description=o["description"],
            top_tasks=c.top_tasks(soc, 5),
            top_skills=_elements(c.skills[c.skills["soc"] == soc], 8),
            top_knowledge=_elements(c.knowledge[c.knowledge["soc"] == soc], 5),
        )

    def occupation(self, soc: str) -> dict:
        """Full record for one occupation (raises ``KeyError`` if unknown)."""
        if soc not in self._occ.index:
            raise KeyError(soc)
        o = self._occ.loc[soc]
        c = self.catalog
        ints = c.interests[c.interests["soc"] == soc]
        return {
            "soc": soc, "title": o["title"], "description": o["description"],
            "major_group": o["major_group"], "major_group_title": MAJOR_GROUPS.get(o["major_group"], ""),
            "job_zone": None if pd.isna(o["job_zone"]) else int(o["job_zone"]),
            "riasec_code": o["riasec_code"],
            "tasks": c.tasks.loc[c.tasks["soc"] == soc, ["text", "task_type", "importance"]]
                     .rename(columns={"text": "task"}).astype(object)
                     .where(lambda df: df.notna(), None).to_dict("records"),
            "detailed_work_activities": c.dwas.loc[c.dwas["soc"] == soc]
                                        .sort_values("n_tasks", ascending=False)["text"].tolist(),
            "skills": _elements(c.skills[c.skills["soc"] == soc], 50),
            "knowledge": _elements(c.knowledge[c.knowledge["soc"] == soc], 50),
            "abilities": _elements(c.abilities[c.abilities["soc"] == soc], 60),
            "interests": [{"name": r.name, "score": round(float(r.score), 2)}
                          for r in ints.sort_values("score", ascending=False).itertuples(index=False)],
        }

    def model_info(self) -> dict:
        return {"name": self.info.get("name", "unnamed"), "spec": self.info.get("spec_string"),
                "onet_release": self.catalog.meta.get("onet_release"),
                "temperature": self.temperature, "source_run": self.info.get("source_run")}


# --------------------------------------------------------------------------- bundling
def write_bundle(
    out_dir: str | Path,
    spec: str,
    catalog: OccupationCatalog,
    *,
    temperature: float,
    fields: Sequence[str] = DEFAULT_FIELDS,
    n_tasks: int = 10,
    name: str | None = None,
    source_run: str | None = None,
    k: int = DEFAULT_K,
    device: str | None = None,
    display_spec: str | None = None,
) -> Path:
    """Write a self-contained serving bundle (copies local model dirs).

    ``display_spec`` is what the bundle reports as its spec (defaults to ``spec``);
    pass a path-free form to avoid leaking local paths through the API.
    """
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    s = parse_spec(spec)
    catalog.save(out / "catalog")
    stored = {"retriever": s["retriever"]}
    docs = catalog.representations(fields, n_tasks=n_tasks)  # production: all tasks
    if s["retriever"] == "dense":
        if Path(s["model"]).is_dir():
            shutil.copytree(s["model"], out / "biencoder", ignore=shutil.ignore_patterns("checkpoints"))
            stored.update(model_dir="biencoder", query_prompt="", doc_prompt="")
        else:
            stored["model"] = s["model"]
        ranker = build_ranker({**stored, "model": str(out / "biencoder") if "model_dir" in stored
                               else s["model"], "rerank": None}, docs, device=device,
                              query_prompt=stored.get("query_prompt"))
        np.save(out / "doc_embeddings.npy", ranker.retriever.doc_embeddings)
    if s["rerank"]:
        if Path(s["rerank"]).is_dir():
            shutil.copytree(s["rerank"], out / "reranker", ignore=shutil.ignore_patterns("checkpoints"))
            stored["rerank_dir"] = "reranker"
        else:
            stored["rerank"] = s["rerank"]
        stored["fuse"] = s["fuse"]
    cfg = {
        "format": BUNDLE_FORMAT, "name": name or display_spec or spec, "spec_string": display_spec or spec, "spec": stored,
        "fields": list(fields), "n_tasks": n_tasks, "k": k, "temperature": temperature,
        "source_run": source_run, "onet": {kk: v for kk, v in catalog.meta.items() if kk.startswith("onet_")},
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (out / "bundle.json").write_text(json.dumps(cfg, indent=2))
    log.info("Wrote bundle %s -> %s", cfg["name"], out)
    return out


_DEFAULT: MatchEngine | None = None
_DEFAULT_LOCK = threading.Lock()


DEFAULT_HF_REPO = "BenjaminChan12/onet-match"


def download_bundle(repo_id: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    repo_id = repo_id or os.environ.get("ONET_MATCH_HF_REPO") or DEFAULT_HF_REPO
    log.info("Downloading match bundle %s from the Hugging Face Hub", repo_id)
    return Path(snapshot_download(repo_id=repo_id))


def resolve_bundle_dir(bundle: str | Path | None = None) -> Path:
    bundle = bundle or os.environ.get("ONET_MATCH_BUNDLE")
    if bundle:
        if str(bundle).startswith("hf:"):
            return download_bundle(str(bundle)[3:])
        return Path(bundle)
    local = Path("models") / "bundle"
    return local if (local / "bundle.json").is_file() else download_bundle()


def default_engine(bundle: str | Path | None = None) -> MatchEngine:
    """Lazily load (and cache) the engine for ``bundle`` / the default bundle location."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if bundle is not None:
            return MatchEngine.from_bundle(resolve_bundle_dir(bundle))
        if _DEFAULT is None:
            _DEFAULT = MatchEngine.from_bundle(resolve_bundle_dir())
        return _DEFAULT


def match(text: str, top_k: int = 5, *, bundle: str | Path | None = None) -> MatchResult:
    """Match one free-text input to its closest O*NET occupations.

    >>> from onet_analysis.matching import match
    >>> result = match("I love weather, maps and tracking storms")
    >>> [(m.title, m.confidence) for m in result.matches]
    """
    return default_engine(bundle).match(text, top_k=top_k)
