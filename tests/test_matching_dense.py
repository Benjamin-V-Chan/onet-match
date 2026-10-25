"""Dense retrieval, bi-encoder/cross-encoder fine-tuning and dense bundles,
using a tiny randomly initialised BERT built locally (no model downloads)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("datasets")

from _matching_data import build_tiny_vocab_model, write_tiny_release  # noqa: E402
from onet_analysis.matching import pipeline as P  # noqa: E402
from onet_analysis.matching import train as T  # noqa: E402
from onet_analysis.matching.engine import MatchEngine, build_ranker  # noqa: E402
from onet_analysis.matching.retrieval import DenseRetriever, default_prompts  # noqa: E402
from onet_analysis.paths import default_paths  # noqa: E402


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("ONET_BULK_DIR", str(tmp_path / "bulk"))
    write_tiny_release(tmp_path / "bulk")
    paths = default_paths(tmp_path / "proj")
    P.build_data(paths)
    catalog, pairs = P.load_inputs(paths)
    texts = list(pairs["text"]) + catalog.representations(("title", "description", "tasks", "skills"))
    enc = build_tiny_vocab_model(tmp_path / "tiny-enc", texts)
    ce = build_tiny_vocab_model(tmp_path / "tiny-ce", texts, cross_encoder=True)
    return paths, enc, ce


def test_default_prompts():
    assert default_prompts("BAAI/bge-small-en-v1.5")[0].startswith("Represent")
    assert default_prompts("intfloat/e5-base-v2") == ("query: ", "passage: ")
    assert default_prompts("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


def test_dense_retriever_scores_are_cosines(project):
    _, enc, _ = project
    r = DenseRetriever(str(enc), device="cpu").fit(["alpha", "beta gamma"])
    s = r.score(["alpha", "gamma"])
    assert s.shape == (2, 2) and (abs(s) <= 1.0001).all()


def test_neighbors_and_examples_exclude_gold(project):
    paths, enc, _ = project
    catalog, pairs = P.load_inputs(paths)
    docs = P.eval_docs(catalog, pairs)
    emb = DenseRetriever(str(enc), device="cpu").fit(docs).doc_embeddings
    nb = T.occupation_neighbors(emb, k=3)
    assert nb.shape == (len(docs), 3) and all(i not in row for i, row in enumerate(nb))
    from onet_analysis.matching import datasets as D

    ex = T.build_biencoder_examples(D.training_pairs(pairs), catalog.socs, docs, neighbors=nb)
    assert list(ex.columns) == ["anchor", "positive", "negative"]
    assert (ex["positive"] != ex["negative"]).all()
    no_neg = T.build_biencoder_examples(D.training_pairs(pairs), catalog.socs, docs)
    assert list(no_neg.columns) == ["anchor", "positive"]


def test_train_biencoder_reranker_eval_and_dense_bundle(project):
    paths, enc, ce = project
    bi = P.train_biencoder(paths, base=str(enc), name="bi", max_steps=2, batch_size=2, bf16=False)
    rec = json.loads((bi / "training.json").read_text())
    assert rec["steps"] == 2 and rec["hard_negatives"] and set(rec["sources"]) == {"title", "task", "dwa", "composite"}
    assert default_prompts(str(bi)) == ("", "")  # fine-tuned models are used without prompts

    rr = P.train_reranker(paths, biencoder=str(bi), base=str(ce), name="rr", n_queries=40, n_negatives=1, max_steps=2,
                          batch_size=2, bf16=False)
    rr_rec = json.loads((rr / "training.json").read_text())
    assert rr_rec["hard_negatives"] == 1 and rr_rec["steps"] == 2

    spec = f"dense={bi}+rerank={rr}"
    m = P.evaluate(paths, spec, run_name="t_ladder5", latency=False)
    assert "training" in json.loads((paths.runs / "t_ladder5" / "config.json").read_text())
    assert 0 <= m["sets"]["real_test"]["overall"]["top1"] <= 1

    out = P.bundle(paths, spec, run_name="t_ladder5")
    assert (out / "biencoder").is_dir() and (out / "reranker").is_dir() and (out / "doc_embeddings.npy").is_file()
    eng = MatchEngine.from_bundle(out, device="cpu")
    res = eng.match("patient medication", top_k=3)
    assert len(res.matches) == 3 and res.model["source_run"] == "t_ladder5"

    fusion = P.tune_fusion(paths, spec, step=0.5)
    assert [g["weight"] for g in fusion["grid"]] == [0.0, 0.5, 1.0]
    tuned = fusion["tuned_spec"]
    assert tuned.endswith(f"+fuse={fusion['chosen']['weight']:g}")
    P.evaluate(paths, tuned, run_name="t_fused", latency=False)
    eng = MatchEngine.from_bundle(P.bundle(paths, tuned, run_name="t_fused"), device="cpu")
    assert eng.ranker.fuse == fusion["chosen"]["weight"] and len(eng.match("patient", top_k=2).matches) == 2


def test_build_ranker_reuses_cached_embeddings(project):
    paths, enc, _ = project
    catalog, pairs = P.load_inputs(paths)
    docs = P.eval_docs(catalog, pairs)
    first = build_ranker(f"dense={enc}", docs, device="cpu")
    cached = build_ranker(f"dense={enc}", docs, device="cpu", doc_embeddings=first.retriever.doc_embeddings)
    q = ["code python"]
    assert (first.rank(q).indices == cached.rank(q).indices).all()
