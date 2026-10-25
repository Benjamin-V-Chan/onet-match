"""Retrieval, metrics, evaluation harness, synthetic generation (mocked LLM),
serving engine, HTTP API and CLI — all offline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _matching_data import OCCUPATIONS, write_tiny_release
from onet_analysis.cli import main
from onet_analysis.matching import datasets as D
from onet_analysis.matching import pipeline as P
from onet_analysis.matching import synth as S
from onet_analysis.matching.catalog import OccupationCatalog
from onet_analysis.matching.engine import MatchEngine, aggregate_profile, parse_spec, write_bundle
from onet_analysis.matching.evaluate import load_run, run_eval
from onet_analysis.matching.metrics import fit_temperature, gold_hits, per_query_frame, reliability, summarize
from onet_analysis.matching.retrieval import BM25Retriever, Ranker, Ranking, TfidfRetriever, softmax
from onet_analysis.paths import default_paths


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project dir with the tiny release cached and data built via the pipeline."""
    monkeypatch.setenv("ONET_BULK_DIR", str(tmp_path / "bulk"))
    write_tiny_release(tmp_path / "bulk")
    paths = default_paths(tmp_path / "proj")
    P.build_data(paths)
    return paths


# ---------------------------------------------------------------- retrieval
@pytest.mark.parametrize("retriever", [TfidfRetriever, BM25Retriever])
def test_lexical_retrievers_rank_matching_doc_first(retriever):
    docs = ["patient care hospital nurse", "python software code api", "logo typography poster"]
    r = retriever().fit(docs)
    s = r.score(["write python code", "design a poster logo"])
    assert s.shape == (2, 3)
    assert s[0].argmax() == 1 and s[1].argmax() == 2


def test_bm25_idf_and_length_normalisation():
    docs = ["apple banana", "apple banana cherry date egg fig grape", "banana kiwi"]
    r = BM25Retriever().fit(docs)
    s = r.score(["apple"])[0]
    assert s[0] > s[1] > s[2] == 0  # same tf: shorter doc wins; absent term scores 0
    assert r.score(["banana"])[0][0] < s[0]  # common term has lower idf than a rarer one


def test_ranker_topk_sorted_and_reranker_applied():
    docs = ["a b", "b c", "c d", "d e"]
    ranker = Ranker(TfidfRetriever(token_pattern=r"\w+", stop_words=None).fit(docs), docs, k=3)
    rk = ranker.rank(["b", "d"])
    assert rk.indices.shape == (2, 3)
    assert (np.diff(rk.scores, axis=1) <= 1e-9).all()

    class Reverse:
        def rerank(self, queries, cands, docs):
            return -np.arange(cands.shape[1], dtype=float)[None, :].repeat(len(queries), 0) * -1

        def config(self):
            return {"type": "reverse"}

    rr = Ranker(ranker.retriever, docs, reranker=Reverse(), k=3).rank(["b"])
    assert rr.indices[0, 0] == rk.indices[0, 2]  # reranker reversed the order


# ---------------------------------------------------------------- metrics
def test_metrics_multilabel_hits_and_calibration():
    socs = ["A", "B", "C"]
    ranking = Ranking(np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]]),
                      np.array([[3.0, 1.0, 0.0], [2.0, 1.9, 0.0], [5.0, 0.0, -1.0]]))
    data = pd.DataFrame({"id": ["q1", "q2", "q3"], "text": ["x", "y", "z"], "source": "title",
                         "input_type": "job_title", "gold": ["A", "A;C", "B"]})
    hits = gold_hits(ranking, socs, [g.split(";") for g in data["gold"]])
    assert hits.tolist() == [[True, False, False], [False, True, True], [False, True, False]]
    t = fit_temperature(ranking.scores, hits)
    assert t > 0
    frame = per_query_frame(ranking, socs, data, t)
    assert frame["hit1"].tolist() == [True, False, False]
    assert frame["rr"].tolist() == [1.0, 0.5, 0.5]
    m = summarize(frame)
    assert m["top1"] == pytest.approx(1 / 3) and m["top5"] == 1.0 and m["mrr"] == pytest.approx(2 / 3)
    assert np.allclose(softmax(ranking.scores, t).sum(axis=1), 1)


def test_reliability_perfectly_calibrated_has_zero_ece():
    conf = np.array([0.95] * 20 + [0.15] * 20)
    correct = np.array([True] * 19 + [False] + [True] * 3 + [False] * 17)
    ece, bins = reliability(conf, correct)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert sum(b["count"] for b in bins) == 40


# ---------------------------------------------------------------- harness + pipeline
def test_run_eval_writes_traceable_outputs(project):
    m = P.evaluate(project, "bm25", run_name="t_bm25")
    cfg, metrics = load_run(project.runs / "t_bm25")
    assert cfg["spec"] == "bm25" and cfg["onet_release"] == "31.0"
    assert metrics["temperature"] > 0
    real = metrics["sets"]["real_test"]
    assert set(real["by_input_type"]) <= {"job_title", "task", "activity", "description"}
    assert 0 <= real["overall"]["top1"] <= real["overall"]["top5"] <= 1
    assert "latency" in metrics and metrics["latency"]["p50_ms"] > 0
    assert (project.runs / "t_bm25" / "predictions_real_test.parquet").is_file()
    assert m == metrics


# ---------------------------------------------------------------- synthetic data (mock LLM)
def test_synth_generation_resume_failures_and_cost(project):
    catalog, pairs = P.load_inputs(project)
    raw = project.matching / "synth" / "raw.jsonl"
    fail_once = {"done": False}

    def fail_on(prompt):
        if "Registered Nurses" in prompt and not fail_once["done"]:
            fail_once["done"] = True
            return True
        return False

    prov = S.MockProvider("claude-haiku-5-5", fail_on=fail_on)
    S.generate(catalog, pairs, prov, raw, role="train", n_per_type=2, workers=2)
    assert len(prov.calls) == len(OCCUPATIONS)
    S.generate(catalog, pairs, prov, raw, role="train", n_per_type=2, workers=2)  # resume: retries failure only
    assert len(prov.calls) == len(OCCUPATIONS) + 1
    df = S.load_generated(raw, catalog, role="train")
    assert set(df["input_type"]) == set(S.INPUT_TYPES)
    assert (df["split"] == "train").all() and set(df["gold"]) == set(OCCUPATIONS)
    usage = S.usage_report(raw)["train:claude-haiku-5-5"]
    assert usage["failed"] == 1 and usage["usd"] > 0


def test_synth_train_prompts_use_only_train_tasks_and_eval_only_test(project):
    catalog, pairs = P.load_inputs(project)
    test_tasks = set(pairs[(pairs.source == "task") & (pairs.split == "test")]["text"])
    train_tasks = set(pairs[(pairs.source == "task") & (pairs.split == "train")]["text"])
    for role, forbidden in (("train", test_tasks), ("eval", train_tasks)):
        prov = S.MockProvider("m")
        S.generate(catalog, pairs, prov, project.matching / f"{role}.jsonl", role=role, n_per_type=1)
        blob = "\n".join(prov.calls)
        assert not any(t in blob for t in forbidden)


def test_synthesize_pilot_reports_cost_and_writes_files(project):
    report = P.synthesize(project, role="train", provider="mock", model="mock-1", pilot=2, n_per_type=1,
                          price_in=1.0, price_out=5.0)
    assert report["snippets"] == 2 * len(S.INPUT_TYPES)
    assert report["estimated_full_run_usd"] > 0
    assert (project.base / "outputs" / "synth" / "pilot_train_mock-1.json").is_file()


def test_render_prompt_mentions_types_and_forbids_title():
    p = S.render_prompt("Graphic Designers", "Make visuals.", ["Draw logos."], ["Design"], 2)
    assert all(f'"{t}"' in p for t in S.INPUT_TYPES)
    assert 'title "Graphic Designers"' in p and "Write 2 distinct" in p


def test_provider_factory_rejects_unknown_and_requires_key(monkeypatch):
    with pytest.raises(ValueError):
        S.make_provider("nope", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        S.make_provider("openai-compatible", "some-model")


# ---------------------------------------------------------------- engine / API / CLI
@pytest.fixture
def engine(project):
    P.evaluate(project, "bm25", run_name="t_bm25", latency=False)
    out = P.bundle(project, "bm25", run_name="t_bm25")
    return MatchEngine.from_bundle(out)


def test_parse_spec():
    assert parse_spec("dense=a/b+rerank=c") == {"retriever": "dense", "model": "a/b", "rerank": "c",
                                                   "fuse": None}
    with pytest.raises(ValueError):
        parse_spec("dense")
    with pytest.raises(ValueError):
        parse_spec("word2vec")


def test_engine_match_returns_confident_relevant_occupation(engine):
    r = engine.match("I keep track of patient medication and vitals on the hospital ward", top_k=3)
    assert r.matches[0].soc == "29-1141.00"
    assert [m.rank for m in r.matches] == [1, 2, 3]
    assert sum(m.confidence for m in r.matches) <= 1.0 + 1e-6
    assert r.matches[0].top_tasks and r.matches[0].top_skills
    assert r.profile["skills"] and len(r.profile["riasec_code"]) == 3
    assert r.model["source_run"] == "t_bm25"
    with pytest.raises(ValueError):
        engine.match("   ")


def test_aggregate_profile_is_importance_weighted(engine):
    cat = engine.catalog
    prof = aggregate_profile(cat, ["15-1252.00"], [1.0], n=50)
    expected = cat.skills[cat.skills["soc"] == "15-1252.00"].set_index("name")["importance"]
    for d in prof["skills"]:
        assert d["weight"] == pytest.approx(expected[d["name"]], abs=1e-3)


def test_occupation_record(engine):
    rec = engine.occupation("27-1024.00")
    assert rec["title"] == "Graphic Designers" and rec["tasks"] and len(rec["interests"]) == 6
    with pytest.raises(KeyError):
        engine.occupation("00-0000.00")


def test_http_api(engine):
    from fastapi.testclient import TestClient

    from onet_analysis.matching.api import create_app

    client = TestClient(create_app(engine))
    assert client.get("/health").json()["occupations"] == len(OCCUPATIONS)
    r = client.post("/match", json={"text": "design a logo and poster typography", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["matches"][0]["soc"] == "27-1024.00" and len(body["matches"]) == 2
    assert body["input_type"] == "auto"
    assert client.post("/match", json={"text": ""}).status_code == 422
    assert client.post("/match", json={"text": "x", "top_k": 50}).status_code == 422
    assert client.post("/match/batch", json={"texts": ["film script", "python api"]}).json()["results"][1][
        "matches"][0]["soc"] == "15-1252.00"
    assert client.get("/occupation/29-1141.00").json()["title"] == "Registered Nurses"
    assert client.get("/occupation/00-0000.00").status_code == 404


def test_cli_match_json(engine, project, capsys):
    bundle = project.bundle
    assert main(["match", "film", "camera", "scene", "--bundle", str(bundle), "--json", "--top-k", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["matches"][0]["soc"] == "27-2012.00"
    assert main(["match", "film", "--bundle", str(bundle)]) == 0
    assert "confidence=" in capsys.readouterr().out


def test_cli_matching_commands(project, capsys):
    base = str(project.base)
    assert main(["--base", base, "matching", "eval", "tfidf", "--run-name", "cli_tfidf", "--no-latency"]) == 0
    assert (project.runs / "cli_tfidf" / "metrics.json").is_file()
    assert main(["--base", base, "matching", "lookup", "logo"]) == 0
    assert "Graphic Designers" in capsys.readouterr().out
    assert main(["match", "x", "--bundle", str(project.base / "nope")]) == 1  # missing bundle -> clean error


def test_write_bundle_rejects_mismatched_run(project):
    P.evaluate(project, "tfidf", run_name="t_tfidf", latency=False)
    with pytest.raises(ValueError):
        P.bundle(project, "bm25", run_name="t_tfidf")


def test_cli_training_flags_do_not_clobber_global_base():
    from onet_analysis.cli import build_parser

    for cmd in (["train-biencoder", "--name", "n"], ["train-reranker", "--name", "n", "--biencoder", "b"]):
        args = build_parser().parse_args(["--base", "/proj", "matching", *cmd])
        assert args.base == "/proj" and args.base_model


def test_spec_fusion_and_score_blend():
    from onet_analysis.matching.engine import parse_spec
    from onet_analysis.matching.retrieval import fuse_scores
    assert parse_spec("dense=m+rerank=r+fuse=0.3")["fuse"] == 0.3
    assert parse_spec("dense=m+rerank=r")["fuse"] is None
    with pytest.raises(ValueError):
        parse_spec("dense=m+fuse=0.3")
    bi, ce = np.array([[3.0, 2.0, 1.0]]), np.array([[1.0, 2.0, 3.0]])
    assert fuse_scores(bi, ce, 0.0).argmax() == 0 and fuse_scores(bi, ce, 1.0).argmax() == 2


def test_bundle_resolution_falls_back_to_hub(tmp_path, monkeypatch):
    from onet_analysis.matching import engine as E

    calls = []
    monkeypatch.setattr(E, "download_bundle", lambda repo_id=None: calls.append(repo_id) or tmp_path)
    monkeypatch.delenv("ONET_MATCH_BUNDLE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert E.resolve_bundle_dir() == tmp_path and calls == [None]
    assert E.resolve_bundle_dir("hf:someone/bundle") == tmp_path and calls[-1] == "someone/bundle"
    assert E.resolve_bundle_dir(tmp_path / "local") == tmp_path / "local"
    (tmp_path / "models" / "bundle").mkdir(parents=True)
    (tmp_path / "models" / "bundle" / "bundle.json").write_text("{}")
    assert E.resolve_bundle_dir() == Path("models") / "bundle" and len(calls) == 2


def test_onet_match_alias():
    import onet_match
    from onet_analysis.matching import match

    assert onet_match.match is match
