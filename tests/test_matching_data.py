"""Catalog, labeled datasets and splits (offline, tiny synthetic release)."""

from __future__ import annotations

import pandas as pd
import pytest

from _matching_data import OCCUPATIONS, write_tiny_release
from onet_analysis.matching import datasets as D
from onet_analysis.matching.catalog import FIELD_PRESETS, OccupationCatalog
from onet_analysis.matching.text import content_tokens, normalize, shares_no_tokens
from onet_data_collector.bulk import load_bulk


@pytest.fixture
def tiny(tmp_path):
    write_tiny_release(tmp_path / "bulk")
    bulk = load_bulk(cache_dir=tmp_path / "bulk")
    catalog = OccupationCatalog.from_bulk(bulk)
    pairs = D.build_real_pairs(catalog, bulk.table("job_titles"), bulk.table("reported_titles"))
    return bulk, catalog, pairs


def test_text_helpers():
    assert normalize("  Sr.  Backend   ENG ") == "sr. backend eng"
    assert content_tokens("Registered Nurses") == {"registered", "nurse"}
    assert shares_no_tokens("Sr. Backend Eng II", "Software Developers")
    assert not shares_no_tokens("Staff Nurse", "Registered Nurses")


def test_catalog_targets_and_tables(tiny, tmp_path):
    _, catalog, _ = tiny
    assert set(catalog.socs) == set(OCCUPATIONS)          # "All Other" (no tasks) excluded
    assert catalog.occupations["riasec_code"].str.len().eq(3).all()
    assert catalog.occupations["job_zone"].notna().all()
    top = catalog.tasks[catalog.tasks["soc"] == "15-1252.00"]["importance"].tolist()
    assert top == sorted(top, reverse=True)                # tasks ranked by importance
    assert set(catalog.skills["name"]) == {"Programming", "Mathematics", "Active Listening",
                                           "Critical Thinking", "Service Orientation"}
    catalog.save(tmp_path / "cat")
    again = OccupationCatalog.load(tmp_path / "cat")
    assert again.socs == catalog.socs and again.meta["onet_release"] == "31.0"


def test_representation_fields(tiny):
    _, catalog, _ = tiny
    t = catalog.representations(FIELD_PRESETS["title"])
    full = catalog.representations(FIELD_PRESETS["title+desc+tasks+skills"])
    assert t[0] == "Software Developers." or t[0].endswith(".")
    assert all("Tasks:" in r and "Skills and knowledge:" in r for r in full)
    with pytest.raises(ValueError):
        catalog.representations(("bogus",))


def test_pairs_sources_labels_and_splits(tiny):
    _, catalog, pairs = tiny
    assert set(pairs["source"]) == {"title", "task", "dwa", "composite"}
    assert set(pairs["split"]) == {"train", "val", "test"}
    # Official titles are dropped; titles of non-target occupations are dropped.
    titles = pairs[pairs["source"] == "title"]
    assert not titles["text"].map(normalize).isin({normalize(t) for t in catalog.titles().values()}).any()
    assert "Computer Guru" not in set(titles["text"])
    # A title shared by all occupations becomes one multi-label row.
    coord = titles[titles["text"] == "Coordinator"]
    assert len(coord) == 1 and len(D.gold_list(coord["gold"].iloc[0])) == len(OCCUPATIONS)
    # Each unique normalised text lives in exactly one split.
    assert not pairs.groupby(["source", pairs["text"].map(normalize)])["split"].nunique().gt(1).any()


def test_composites_use_tasks_from_their_own_split(tiny):
    _, _, pairs = tiny
    tasks = pairs[pairs["source"] == "task"]
    split_of = dict(zip(tasks["text"], tasks["split"]))
    for row in pairs[pairs["source"] == "composite"].itertuples():
        parts = [s.strip() + "." for s in row.text.split(".") if s.strip()]
        assert {split_of[p] for p in parts} == {row.split}


def test_hard_flag_means_no_shared_tokens_with_official_title(tiny):
    _, catalog, pairs = tiny
    titles = catalog.titles()
    for row in pairs[pairs["hard"]].itertuples():
        assert all(shares_no_tokens(row.text, titles[g]) for g in D.gold_list(row.gold))
    assert pairs.loc[pairs["source"] != "title", "hard"].eq(False).all()


def test_heldout_tasks_never_appear_in_eval_documents(tiny):
    _, catalog, pairs = tiny
    held = D.heldout_task_ids(pairs, catalog)
    assert held
    docs = " ".join(catalog.representations(exclude_task_ids=held, n_tasks=100))
    heldout_text = pairs[(pairs["source"] == "task") & (pairs["split"] != "train")]["text"]
    assert not any(t in docs for t in heldout_text)


def test_training_pairs_cap_dwa_labels_and_use_train_only(tiny):
    _, _, pairs = tiny
    tp = D.training_pairs(pairs)
    train_texts = set(pairs.loc[pairs["split"] == "train", "text"])
    assert set(tp["anchor"]) <= train_texts
    assert tp.groupby("anchor").size().max() <= max(D.MAX_DWA_LABELS_PER_TRAIN_PAIR, len(OCCUPATIONS))
    assert set(tp["source"]) <= {"title", "task", "dwa", "composite"}


def test_save_and_load_pairs(tiny, tmp_path):
    _, _, pairs = tiny
    path = D.save_pairs(pairs, tmp_path / "m" / "pairs.parquet")
    assert path.with_suffix(".summary.json").is_file()
    pd.testing.assert_frame_equal(D.load_pairs(path), pairs, check_dtype=False)
    with pytest.raises(FileNotFoundError):
        D.load_pairs(tmp_path / "missing.parquet")
