"""Tiny synthetic O*NET bulk release + helpers for offline matching tests."""

from __future__ import annotations

import json
from pathlib import Path

# Six occupations across three major groups, each with a distinctive vocabulary.
OCCUPATIONS = {
    "15-1252.00": ("Software Developers", "Design and build computer software applications.",
                   ["code", "software", "python", "debug", "api", "deploy", "database", "testing"]),
    "15-2051.00": ("Data Scientists", "Analyze data with statistics and machine learning.",
                   ["statistics", "model", "dataset", "regression", "analytics", "forecast", "survey", "plots"]),
    "29-1141.00": ("Registered Nurses", "Provide and coordinate patient care.",
                   ["patient", "medication", "care", "vitals", "hospital", "wound", "triage", "bedside"]),
    "29-1215.00": ("Family Medicine Physicians", "Diagnose and treat patients of all ages.",
                   ["diagnose", "prescribe", "exam", "clinic", "illness", "referral", "chronic", "physical"]),
    "27-1024.00": ("Graphic Designers", "Create visual concepts for print and digital media.",
                   ["layout", "logo", "typography", "illustration", "branding", "color", "poster", "sketch"]),
    "27-2012.00": ("Producers and Directors", "Produce or direct stage, film and television productions.",
                   ["film", "script", "casting", "rehearsal", "camera", "scene", "studio", "stage"]),
}
SKILLS = ["Programming", "Mathematics", "Active Listening", "Critical Thinking", "Service Orientation"]
KNOWLEDGE = ["Computers and Electronics", "Medicine and Dentistry", "Design", "Mathematics", "Psychology"]
RIASEC = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]


def _tsv(path: Path, header: list[str], rows: list[list]) -> None:
    lines = ["\t".join(header)] + ["\t".join(str(x) for x in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tiny_release(root: Path, release: str = "31.0") -> Path:
    d = root / f"db_{release.replace('.', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    occ, titles, reported, tasks, ratings, t2d, dwa_ref = [], [], [], [], [], [], []
    ess, trans, know, abil, ints, jz = [], [], [], [], [], []
    task_id, dwa_id = 1000, 0
    for k, (soc, (title, desc, words)) in enumerate(OCCUPATIONS.items()):
        occ.append([soc, title, desc])
        for i, w in enumerate(words):
            titles.append([soc, f"{w.title()} Specialist", "n/a", "08"])
            if i < 3:
                reported.append([soc, f"{w.title()} Specialist", "Y"])
        titles.append([soc, "Coordinator", "n/a", "10"])  # shared title -> multi-label gold
        titles.append([soc, title, "n/a", "10"])           # identical to official title -> dropped
        for i in range(10):
            a, b = words[i % len(words)], words[(i + 3) % len(words)]
            tasks.append([soc, task_id, f"Handle {a} work and {b} duties number {i}.", "Core", 50, "08/2023", "Incumbent"])
            ratings.append([soc, task_id, "IM", "n/a", 5 - i * 0.3, 10, 0, 0, 0, "N", "08/2023", "Incumbent"])
            dwa = f"4.A.{k}.{i % 3}"
            t2d.append([soc, task_id, dwa, "07/2014", "Analyst"])
            task_id += 1
        for i in range(3):
            dwa_ref.append(["4.A", "4.A.x", f"4.A.{k}.{i}", f"Perform {words[i]} activity."])
        dwa_ref.append(["4.A", "4.A.x", "4.A.shared", "Communicate with others."])
        t2d.append([soc, task_id - 1, "4.A.shared", "07/2014", "Analyst"])
        for j, name in enumerate(SKILLS):
            table = ess if j < 2 else trans
            table.append([soc, f"2.A.{j}", name, "IM", 1 + (j + k) % 5, 8, 0, 0, 0, "N", "n/a", "08/2023", "Analyst"])
            table.append([soc, f"2.A.{j}", name, "LV", 2 + (j + k) % 5, 8, 0, 0, 0, "N", "N", "08/2023", "Analyst"])
        for j, name in enumerate(KNOWLEDGE):
            know.append([soc, f"2.C.{j}", name, "IM", 1 + (j * 2 + k) % 5, 8, 0, 0, 0, "N", "n/a", "08/2023", "Analyst"])
        abil.append([soc, "1.A.1", "Oral Comprehension", "IM", 3.5, 8, 0, 0, 0, "N", "n/a", "08/2023", "Analyst"])
        for j, name in enumerate(RIASEC):
            ints.append([soc, f"1.B.1.{j}", name, "OI", 1 + (j + k) % 7, "02/2026", "ML"])
        for j in range(3):
            ints.append([soc, f"1.B.1.{'ghi'[j]}", ["First", "Second", "Third"][j] + " Interest High-Point",
                         "IH", (k + j) % 6 + 1, "02/2026", "ML"])
        jz.append([soc, 2 + k % 3, "08/2023", "Analyst"])
    # An "All Other" occupation with no tasks: must be excluded from targets.
    occ.append(["15-1299.00", "Computer Occupations, All Other", "All other computer occupations."])
    titles.append(["15-1299.00", "Computer Guru", "n/a", "10"])

    rated = ["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value", "N", "Standard Error",
             "Lower CI Bound", "Upper CI Bound", "Recommend Suppress", "Not Relevant", "Date", "Domain Source"]
    _tsv(d / "Occupation Data.txt", ["O*NET-SOC Code", "Title", "Description"], occ)
    _tsv(d / "Job Titles.txt", ["O*NET-SOC Code", "Job Title", "Short Title", "Source(s)"], titles)
    _tsv(d / "Sample of Reported Titles.txt", ["O*NET-SOC Code", "Reported Job Title", "Shown in My Next Move"], reported)
    _tsv(d / "Task Statements.txt", ["O*NET-SOC Code", "Task ID", "Task", "Task Type", "Incumbents Responding",
                                     "Date", "Domain Source"], tasks)
    _tsv(d / "Task Ratings.txt", ["O*NET-SOC Code", "Task ID", "Scale ID", "Category", "Data Value", "N",
                                  "Standard Error", "Lower CI Bound", "Upper CI Bound", "Recommend Suppress",
                                  "Date", "Domain Source"], ratings)
    _tsv(d / "Tasks to DWAs.txt", ["O*NET-SOC Code", "Task ID", "DWA Element ID", "Date", "Domain Source"], t2d)
    _tsv(d / "GWAs to IWAs to DWAs.txt", ["GWA Element ID", "IWA Element ID", "DWA Element ID", "DWA Element Name"],
         dwa_ref)
    _tsv(d / "Essential Skills.txt", rated, ess)
    _tsv(d / "Transferable Skills.txt", rated, trans)
    _tsv(d / "Knowledge.txt", rated, know)
    _tsv(d / "Abilities.txt", rated, abil)
    _tsv(d / "Career Interest Types.txt", ["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value",
                                           "Date", "Domain Source"], ints)
    _tsv(d / "Job Zones.txt", ["O*NET-SOC Code", "Job Zone", "Date", "Domain Source"], jz)
    (d / "manifest.json").write_text(json.dumps({"release": release, "source_url": "test://tiny",
                                                 "sha256": "0" * 64, "downloaded_at": "2026-10-23T00:00:00+00:00"}))
    return d


def build_tiny_vocab_model(out_dir: Path, texts: list[str], *, cross_encoder: bool = False) -> Path:
    """A randomly initialised 2-layer BERT saved locally (no downloads)."""
    import re

    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

    words = sorted({w for t in texts for w in re.findall(r"[a-z0-9]+", t.lower())})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words, ".", ",", "'"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(out_dir / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(out_dir)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=256, num_labels=1)
    torch.manual_seed(0)
    model = (BertForSequenceClassification if cross_encoder else BertModel)(cfg)
    model.save_pretrained(out_dir)
    return out_dir
