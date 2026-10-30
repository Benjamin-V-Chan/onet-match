"""Build the in-browser demo (static Hugging Face Space) from a match bundle.

    python demo/build.py data  models/bundle demo/dist      # catalog, documents, embeddings
    python demo/build.py onnx  models/bundle demo/dist      # needs: pip install "optimum-onnx[onnxruntime]"

demo/dist plus demo/index.html, demo/app.js and demo/README.md is the Space.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def export_data(bundle: Path, out: Path) -> None:
    import pandas as pd

    from onet_analysis.matching.catalog import MAJOR_GROUPS, RIASEC, OccupationCatalog

    cfg = json.loads((bundle / "bundle.json").read_text())
    cat = OccupationCatalog.load(bundle / "catalog")
    docs = cat.representations(cfg["fields"], n_tasks=cfg["n_tasks"])
    emb = np.load(bundle / "doc_embeddings.npy").astype("<f4")
    assert emb.shape[0] == len(docs) == len(cat.socs)

    def ratings(df: pd.DataFrame, value: str, names=None) -> tuple[list[str], dict[str, list]]:
        names = list(names or sorted(df["name"].unique()))
        col = {n: i for i, n in enumerate(names)}
        per = {}
        for soc, g in df.groupby("soc"):
            row = [None] * len(names)
            for n, v in zip(g["name"], g[value]):
                row[col[n]] = round(float(v), 3)
            per[soc] = row
        return names, per

    skill_names, skills = ratings(cat.skills, "importance")
    know_names, know = ratings(cat.knowledge, "importance")
    _, interests = ratings(cat.interests, "score", RIASEC)

    def top(df: pd.DataFrame, soc: str, n: int) -> list[str]:
        return df[df["soc"] == soc].sort_values("importance", ascending=False)["name"].head(n).tolist()

    occ = []
    for r in cat.occupations.itertuples(index=False):
        occ.append({
            "soc": r.soc, "title": r.title, "description": r.description,
            "major_group": MAJOR_GROUPS.get(r.major_group, ""),
            "job_zone": None if pd.isna(r.job_zone) else int(r.job_zone), "riasec_code": r.riasec_code,
            "tasks": cat.top_tasks(r.soc, 5), "skills": top(cat.skills, r.soc, 8),
            "knowledge": top(cat.knowledge, r.soc, 5),
            "r_skills": skills.get(r.soc), "r_knowledge": know.get(r.soc), "r_interests": interests.get(r.soc),
        })
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    meta = {"k": cfg["k"], "fuse": cfg["spec"]["fuse"], "temperature": cfg["temperature"],
            "dim": int(emb.shape[1]), "riasec": list(RIASEC), "skill_names": skill_names,
            "knowledge_names": know_names, "onet_release": cat.meta.get("onet_release"),
            "model": cfg["name"]}
    (out / "data" / "meta.json").write_text(json.dumps(meta))
    (out / "data" / "occupations.json").write_text(json.dumps(occ, separators=(",", ":")))
    (out / "data" / "docs.json").write_text(json.dumps(docs, separators=(",", ":")))
    (out / "data" / "embeddings.bin").write_bytes(emb.tobytes())
    print(f"data: {len(occ)} occupations, embeddings {emb.shape}")


def export_onnx(bundle: Path, out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Per-channel int8 keeps the bi-encoder within noise of the PyTorch model; plain uint8 cost
    # ~1.3 points top-1. The reranker is fine with uint8.
    for name, task, per_channel in [("biencoder", "feature-extraction", True),
                                    ("reranker", "text-classification", False)]:
        d = out / "models" / name
        shutil.rmtree(d, ignore_errors=True)
        subprocess.run(["optimum-cli", "export", "onnx", "--model", str(bundle / name), "--task", task, str(d)],
                       check=True)
        (d / "onnx").mkdir()
        (d / "model.onnx").rename(d / "onnx" / "model.onnx")
        quantize_dynamic(d / "onnx" / "model.onnx", d / "onnx" / "model_quantized.onnx",
                         weight_type=QuantType.QInt8 if per_channel else QuantType.QUInt8,
                         per_channel=per_channel)
        (d / "onnx" / "model.onnx").unlink()  # the browser loads the int8 model only
        print(f"onnx: {name} -> {d}")


if __name__ == "__main__":
    step, bundle, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {"data": export_data, "onnx": export_onnx}[step](bundle, out)
