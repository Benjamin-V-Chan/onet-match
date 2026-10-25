"""Occupation catalog: the matching targets and everything we return about them.

Built once from the O*NET bulk release (see
:mod:`onet_data_collector.bulk`) and saved as a small directory of parquet
files, so serving needs neither the raw release nor the collector at runtime.

Matching targets are the O*NET-SOC occupations that have task statements
(923 in release 31.0). The remaining ~90 codes are mostly "..., All Other"
residual categories with no tasks, ratings, skills or interests, so there is
nothing to match against or to return for them.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# SOC 2018 major groups (first two digits of the O*NET-SOC code).
MAJOR_GROUPS: dict[str, str] = {
    "11": "Management",
    "13": "Business and Financial Operations",
    "15": "Computer and Mathematical",
    "17": "Architecture and Engineering",
    "19": "Life, Physical, and Social Science",
    "21": "Community and Social Service",
    "23": "Legal",
    "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, and Media",
    "29": "Healthcare Practitioners and Technical",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation and Serving Related",
    "37": "Building and Grounds Cleaning and Maintenance",
    "39": "Personal Care and Service",
    "41": "Sales and Related",
    "43": "Office and Administrative Support",
    "45": "Farming, Fishing, and Forestry",
    "47": "Construction and Extraction",
    "49": "Installation, Maintenance, and Repair",
    "51": "Production",
    "53": "Transportation and Material Moving",
    "55": "Military Specific",
}

RIASEC = ("Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional")

# Representation field presets, used by the occupation-representation ablation.
FIELD_PRESETS: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "title+desc": ("title", "description"),
    "title+desc+tasks": ("title", "description", "tasks"),
    "title+desc+tasks+skills": ("title", "description", "tasks", "skills"),
}
DEFAULT_FIELDS = FIELD_PRESETS["title+desc+tasks"]

_TABLES = ("occupations", "tasks", "dwas", "skills", "knowledge", "abilities", "interests")


def major_group(soc: str) -> str:
    return str(soc)[:2]


def _rated(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot an element-rating table (IM/LV scales) to one row per (soc, element)."""
    df = df[df["scale_id"].isin(["IM", "LV"])].copy()
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    wide = (
        df.pivot_table(index=["soc_code", "element_id", "element_name"], columns="scale_id",
                       values="data_value", aggfunc="first")
        .reset_index()
        .rename(columns={"soc_code": "soc", "element_name": "name", "IM": "importance", "LV": "level"})
    )
    wide.columns.name = None
    for col in ("importance", "level"):
        if col not in wide:
            wide[col] = np.nan
    return wide[["soc", "element_id", "name", "importance", "level"]]


@dataclass
class OccupationCatalog:
    """In-memory catalog. All tables are keyed by ``soc`` (O*NET-SOC code)."""

    occupations: pd.DataFrame   # soc, title, description, major_group, job_zone, riasec_code
    tasks: pd.DataFrame         # soc, task_id, text, task_type, importance
    dwas: pd.DataFrame          # soc, dwa_id, text, n_tasks
    skills: pd.DataFrame        # soc, element_id, name, importance, level
    knowledge: pd.DataFrame
    abilities: pd.DataFrame
    interests: pd.DataFrame     # soc, name (RIASEC type), score (1-7)
    meta: dict

    # ------------------------------------------------------------------ build
    @classmethod
    def from_bulk(cls, release) -> "OccupationCatalog":
        """Build from an :class:`onet_data_collector.bulk.BulkRelease`."""
        occ = release.table("occupation_data").rename(columns={"soc_code": "soc"})

        ts = release.table("task_statements").rename(columns={"soc_code": "soc", "task": "text"})
        tr = release.table("task_ratings")
        im = tr[tr["scale_id"] == "IM"][["soc_code", "task_id", "data_value"]].rename(
            columns={"soc_code": "soc", "data_value": "importance"})
        im["importance"] = pd.to_numeric(im["importance"], errors="coerce")
        tasks = ts.merge(im, on=["soc", "task_id"], how="left")
        tasks["task_type"] = tasks["task_type"].replace("", "n/a")
        tasks = tasks[["soc", "task_id", "text", "task_type", "importance"]]
        # Rank tasks: importance desc (unrated last), Core before Supplemental.
        tasks["_core"] = (tasks["task_type"] == "Core").astype(int)
        tasks = (tasks.sort_values(["soc", "importance", "_core", "task_id"],
                                   ascending=[True, False, False, True], na_position="last")
                 .drop(columns="_core").reset_index(drop=True))

        targets = sorted(tasks["soc"].unique())
        occ = occ[occ["soc"].isin(targets)].copy()
        occ["major_group"] = occ["soc"].map(major_group)

        jz = release.table("job_zones").rename(columns={"soc_code": "soc"})
        occ = occ.merge(jz[["soc", "job_zone"]], on="soc", how="left")
        occ["job_zone"] = pd.to_numeric(occ["job_zone"], errors="coerce").astype("Int64")

        t2d = release.table("tasks_to_dwas").rename(columns={"soc_code": "soc"})
        ref = release.table("dwa_reference")[["dwa_id", "dwa_title"]].drop_duplicates("dwa_id")
        dwas = (t2d.merge(ref, on="dwa_id", how="left")
                .groupby(["soc", "dwa_id", "dwa_title"]).size().reset_index(name="n_tasks")
                .rename(columns={"dwa_title": "text"}))
        dwas = dwas[dwas["soc"].isin(targets)].reset_index(drop=True)

        skills = _rated(release.table("skills"))
        knowledge = _rated(release.table("knowledge"))
        abilities = _rated(release.table("abilities"))

        ints = release.table("interests").rename(columns={"soc_code": "soc"})
        ints["data_value"] = pd.to_numeric(ints["data_value"], errors="coerce")
        oi = ints[(ints["scale_id"] == "OI") & ints["element_name"].isin(RIASEC)]
        interests = oi[["soc", "element_name", "data_value"]].rename(
            columns={"element_name": "name", "data_value": "score"}).reset_index(drop=True)
        # High-point code (e.g. "IRC"): IH rows hold 1..6 = R..C, 0 = none.
        ih = ints[ints["scale_id"] == "IH"].sort_values(["soc", "element_id"])
        codes = (ih.groupby("soc")["data_value"]
                 .apply(lambda v: "".join("RIASEC"[int(x) - 1] for x in v if 1 <= x <= 6)))
        occ["riasec_code"] = occ["soc"].map(codes).fillna("")

        occ = occ[["soc", "title", "description", "major_group", "job_zone", "riasec_code"]]
        occ = occ.sort_values("soc").reset_index(drop=True)
        keep = lambda df: df[df["soc"].isin(targets)].reset_index(drop=True)  # noqa: E731
        meta = {
            "onet_release": release.release,
            "onet_source_url": release.manifest.get("source_url"),
            "onet_sha256": release.manifest.get("sha256"),
            "onet_downloaded_at": release.manifest.get("downloaded_at"),
            "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "n_occupations": len(occ),
        }
        return cls(occ, keep(tasks), dwas, keep(skills), keep(knowledge), keep(abilities),
                   keep(interests), meta)

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name in _TABLES:
            getattr(self, name).to_parquet(d / f"{name}.parquet", index=False)
        (d / "catalog.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "OccupationCatalog":
        d = Path(directory)
        if not (d / "catalog.json").is_file():
            raise FileNotFoundError(f"No occupation catalog at {d}. Run `onet-analyze matching build-data`.")
        tables = {name: pd.read_parquet(d / f"{name}.parquet") for name in _TABLES}
        meta = json.loads((d / "catalog.json").read_text(encoding="utf-8"))
        return cls(**tables, meta=meta)

    # ---------------------------------------------------------------- queries
    @property
    def socs(self) -> list[str]:
        return self.occupations["soc"].tolist()

    def titles(self) -> dict[str, str]:
        return dict(zip(self.occupations["soc"], self.occupations["title"]))

    def top_tasks(self, soc: str, n: int = 5) -> list[str]:
        return self.tasks.loc[self.tasks["soc"] == soc, "text"].head(n).tolist()

    def representations(
        self,
        fields: Sequence[str] = DEFAULT_FIELDS,
        *,
        n_tasks: int = 10,
        n_skills: int = 10,
        exclude_task_ids: Iterable[str] = (),
    ) -> list[str]:
        """One text per occupation (in :attr:`socs` order) built from ``fields``.

        ``exclude_task_ids`` removes held-out tasks so evaluation never matches a
        query against its own text.
        """
        unknown = set(fields) - {"title", "description", "tasks", "skills"}
        if unknown:
            raise ValueError(f"Unknown representation fields: {sorted(unknown)}")
        excluded = set(exclude_task_ids)
        tasks = self.tasks[~self.tasks["task_id"].isin(excluded)] if excluded else self.tasks
        task_map = tasks.groupby("soc")["text"].apply(lambda s: list(s.head(n_tasks))).to_dict()
        if "skills" in fields:
            sk = pd.concat([self.skills, self.knowledge])
            sk = sk.sort_values(["soc", "importance"], ascending=[True, False])
            skill_map = sk.groupby("soc")["name"].apply(lambda s: list(s.head(n_skills))).to_dict()

        out = []
        for row in self.occupations.itertuples(index=False):
            parts = []
            if "title" in fields:
                parts.append(f"{row.title}.")
            if "description" in fields:
                parts.append(str(row.description))
            if "tasks" in fields and task_map.get(row.soc):
                parts.append("Tasks: " + " ".join(task_map[row.soc]))
            if "skills" in fields and skill_map.get(row.soc):
                parts.append("Skills and knowledge: " + ", ".join(skill_map[row.soc]) + ".")
            out.append(" ".join(parts))
        return out
