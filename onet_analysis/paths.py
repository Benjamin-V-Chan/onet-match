"""Canonical on-disk layout for the analytics pipeline.

All paths are relative to a configurable base directory (default: current working
directory), so the same code runs locally, in notebooks, or in CI.

    <base>/
      data/raw/        keyword_search_results.csv, job_details.json
      data/processed/  condensed_job_details.csv, coverage tables, cluster assignments
      data/results/    keyword_analysis_results.csv
      results/         plots (.png) and summary text
      data/matching/   occupation catalog + labeled matching datasets
      models/          fine-tuned matching models and the serving bundle
      outputs/runs/    one directory per benchmark run (metrics + predictions)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Resolved paths for every pipeline artifact."""

    base: Path

    @property
    def raw(self) -> Path:
        return self.base / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.base / "data" / "processed"

    @property
    def data_results(self) -> Path:
        return self.base / "data" / "results"

    @property
    def results(self) -> Path:
        return self.base / "results"

    # Individual artifacts -------------------------------------------------
    @property
    def keyword_search_csv(self) -> Path:
        return self.raw / "keyword_search_results.csv"

    @property
    def job_details_json(self) -> Path:
        return self.raw / "job_details.json"

    @property
    def condensed_csv(self) -> Path:
        return self.processed / "condensed_job_details.csv"

    @property
    def keyword_analysis_csv(self) -> Path:
        return self.data_results / "keyword_analysis_results.csv"

    # Semantic matching ------------------------------------------------------
    @property
    def matching(self) -> Path:
        return self.base / "data" / "matching"

    @property
    def catalog_dir(self) -> Path:
        return self.matching / "catalog"

    @property
    def pairs(self) -> Path:
        return self.matching / "pairs.parquet"

    @property
    def models(self) -> Path:
        return self.base / "models"

    @property
    def bundle(self) -> Path:
        return self.models / "bundle"

    @property
    def runs(self) -> Path:
        return self.base / "outputs" / "runs"

    def make_all(self) -> "Paths":
        """Create every directory in the layout and return self."""
        for d in (self.raw, self.processed, self.data_results, self.results):
            d.mkdir(parents=True, exist_ok=True)
        return self


def default_paths(base: str | Path = ".") -> Paths:
    return Paths(Path(base).resolve())
