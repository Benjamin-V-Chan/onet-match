"""Semantic matching: free text about work -> closest O*NET occupations.

The input can be anything a person writes about work — a job title, a messy job
posting, an experience bullet, a project, an interests/strengths statement or a
list of skills. Each match carries a calibrated confidence plus the
occupation's key tasks, skills and knowledge, and the result aggregates
skills / knowledge / RIASEC interests across the matches.

    from onet_analysis.matching import match
    result = match("ran the robotics club's CAD team", top_k=5)

Modules: :mod:`catalog` (targets), :mod:`datasets` (labeled data + splits),
:mod:`synth` (LLM-generated inputs), :mod:`retrieval` (TF-IDF / BM25 / dense /
cross-encoder), :mod:`train`, :mod:`metrics`, :mod:`evaluate`, :mod:`engine`
(serving), :mod:`api` (FastAPI), :mod:`report` (write-up tables).

Heavy dependencies (torch, sentence-transformers, FastAPI) are optional extras
and only imported when a dense model, training or the HTTP service is used.
"""

from __future__ import annotations

from .engine import MatchEngine, MatchResult, OccupationMatch, default_engine, match

__all__ = ["match", "MatchEngine", "MatchResult", "OccupationMatch", "default_engine"]
