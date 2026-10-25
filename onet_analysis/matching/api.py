"""HTTP service for the matching engine (FastAPI). Contract: ``docs/API.md``.

    onet-analyze serve --bundle models/bundle --port 8000
    # or: uvicorn onet_analysis.matching.api:app   (reads $ONET_MATCH_BUNDLE)
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import MatchEngine, default_engine

API_VERSION = "1"
MAX_BATCH = 64


class MatchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000,
                      description="Any free text about work: a title, posting, experience, project, "
                                  "interests/strengths statement or a skill list.")
    top_k: int = Field(5, ge=1, le=20, description="Number of occupations to return.")
    input_type: Literal["auto", "job_title", "description", "experience", "project", "interest",
                        "skills"] = Field("auto", description="Optional hint; echoed back, not used for "
                                                              "routing (the model is type-agnostic).")


class BatchMatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    top_k: int = Field(5, ge=1, le=20)


def create_app(engine: MatchEngine | None = None) -> FastAPI:
    """Build the app. ``engine`` is injectable for tests; otherwise loaded lazily."""
    app = FastAPI(title="O*NET semantic occupation matching", version=API_VERSION,
                  description="Match free text about work to O*NET occupations.")
    state = {"engine": engine}

    def get_engine() -> MatchEngine:
        if state["engine"] is None:
            try:
                state["engine"] = default_engine()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["engine"]

    @app.get("/health")
    def health() -> dict:
        eng = get_engine()
        return {"status": "ok", "api_version": API_VERSION, "model": eng.model_info(),
                "occupations": len(eng.catalog.socs)}

    @app.post("/match")
    def match(req: MatchRequest) -> dict:
        if not req.text.strip():
            raise HTTPException(status_code=422, detail="text must contain non-whitespace characters")
        result = get_engine().match(req.text, top_k=req.top_k).to_dict()
        result["input_type"] = req.input_type
        return result

    @app.post("/match/batch")
    def match_batch(req: BatchMatchRequest) -> dict:
        if any(not t.strip() for t in req.texts):
            raise HTTPException(status_code=422, detail="every text must contain non-whitespace characters")
        results = get_engine().match_many([t[:8000] for t in req.texts], top_k=req.top_k)
        return {"results": [r.to_dict() for r in results]}

    @app.get("/occupation/{soc}")
    def occupation(soc: str) -> dict:
        try:
            return get_engine().occupation(soc)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown O*NET-SOC code {soc!r}") from exc

    return app


app = create_app()
