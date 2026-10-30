"""MCP server for the matching engine, so AI assistants can call it as a tool.

    onet-match mcp                      # stdio, for desktop MCP clients
    onet-match mcp --transport streamable-http --port 8001
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .. import __version__
from .engine import MatchEngine, default_engine

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)


def _compact(m) -> dict[str, Any]:
    return {"rank": m.rank, "soc": m.soc, "title": m.title, "confidence": m.confidence,
            "major_group": m.major_group_title, "job_zone": m.job_zone, "riasec_code": m.riasec_code,
            "description": m.description, "top_tasks": m.top_tasks[:3],
            "top_skills": [s["name"] for s in m.top_skills[:5]]}


def create_server(engine: MatchEngine | None = None) -> MCPServer:
    """Build the server. ``engine`` is injectable for tests; otherwise loaded on first call."""
    server = MCPServer(
        "onet-match", version=__version__,
        instructions="Map free text about work (job titles, resume lines, job postings, interests, skill "
                     "lists) to O*NET-SOC occupations. Use match_occupations first, then get_occupation "
                     "for the full record of a returned SOC code.",
        website_url="https://github.com/Benjamin-V-Chan/onet-match",
    )

    def get_engine() -> MatchEngine:
        return engine or default_engine()

    @server.tool(annotations=READ_ONLY)
    def match_occupations(text: str, top_k: int = 5) -> dict[str, Any]:
        """Match free text about work to the closest O*NET occupations.

        text: a job title, resume bullet, job posting, project description, interest statement or
        skill list. top_k: number of occupations to return (1-20). Returns ranked occupations with
        SOC codes, calibrated confidence, key tasks and skills, plus an aggregate skill and RIASEC
        interest profile.
        """
        if not text.strip():
            raise ToolError("text must contain non-whitespace characters")
        r = get_engine().match(text[:8000], top_k=max(1, min(int(top_k), 20)))
        p = r.profile
        return {"query": r.query, "matches": [_compact(m) for m in r.matches],
                "profile": {"riasec_code": p["riasec_code"],
                            "skills": [d["name"] for d in p["skills"][:8]],
                            "knowledge": [d["name"] for d in p["knowledge"][:5]]}}

    @server.tool(annotations=READ_ONLY)
    def get_occupation(soc: str) -> dict[str, Any]:
        """Full O*NET record for one occupation by SOC code (e.g. "15-1252.00"): description, tasks,
        detailed work activities, skills, knowledge, abilities and RIASEC interests."""
        try:
            return get_engine().occupation(soc.strip())
        except KeyError:
            raise ToolError(f"Unknown O*NET-SOC code {soc!r}") from None

    return server


def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8001) -> None:
    import threading

    server = create_server()
    # Download and load the model while the client connects, so the first call isn't slow.
    threading.Thread(target=default_engine, daemon=True).start()
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run(transport, host=host, port=port)
