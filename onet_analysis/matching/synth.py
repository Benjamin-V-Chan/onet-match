"""LLM-generated messy inputs (job descriptions, experience bullets, projects,
interest statements, skill lists), each labeled with its source occupation.

* ``role="train"`` examples are used **for training only**; the prompt shows the
  LLM only *train-split* tasks.
* ``role="eval"`` examples form the held-out synthetic test set. They must come
  from a **different LLM** than the training examples (to avoid rewarding a
  model for learning one generator's style) and the prompt shows only
  *test-split* tasks.

Every call is logged verbatim to a JSONL file (prompt version, model, usage) so
datasets can be rebuilt, audited and costed. The prompt is versioned in
:data:`PROMPT_VERSION`; :func:`render_prompt` is the single source of the text
(documented in ``docs/SYNTHETIC_DATA.md``).

Providers are pluggable (:class:`Provider`): Anthropic, any OpenAI-compatible
endpoint, and :class:`MockProvider` for offline tests. API keys are read from
environment variables only.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .catalog import OccupationCatalog
from .text import content_tokens, normalize

log = logging.getLogger("onet_analysis.matching.synth")

PROMPT_VERSION = "v1"
INPUT_TYPES: dict[str, str] = {
    "description": "an informal job description or job-posting excerpt, as a recruiter, hiring manager "
                   "or small-business owner might write it (2-5 sentences; may mention perks, schedule, "
                   "abbreviations or small typos; may use an informal or company-specific job title, but "
                   "not the exact official occupation title)",
    "experience": "a person describing their own past work, volunteer, military or school experience — "
                  "resume-style bullets or one or two first-person sentences (e.g. \"ran the robotics "
                  "club's CAD team\")",
    "project": "a description of a specific project someone did at school, at work or as a hobby "
               "(1-3 sentences, first or third person)",
    "interest": "an interests or strengths statement from a student or career changer that points toward "
                "this kind of work without naming any job (e.g. \"I love drawing and figuring out how "
                "machines work\")",
    "skills": "a list of 4-10 skills, tools or certifications as someone might type them into a profile "
              "(comma-, slash- or newline-separated; mixed capitalisation is fine)",
}

SYSTEM_PROMPT = (
    "You write realistic, varied, human-sounding text snippets that people type into career and hiring "
    "tools. Your snippets are used to train and test a system that maps free text to occupations, so "
    "they must sound like real people (students, career changers, recruiters, experienced workers) rather "
    "than like an occupational database. Paraphrase in everyday language; never copy the reference "
    "material verbatim."
)

# USD per million tokens (input, output). Anthropic rates from the Claude API
# model table (cached 2026-10-06). Other providers: pass prices explicitly.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def render_prompt(title: str, description: str, tasks: Sequence[str], skills: Sequence[str],
                  n_per_type: int, input_types: Sequence[str] = tuple(INPUT_TYPES)) -> str:
    """The user prompt for one occupation (all requested input types in one call)."""
    task_lines = "\n".join(f"- {t}" for t in tasks) or "- (none listed)"
    type_lines = "\n".join(f"{i + 1}. \"{t}\": {INPUT_TYPES[t]}" for i, t in enumerate(input_types))
    return f"""Reference occupation (for you only — the writers must not name it):
Title: {title}
What it does: {description}
Example tasks:
{task_lines}
Important skills and knowledge: {", ".join(skills) or "(none listed)"}

Write {n_per_type} distinct snippet(s) for EACH of these input types:
{type_lines}

Rules:
- Each snippet should be most consistent with the reference occupation, but realistic: vary voice, age, \
education level, length and specificity; some can be vague, terse or slightly off-topic, like real input.
- Do not use the words of the official title "{title}" (the "description" type may use a different, \
informal job title).
- Paraphrase; do not copy task statements.
- Return JSON only, with one key per input type, each mapping to a list of {n_per_type} string(s)."""


def response_schema(input_types: Sequence[str] = tuple(INPUT_TYPES)) -> dict:
    return {
        "type": "object",
        "properties": {t: {"type": "array", "items": {"type": "string"}} for t in input_types},
        "required": list(input_types),
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- providers
@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class Provider:
    """Interface: ``complete(system, user, schema) -> Completion``."""

    name = "base"

    def __init__(self, model: str):
        self.model = model

    def complete(self, system: str, user: str, schema: dict) -> Completion:  # pragma: no cover
        raise NotImplementedError


class AnthropicProvider(Provider):
    """Claude via the official SDK; credentials from ``ANTHROPIC_API_KEY`` (or an ant profile)."""

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 4000):
        super().__init__(model)
        import anthropic

        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens

    def complete(self, system, user, schema):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Model refused: {response.stop_details}")
        text = next(b.text for b in response.content if b.type == "text")
        return Completion(text, response.usage.input_tokens, response.usage.output_tokens)


class OpenAICompatibleProvider(Provider):
    """Any OpenAI-compatible chat-completions endpoint (OpenAI, Gemini, Together, vLLM, ...).

    Reads the key from ``api_key_env`` (default ``OPENAI_API_KEY``) and an
    optional ``base_url``.
    """

    name = "openai-compatible"

    def __init__(self, model: str, *, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 max_tokens: int = 4000):
        super().__init__(model)
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Set {api_key_env} in the environment for provider {self.name}.")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.max_tokens = max_tokens

    def complete(self, system, user, schema):
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user + "\n\nJSON schema:\n" + json.dumps(schema)}],
            response_format={"type": "json_object"},
        )
        usage = response.usage
        return Completion(response.choices[0].message.content or "",
                          getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))


class MockProvider(Provider):
    """Deterministic offline provider for tests: echoes reference words per type."""

    name = "mock"

    def __init__(self, model: str = "mock-1", fail_on: Callable[[str], bool] | None = None):
        super().__init__(model)
        self.calls: list[str] = []
        self.fail_on = fail_on

    def complete(self, system, user, schema):
        self.calls.append(user)
        if self.fail_on and self.fail_on(user):
            raise RuntimeError("mock failure")
        n = int(user.split("Write ", 1)[1].split(" ", 1)[0])
        what = user.split("What it does: ", 1)[1].split("\n", 1)[0]
        out = {t: [f"{t} {i}: {what}" for i in range(n)] for t in schema["required"]}
        return Completion(json.dumps(out), len(user) // 4, 50 * n)


def make_provider(provider: str, model: str, **kwargs) -> Provider:
    if provider == "anthropic":
        return AnthropicProvider(model)
    if provider in ("openai", "openai-compatible"):
        return OpenAICompatibleProvider(model, **kwargs)
    if provider == "mock":
        return MockProvider(model)
    raise ValueError(f"Unknown provider {provider!r} (anthropic | openai-compatible | mock).")


# --------------------------------------------------------------------------- generation
def _reference(catalog: OccupationCatalog, pairs: pd.DataFrame, role: str, n_tasks: int, n_skills: int,
               seed: int) -> dict[str, dict]:
    """Per occupation: title, description, tasks from the role's split, top skills."""
    split = "train" if role == "train" else "test"
    tk = pairs[(pairs["source"] == "task") & (pairs["split"] == split)]
    by_soc: dict[str, list[str]] = {}
    for text, gold in zip(tk["text"], tk["gold"]):
        for g in gold.split(";"):
            by_soc.setdefault(g, []).append(text)
    sk = pd.concat([catalog.skills, catalog.knowledge]).sort_values(["soc", "importance"],
                                                                     ascending=[True, False])
    skills = sk.groupby("soc")["name"].apply(lambda s: list(s.head(n_skills))).to_dict()
    rng = np.random.default_rng(seed)
    ref = {}
    for row in catalog.occupations.itertuples(index=False):
        tasks = by_soc.get(row.soc, [])
        if len(tasks) > n_tasks:
            tasks = [tasks[i] for i in sorted(rng.choice(len(tasks), n_tasks, replace=False))]
        ref[row.soc] = {"title": row.title, "description": row.description, "tasks": tasks,
                        "skills": skills.get(row.soc, [])}
    return ref


def _parse(text: str, input_types: Sequence[str]) -> dict[str, list[str]]:
    data = json.loads(text)
    return {t: [s.strip() for s in data.get(t, []) if isinstance(s, str) and s.strip()] for t in input_types}


def generate(
    catalog: OccupationCatalog,
    pairs: pd.DataFrame,
    provider: Provider,
    raw_path: str | Path,
    *,
    role: str = "train",
    socs: Sequence[str] | None = None,
    n_per_type: int = 3,
    input_types: Sequence[str] = tuple(INPUT_TYPES),
    n_tasks: int = 8,
    n_skills: int = 8,
    workers: int = 8,
    seed: int = 13,
) -> Path:
    """Generate snippets for ``socs`` (default: all occupations), appending to ``raw_path``.

    Resumable: occupations already present in ``raw_path`` (same role, model and
    prompt version) are skipped. Failures are logged and can be retried by
    re-running.
    """
    if role not in ("train", "eval"):
        raise ValueError("role must be 'train' or 'eval'")
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if raw_path.is_file():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("ok") and rec["role"] == role and rec["model"] == provider.model \
                    and rec["prompt_version"] == PROMPT_VERSION:
                done.add(rec["soc"])
    ref = _reference(catalog, pairs, role, n_tasks, n_skills, seed)
    todo = [s for s in (socs or catalog.socs) if s not in done]
    log.info("Generating %s snippets with %s/%s for %d occupations (%d already done).",
             role, provider.name, provider.model, len(todo), len(done))
    schema = response_schema(input_types)
    lock = threading.Lock()

    def one(soc: str) -> dict:
        r = ref[soc]
        prompt = render_prompt(r["title"], r["description"], r["tasks"], r["skills"], n_per_type, input_types)
        rec = {"soc": soc, "role": role, "provider": provider.name, "model": provider.model,
               "prompt_version": PROMPT_VERSION, "n_per_type": n_per_type,
               "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
               "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        try:
            c = provider.complete(SYSTEM_PROMPT, prompt, schema)
            rec.update(ok=True, items=_parse(c.text, input_types), input_tokens=c.input_tokens,
                       output_tokens=c.output_tokens)
        except Exception as exc:  # noqa: BLE001 - logged and retried on the next run
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, open(raw_path, "a", encoding="utf-8") as fh:
        futures = [pool.submit(one, s) for s in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if not rec["ok"]:
                log.warning("%s failed: %s", rec["soc"], rec["error"])
            if i % 50 == 0:
                log.info("  %d / %d occupations", i, len(todo))
    return raw_path


def load_generated(raw_path: str | Path, catalog: OccupationCatalog, *, role: str | None = None,
                   model: str | None = None) -> pd.DataFrame:
    """Flatten a raw JSONL log into labeled rows (same columns as the real pairs).

    Exact duplicates (after normalisation) are dropped. ``title_leak`` flags
    non-description snippets containing every content word of the official title.
    """
    titles = catalog.titles()
    rows = []
    for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("ok") or (role and rec["role"] != role) or (model and rec["model"] != model):
            continue
        for itype, texts in rec["items"].items():
            for text in texts:
                rows.append({"text": text, "source": "synthetic", "input_type": itype, "gold": rec["soc"],
                             "primary_soc": rec["soc"], "split": "train" if rec["role"] == "train" else "test",
                             "task_id": "", "hard": False, "generator": f'{rec["provider"]}/{rec["model"]}',
                             "prompt_version": rec["prompt_version"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["norm"] = df["text"].map(normalize)
    df = df.drop_duplicates("norm").drop(columns="norm").reset_index(drop=True)
    df["title_leak"] = [
        it != "description" and bool(content_tokens(titles[g])) and content_tokens(titles[g]) <= content_tokens(t)
        for t, g, it in zip(df["text"], df["gold"], df["input_type"])
    ]
    df["id"] = [f"syn-{df['split'][i]}-{i:06d}" for i in range(len(df))]
    return df


def usage_report(raw_path: str | Path, *, price_in: float | None = None, price_out: float | None = None) -> dict:
    """Token usage and cost per (role, model) in a raw log."""
    agg: dict[tuple, dict] = {}
    for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        key = (rec["role"], rec["model"])
        a = agg.setdefault(key, {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "snippets": 0})
        a["calls"] += 1
        if not rec.get("ok"):
            a["failed"] += 1
            continue
        a["input_tokens"] += rec["input_tokens"]
        a["output_tokens"] += rec["output_tokens"]
        a["snippets"] += sum(len(v) for v in rec["items"].values())
    out = {}
    for (role, model), a in agg.items():
        pin, pout = (price_in, price_out) if price_in is not None else PRICES.get(model, (None, None))
        a["usd"] = None if pin is None else a["input_tokens"] / 1e6 * pin + a["output_tokens"] / 1e6 * pout
        a["price_per_mtok"] = None if pin is None else {"input": pin, "output": pout}
        out[f"{role}:{model}"] = a
    return out
