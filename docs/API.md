# Matching API — HTTP contract

The FastAPI service in `onet_analysis/matching/api.py` exposes the matching
engine over HTTP. API version: **1**. All bodies are JSON (UTF-8).

```bash
onet-match serve --host 0.0.0.0 --port 8000      # pretrained bundle from the Hugging Face Hub
onet-match serve --bundle models/bundle          # or a local bundle
# or
ONET_MATCH_BUNDLE=models/bundle uvicorn onet_analysis.matching.api:app --port 8000
# or Docker (the image downloads the pretrained bundle at build time)
docker build -t onet-match .
docker run -p 8000:8000 onet-match
```

Interactive OpenAPI docs are served at `/docs`, and the schema at `/openapi.json`.

---

## `POST /match`

Match one free-text input to the closest O*NET occupations.

### Request

| field        | type    | required | constraints | description |
|--------------|---------|----------|-------------|-------------|
| `text`       | string  | yes      | 1–8000 chars, not only whitespace | Any text about work: a job title, a posting, an experience bullet, a project, an interests/strengths statement or a skill list. |
| `top_k`      | integer | no (5)   | 1–20        | Number of occupations to return. |
| `input_type` | string  | no (`"auto"`) | one of `auto`, `job_title`, `description`, `experience`, `project`, `interest`, `skills` | A hint that is echoed back. The model doesn't use it: one model handles every input type. |

```json
{"text": "ran the robotics club's CAD team", "top_k": 3}
```

### Response `200`

| field | type | description |
|---|---|---|
| `query` | string | The input text. |
| `input_type` | string | Echo of the request hint. |
| `matches` | array | Ranked occupations, best first (see below). |
| `profile` | object | Skills / knowledge / interests aggregated across `matches`. |
| `model` | object | `name`, `spec`, `onet_release`, `temperature`, `source_run` (the benchmark run the model and its calibration come from). |

Each element of `matches`:

| field | type | description |
|---|---|---|
| `rank` | int | 1-based rank. |
| `soc` | string | O*NET-SOC code, e.g. `17-2199.08`. |
| `title` | string | Official O*NET title. |
| `confidence` | float 0–1 | Calibrated probability that this occupation is the correct one, among the top-20 candidates. Confidences across returned matches sum to ≤ 1. |
| `score` | float | Raw model score (only comparable within one model). |
| `major_group`, `major_group_title` | string | SOC 2018 major group (`"17"`, `"Architecture and Engineering"`). |
| `job_zone` | int or null | O*NET Job Zone 1–5 (preparation needed). |
| `riasec_code` | string | O*NET interest high-point code, e.g. `"RIC"`. |
| `description` | string | O*NET occupation description. |
| `top_tasks` | string[] | Up to 5 most important tasks (O*NET task importance ratings). |
| `top_skills` | object[] | Up to 8 `{name, importance (1–5), level (0–7)}`, by importance. |
| `top_knowledge` | object[] | Up to 5 `{name, importance, level}`. |

`profile`:

| field | type | description |
|---|---|---|
| `skills` | object[] | Up to 10 `{name, weight}`. `weight` is the confidence-weighted mean O*NET importance (1–5) across the matched occupations. |
| `knowledge` | object[] | Same, for knowledge areas. |
| `interests` | object[] | All six RIASEC types `{name, weight}`. `weight` is the confidence-weighted mean occupational-interest score (1–7). |
| `riasec_code` | string | The top-3 interest initials of the aggregate, e.g. `"RIA"`. |
| `weighting` | string | A description of the weighting. |

Abridged example (real output of the served model, saved in `outputs/examples/http_match.json`):

```json
{
  "query": "ran the robotics club's CAD team",
  "input_type": "auto",
  "matches": [
    {
      "rank": 1,
      "soc": "17-2199.08",
      "title": "Robotics Engineers",
      "confidence": 0.5217,
      "score": 2.2739,
      "major_group": "17",
      "major_group_title": "Architecture and Engineering",
      "job_zone": 4,
      "riasec_code": "RIC",
      "description": "...",
      "top_tasks": [
        "Review or approve designs, calculations, or cost estimates.",
        "..."
      ],
      "top_skills": [
        {
          "name": "Critical Thinking",
          "importance": 4.0,
          "level": 4.62
        }
      ],
      "top_knowledge": [
        {
          "name": "Engineering and Technology",
          "importance": 4.79,
          "level": 6.13
        }
      ]
    }
  ],
  "profile": {
    "skills": [
      {
        "name": "Critical Thinking",
        "weight": 3.937
      }
    ],
    "knowledge": [
      {
        "name": "Engineering and Technology",
        "weight": 4.578
      }
    ],
    "interests": [
      {
        "name": "Realistic",
        "weight": 6.456
      },
      {
        "name": "Investigative",
        "weight": 5.031
      }
    ],
    "riasec_code": "RIC",
    "weighting": "confidence-weighted mean of O*NET importance (skills/knowledge, 1-5) and occupational-interest scores (RIASEC, 1-7)"
  },
  "model": {
    "name": "L5b_ft-bge-base+synth+ce+fuse",
    "spec": "dense=models/L4b_ft-bge-base+synth+rerank=models/L5s_ft-bge-small+synth+ce+fuse=0.3",
    "onet_release": "31.0",
    "temperature": 0.6147757481308899,
    "source_run": "L5b_ft-bge-base+synth+ce+fuse"
  }
}
```

### Errors

| status | when |
|---|---|
| `422` | Validation error: empty or whitespace-only `text`, `text` over 8000 chars, `top_k` outside 1–20, unknown `input_type`. The body uses FastAPI's standard `{"detail": [...]}`. |
| `503` | No model bundle is available (`detail` explains how to build one). |

---

## `POST /match/batch`

Match up to 64 inputs in one call. This is more efficient than separate calls because the model runs them as one batch.

Request: `{"texts": ["...", "..."], "top_k": 5}` (`texts`: 1–64 non-empty strings, each truncated to 8000 chars).
Response `200`: `{"results": [<same object as /match, without input_type>, ...]}`, in input order.

---

## `GET /occupation/{soc}`

Return the full O*NET record for one occupation.

Response `200`:

| field | type |
|---|---|
| `soc`, `title`, `description`, `major_group`, `major_group_title`, `riasec_code` | string |
| `job_zone` | int or null |
| `tasks` | `{task, task_type ("Core"/"Supplemental"/"n/a"), importance (1–5 or null)}[]`, by importance |
| `detailed_work_activities` | string[] |
| `skills`, `knowledge`, `abilities` | `{name, importance, level}[]`, by importance |
| `interests` | `{name, score (1–7)}[]` for the six RIASEC types |

Errors: `404` if the code is not a match-target occupation. Residual "All Other" codes with no task data are not targets.

---

## `GET /health`

`200` → `{"status": "ok", "api_version": "1", "model": {...}, "occupations": 923}`. Returns `503` if no bundle can be loaded.

---

## Notes

- **Stateless:** the service doesn't store inputs. Add authentication, rate limiting and request-size limits at your gateway before you expose it publicly.
- **Concurrency:** model inference is serialized inside a process, so scale with more worker processes or replicas.
- **Versioning:** breaking changes to this contract will bump `api_version`. Adding fields is non-breaking.
- **Data licence:** O*NET content is CC BY 4.0. Attribute O*NET when you display it.
