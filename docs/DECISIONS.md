# Decision log

This log records the decisions that shaped onet-match, in the order they were made.
Each decision cites the saved run, doc or code file its numbers come from.
"Not measured" means no saved run covers the question.
"(rationale reconstructed)" means the reason was not written down at the time, so the entry gives the engineering reason that fits the evidence.

## The system at a glance

- **Taxonomy and targets.** The 923 O\*NET 31.0 occupations that have task statements. Decisions [2](#2-frame-the-task-as-ranking-all-923-onet-310-occupations), [4](#4-build-an-occupation-catalog-of-the-923-occupations-with-tasks).
- **Data source.** The O\*NET bulk database release, downloaded once, verified and cached outside git. Decision [3](#3-load-onet-from-the-bulk-database-release).
- **Occupation catalog.** Parquet tables for occupations, tasks, work activities, skills, knowledge, abilities and RIASEC interests. Decision [4](#4-build-an-occupation-catalog-of-the-923-occupations-with-tasks).
- **Occupation documents.** Title + description + 10 most important tasks. Decision [7](#7-represent-each-occupation-as-title--description--top-10-tasks).
- **Labeled data.** Real O\*NET titles, tasks, work activities and composite descriptions, with multi-label gold sets. Decision [5](#5-build-labeled-pairs-from-real-onet-text).
- **Splits.** 80/10/10 per unique text, stratified by occupation, with held-out tasks removed from the searched documents. Decision [6](#6-split-by-unique-text-stratified-by-occupation-and-remove-held-out-tasks-from-documents).
- **Synthetic messy text.** Written by one LLM for training and by a different LLM for the test set. Decisions [20](#20-generate-messy-training-text-with-an-llm-five-input-types-one-call-per-occupation), [21](#21-use-a-different-llm-for-the-test-set-within-a-65-budget).
- **Baselines.** TF-IDF (the original approach) and BM25. Decision [9](#9-use-tf-idf-and-bm25-as-baselines).
- **Bi-encoder.** Fine-tuned bge-base-en-v1.5, CLS pooling, normalized embeddings, max length 256. Decisions [14](#14-choose-the-bge-family-for-the-bi-encoder), [15](#15-fine-tune-with-multiplenegativesrankingloss-one-epoch-on-the-laptop-gpu), [16](#16-mine-hard-negatives-from-neighboring-occupations), [24](#24-fine-tune-bge-base-on-real-plus-synthetic-data-and-reuse-the-bge-small-reranker).
- **Reranker.** Fine-tuned ms-marco-MiniLM-L6-v2 cross-encoder, max length 320, raw logits. Decision [17](#17-add-a-fine-tuned-cross-encoder-reranker).
- **Candidates.** k = 20 from the bi-encoder. Decision [18](#18-retrieve-k--20-candidates).
- **Score fusion.** Per-query z-scores blended with reranker weight 0.3. Decision [23](#23-blend-reranker-and-bi-encoder-scores-with-per-query-z-scores-weight-chosen-on-validation).
- **Calibration.** Temperature-scaled softmax over the top 20, T ≈ 0.6148. Decision [10](#10-calibrate-with-one-temperature-fitted-on-real-validation-data).
- **Evaluation.** Top-1, top-5, MRR@20, major-group top-1, ECE and latency, per input type, on a real test set and a synthetic test set. Decisions [8](#8-evaluate-every-model-on-the-same-held-out-sets-select-only-on-validation-and-generate-all-numbers-from-saved-runs), [22](#22-run-ablations-and-learning-curves-on-bge-small).
- **Served model.** `L5b_ft-bge-base+synth+ce+fuse`, chosen on validation data. Decision [25](#25-serve-the-model-with-the-best-validation-top-1-then-mrr).
- **Bundle.** A self-contained directory, hosted on the Hugging Face Hub. Decisions [12](#12-package-a-served-model-as-a-self-contained-bundle), [28](#28-host-the-model-bundle-on-the-hugging-face-hub).
- **Interfaces.** Python API, CLI, REST API (FastAPI) and Docker. Decision [13](#13-serve-through-a-python-api-a-cli-a-rest-api-and-docker).
- **Package.** `onet-match` on PyPI, imported as `onet_match`. Decisions [27](#27-package-as-one-distribution-named-onet-match), [30](#30-publish-to-pypi-with-a-project-scoped-token-uploaded-only-with-the-owners-approval).
- **MCP server.** Two read-only tools, listed in the official MCP Registry. Decisions [31](#31-add-an-mcp-server-with-two-read-only-tools-using-the-mcp-2x-mcpserver-api), [35](#35-list-the-mcp-server-in-the-official-mcp-registry).
- **In-browser demo.** A static Hugging Face Space running int8 ONNX models with transformers.js. Decisions [32](#32-host-the-demo-as-a-free-static-space-with-in-browser-inference), [33](#33-quantize-to-int8-onnx-per-channel-qint8-for-the-bi-encoder-quint8-for-the-reranker), [34](#34-run-on-multi-threaded-wasm-mirror-the-python-pipeline-exactly-and-render-progressively).
- **Repository.** One repo, with the former data collector merged in. Decisions [1](#1-build-on-the-existing-2024-pipeline-instead-of-starting-over), [26](#26-merge-the-data-collector-into-this-repository-keeping-both-histories).
- **Tests.** Fully offline, with tiny local models and a mocked LLM. Decision [11](#11-test-fully-offline-with-tiny-local-models-and-a-mocked-llm).
- **Licensing.** MIT code and weights; O\*NET data under CC BY 4.0 with attribution. Decision [29](#29-license-the-code-and-weights-as-mit-and-attribute-onet-under-cc-by-40).
- **README examples.** Real outputs, chosen to cover use cases and input styles. Decision [36](#36-show-only-real-model-outputs-in-the-readme-chosen-to-cover-use-cases-and-input-styles).
- **Dependency floor.** `sentence-transformers>=6.2`, the version the bundle was saved with. Decision [37](#37-require-sentence-transformers-62-the-version-the-bundle-was-saved-with).
- **Worked example.** A separate repository that uses the PyPI package on 3,000 real job postings, also published on Kaggle. Decision [38](#38-put-the-worked-example-in-its-own-repository-that-installs-onet-match-from-pypi).

---

## 1. Build on the existing 2024 pipeline instead of starting over

**Decision.** Extend the existing two-package codebase (`onet_data_collector` for ingestion, `onet_analysis` for analysis) with a new `onet_analysis/matching/` package. Keep the existing commands and tests working. Use the project's existing TF-IDF setup as the baseline.

**Context.** The project was built in 2024 for EDA, occupation clustering and keyword attribution on O\*NET Web Services data. Both packages had been rebuilt into tested libraries before the matching work began (git: "Rebuild analytics into the onet_analysis package", "Rebuild collector into a robust, tested ETL package"). The 2026 goal was different: map any free text about work to O\*NET occupations.

**Rationale.**
- The existing code already handled O\*NET data and had tests, so new work could build on it.
- Keeping the TF-IDF approach as rung 1 of the model ladder makes every later gain measurable against what the project did before (docs/RESULTS.md, "Model ladder").
- The original analysis tests still pass alongside the new ones.

**Alternatives considered.**

| option | why not |
|---|---|
| New repository from scratch | Loses the history and the working ingestion code. Breaks existing commands. |
| Replace TF-IDF clustering with embeddings | The classic analysis commands (`eda`, `keywords`, `cluster`) still have users. Matching was added beside them instead. |

**Consequences.** The repository now holds three packages: `onet_match` (public alias), `onet_analysis` and `onet_data_collector` (AGENTS.md).

## 2. Frame the task as ranking all 923 O\*NET 31.0 occupations

**Decision.** Treat matching as retrieval. Each of the 923 occupations is one document. A query is scored against all of them and the top k are returned with calibrated confidence. Any input type is accepted: titles, postings, experience bullets, project descriptions, interest statements and skill lists. One model handles all types. The `input_type` field in the REST API is echoed back, not used (docs/API.md).

**Context.** The engine had to accept free text of any style and return occupations with skills, tasks and a confidence score. Job titles are only one input type.

**Rationale.**
- With about 900 targets, a full score matrix is small, so no approximate-nearest-neighbor index is needed (onet_analysis/matching/retrieval.py, module docstring).
- Retrieval lets off-the-shelf embedding models work zero-shot from the occupation text. This gave a 36.1% top-1 starting point before any training (outputs/runs/L3_bge-base/metrics.json).
- Many inputs legitimately belong to several occupations. Ranking handles this naturally; 6,964 labeled rows have more than one gold code (docs/RESULTS.md, "Data").
- Users rarely label their input type, so a single type-agnostic model is simpler to call. (rationale reconstructed)

**Alternatives considered.**

| option | why not |
|---|---|
| 923-way classifier | Needs labeled examples for every class and ignores the occupation text. Not measured. (rationale reconstructed) |
| Separate models per input type | More models to train and serve, and callers would have to classify their own input. Not measured. (rationale reconstructed) |
| Include all ~1,016 O\*NET-SOC codes | About 90 codes, mostly "..., All Other", have no tasks, ratings, skills or interests, so there is nothing to match against or return (onet_analysis/matching/catalog.py, module docstring). |

## 3. Load O\*NET from the bulk database release

**Decision.** Add `onet_data_collector/bulk.py`. It downloads the current O\*NET database text release from onetcenter.org, verifies it, and caches it outside git (`~/.cache/onet_data_collector/bulk/db_<version>/`, or `$ONET_BULK_DIR`). A `manifest.json` records the release, source URL, SHA-256 and download time.

**Context.** The 2024 pipeline used O\*NET Web Services keyword search, which needs an account and only returns occupations a keyword surfaces. Matching needs every occupation and every table: titles, tasks, task ratings, work activities, skills, knowledge, abilities, interests and job zones.

**Rationale.**
- The bulk release needs no credentials and contains every table at once (README.md, "O\*NET data pipeline").
- Provenance travels with the data. The served bundle records release 31.0, its source URL and SHA-256 (models/bundle/bundle.json).
- About 100 MB of data stays out of git (onet_data_collector/bulk.py, module docstring).
- Release 31.0 changed file names. For example, "Job Titles" replaces "Alternate Titles" and skills are split into essential and transferable. A logical-name-to-file map (`TABLES`) handles both the new and old layouts.

**Alternatives considered.**

| option | why not |
|---|---|
| O\*NET Web Services API | Needs an account and is query-driven, so it can't cheaply return every table for every occupation. (rationale reconstructed) |
| Commit the data to git | Large, and it would mix data versions with code history. |

**Consequences.** Downloads use the `certifi` CA bundle when available, because the local Python build lacked CA certificates. Extraction guards against zip-slip.

## 4. Build an occupation catalog of the 923 occupations with tasks

**Decision.** `OccupationCatalog.from_bulk` builds parquet tables: occupations (SOC, title, description, major group, job zone, RIASEC code), ranked tasks, detailed work activities, skills, knowledge, abilities and interests. Targets are the occupations that have task statements: 923 in release 31.0 (docs/RESULTS.md).

**Context.** Serving must return tasks, skills, knowledge and interests for every match without the raw release.

**Rationale.**
- Tasks are ranked by importance, with Core before Supplemental, so "top tasks" and the occupation documents use the most important ones (onet_analysis/matching/catalog.py).
- Skills, knowledge and abilities keep both the importance and level scales, which the API returns (docs/API.md).
- The RIASEC high-point code comes from O\*NET's own interest high-point table rather than being recomputed.
- The catalog is small and self-contained, so it ships inside the model bundle (decision [12](#12-package-a-served-model-as-a-self-contained-bundle)).

**Alternatives considered.** Keeping residual codes as targets was rejected for the reason in decision [2](#2-frame-the-task-as-ranking-all-923-onet-310-occupations).

## 5. Build labeled pairs from real O\*NET text

**Decision.** Build (text → SOC codes) pairs from four real sources:

| source | input type it stands in for | content |
|---|---|---|
| `title` | job title | O\*NET Job Titles and Sample of Reported Titles |
| `task` | task | Task statements |
| `dwa` | activity | Detailed work activities |
| `composite` | description | 2–4 tasks of one occupation joined together |

Each row stores a set of gold codes. A prediction is correct if it hits any of them. Titles identical to an official title are dropped. Training pairs keep at most 10 random labels per work activity (onet_analysis/matching/datasets.py).

**Context.** There is no public labeled dataset of free text mapped to O\*NET 31.0. O\*NET itself contains many labeled texts.

**Rationale.**
- The sources give 39,473 unique titles, 17,578 tasks, 2,085 work activities and about 10,900 composites (docs/RESULTS.md, "Data").
- Multi-label gold is necessary: 6,964 rows have more than one valid code (docs/RESULTS.md).
- Titles that match an official title exactly are trivially solved, so they would inflate scores.
- Some work activities belong to over 100 occupations. Capping their labels stops generic activities from dominating training (onet_analysis/matching/datasets.py, `training_pairs`).
- Composites imitate multi-sentence job descriptions using only real text.

**Alternatives considered.**

| option | why not |
|---|---|
| Single-label gold (primary occupation only) | Would mark correct answers wrong for thousands of shared titles and generic activities. |
| Titles only | Tested as an ablation. Real test top-1 was 42.4% vs 45.6% with all four sources (outputs/runs/abl_data_titles/metrics.json, outputs/runs/L4_ft-bge-small/metrics.json). |

## 6. Split by unique text, stratified by occupation, and remove held-out tasks from documents

**Decision.**
- Assign train/val/test per unique normalized text, 80/10/10, stratified by primary occupation, with seed 13.
- Work activities are split randomly, without stratification, because they are generic.
- Composites are built only from tasks in their own split: 10 per occupation in train, 2 each in val and test.
- Every val and test task is removed from the occupation documents used in evaluation and training (`heldout_task_ids`, `eval_docs`).
- A "hard" title subset holds test titles that share no content token with the official title of any gold occupation, after removing stop words and stripping plurals (onet_analysis/matching/text.py).

**Context.** Occupation documents include task text. A held-out task would otherwise match itself in its own occupation's document.

**Rationale.**
- Splitting by unique text means the same string can't appear in two splits.
- Stratifying means every occupation contributes about 10% of its titles and tasks to val and to test.
- Removing held-out tasks from documents means no test text is visible as training data or as part of a searched document (onet_analysis/matching/datasets.py, module docstring; README.md, "How it works").
- The hard set measures what token overlap can't solve. TF-IDF scores 6.5% top-1 on it, the served model 38.1% (docs/RESULTS.md).

**Alternatives considered.**

| option | why not |
|---|---|
| Split by occupation (unseen occupations at test time) | Every occupation is a valid target at serving time, so this would test a situation that never occurs. (rationale reconstructed) |
| Keep held-out tasks in documents | Leaks test text into retrieval and inflates task and description scores. |

**Consequences.** The served bundle uses all tasks in its documents, because no evaluation runs against it (onet_analysis/matching/engine.py, `write_bundle`). Accuracy against the served bundle is therefore higher than the held-out benchmark; outputs/demo/browser_parity.json notes this.

## 7. Represent each occupation as title + description + top 10 tasks

**Decision.** Each occupation document is `"{title}. {description} Tasks: {top 10 tasks}"` (`DEFAULT_FIELDS = title+desc+tasks`, `n_tasks = 10`; onet_analysis/matching/catalog.py, models/bundle/bundle.json).

**Context.** The document is what every retriever and the reranker compare a query against. Fields were chosen with an ablation over off-the-shelf models.

**Rationale.** Real test top-1 (docs/RESULTS.md, "Occupation representation fields"; outputs/runs/abl_fields_*/metrics.json):

| model | title | +desc | +tasks | +skills |
|---|---|---|---|---|
| BM25 | 12.5 | 16.4 | 20.7 | 20.5 |
| bge-small | 27.5 | 31.0 | 32.9 | 32.9 |
| bge-base | 29.9 | 34.5 | 36.1 | 36.1 |

Adding tasks helped every model. Adding skills helped none.

**Alternatives considered.**
- **+ skills and knowledge.** No gain (above), and it makes documents longer. Rejected.
- **Several short texts per occupation (one per title or task), scored by max or mean.** Not measured.
- **More than 10 tasks.** Not measured. Documents are truncated at 256 tokens by the bi-encoder (decision [15](#15-fine-tune-with-multiplenegativesrankingloss-one-epoch-on-the-laptop-gpu)), so extra tasks would often be cut. (rationale reconstructed)

## 8. Evaluate every model on the same held-out sets, select only on validation, and generate all numbers from saved runs

**Decision.**
- Metrics: top-1, top-5, MRR@20, major-group top-1, ECE (10 bins) and single-query p50 latency over 200 queries, each broken down by input type (onet_analysis/matching/metrics.py, evaluate.py).
- Test sets: `real_test` (6,792 rows), the hard-title subset (1,556), and `synthetic_test` (4,703) once it existed (decision [21](#21-use-a-different-llm-for-the-test-set-within-a-65-budget)).
- Validation (6,813 real rows) is used only for model selection, the fusion weight and calibration.
- Every run writes `outputs/runs/<run>/config.json` and `metrics.json`. `docs/RESULTS.md` is generated from them by `onet-match matching report` and is never edited by hand (AGENTS.md).

**Context.** The project rule was "no invented numbers": every number must trace to a saved run in `outputs/`.

**Rationale.**
- Major-group accuracy shows whether a miss is close (for example a neighboring production job) or far off.
- Per-input-type tables expose weak spots. The served model gets 81.0% top-1 on synthetic job descriptions but only 37.6% on raw task statements (docs/RESULTS.md).
- A generated report can't drift from the runs. Missing runs show as "not yet run" instead of placeholder numbers.

**Alternatives considered.**

| option | why not |
|---|---|
| Hand-written results tables | Numbers drift from the runs. |
| Select models on test scores | Overfits the test set. |
| Several seeds per configuration | Not run because of training time. Each configuration ran once with seed 13, so differences of about one point or less are treated as noise (docs/RESULTS.md, "Limitations"). |

## 9. Use TF-IDF and BM25 as baselines

**Decision.**
- Rung 1: the project's existing TF-IDF setup (unigrams and bigrams, English stop words, cosine similarity).
- Rung 2: a custom Okapi BM25 over sparse matrices (k1 = 1.5, b = 0.75) (onet_analysis/matching/retrieval.py; outputs/runs/L2_bm25/config.json).

**Context.** Improvements need a fixed reference point, and TF-IDF is what the project used before.

**Rationale.**

| run | real test top-1 / top-5 | synthetic test top-1 | p50 latency |
|---|---|---|---|
| TF-IDF (`L1_tfidf`) | 20.5 / 40.5 | 17.4 | 0.5 ms |
| BM25 (`L2_bm25`) | 20.7 / 40.1 | 24.1 | 0.2 ms |

(outputs/runs/L1_tfidf/metrics.json, outputs/runs/L2_bm25/metrics.json)

BM25 was barely better on O\*NET-style text but 6.7 points better on messy text. Neither handles paraphrase, which motivated dense models (decision [14](#14-choose-the-bge-family-for-the-bi-encoder)).

**Alternatives considered.** A BM25 library dependency. The vectorized version is short and has no extra dependency. (rationale reconstructed)

## 10. Calibrate with one temperature, fitted on real validation data

**Decision.** Confidence is a softmax over the top-20 scores with one temperature T. T is grid-searched over 400 log-spaced values, relative to the score spread, to minimize the negative log of the total probability on gold candidates. It is fitted per run on the real validation split. The served model uses T = 0.6147757 (models/bundle/bundle.json).

**Context.** Raw cosine similarities and reranker logits are not probabilities, and their scales differ a lot: cosine scores sit about 0.01 apart, logits several units apart (onet_analysis/matching/metrics.py).

**Rationale.**
- Temperature scaling has one parameter and doesn't change the ranking.
- A grid relative to the score spread works for every score type. Fitted temperatures span orders of magnitude, for example 0.0117 for e5-base cosine scores and 4.02 for BM25 (outputs/runs/L3_e5-base/config.json, outputs/runs/L2_bm25/config.json).
- The gold-set objective fits multi-label rows.
- Result for the served model: ECE 0.037 on `real_test` and 0.031 on validation (outputs/runs/L5b_ft-bge-base+synth+ce+fuse/metrics.json).
- Synthetic data never enters calibration, so the synthetic test set stays fully held out (docs/SYNTHETIC_DATA.md).

**Alternatives considered.**

| option | why not |
|---|---|
| Report raw scores | Not comparable across models and not probabilities. |
| Isotonic regression or Platt scaling | Not measured. More parameters, and they handle multi-label gold sets less naturally. (rationale reconstructed) |
| Fit on synthetic or mixed data | Would contaminate the held-out synthetic test set. |

**Consequences.** Calibration is weaker on messy text: ECE is 0.102 on `synthetic_test` (same metrics file). The docs say to check the reliability curve per input style before thresholding on confidence (docs/RESULTS.md, "Limitations").

## 11. Test fully offline with tiny local models and a mocked LLM

**Decision.**
- Tests build a tiny synthetic O\*NET release (6 occupations plus one "All Other" code).
- Dense training and reranking tests use a randomly initialized 2-layer BERT built locally and seeded.
- LLM generation uses a `mock` provider.
- `tests/test_matching_docs.py` checks that docs/API.md and docs/SYNTHETIC_DATA.md stay in sync with the code.

`pytest` needs no network, downloads or API keys (README.md, "Train your own"; tests/).

**Context.** Real models are hundreds of MB, and real generation costs money.

**Rationale.**
- The whole pipeline runs end to end in tests: data, training, fusion tuning, bundling, serving, HTTP API, CLI and MCP (tests/test_matching_dense.py, tests/test_matching_models.py).
- Seeding the tiny models fixed a flaky test in which the reranker trained for 0 steps.
- Doc-sync tests stop the published prompt and HTTP contract from drifting away from the code.

**Alternatives considered.**

| option | why not |
|---|---|
| Tests that download real models | Slow, need the network, and break offline. |
| CI pipeline, GitHub Actions, lint gates | Left out on purpose to keep the project lightweight. The full suite runs locally in about 11 s. |

## 12. Package a served model as a self-contained bundle

**Decision.** `onet-match matching bundle <spec> --run <run>` writes one directory:
- `bundle.json` (format 1): spec, fields, `n_tasks`, k, fusion weight, temperature, source run and O\*NET provenance;
- `catalog/`;
- `biencoder/`;
- `reranker/`;
- `doc_embeddings.npy`.

The temperature is copied from the named run. The command refuses to bundle if the spec or fields differ from that run's (onet_analysis/matching/pipeline.py, `bundle`; engine.py).

**Context.** Serving should need neither the O\*NET release nor the training data, and the served numbers should trace back to one benchmark run.

**Rationale.**
- One directory is easy to cache, upload and mount in Docker.
- Precomputed occupation embeddings mean only the query is encoded at serving time.
- The spec check stops a bundle from carrying a temperature fitted for a different model.
- The stored spec uses relative paths (`spec_string` in models/bundle/bundle.json), so it doesn't leak local file paths.

**Alternatives considered.** Loading models and catalog from separate locations at runtime. This is more fragile and can't guarantee the models and calibration match. (rationale reconstructed)

## 13. Serve through a Python API, a CLI, a REST API and Docker

**Decision.**
- **Python:** `match(text, top_k)` returns ranked matches plus a confidence-weighted profile of skills, knowledge and RIASEC interests. `default_engine()` caches the loaded engine (AGENTS.md).
- **CLI:** `onet-match match "<text>" [--json]` and `onet-match serve`.
- **REST (FastAPI):** `POST /match`, `POST /match/batch` (up to 64 texts), `GET /occupation/{soc}` and `GET /health`. Text is limited to 1–8,000 characters and `top_k` to 1–20. Returns 503 when no bundle is available (onet_analysis/matching/api.py, docs/API.md).
- **Docker:** `python:3.12-slim` with CPU-only torch. The bundle is downloaded at build time, the container runs offline as a non-root user, and it has a health check (Dockerfile).

**Context.** The required outputs were a Python API, a CLI subcommand, a small HTTP service with a Dockerfile, and a documented HTTP contract.

**Rationale.**
- The profile weights each matched occupation's O\*NET ratings by its confidence, so the result says more than one occupation label.
- The HTTP contract is documented field by field. Its example is real output saved in `outputs/examples/http_match.json`.
- Baking the bundle into the image means containers start without network access.

**Alternatives considered.** A heavier serving stack (model server, GPU image). Single-query latency is 46.7 ms on a laptop (outputs/runs/L5b_ft-bge-base+synth+ce+fuse/metrics.json), so a plain FastAPI process is enough. (rationale reconstructed)

**Consequences.** Docker was not installed on the development machine when the Dockerfile was written. No source records a full image build test.

## 14. Choose the bge family for the bi-encoder

**Decision.** Use BAAI bge-en-v1.5 models as the bi-encoder base: bge-small first, bge-base later (decision [24](#24-fine-tune-bge-base-on-real-plus-synthetic-data-and-reuse-the-bge-small-reranker)).

**Context.** Four off-the-shelf models were benchmarked with the same occupation documents. bge and e5 used their documented query prompts (outputs/runs/L3_*/config.json).

**Rationale.** (outputs/runs/L3_*/metrics.json)

| model | real test top-1 | synthetic test top-1 | p50 latency |
|---|---|---|---|
| all-MiniLM-L6-v2 | 31.7 | 34.3 | 3.1 ms |
| bge-small-en-v1.5 | 32.9 | 37.1 | 5.5 ms |
| bge-base-en-v1.5 | 36.1 | 45.4 | 6.9 ms |
| e5-base-v2 | 35.5 | 42.2 | 6.8 ms |

- bge-base was best on both test sets.
- bge-small beat MiniLM at a similar size.
- bge-base is MIT-licensed (models/bundle/README.md).
- Off-the-shelf bge-base alone beat TF-IDF by 15.6 points (docs/RESULTS.md, "Key findings").

**Alternatives considered.**
- **e5-base-v2.** Close on real text (−0.6 points) but 3.2 points worse on messy text. It also needs "query:"/"passage:" prefixes.
- **MiniLM.** Fastest but weakest.
- **Larger models (for example bge-large).** Not measured. Training time on the laptop was already near the one-hour limit for bge-base (decision [24](#24-fine-tune-bge-base-on-real-plus-synthetic-data-and-reuse-the-bge-small-reranker)).

**Consequences.** Fine-tuned models are trained and served without prompts (`default_prompts` in onet_analysis/matching/retrieval.py).

## 15. Fine-tune with MultipleNegativesRankingLoss, one epoch, on the laptop GPU

**Decision.** Train with the sentence-transformers trainer:
- loss: `MultipleNegativesRankingLoss`, with in-batch negatives plus one mined hard negative per pair (decision [16](#16-mine-hard-negatives-from-neighboring-occupations));
- `NO_DUPLICATES` batch sampling;
- 1 epoch, learning rate 2e-5, warmup ratio 0.1, max sequence length 256, bf16;
- batch size 64 for bge-small and 32 for bge-base;
- seed 13, on Apple Silicon (MPS).

(onet_analysis/matching/train.py; outputs/runs/L4_ft-bge-small/config.json, outputs/runs/L4b_ft-bge-base+synth/config.json)

**Context.** Off-the-shelf models were trained on general web text, not on the mapping from job text to O\*NET occupations.

**Rationale.**
- **Result.** Fine-tuning bge-small raised real test top-1 from 32.9% to 45.6% (+12.7) (docs/RESULTS.md, "Key findings").
- **Loss.** MNRL uses every other document in a batch as a negative, which suits one positive document per query.
- **Sampler.** `NO_DUPLICATES` keeps the same occupation document from appearing twice in a batch, where it would turn a positive into a false negative (onet_analysis/matching/train.py, module docstring).
- **bf16.** Training speed on MPS went from about 2 s per step to about 1.46 s per step.
- **One epoch.** bge-small took 26.1 minutes for one epoch (docs/RESULTS.md, "Learning curve") and bge-base took 58 minutes (README.md). The project rule required approval for runs over about an hour, so one epoch kept every run within budget. (rationale reconstructed)
- **Batch 32 for bge-base.** At batch 64 it stalled at step 4 under memory pressure (about 18 GB of swap). At 32 it ran at about 1.3 s per step.

**Alternatives considered.**

| option | why not |
|---|---|
| More epochs | Not measured. The learning curve suggests more data helps (decision [22](#22-run-ablations-and-learning-curves-on-bge-small)); more epochs over the same data is untested. |
| Larger batches for bge-base | Ran out of memory on the laptop (above). |
| Training on a cloud GPU | Not used. Every run fit on the laptop at no cost. (rationale reconstructed) |

**Consequences.** Training only ran when the GPU was idle. Other local jobs that used the GPU slowed a step to about 71–90 s, so runs waited for a free machine instead of stopping those jobs.

## 16. Mine hard negatives from neighboring occupations

**Decision.** Each training pair gets one negative document, drawn at random from the gold occupation's 10 nearest occupations by the base model's document embeddings. Every gold code of the anchor is excluded (`N_NEIGHBORS = 10`; onet_analysis/matching/train.py).

**Context.** The common errors are confusions between similar occupations, for example Production vs Construction and Extraction, the most frequent major-group confusion (docs/RESULTS.md, "Error analysis").

**Rationale.** The setup was specified up front. The ablation shows a small effect (docs/RESULTS.md, "Hard-negative mining"):

| variant | real test top-1 | synthetic test top-1 | hard titles top-1 | training time |
|---|---|---|---|---|
| hard negatives off (`abl_hardneg_off`) | 45.2 | 43.5 | 29.5 | 864.9 s |
| hard negatives on (`L4_ft-bge-small`) | 45.6 | 43.1 | 29.8 | 1,563.7 s |

(training times from the two runs' config.json)

They were kept because they did no measurable harm. Differences this small are within single-seed noise (docs/RESULTS.md, "Limitations").

**Alternatives considered.**
- **No hard negatives.** Equivalent within noise and about 45% faster to train (above). A defensible choice too.
- **Negatives mined from the fine-tuned model in a second round.** Not measured.

## 17. Add a fine-tuned cross-encoder reranker

**Decision.** Rescore the bi-encoder's top 20 with `cross-encoder/ms-marco-MiniLM-L6-v2`, fine-tuned:
- loss: listwise cross-encoder MNRL;
- each example has the query, a gold document and 4 hard negatives taken from the fine-tuned bi-encoder's own top 20, plus 2 in-batch negatives;
- max length 320;
- 30,000 training queries sampled per source;
- 1 epoch, batch 32, learning rate 2e-5, bf16.

Scores are raw logits (`activation_fn=None`). (onet_analysis/matching/train.py, retrieval.py; outputs/runs/L5s_ft-bge-small+synth+ce/config.json)

**Context.** A cross-encoder reads the query and document together, so it can settle close calls a bi-encoder misses.

**Rationale.**
- Negatives from the bi-encoder's own candidates teach the reranker exactly the confusions it will see at serving time.
- A 6-layer MiniLM reranker trained in about 24 minutes (1,436.7 s; same config.json) and is Apache-2.0 (models/bundle/README.md).
- 320 tokens fits a short query plus the title, description and most of the tasks. (rationale reconstructed)
- Raw logits keep the spread that z-score fusion needs (decision [23](#23-blend-reranker-and-bi-encoder-scores-with-per-query-z-scores-weight-chosen-on-validation)). A sigmoid would squash it. (rationale reconstructed)

**Alternatives considered.**
- **Larger cross-encoders.** Not measured. The reranker dominates latency: 46.7 ms with it vs 6.8 ms for the bi-encoder alone (docs/RESULTS.md).
- **Using reranker scores alone.** Rejected after measurement; see decision [23](#23-blend-reranker-and-bi-encoder-scores-with-per-query-z-scores-weight-chosen-on-validation).

**Consequences.** On its own the reranker hurt: real test top-1 fell from 45.6% to 38.9% (outputs/runs/L5_ft-bge-small+ce/metrics.json). This led to score fusion.

## 18. Retrieve k = 20 candidates

**Decision.** The bi-encoder returns 20 candidates. The reranker rescores those 20, and calibration normalizes over them (`k = 20` in models/bundle/bundle.json).

**Context.** k sets how much work the reranker does and the support set for confidence.

**Rationale.**
- The served model has the gold occupation in its top 5 79.1% of the time on real text and 92.9% on messy text (docs/RESULTS.md), so 20 leaves margin for the reranker. (rationale reconstructed)
- Reranker cost grows with k: 20 pairs per query is most of the 46.7 ms latency, and about 6.5 s in single-threaded browser WASM (decision [34](#34-run-on-multi-threaded-wasm-mirror-the-python-pipeline-exactly-and-render-progressively)).

**Alternatives considered.** Larger or smaller k. Not measured. Recall at 20 is not reported as a separate metric; MRR is computed at 20.

## 19. Do not ship a human-written gold set yet

**Decision.** The planned 100-row human-written gold set was removed from the repo and from all public material. Results are reported on held-out real O\*NET text, the hard titles and the LLM-written test set.

**Context.** The gold set had to be written and labeled by a person, not generated. Its author chose to fill it in later.

**Rationale.**
- A gold set written by the project's own tooling would grade the model on its own kind of text.
- Leaving a template or empty hook in public files would advertise an evaluation that doesn't exist.

**Alternatives considered.**

| option | why not |
|---|---|
| Generate the gold set with an LLM | Not human-written, which defeats its purpose. |
| Keep an empty template in the repo | Misleading in public docs. |

**Consequences.** The limitations say plainly that there is no human-written test set yet (docs/RESULTS.md, models/bundle/README.md).

## 20. Generate messy training text with an LLM, five input types, one call per occupation

**Decision.** Paraphrase O\*NET content into five input types: `description`, `experience`, `project`, `interest` and `skills`.
- One call per occupation asks for all five types at once, constrained by a JSON schema.
- The prompt shows the title, description, up to 8 tasks and top skills and knowledge, and forbids the words of the official title.
- Train prompts use only train-split tasks. Eval prompts use only test-split tasks.
- Exact duplicates are dropped. A `title_leak` flag is reported but flagged rows are kept.
- Prompt version `v1` is published verbatim and checked against the code by a test.

(docs/SYNTHETIC_DATA.md; onet_analysis/matching/synth.py)

**Context.** Real O\*NET text is clean and formal. People type things like "ran the robotics club's CAD team". Fine-tuning bge-small on real data alone raised messy-text top-1 only from 37.1% to 43.1%, still below off-the-shelf bge-base at 45.4% (docs/RESULTS.md).

**Rationale.**
- **Result.** Adding 14,574 synthetic snippets raised synthetic test top-1 from 43.1% to 57.9% (+14.9), with no change on real text (45.6% → 45.6%) (docs/RESULTS.md, "Key findings").
- One call for all five types keeps the cost to 923 calls per role (outputs/synth/train_\*.json).
- Using split-specific tasks means eval snippets paraphrase tasks that never appear in training data or in searched documents.
- Title leakage was rare: 0.10% of training snippets and 0.02% of eval snippets (outputs/synth/\*.json, `title_leak_rate`).

**Alternatives considered.**

| option | why not |
|---|---|
| No synthetic data | Messy-text accuracy stays near off-the-shelf levels (above). |
| Separate calls per input type | Five times the calls for the same content. (rationale reconstructed) |
| Template-based augmentation | Not measured. Templates can't produce the variety of voice, length and vagueness real input has. (rationale reconstructed) |

## 21. Use a different LLM for the test set, within a $65 budget

**Decision.**
- Training text: Anthropic `claude-opus-5-5`, 3 snippets per type, 14,574 snippets, $23.99.
- Test text: Anthropic `claude-sonnet-5-5`, 1 snippet per type, 4,703 snippets, $6.74.
- Two 7-occupation pilots cost $0.24.
- Total spend: $30.97 of a $65 cap.

(docs/SYNTHETIC_DATA.md; outputs/synth/\*.json)

**Context.** If one model wrote both sets, a fine-tuned matcher could score well just by learning that model's phrasing. The project required a pilot of about 100 examples, with cost and quality reported, before any full generation.

**Rationale.**
- A different model for the test set reduces style leakage.
- The pilots produced 111 training and 35 eval snippets. They estimated the full training run at $25.49 and the eval run at $6.74 (outputs/synth/pilot_\*.json), which fit the budget.
- The stronger model wrote the larger training set. The cheaper model wrote the smaller test set.
- All 923 calls per role succeeded with no failures (outputs/synth/\*.json).
- The remaining budget was not spent. The next gain came from a larger bi-encoder (decision [24](#24-fine-tune-bge-base-on-real-plus-synthetic-data-and-reuse-the-bge-small-reranker)), not from more data.

**Alternatives considered.**

| option | why not |
|---|---|
| Different vendor for the test set (an OpenAI model) | Preferred, and the generator supports any OpenAI-compatible endpoint. No OpenAI key was available, so the test set uses a different model from the same vendor. This is disclosed as a limitation (docs/RESULTS.md, docs/SYNTHETIC_DATA.md). |
| Same model for both sets | Style leakage (above). |
| Local open models through Ollama | Free and keyless, but lower quality and slower. Not used once API keys were provided. |
| Smallest, cheapest model for training text | Not measured. The budget allowed the stronger model. |

**Consequences.** API keys were stored in the macOS Keychain and loaded only for the command that needed them. Sampling parameters were left at defaults, because the API rejected temperature settings for these models at the time.

## 22. Run ablations and learning curves on bge-small

**Decision.** Run the ablations on fine-tuned bge-small, the cheaper model:
- training-data sources: titles only, + tasks, work activities and composites, + synthetic;
- hard negatives on vs off (decision [16](#16-mine-hard-negatives-from-neighboring-occupations));
- reranker scoring: bi-encoder only, reranker only, blended;
- learning curve at 10%, 25%, 50% and 100% of the real pairs.

(scripts/matching/run_finetune.sh)

**Context.** The write-up had to show which data and which steps mattered.

**Rationale.** Results (docs/RESULTS.md, "Ablations" and "Learning curve"):

| question | finding | runs |
|---|---|---|
| Do tasks and activities help beyond titles? | Yes: real top-1 42.4 → 45.6; synthetic top-1 39.7 → 43.1 | `abl_data_titles`, `L4_ft-bge-small` |
| Does synthetic text help? | Synthetic top-1 43.1 → 57.9; real unchanged | `L4_ft-bge-small`, `L4s_ft-bge-small+synth` |
| Does more real data help? | 39.3 → 42.0 → 43.8 → 45.6 at 10/25/50/100%, still rising | `lc_frac*`, `L4_ft-bge-small` |
| Does the reranker help on its own? | No: 38.9 vs 45.6 | `L5_ft-bge-small+ce` |

bge-small trains in about half the time of bge-base (26.1 vs 58 minutes), so it was the practical choice for many runs.

**Alternatives considered.** Running the ablations on bge-base. That would have taken about twice the GPU time, and the effects being tested don't depend on model size in an obvious way. Not measured. (rationale reconstructed)

**Consequences.** The rising learning curve suggests more labeled data would still help. Ablation differences under about one point are treated as noise (single seed).

## 23. Blend reranker and bi-encoder scores with per-query z-scores, weight chosen on validation

**Decision.** Final score = (1 − w) · z(bi-encoder) + w · z(reranker), with z-scores computed per query over the 20 candidates (population std + 1e-6).
- w is chosen by validation top-1 on a 0.05 grid from 0 to 1. Ties go to the weight nearest 0.5.
- The grid is saved to `outputs/fusion/<reranker>.json`.
- The served weight is 0.3.

(onet_analysis/matching/retrieval.py, pipeline.py `tune_fusion`)

**Context.** Reranker scores alone lowered validation top-1 from 46.2% to 40.0% for the bge-small pair (decision [17](#17-add-a-fine-tuned-cross-encoder-reranker)).

**Rationale.**
- The two scores have very different scales (cosine vs logits), so z-scoring each per query makes them comparable before blending.
- Validation top-1 by weight:

| pair | w = 0 (bi-encoder only) | chosen w | w = 1 (reranker only) | source |
|---|---|---|---|---|
| bge-small + reranker | 46.2 | 48.3 at 0.4 | 40.0 | outputs/fusion/L5_ft-bge-small+ce.json |
| bge-base + reranker (served) | 50.9 | 51.7 at 0.3 | 39.9 | outputs/fusion/L5s_ft-bge-small+synth+ce.json |

- On the real test set, blending beat both parts: 47.3% vs 45.6% (bi-encoder) and 38.9% (reranker) (docs/RESULTS.md, "Reranker scoring").

**Alternatives considered.**

| option | why not |
|---|---|
| Reranker score only | 38.9% vs 47.3% blended on real test (above). |
| Raw score sum without normalization | Scales differ by orders of magnitude, so one model would dominate. (rationale reconstructed) |
| Reciprocal rank fusion | Not measured. It discards score margins. (rationale reconstructed) |
| Choose w on test data | Overfits the test set. |

**Consequences.** For the served bge-base model the reranker adds little: real test top-1 rises from 50.2% to 51.1% and synthetic from 65.2% to 65.4%, while latency rises from 6.8 ms to 46.7 ms (docs/RESULTS.md). The gain is about one point, so per the single-seed caveat it is not clearly beyond noise. ECE also rose from 0.024 to 0.037 on real text (same source).

## 24. Fine-tune bge-base on real plus synthetic data, and reuse the bge-small reranker

**Decision.** Fine-tune bge-base-en-v1.5 on all five sources (86,430 examples, batch 32, 2,700 steps, 3,487 s). Blend it with the existing reranker trained on bge-small's candidates (`L5s_ft-bge-small+synth+ce`) at fusion weight 0.3 (outputs/runs/L4b_ft-bge-base+synth/config.json, outputs/runs/L5b_ft-bge-base+synth+ce+fuse/config.json).

**Context.** bge-small was fine-tuned first because bge-base was estimated at about three times the training time, over the one-hour approval threshold. Once the machine was free, bge-base was the best remaining lever: it had been the strongest off-the-shelf model (decision [14](#14-choose-the-bge-family-for-the-bi-encoder)).

**Rationale.**
- **Real text.** Top-1 rose from 45.6% to 50.2% (+4.6) over the same setup on bge-small.
- **Messy text.** Top-1 rose from 57.9% to 65.2% (+7.3).
- This was the largest single gain after fine-tuning itself (docs/RESULTS.md, "Key findings").
- Training took 58 minutes, inside the one-hour budget at batch 32 (README.md).
- Reusing the reranker avoided another training run of about 24 minutes. Fusion still lifted validation top-1 from 50.9% to 51.7% (outputs/fusion/L5s_ft-bge-small+synth+ce.json).

**Alternatives considered.**

| option | why not |
|---|---|
| Stay on bge-small | 4.6 points worse on real text and 7.3 on messy text (above). |
| bge-base on real data only | Not run. Synthetic data had already shown a large messy-text gain with no real-text cost on bge-small (decision [20](#20-generate-messy-training-text-with-an-llm-five-input-types-one-call-per-occupation)). |
| Retrain the reranker on bge-base's candidates | Not measured. |
| More synthetic data with the remaining budget | Not needed for this gain. Not measured. |

## 25. Serve the model with the best validation top-1, then MRR

**Decision.** The served model is chosen by validation top-1, with validation MRR as the tie-break. This selected `L5b_ft-bge-base+synth+ce+fuse`: validation top-1 51.7% vs 48.3% for the previous served model, `L5s_ft-bge-small+synth+ce+fuse` (outputs/runs/L5b_ft-bge-base+synth+ce+fuse/metrics.json; docs/RESULTS.md, "Error analysis" header).

**Context.** Several ladder runs were close. The earlier bge-small blends tied on validation top-1, and MRR broke the tie.

**Rationale.**
- Choosing on validation keeps the test sets honest.
- The served model's results:

| test set | n | top-1 | top-5 | MRR@20 | ECE |
|---|---|---|---|---|---|
| real O\*NET text | 6,792 | 51.1% | 79.1% | 0.633 | 0.037 |
| messy text, different LLM | 4,703 | 65.4% | 92.9% | 0.774 | 0.102 |
| TF-IDF baseline (top-1) | | 20.5% / 17.4% | | | |

(models/bundle/README.md; docs/RESULTS.md)

**Alternatives considered.**
- **The bi-encoder alone (`L4b`).** About 7× faster at 6.8 ms vs 46.7 ms, and 0.9 points lower on real top-1 (docs/RESULTS.md). The selection rule favors accuracy.
- **Choosing on synthetic test scores.** Rejected: that would turn a test set into a selection set.

## 26. Merge the data collector into this repository, keeping both histories

**Decision.** Merge ONET-Data-Collector into this repository with `git merge --allow-unrelated-histories`, so its 2024 commits stay visible (git: "Merge ONET-Data-Collector into this repo"). Archive the old repository with a pointer README rather than deleting it.

**Context.** Matching depended on the collector's bulk loader. Two repos meant two installs. The Dockerfile needed a second build context (`--build-context collector=../ONET-Data-Collector`). Discoverability was split across two pages.

**Rationale.**
- One `pip install` and one Docker build context.
- Keeping both histories preserves the project's full 2024-to-2026 timeline.
- Archiving keeps old links working.

**Alternatives considered.**

| option | why not |
|---|---|
| Keep two repos and cross-link them | Install and Docker friction stays, and attention is split. |
| Delete the old repo | Its history is already in the merged repo, but deleting breaks old links for no gain. The owner chose to keep it. |
| Squash the collector in without history | Loses the 2024 commits. |

## 27. Package as one distribution named onet-match

**Decision.**
- Distribution: `onet-match`, containing `onet_match` (a thin public alias), `onet_analysis` and `onet_data_collector`.
- Commands: `onet-match` (with `onet-analyze` kept as an alias) and `onet-collect`.
- Heavy dependencies are optional extras: `matching` (torch, sentence-transformers, transformers, huggingface_hub), `serve`, `mcp`, `synth` and `dev`. TF-IDF and BM25 need none of them (pyproject.toml).

**Context.** The old repo name, data-science-onet, and its description ("EDA on O\*NET API data") hid what the project now does.

**Rationale.** The name went through three rounds:

| round | candidates | outcome |
|---|---|---|
| 1 | `onet-match` vs keeping `data-science-onet` | `onet-match`: short, searchable, says what it does |
| 2 ("more technical") | `onet-semantic-search`, `occupation-retrieval`, `onet-retrieval`, `onet-match` | Briefly renamed to `onet-semantic-search` (git: "Rename package to onet-semantic-search") |
| 3 ("more catchy") | `job2soc` (recommended), `occurank`, `occubert`, `text2soc` | The owner chose `onet-match` (git: "Rename to onet-match") |

- `onet-match` keeps "O\*NET", the term people search for, and matches the import and CLI names.
- The rename happened before anything was published to PyPI or the Hub, when it was cheapest.
- GitHub redirects old URLs.
- Optional extras keep a basic install small.

**Alternatives considered.**
- **`job2soc`.** Reads like an ML tool and was free on PyPI, GitHub and the Hub. But "SOC" also means security operations center or system-on-chip, and the name drops "O\*NET".
- **`onet-semantic-search`.** Technical but long, and judged not catchy.
- **Separate packages for the collector and the matcher.** Rejected along with the two-repo layout (decision [26](#26-merge-the-data-collector-into-this-repository-keeping-both-histories)).

**Consequences.** `requires-python >= 3.10` (pyproject.toml). The `mcp>=2.3` extra needs a modern Python. (rationale reconstructed)

## 28. Host the model bundle on the Hugging Face Hub

**Decision.** Upload the bundle, with a model card, to `BenjaminChan12/onet-match`. Bundle resolution order:
1. `$ONET_MATCH_BUNDLE`;
2. `./models/bundle`;
3. the Hub repo (`$ONET_MATCH_HF_REPO`, default `BenjaminChan12/onet-match`), downloaded once into the Hugging Face cache.

`hf:<owner>/<repo>` selects a Hub bundle explicitly (onet_analysis/matching/engine.py).

**Context.** Trained models were not in git, so anyone cloning the repo would have had to retrain for hours before matching anything.

**Rationale.**
- `match()` works right after `pip install`. The first call downloads about 0.5 GB (README.md).
- The Hub is where ML users and agents look for models. The model card carries the evaluation table, limitations and O\*NET attribution (models/bundle/README.md).
- The resolution order lets users serve their own bundle or work offline.

**Alternatives considered.**

| option | why not |
|---|---|
| GitHub Release asset | No extra account needed, but less discoverable and no model card or Hub tooling. |
| Don't host; users retrain | Hours of training before first use. |

## 29. License the code and weights as MIT, and attribute O\*NET under CC BY 4.0

**Decision.**
- Code and model weights are MIT (LICENSE, models/bundle/README.md).
- The base models' licenses are listed: bge-base-en-v1.5 (MIT) and ms-marco-MiniLM-L6-v2 (Apache-2.0).
- The README, the model card and the demo Space carry the O\*NET attribution: "This product includes information from the O\*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under the CC BY 4.0 license." They also carry the trademark and no-endorsement notice.
- `CITATION.cff` and a Zenodo DOI support citation.

**Context.** The served bundle redistributes O\*NET content in its catalog and in the derived embeddings.

**Rationale.**
- CC BY 4.0 requires attribution wherever the data is redistributed. That includes the bundle and the Space, not only the repo.
- MIT matches the repository's existing license and the bi-encoder base model.

**Alternatives considered.** A more restrictive license. Not considered necessary for an open-source library. (rationale reconstructed)

## 30. Publish to PyPI with a project-scoped token, uploaded only with the owner's approval

**Decision.** Build the wheel and sdist, verify them with `twine check`, and upload with a PyPI token scoped to the `onet-match` project only, stored in the macOS Keychain and loaded per command. Every upload happens only with the owner's approval. Install lines switched to `pip install "onet-match[matching]"` once it was live (git: "Install from PyPI"). Releases: 0.3.0, 0.3.1 (MCP server), 0.3.2 (registry metadata), 0.3.3 (new examples, DOI) and 0.3.4 (dependency floor, decision [37](#37-require-sentence-transformers-62-the-version-the-bundle-was-saved-with)) (pyproject.toml; GitHub releases).

**Context.** Installing from a git URL is awkward. MCP clients launch servers with `uvx --from onet-match[mcp]`, which needs a PyPI package.

**Rationale.**
- The owner keeps control of the publishing account.
- A project-scoped token limits the damage if it leaks.
- Each release was verified with a clean install from PyPI.

**Alternatives considered.**

| option | why not |
|---|---|
| Install from GitHub only | Worse discoverability, and the MCP `uvx` install depends on PyPI. |
| An account-wide token | A leaked token could publish to every project on the account. The project-scoped token can only touch onet-match. |

**Consequences.** The first upload failed with a 403 because the Keychain prompt cut the token at 128 characters. Storing it with `-w "$(pbpaste)"` fixed it.

## 31. Add an MCP server with two read-only tools, using the mcp 2.x MCPServer API

**Decision.** `onet-match mcp` runs a Model Context Protocol server built on `mcp.server.MCPServer` (`mcp>=2.3`). It has two tools:
- `match_occupations(text, top_k)`;
- `get_occupation(soc)`.

Details (onet_analysis/matching/mcp_server.py):
- Both tools are annotated read-only, non-destructive, idempotent and closed-world.
- Errors meant for the client are raised as `ToolError`.
- Text is truncated to 8,000 characters and `top_k` is clamped to 1–20.
- Results are compact: top 3 tasks, top 5 skills and a short profile.
- The model loads on a background thread at startup.
- Transports: stdio by default, or `--transport streamable-http`.

**Context.** LLM agents asked about occupations tend to answer from memory. A tool lets them ground answers in O\*NET.

**Rationale.**
- `FastMCP` was removed in mcp v2, so `MCPServer` is the current API.
- Read-only annotations tell clients the tools are safe to call without confirmation.
- Without `ToolError`, a plain `ValueError` reached the client as a generic unexpected-tool error.
- Compact results save the agent's context. `get_occupation` returns the full record on demand.
- Preloading the model means the first call isn't slow.

**Alternatives considered.**

| option | why not |
|---|---|
| `FastMCP` (mcp v1 API) | Removed in mcp v2. |
| Ask agents to call the REST API | Needs a running server and custom client code. MCP clients launch the server themselves. |
| More tools (batch matching, search) | Not needed for the main use. Two tools keep the choice obvious for the agent. (rationale reconstructed) |

## 32. Host the demo as a free static Space with in-browser inference

**Decision.** Publish the demo as a static Hugging Face Space. The models run in the visitor's browser with transformers.js 4.3.2 (demo/README.md, demo/app.js). `demo/build.py` builds it from the served bundle.

**Context.** A Gradio demo was built and tested locally, but Hugging Face returned "402: PRO required" when creating Gradio and Docker Spaces.

**Rationale.**
- Static Spaces are free, have no cold starts and no server to keep running.
- Text never leaves the visitor's device (demo/README.md).

**Alternatives considered.**

| option | why not |
|---|---|
| Gradio Space on HF PRO | Costs a monthly subscription. |
| Colab notebook only | Visitors must run cells and wait for installs. |
| No hosted demo | Loses the zero-install way to try the model. |

## 33. Quantize to int8 ONNX: per-channel QInt8 for the bi-encoder, QUInt8 for the reranker

**Decision.** Export both models to ONNX with optimum, then quantize them dynamically:
- bi-encoder: per-channel `QInt8`;
- reranker: per-tensor `QUInt8`.

Only the int8 files ship, about 130 MB in total, cached after the first visit (demo/build.py; README.md, "In-browser demo").

**Context.** The fp32 models were too large for a fast first load in a browser.

**Rationale.**
- Plain `QUInt8` on the bi-encoder cost about 1.3 points of top-1. Per-channel `QInt8` keeps it within noise of PyTorch (demo/build.py comment).
- The reranker was fine with `QUInt8`.
- Python ONNX int8 picks the same top match as PyTorch 96.8% of the time on both real and synthetic queries (outputs/demo/browser_parity.json).

**Alternatives considered.**

| option | why not |
|---|---|
| Plain QUInt8 for both models | About 1.3 points of top-1 lost on the bi-encoder. |
| fp32 | Much larger download. |
| fp32 on WebGPU | See decision [34](#34-run-on-multi-threaded-wasm-mirror-the-python-pipeline-exactly-and-render-progressively). |

## 34. Run on multi-threaded WASM, mirror the Python pipeline exactly, and render progressively

**Decision.**
- `demo/pipeline.js` reimplements the ranking: CLS embedding, L2 normalization, cosine against precomputed embeddings, top 20, cross-encoder logits at max length 320, z-score blend with weight 0.3, and a temperature softmax with stable sorting.
- `restoreSep` puts back the final `[SEP]` token that transformers.js drops when it truncates.
- The Space sends COOP/COEP/CORP headers, so the page is cross-origin isolated and WASM can use up to 8 threads.
- The UI shows the bi-encoder ranking first, then swaps in the reranked result.

(demo/pipeline.js, demo/app.js, demo/README.md)

**Context.**
- The first browser build took 6–8 s per match. The reranker was the bottleneck: about 6.5 s for 20 pairs of up to 320 tokens on one thread, vs about 0.1 s for the bi-encoder.
- Results also differed slightly from Python because of the truncation behavior.

**Rationale.**
- With threads, a match took about 0.6 s locally and about 2 s on the live Space.
- After `restoreSep`, the Node run of the browser pipeline matches Python ONNX exactly: 100% top-1 agreement and 0.0 max confidence difference (outputs/demo/browser_parity.json).
- The pipeline file avoids browser APIs, so it can be checked against Python in Node.
- Measured on 800 test queries in Chrome WASM: top-1 61.5% vs 62.6% for PyTorch, 95.4% top-1 agreement, median absolute confidence difference 0.014 and p95 0.063 (outputs/demo/browser_parity.json). The page says the Python package is the reference.

**Alternatives considered.**

| option | why not |
|---|---|
| WebGPU | fp32 on WebGPU brought the reranker to about 1 s. That is slower than threaded WASM and needs a larger fp32 download. |
| Single-threaded WASM | About 6.5 s per match. |
| Bi-encoder only in the browser | Diverges from the served pipeline. The progressive UI gives the fast first result anyway. |

## 35. List the MCP server in the official MCP Registry

**Decision.** Add `server.json` with the name `io.github.Benjamin-V-Chan/onet-match`. It points to the PyPI package with `runtimeHint: uvx`, `--from onet-match[mcp]`, the positional argument `mcp` and stdio transport. A hidden `<!-- mcp-name: ... -->` line in the README proves package ownership. The entry was validated and published with the registry's `mcp-publisher` tool (server.json, README.md; git: "Add MCP registry metadata (server.json), v0.3.2").

**Context.** An MCP server is only useful if agents and users can find it.

**Rationale.**
- The official registry stores a short entry that points to PyPI.
- Other directories and client app browsers copy from it, so one submission reaches several catalogs.
- Ownership is proven through the README already published on PyPI, with no visible text added.

**Alternatives considered.** Submitting only to individual third-party directories. These were done as well, but the official registry feeds the others, so it came first.

**Consequences.** `server.json` is bumped with each release and republished with `mcp-publisher publish`, which needs a fresh `mcp-publisher login github` when the registry token expires.

## 36. Show only real model outputs in the README, chosen to cover use cases and input styles

**Decision.** The README shows 8 examples, each labeled with a use case: career exploration, skills-based hiring, resume parsing, career counseling and job-posting normalization. They use several input styles: free text, a skill list, a structured profile and a JSON posting. Every example is real output of the served model, saved in `outputs/examples/readme_examples.json`. Known failure cases, such as seniority-coded titles, were taken out of the README. They stay documented in the error analysis (docs/RESULTS.md) and the model card's limitations (models/bundle/README.md).

**Context.** Earlier README examples included misses, such as "Sr. Backend Eng II" → Web Developers, and examples that didn't map to clear use cases.

**Rationale.**
- The README is the project's landing page, so its examples show what the tool is for, across the audiences in the "Use cases" section.
- Saving the outputs keeps the "no invented numbers" rule: each confidence shown traces to a file.
- Failures are still disclosed where people judge accuracy: the results report, and the model card's "Weakest on raw task statements … and seniority-coded titles".

**Alternatives considered.**

| option | why not |
|---|---|
| Keep failure examples in the README table | The owner preferred the landing page to show intended use. Failures are covered in the results and limitations. |
| Hand-written illustrative examples | Would break the traceability rule. |

**Consequences.** The same examples drive the demo's example buttons, the model card's quick start and the social preview image, so every surface shows the same real outputs (demo/app.js, models/bundle/README.md, docs/assets/social-preview.png).

## 37. Require sentence-transformers 6.2, the version the bundle was saved with

**Decision.** Raise the `matching` extra from `sentence-transformers[train]>=5.0` to `>=6.2` and release 0.3.4 (pyproject.toml; GitHub release v0.3.4).

**Context.** The worked-example notebook (decision [38](#38-put-the-worked-example-in-its-own-repository-that-installs-onet-match-from-pypi)) failed on Kaggle, whose image ships sentence-transformers 5.7.0. Loading the bundle raised `No module named 'sentence_transformers.base.modules.normalize'`. The bundle was saved with 6.2.0, and its `modules.json` names module paths that only exist from 6.x on (models/bundle/biencoder/modules.json). The old `>=5.0` floor let pip keep the incompatible 5.7.

**Rationale.**
- The floor now matches what the saved files actually need, so pip upgrades older installs instead of failing at load time.
- Verified in a clean environment that started with sentence-transformers 5.7.0: installing the 0.3.4 wheel upgraded it to 6.2.0, and `match("I love weather, maps and tracking storms")` returned Atmospheric and Space Scientists at 0.87, the same output as the README.
- All 92 tests pass with the new floor.

**Alternatives considered.**

| option | why not |
|---|---|
| Rewrite the bundle's `modules.json` to the older module paths | Would need testing on both major versions and could break 6.x loading. A dependency floor is simpler and explicit. |
| Pin an exact version (`==6.2.0`) | Blocks compatible bug-fix releases and conflicts with other packages in users' environments. |
| Document the requirement only | Users would still hit the error before reading the docs. |

## 38. Put the worked example in its own repository that installs onet-match from PyPI

**Decision.** The LinkedIn job-postings analysis lives in a separate public repository, `onet-match-job-postings`. It has its own environment that installs `onet-match[matching]` from PyPI, and the same notebook is published on Kaggle with the source dataset attached. The onet-match README links to it under "Use cases".

**Context.** The notebook codes a random sample of 3,000 postings from the public LinkedIn Job Postings (2023–2024) dataset and compares posted pay by SOC major group. The 108 postings coded as Registered Nurses used 92 distinct titles. Median posted pay ran from $53k (Office and Administrative Support) to $116k (Computer and Mathematical). Matching took 137 s on a laptop (onet-match-job-postings README and notebook outputs).

**Rationale.**
- A separate project that only depends on the published package proves the PyPI install works end to end, as an outside user would run it. Running it on Kaggle is what exposed the dependency issue in decision [37](#37-require-sentence-transformers-62-the-version-the-bundle-was-saved-with).
- Keeps a 0.5 GB dataset and notebook outputs out of the library repository.
- The postings data is CC BY-SA 4.0, so it is downloaded at run time rather than redistributed.

**Alternatives considered.**

| option | why not |
|---|---|
| An `examples/` folder in this repository | Runs against the local editable install, so it wouldn't catch packaging problems. Mixes analysis outputs into the library repo. |
| Kaggle only | No version-controlled source, and Kaggle needs an account to view comfortably. |

