# Applications

`onet_analysis.matching` turns any free text about work into concrete O*NET
occupations, each with a calibrated confidence and the tasks, skills, knowledge
and RIASEC interests O*NET records for it. It is a general-purpose, open-source
building block (MIT-licensed code, CC BY 4.0 O*NET data), and nothing in it is
specific to one organization or product.

This page describes applications the library supports. It does not claim that
any organization uses it.

Every application uses the same three calls:

```python
from onet_analysis.matching import match, default_engine

result = match(text, top_k=5)            # ranked occupations + aggregated profile
engine = default_engine()
engine.match_many(texts, top_k=5)        # batches
engine.occupation("17-2199.08")          # full record for one occupation
```

Or over HTTP: `POST /match`, `POST /match/batch`, `GET /occupation/{soc}` (see [API.md](API.md)).

---

## 1. EdTech platforms and student career exploration

Many education platforms collect free-form records about each student. Systems
like Project Leo store student data about interests, strengths, past experiences
and projects, and portfolio, advising and project-based-learning tools collect
similar material. These texts are rich but unstructured, written in a student's
own words:

- *"I love drawing and figuring out how machines work"*
- *"ran the robotics club's CAD team"*
- *"built a website for my mom's bakery and ran their Instagram"*
- *"good at explaining math to younger kids; patient"*

The library maps text like this to occupations, skills and pathways.

**From interest or strength statements to careers.** You can pass an interest
statement straight to `match`. The result is a ranked list of occupations such
as engineering, drafting or industrial-design roles for the first example, each
with a confidence. The `profile` aggregates the RIASEC interests of those
occupations, so a platform can show "your statement leans Realistic /
Investigative / Artistic" next to concrete careers. That way the RIASEC framing
students may already know from interest inventories connects to real job
titles.

**From past projects and experiences to the skills careers require.** A student
project ("designed and 3D-printed a prosthetic hand for a science fair") matches
occupations whose O*NET tasks resemble the work. Each match carries the
occupation's most important tasks, skills and knowledge areas, with O*NET
importance and level ratings. A platform can use these to show:

- what the project already demonstrates, by surfacing matched tasks that read
  like what the student did, and
- what the matched careers also require: higher-importance skills and
  knowledge areas the student hasn't shown evidence of yet, which are candidate
  next projects or courses.

**From matches to pathways.** Each match includes its O*NET **Job Zone**, which
summarizes how much preparation an occupation typically needs. Job Zones range
from little or no preparation (1) to extensive preparation such as graduate
degrees (5). Combined with the SOC major group, this lets a platform group
suggestions into near-term and longer-term options, or into career clusters.
`GET /occupation/{soc}` returns the full task, activity, skill, knowledge,
ability and interest record to build a career page from.

**Aggregating across a student record.** A student usually has many entries.
Call `match_many` over all of a student's interests, projects and experiences.
Then combine the per-entry results, for example by summing confidences per
occupation, or by averaging the aggregated skill and interest profiles. Recurring
occupations and skills give a steadier picture than any single entry.

**Design notes for EdTech use:**

- Treat output as a **suggestion for exploration**, not advice or a placement
  decision. Show several matches, not just the top one, and use the
  confidence to decide how strongly to phrase suggestions. The reliability
  curves in `notes/matching-results.md` show how far confidence can be trusted
  for each input style.
- Short interest statements are naturally ambiguous. A spread of moderate
  confidences is the correct answer for them, not a failure.
- O*NET occupations are US labor-market categories. Some student interests,
  like particular hobbies or emerging roles, map only loosely.
- Student data is sensitive. The library runs fully locally, either in-process
  or as a self-hosted container, so text never has to leave your
  infrastructure. The HTTP service stores nothing.
- Check the matches for bias before deploying: language that correlates with
  demographics can shift results. Test it on examples from your own
  population first.

---

## 2. Resume and profile parsing

Experience bullets, project descriptions and skill lists from resumes or
profiles can be matched one at a time, then aggregated, to:

- infer the occupations a person's history most resembles, which is useful
  when job titles are idiosyncratic ("Ninja", "Associate II") or missing
- attach standardized O*NET skills and knowledge areas to free-text experience
- support skills-based matching between candidates and roles through the shared
  occupation and skill vocabulary, rather than relying on exact title matches

## 3. Job-posting normalization

Titles and descriptions in job postings vary endlessly ("Sr. Backend Eng II",
"Software Ninja", "Developer, Platform"). Mapping each posting to an O*NET-SOC
code, using the title alone or title plus description, makes postings
comparable across employers and sources. It also lets aggregators de-duplicate
and categorize listings. The `major_group` field gives a coarse fallback when
the fine-grained occupation is uncertain, and the evaluation reports
major-group accuracy separately for that reason.

## 4. Workforce and labor-market analytics

Batch-code large text collections (job ads, internal HR role catalogs, survey
"what do you do?" responses, training-program descriptions) to O*NET occupations.
Then join them to any data keyed by SOC code, such as employment and wage
statistics, projections, or automation-exposure research. Use the confidence to
send low-confidence cases to human coders, a common pattern in occupational
coding.

## 5. Career counseling and guidance tools

Counselors and self-service career tools can take a client's description of
past work or what they enjoy and show concrete occupations, what those jobs
involve (top tasks), what they require (skills, knowledge, Job Zone), and how
they align with the client's interests (RIASEC). Career changers can match
descriptions of their past roles to find related occupations, then compare skill
profiles to see transferable skills and gaps. Veterans can do the same with
descriptions of their military duties.

## 6. Other uses

- **Training and curriculum mapping:** match course or program descriptions to
  the occupations and skills they prepare for.
- **Search and recommendation backends:** use the matcher as a semantic first
  stage for job or course search.
- **Research:** reproducible occupational coding of free text, with a documented
  benchmark and calibration.

---

## Choosing a model

`notes/matching-results.md` reports accuracy, calibration and latency for every
model on each input type. The lexical bundle (`onet-analyze matching bundle bm25
--run L2_bm25`) needs no PyTorch and is the fastest option (see its latency in the
results), but it is much less
accurate on messy text. The fine-tuned dense models, optionally with the
reranker, are the accurate choice. Pick based on the input types you care about
and your latency budget.
