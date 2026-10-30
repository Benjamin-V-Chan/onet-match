---
title: onet-match
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
pinned: false
license: mit
short_description: Map job titles, resumes and skills to O*NET occupations
models:
  - BenjaminChan12/onet-match
tags:
  - onet
  - occupation-classification
  - job-title-normalization
  - semantic-search
  - career-exploration
  - transformers.js
---

# onet-match demo

Type any free text about work (a job title, resume bullet, job posting,
interest statement or skill list) and get the closest O\*NET occupations, with
SOC codes, calibrated confidence, key tasks, skills and RIASEC interests.

It runs entirely in your browser with [transformers.js](https://huggingface.co/docs/transformers.js):
the fine-tuned bi-encoder and cross-encoder reranker are compressed to int8
ONNX (about 130 MB, cached after the first visit). Text you type never leaves
your device.

- Code, docs and accuracy: https://github.com/Benjamin-V-Chan/onet-match
- Install: `pip install "onet-match[matching]"`
- Model: https://huggingface.co/BenjaminChan12/onet-match
- Built from the repo with `demo/build.py`

This product includes information from the O\*NET 31.0 Database by the U.S.
Department of Labor, Employment and Training Administration (USDOL/ETA), used
under the CC BY 4.0 license. O\*NET® is a trademark of USDOL/ETA.
