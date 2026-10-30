// onet-match in the browser: loads the ONNX models with transformers.js and renders results.
// The ranking itself lives in pipeline.js.
import {
  AutoTokenizer, AutoModel, AutoModelForSequenceClassification, env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

import { profile, rerank, retrieve } from "./pipeline.js";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "./models/"; // must be relative: transformers.js skips local lookups for full URLs
// Threads need cross-origin isolation (COOP/COEP headers, set in the Space config); otherwise 1 thread.
env.backends.onnx.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 8);

const $ = (id) => document.getElementById(id);
let M = null; // loaded models and data

const EXAMPLES = [
  "designed our robot in CAD and wired its circuits for my high school robotics team",
  "I love weather, maps and tracking storms",
  "Must know survival analysis, R, SAS and clinical trial data",
  "Must know Solidity, smart contracts and Ethereum",
  "Led a team of 12 nurses on a cardiac ICU night shift",
  "Excel, QuickBooks, payroll, AP/AR",
  "skills: Python, SQL, Tableau; interests: sports statistics",
];

function status(text, busy = true) {
  $("status").textContent = text;
  $("status").classList.toggle("busy", busy);
}

const progress = {};
function onProgress(p) {
  if (p.status !== "progress" || !p.total) return;
  progress[p.file + p.name] = [p.loaded, p.total];
  const [a, b] = Object.values(progress).reduce(([x, y], [l, t]) => [x + l, y + t], [0, 0]);
  status(`Downloading the model into your browser… ${(a / 1e6).toFixed(0)} / ${(b / 1e6).toFixed(0)} MB (cached after the first visit)`);
}

async function load() {
  const json = (f) => fetch(`./data/${f}`).then((r) => r.json());
  const [meta, occ, docs, embBuf] = await Promise.all([
    json("meta.json"), json("occupations.json"), json("docs.json"),
    fetch("./data/embeddings.bin").then((r) => r.arrayBuffer()),
  ]);
  const opts = { dtype: "q8", device: "wasm", progress_callback: onProgress };
  const [btok, bmodel, rtok, rmodel] = await Promise.all([
    AutoTokenizer.from_pretrained("biencoder"),
    AutoModel.from_pretrained("biencoder", opts),
    AutoTokenizer.from_pretrained("reranker"),
    AutoModelForSequenceClassification.from_pretrained("reranker", opts),
  ]);
  M = { meta, occ, docs, emb: new Float32Array(embBuf), btok, bmodel, rtok, rmodel };
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function render(text, ranked, topK, final) {
  const matches = ranked.slice(0, topK);
  const prof = final ? profile(M, matches) : null;
  const card = (m, r) => {
    const o = M.occ[m.idx];
    const conf = final
      ? `<span class="conf"><span class="bar"><i style="width:${(m.conf * 100).toFixed(1)}%"></i></span>${m.conf.toFixed(2)}</span>`
      : "";
    return `<details class="match" ${r === 0 && final ? "open" : ""}>
      <summary>
        <span class="rank">${r + 1}</span>
        <span class="title">${esc(o.title)}<small>${esc(o.soc)} · ${esc(o.major_group)}</small></span>
        ${conf}
      </summary>
      <div class="body">
        <p>${esc(o.description)}</p>
        <h4>Key tasks</h4><ul>${o.tasks.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
        <p><b>Top skills:</b> ${esc(o.skills.join(", ") || "n/a")}</p>
        <p><b>Knowledge:</b> ${esc(o.knowledge.join(", ") || "n/a")}</p>
        <p class="muted">Job zone ${o.job_zone ?? "n/a"} · RIASEC ${esc(o.riasec_code)} ·
          <a href="https://www.onetonline.org/link/summary/${esc(o.soc)}" target="_blank" rel="noopener">O*NET OnLine ↗</a></p>
      </div>
    </details>`;
  };
  const head = `<h3>${final ? "Top matches" : "First pass (refining…)"} for “${esc(text)}”</h3>`;
  if (!final) {
    $("results").innerHTML = `<div class="prelim">${head}${matches.map(card).join("")}</div>`;
    return;
  }
  const maxI = Math.max(...prof.interests.map((d) => d.weight), 1);
  $("results").innerHTML = `
    ${head}
    ${matches.map(card).join("")}
    <div class="profile">
      <h3>Profile across these matches</h3>
      <p><b>Interest code ${esc(prof.riasec)}</b> (RIASEC, O*NET interest scores 1–7)</p>
      <div class="riasec">${prof.interests.map((d) => `
        <div><span>${esc(d.name)}</span><span class="bar"><i style="width:${(d.weight / maxI) * 100}%"></i></span><span>${d.weight.toFixed(1)}</span></div>`).join("")}
      </div>
      <p><b>Skills:</b> ${esc(prof.skills.map((d) => d.name).join(", "))}</p>
    </div>`;
}

async function run() {
  const text = $("text").value;
  if (!text.trim()) return status("Type a job title, resume line, interest or skill list.", false);
  if (!M) return status("Still loading the model…");
  $("go").disabled = true;
  status("Matching…");
  const t0 = performance.now();
  try {
    const topK = Number($("topk").value);
    const first = await retrieve(M, text);
    render(text, first.top.map((idx) => ({ idx })), topK, false);
    status("Refining with the cross-encoder reranker…");
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0))); // paint the first pass
    render(text, await rerank(M, first), topK, true);
    status(`Matched in ${((performance.now() - t0) / 1000).toFixed(1)} s, on your device.`, false);
  } catch (err) {
    console.error(err);
    status(`Something went wrong: ${err.message}`, false);
  } finally {
    $("go").disabled = false;
  }
}

$("examples").innerHTML = EXAMPLES.map((e) => `<button type="button" class="chip">${esc(e)}</button>`).join("");
$("examples").addEventListener("click", (ev) => {
  if (ev.target.classList.contains("chip")) { $("text").value = ev.target.textContent; run(); }
});
$("go").addEventListener("click", run);
$("text").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); run(); }
});
$("topk").addEventListener("input", () => { $("topk-val").textContent = $("topk").value; });

status("Loading…");
load()
  .then(() => { status("Ready. Everything runs locally in your browser.", false); $("go").disabled = false; })
  .catch((err) => { console.error(err); status(`Could not load the model: ${err.message}`, false); });
