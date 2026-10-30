// The matching pipeline, kept free of browser APIs so it can be checked against Python in Node.
// Mirrors onet_analysis/matching: bi-encoder retrieve -> cross-encoder rerank -> z-scored blend ->
// temperature softmax, then the confidence-weighted profile.

const zscore = (x) => {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const std = Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length);
  return x.map((v) => (v - mean) / (std + 1e-6));
};

// transformers.js truncates after adding special tokens, so a truncated row ends in a word piece
// instead of [SEP] (Hugging Face tokenizers keep the [SEP]). Restore it to match Python exactly.
function restoreSep(enc, sepId, maxLength) {
  const [rows, cols] = enc.input_ids.dims;
  if (cols < maxLength) return enc;
  const ids = enc.input_ids.data, mask = enc.attention_mask.data;
  for (let r = 0; r < rows; r++) {
    const last = r * cols + cols - 1;
    if (mask[last] && ids[last] !== BigInt(sepId)) ids[last] = BigInt(sepId);
  }
  return enc;
}

// 1. Bi-encoder: CLS embedding, L2-normalised, cosine against precomputed occupation embeddings.
export async function retrieve(M, text) {
  const { meta, docs, emb, btok, bmodel } = M;
  const q = text.slice(0, 8000).trim();
  const qenc = restoreSep(await btok(q, { truncation: true, max_length: 256 }), btok.sep_token_id, 256);
  const { last_hidden_state: h } = await bmodel(qenc);
  const cls = h.data.slice(0, meta.dim);
  const norm = Math.hypot(...cls);
  const n = docs.length;
  const sims = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let d = 0, o = j * meta.dim; d < meta.dim; d++) s += emb[o + d] * cls[d];
    sims[j] = s / norm;
  }
  const top = [...sims.keys()].sort((a, b) => sims[b] - sims[a] || a - b).slice(0, meta.k);
  return { q, top, sims: top.map((j) => sims[j]) };
}

// 2. Cross-encoder rescoring of the top k, 3. z-scored blend, 4. temperature softmax.
export async function rerank(M, { q, top, sims }) {
  const { meta, docs, rtok, rmodel } = M;
  const enc = restoreSep(await rtok(top.map(() => q), {
    text_pair: top.map((j) => docs[j]), padding: true, truncation: true, max_length: 320,
  }), rtok.sep_token_id, 320);
  const ce = Array.from((await rmodel(enc)).logits.data);
  const zr = zscore(sims);
  const zc = zscore(ce);
  const fused = top.map((_, i) => (1 - meta.fuse) * zr[i] + meta.fuse * zc[i]);
  const order = [...fused.keys()].sort((a, b) => fused[b] - fused[a] || a - b);
  const f = order.map((i) => fused[i]);
  const mx = Math.max(...f);
  const e = f.map((v) => Math.exp((v - mx) / meta.temperature));
  const sum = e.reduce((a, b) => a + b, 0);
  return order.map((i, r) => ({ idx: top[i], conf: e[r] / sum }));
}

export async function rank(M, text) {
  return rerank(M, await retrieve(M, text));
}

// Confidence-weighted mean of O*NET ratings across the matches (same as aggregate_profile).
export function profile(M, matches) {
  const { meta, occ } = M;
  let w = matches.map((m) => m.conf);
  const tot = w.reduce((a, b) => a + b, 0) || 1;
  w = w.map((v) => v / tot);
  const agg = (key, names) => {
    const acc = new Array(names.length).fill(0);
    let cover = 0;
    matches.forEach((m, i) => {
      const r = occ[m.idx][key];
      if (!r) return;
      cover += w[i];
      r.forEach((v, k) => { if (v !== null) acc[k] += w[i] * v; });
    });
    if (!cover) return [];
    return names.map((name, k) => ({ name, weight: acc[k] / cover }));
  };
  const interests = agg("r_interests", meta.riasec)
    .sort((a, b) => b.weight - a.weight || meta.riasec.indexOf(a.name) - meta.riasec.indexOf(b.name));
  return {
    interests,
    riasec: interests.slice(0, 3).map((d) => d.name[0]).join(""),
    skills: agg("r_skills", meta.skill_names).sort((a, b) => b.weight - a.weight).slice(0, 8),
  };
}
