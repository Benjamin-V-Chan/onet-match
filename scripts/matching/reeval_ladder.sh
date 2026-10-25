#!/usr/bin/env bash
# Re-score existing models (no retraining), e.g. after the synthetic test set is added.
set -euo pipefail
A=${ONET_ANALYZE:-onet-analyze}
$A matching eval tfidf --run-name L1_tfidf
$A matching eval bm25  --run-name L2_bm25
$A matching eval dense=sentence-transformers/all-MiniLM-L6-v2 --run-name L3_minilm
$A matching eval dense=BAAI/bge-small-en-v1.5 --run-name L3_bge-small
$A matching eval dense=BAAI/bge-base-en-v1.5  --run-name L3_bge-base
$A matching eval dense=intfloat/e5-base-v2    --run-name L3_e5-base
for m in L4_ft-bge-small L4s_ft-bge-small+synth L4b_ft-bge-base+synth abl_hardneg_off abl_data_titles lc_frac0.1 lc_frac0.25 lc_frac0.5; do
  [[ -d models/$m ]] && $A matching eval "dense=models/$m" --run-name "$m"
done
[[ -d models/L5_ft-bge-small+ce ]] && \
  $A matching eval "dense=models/L4_ft-bge-small+rerank=models/L5_ft-bge-small+ce" --run-name L5_ft-bge-small+ce
[[ -d models/L5s_ft-bge-small+synth+ce ]] && \
  $A matching eval "dense=models/L4s_ft-bge-small+synth+rerank=models/L5s_ft-bge-small+synth+ce" \
      --run-name L5s_ft-bge-small+synth+ce
for r in L5_ft-bge-small+ce+fuse L5s_ft-bge-small+synth+ce+fuse L5b_ft-bge-base+synth+ce+fuse; do
  cfg=outputs/runs/$r/config.json
  [[ -f $cfg ]] && $A matching eval "$(python3 -c "import json; print(json.load(open('$cfg'))['spec'])")" --run-name "$r"
done
exit 0
