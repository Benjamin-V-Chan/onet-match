#!/usr/bin/env bash
# Ladder step 3 (off-the-shelf embeddings) + occupation-representation ablation.
set -euo pipefail
A=${ONET_ANALYZE:-onet-analyze}
$A matching eval dense=sentence-transformers/all-MiniLM-L6-v2 --run-name L3_minilm
$A matching eval dense=BAAI/bge-small-en-v1.5               --run-name L3_bge-small
$A matching eval dense=BAAI/bge-base-en-v1.5                --run-name L3_bge-base
$A matching eval dense=intfloat/e5-base-v2                  --run-name L3_e5-base
for f in title title+desc title+desc+tasks title+desc+tasks+skills; do
  $A matching eval bm25                       --fields "$f" --run-name "abl_fields_bm25_$f"       --no-latency
  $A matching eval dense=BAAI/bge-small-en-v1.5 --fields "$f" --run-name "abl_fields_bge-small_$f" --no-latency
  $A matching eval dense=BAAI/bge-base-en-v1.5  --fields "$f" --run-name "abl_fields_bge-base_$f"  --no-latency
done
