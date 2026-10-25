#!/usr/bin/env bash
# Reproduce every number in notes/matching-results.md, end to end.
#
#   scripts/matching/reproduce.sh            # real-data ladder + ablations (no API keys needed)
#   WITH_SYNTH=1 TRAIN_LLM=... EVAL_LLM=... EVAL_PROVIDER=... scripts/matching/reproduce.sh
#
# Synthetic data needs API keys in the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY).
set -euo pipefail
cd "$(dirname "$0")/../.."
A=${ONET_ANALYZE:-onet-analyze}
export ONET_ANALYZE=$A

$A matching build-data                                   # O*NET bulk download + catalog + pairs
$A matching eval tfidf --run-name L1_tfidf               # 1. baseline (existing TF-IDF setup)
$A matching eval bm25  --run-name L2_bm25                # 2. BM25
scripts/matching/run_offtheshelf.sh                      # 3. off-the-shelf embeddings + field ablation
scripts/matching/run_finetune.sh                         # 4-5. fine-tuned bi-encoder, reranker, ablations

if [[ "${WITH_SYNTH:-0}" == "1" ]]; then
  $A matching synth --role train --provider "${TRAIN_PROVIDER:-anthropic}" --model "$TRAIN_LLM"
  $A matching synth --role eval  --provider "${EVAL_PROVIDER:-openai-compatible}" --model "$EVAL_LLM" --per-type 1 \
      ${EVAL_PRICE_IN:+--price-in $EVAL_PRICE_IN} ${EVAL_PRICE_OUT:+--price-out $EVAL_PRICE_OUT}
  ALL=title,task,dwa,composite,synthetic
  $A matching train-biencoder --name L4s_ft-bge-small+synth --sources $ALL --eval
  $A matching train-reranker --name L5s_ft-bge-small+synth+ce --biencoder models/L4s_ft-bge-small+synth \
      --sources $ALL --eval
  $A matching tune-fusion "dense=models/L4s_ft-bge-small+synth+rerank=models/L5s_ft-bge-small+synth+ce" \
      --eval-as L5s_ft-bge-small+synth+ce+fuse
  $A matching train-biencoder --name L4b_ft-bge-base+synth --base-model BAAI/bge-base-en-v1.5 --sources $ALL \
      --batch-size 32 --eval
  $A matching tune-fusion "dense=models/L4b_ft-bge-base+synth+rerank=models/L5s_ft-bge-small+synth+ce" \
      --eval-as L5b_ft-bge-base+synth+ce+fuse
  # Re-evaluate earlier rungs so every model is scored on the synthetic set too.
  scripts/matching/reeval_ladder.sh
fi

$A matching report                                       # -> notes/matching-results.md
