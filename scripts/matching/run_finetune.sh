#!/usr/bin/env bash
set -euo pipefail
A=${ONET_ANALYZE:-onet-analyze}
REAL=title,task,dwa,composite

$A matching train-biencoder --name L4_ft-bge-small --sources $REAL --eval
$A matching train-reranker --name pilot-ce --biencoder models/L4_ft-bge-small --max-steps 20
$A matching train-reranker --name L5_ft-bge-small+ce --biencoder models/L4_ft-bge-small --sources $REAL --eval
$A matching tune-fusion "dense=models/L4_ft-bge-small+rerank=models/L5_ft-bge-small+ce" --eval-as L5_ft-bge-small+ce+fuse

$A matching train-biencoder --name abl_hardneg_off --sources $REAL --no-hard-negatives --eval
$A matching train-biencoder --name abl_data_titles --sources title --eval
for f in 0.1 0.25 0.5; do
  $A matching train-biencoder --name "lc_frac$f" --sources $REAL --fraction "$f" --eval
done
