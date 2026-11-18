#!/usr/bin/env bash
# Full inference on one French medical report: Report-Extractor (GLiNER) -> retriever top 100 -> reranker.
#   bash run.sh REPORT.txt [GPU] [options of link_entities.py, e.g. --no-rerank --top 10 --keep gpu]
# Writes outputs/<name>_entities.json (step 1) and outputs/<name>_linked.{json,tsv} (step 2).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/local.env" ] || { echo "copy local.env.example to local.env and fill it in"; exit 1; }
set -a; . "$HERE/local.env"; set +a
REPORT=$(realpath "$1"); GPU=${2:-0}; shift; [ $# -gt 0 ] && shift
NAME=$(basename "${REPORT%.*}")
mkdir -p "$HERE/outputs"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "== step 1: entity extraction (GLiNER, CPU)"
CUDA_VISIBLE_DEVICES="" "$GLINER_PYTHON" "$HERE/Report-Extractor/extract_medical_entities.py" \
    "$REPORT" --model "${GLINER_MODEL:-almanach/camembert-bio-gliner-v0.1}" --json > "$HERE/outputs/${NAME}_entities.json"

echo "== step 2: linking (GPU $GPU)"
cd "$HERE"
CUDA_VISIBLE_DEVICES=$GPU "$LINK_PYTHON" "$HERE/link_entities.py" \
    "$HERE/outputs/${NAME}_entities.json" --out "$HERE/outputs/${NAME}_linked" "$@"
