#!/bin/bash
# ninfer 基准: ./ninfer-bench.sh [host] [port] [tag] [model_id]
BIN=/home/adam/vllm-venv/bin/vllm
TOK=/home/adam/ai-models/qwen38-27b-w4a16-vl
OUT=/home/adam/bench_results
HOST=${1:-127.0.0.1}
PORT=${2:-8120}
TAG=${3:-ninfer_mtp}
MODEL=${4:-qwen3.8-27b}
for C in 1 2 4 8 16; do
  P=$(( C * 4 )); [ $P -gt 128 ] && P=128; [ $P -lt 8 ] && P=8
  echo "#### [${TAG}] concurrency=$C prompts=$P $(date '+%T')"
  "$BIN" bench serve \
    --model "$MODEL" --tokenizer "$TOK" \
    --host "$HOST" --port "$PORT" \
    --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name random --input-len 2048 --output-len 256 \
    --max-concurrency "$C" --num-prompts "$P" \
    --seed 42 --disable-tqdm \
    --percentile-metrics ttft,tpot,itl \
    --save-result --result-dir "$OUT" --result-filename "bench_${TAG}_c${C}.json" \
    2>&1 | grep -E "Successful|Failed|Median TTFT|Median TPOT|P99 ITL|Output token throughput|Total token throughput|Benchmark duration|error"
  sleep 5
done
echo "#### [${TAG}] DONE $(date '+%T')"
