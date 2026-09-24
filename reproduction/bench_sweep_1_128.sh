#!/bin/bash
# 8100 TP2 全档并发速率: 1-128, 同历史口径(random 2048/256, 采样, 关thinking)
BIN=/home/adam/vllm-venv/bin/vllm
TOK=/home/adam/ai-models/qwen38-27b-w4a16-vl
OUT=/home/adam/bench_results
for C in 1 2 4 8 16 32 64 128; do
  P=$(( C * 4 )); [ $P -gt 128 ] && P=128
  echo "#### c=$C prompts=$P $(date '+%T')"
  "$BIN" bench serve \
    --model qwen38-27b --tokenizer "$TOK" \
    --host 127.0.0.1 --port 8100 \
    --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name random --input-len 2048 --output-len 256 \
    --max-concurrency "$C" --num-prompts "$P" \
    --seed 42 --disable-tqdm \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --percentile-metrics ttft,tpot,itl \
    --save-result --result-dir "$OUT" --result-filename "bench_prod_sweep_c${C}.json" \
    2>&1 | grep -E "Successful|Failed|Median TTFT|Median TPOT|Output token throughput|Total token throughput|Benchmark duration"
  sleep 5
done
echo "#### SWEEP DONE $(date '+%T')"
