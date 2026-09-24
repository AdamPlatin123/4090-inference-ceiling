#!/bin/bash
# 冲刺v3驱动: python生成变体→逐轮(起/测/冒烟/停)
set -u
LOG=~/sprint-results.txt
: > $LOG
for name in base0 async1 fuseAR fuseQK mnbt65; do
  echo "#### [$name] $(date '+%T')" | tee -a $LOG
  python3 ~/sprint3-gen.py $name /tmp/sprint-tmp.sh >/dev/null || { echo '生成失败' | tee -a $LOG; continue; }
  cd /home/adam; nohup bash /tmp/sprint-tmp.sh > ~/vllm-sprint.log 2>&1 &
  for i in $(seq 1 55); do sleep 10; ss -tln 2>/dev/null | grep -q ':8100 ' && break; done
  if ! ss -tln 2>/dev/null | grep -q ':8100 '; then echo "启动失败: $(grep -m1 -iE 'rror' ~/vllm-sprint.log | head -c 200)" | tee -a $LOG; continue; fi
  ~/vllm-venv/bin/vllm bench serve --model qwen38-27b --tokenizer ~/ai-models/qwen38-27b-w4a16-vl \
    --host 127.0.0.1 --port 8100 --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name random --input-len 512 --output-len 1000 --ignore-eos \
    --max-concurrency 128 --num-prompts 128 --seed 42 --disable-tqdm \
    --chat-template-kwargs '{"enable_thinking": false}' --percentile-metrics ttft,tpot \
    --save-result --result-dir ~/bench_results --result-filename "sprint_${name}.json" \
    2>&1 | grep -E 'Output token|Median TPOT' | tee -a $LOG
  curl -s --max-time 30 http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"12*12=?"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | grep -o '144' | head -1 | xargs -I{} echo "正确性: {}" | tee -a $LOG
  P=$(ss -tlnp 2>/dev/null | grep ':8100 ' | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$P" ] && kill $P
  sleep 8; pkill -9 -f 'VLLM::Worke[r]' 2>/dev/null; sleep 4
done
echo "#### 冲刺完成 $(date '+%T')" | tee -a $LOG
