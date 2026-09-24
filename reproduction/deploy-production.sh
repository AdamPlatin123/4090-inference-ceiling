#!/bin/bash
# 生产一键部署 v2 (2026-09-17): 8100 TP2@0.93(GPU2/3) + emb8101/rerank8102(GPU1=PH402)
# 用法: 清空服务后 bash ~/deploy-production.sh
set -u

# --- 1. 8100 主模型: 0.93 顶格(尾巴不留工具, KV最大化) ---
nohup bash ~/vllm-qwen38-serve-093.sh > ~/vllm-qwen38.log 2>&1 &
echo "8100 TP2@0.93 启动(约4分钟)"
for i in $(seq 1 60); do sleep 10; ss -tln 2>/dev/null | grep -q ':8100 ' && echo "8100 ready" && break; done

# --- 2. emb(8101) + rerank(8102) 于 GPU1(PH402, torch cu126) ---
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MODE=emb PORT=8101 \
  nohup ~/p402-venv/bin/python ~/p402-serve.py > ~/p402-emb.log 2>&1 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MODE=rer PORT=8102 \
  nohup ~/p402-venv/bin/python ~/p402-serve.py > ~/p402-rer.log 2>&1 &
echo "PH402 双服务启动(约1.5分钟)"
for i in $(seq 1 30); do R=0; ss -tln 2>/dev/null | grep -q ':8101 ' && R=$((R+1)); ss -tln 2>/dev/null | grep -q ':8102 ' && R=$((R+1)); [ "$R" -eq 2 ] && echo "8101+8102 ready" && break; sleep 10; done

echo "=== 部署完成 ==="; ss -tln 2>/dev/null | grep -E ':(8100|8101|8102) '
# 验证: curl :8100 chat / :8101 /v1/embeddings / :8102 /rerank
