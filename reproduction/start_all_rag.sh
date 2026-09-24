#!/bin/bash
# 三服务开机自启编排：main(8100) → wemm(8101) → qrer(8102) 严格串行
# 由 crontab @reboot 触发；幂等（已在跑则跳过）
# 2026-09-14: main 改用 ~/vllm-qwen38-serve.sh（带 --enable-auto-tool-choice
#   --tool-call-parser step3p5）。此前引用的 serve_w4vl_kvfp8.sh 是 9-08 加 tool
#   支持前的旧副本，9-13 重启后被 @reboot 拉起导致 tool 功能丢失。
#   8100 健康判定增加参数守卫 main_ok：在线但缺 --tool-call-parser 视为故障。
exec >> /tmp/rag_autostart.log 2>&1
echo "=== autostart $(date) ==="

# 带重试的健康探测：排除服务预热期 curl 瞬时超时导致的误判
is_up() {
  local i
  for i in 1 2 3; do
    curl -s -m 5 "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -q '"id"' && return 0
    [ "$i" = "3" ] || sleep 3
  done
  return 1
}

# main(8100) 守卫：不仅 /v1/models 在线，还要求监听进程 cmdline 带
# --tool-call-parser —— 防止旧参数脚本再次顶替造成 tool 功能静默丢失
main_ok() {
  is_up 8100 || return 1
  local pid=$(ss -tlnp 2>/dev/null | grep ":8100 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
  [ -n "$pid" ] && tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q -- "--tool-call-parser"
}

kill_stale() {
  # 只杀本用户的 vllm 服务进程（按端口匹配，避免误杀）
  for port in "$@"; do
    for pid in $(pgrep -f "port $port" 2>/dev/null); do
      [ "$(readlink /proc/$pid/exe 2>/dev/null | grep -c python)" = "1" ] && kill "$pid" 2>/dev/null
    done
  done
  sleep 3
  # GPU 残留：EngineCore/Worker 的 cmdline 被 vLLM 改写为大写 "VLLM::..."，
  # 匹配须大小写不敏感（原写法 grep "vllm" 实际杀不到任何残留进程）
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    ps -o args= -p $pid 2>/dev/null | grep -iq "vllm" && kill -9 "$pid" 2>/dev/null
  done
  sleep 3
}

start_and_wait() {  # $1=script $2=port $3=timeout_sec $4=log $5=check_fn(默认 is_up)
  local check=${5:-is_up}
  local t0=$(date +%s)
  setsid env PYTHONUNBUFFERED=1 nohup "$1" > "$4" 2>&1 < /dev/null &
  while true; do
    "$check" "$2" && { echo "[$2] UP in $(( $(date +%s) - t0 ))s"; return 0; }
    [ $(( $(date +%s) - t0 )) -gt "$3" ] && { echo "[$2] TIMEOUT after ${3}s"; return 1; }
    sleep 15
  done
}

# 幂等：全部已在线（main 需过参数守卫）则退出
if main_ok && is_up 8101 && is_up 8102; then echo "all already up"; exit 0; fi

kill_stale 8100 8101 8102

# 串行链（main 失败即退出，与原逻辑一致；侧服务由保活 cron 兜底）
start_and_wait ~/vllm-qwen38-serve-093-fast.sh 8100 2400 /tmp/vllm_serve.log main_ok || {
  echo "=== autostart FAILED: main 8100 未就绪 ==="; exit 1
}
# PH402(GPU1)双工具: torch cu126方案(vLLM在Pascal不可用), 2026-09-23起
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MODE=emb PORT=8101 setsid env PYTHONUNBUFFERED=1 nohup ~/p402-venv/bin/python ~/p402-serve.py > /tmp/p402_emb.log 2>&1 < /dev/null &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MODE=rer PORT=8102 setsid env PYTHONUNBUFFERED=1 nohup ~/p402-venv/bin/python ~/p402-serve.py > /tmp/p402_rer.log 2>&1 < /dev/null &
for i in $(seq 1 60); do sleep 10; is_up 8101 && is_up 8102 && break; done
echo "=== autostart done (8100:$(main_ok && echo OK || echo FAIL) 8101:$(is_up 8101 && echo OK || echo FAIL) 8102:$(is_up 8102 && echo OK || echo FAIL)) ==="
