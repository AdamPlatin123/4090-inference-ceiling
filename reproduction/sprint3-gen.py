#!/usr/bin/env python3
"""冲刺v3: 从093-fast母本生成参数变体脚本(保留全部环境), 供bash驱动逐轮跑"""
import sys, os
BASE = open(os.path.expanduser('~/vllm-qwen38-serve-093-fast.sh')).read()
ANCHOR = '--reasoning-parser qwen3'
assert ANCHOR in BASE, '母本锚点缺失'

def gen(extra, path):
    assert BASE.count(ANCHOR) == 1
    s = BASE.replace(ANCHOR, ANCHOR + (' ' + extra if extra else ''))
    open(path, 'w').write(s)
    os.chmod(path, 0o755)

import json as _j
def _cc(d):  # 编译配置参数(单引号包JSON, shell安全)
    return "--compilation-config '" + _j.dumps({"pass_config": d}, separators=(",", ":")) + "'"

ROUNDS = {
    "base0":  "",
    "async1": "--async-scheduling",
    "fuseAR": "--async-scheduling " + _cc({"fuse_allreduce_rms": True}),
    "fuseQK": "--async-scheduling " + _cc({"fuse_allreduce_rms": True, "enable_qk_norm_rope_fusion": True}),
    "fuseKVC": "--async-scheduling " + _cc({"fuse_allreduce_rms": True, "enable_qk_norm_rope_fusion": True, "fuse_qk_norm_rope_kvcache": True}),
    "mnbt65": "--async-scheduling --max-num-batched-tokens 65536 " + _cc({"fuse_allreduce_rms": True, "enable_qk_norm_rope_fusion": True, "fuse_qk_norm_rope_kvcache": True}),
}
if __name__ == '__main__':
    name, path = sys.argv[1], sys.argv[2]
    gen(ROUNDS[name], path)
    print(f'{name} -> {path} (+{ROUNDS[name][:80]})')
