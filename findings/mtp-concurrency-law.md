# Finding 1: MTP Speculative Decoding Has a Concurrency Boundary

Counter-intuitive, fully measured on Qwen3.8-27B (W4A16) @ RTX 4090 48G:

| Concurrency | no-MTP | MTP3 | MTP effect |
|---:|---:|---:|---:|
| 1 (single) | 52 | 139 | **+167%** ✅ |
| 8 (batch) | **412** | 315 | **-24%** ❌ |

## Why it inverts
An MTP step = 4 forwards (3 draft + 1 verify of T=32) producing ~21 tokens for 8 lanes.
Without MTP, one forward produces 8 tokens at 91% of bandwidth ceiling (19.4 ms/step,
measured over 30 consecutive steps, p25-p75 = 19.37-19.44 ms). The verify batch's
W4A16 GEMM path runs at 331-365 GB/s (vs 590-660 for T≤16), eating the acceptance gain.

## Practical law
- Interactive single-user serving → speculative decoding on
- Batched serving (≥4 concurrent) → speculative decoding **off** (we saw -24%)
- Community benchmark reports quoting MTP speedups are almost always single-stream

## Draft-token sweep (c8, greedy)
draft1: 268 / draft2: 263 / **draft3: 315** / draft5: 116 tok/s — draft3 is the optimum,
acceptance rate at 2k context caps at ~59% (draft head quality ceiling, not sampling).
