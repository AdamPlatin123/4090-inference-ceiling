# Finding 3: Where the Ceilings Actually Are (RTX 4090 × 27B W4A16)

Roofline for decode: every step must stream the full weight set once.

```
Bandwidth wall:  17.5 GB ÷ 1008 GB/s = 17.4 ms/step (single GPU), 8.7 ms (TP2)
Compute wall:    27e9 × B FLOP/step ÷ (165 TF × ~65% dequant efficiency) ≈ 4,000 tok/s
```

| batch | bandwidth ceiling | compute ceiling | measured | position |
|---:|---:|---:|---:|---|
| 1 | 57.5 | — | 52 (ninfer) | **91%** |
| 8 | 460 | ~2,000 | **412** (ninfer) | **91%** |
| 64 | 7,360 | ~4,000 | 630 (vLLM TP2) | 18-20% |
| 128 | 14,700 | ~4,000 | **1,120** (vLLM TP2) | 28-37% |

## What fills the gap at high batch (all measured/eliminated)
- GDN (gated-delta-net) state update: O(batch), does not amortize — prime suspect
- vLLM pass_config fusions (allreduce+RMS, QK-norm+RoPE, KV-cache): **measured ±0.5%,
  i.e. no effect at TPOT 88 ms** — microsecond fusions drown in GEMM-dominated steps
- async-scheduling: ±0.1%
- lm_head: already W8 in artifact (1.35 GB), Q6 would save 2% — not worth it

## Claim audit (why published higher numbers don't contradict this)
Every higher community number we found uses a smaller model, speculative decoding
(legit — see finding 1 for where it stops working), or total-token throughput
(includes prefill) instead of decode. At 91% of the bandwidth roofline there is
simply no headroom left for a same-model same-quant single-stream implementation.
