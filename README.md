# Pushing a 27B Model to 91% of the RTX 4090's Physical Ceiling

An engineering archive — every number reproducible, every claim audited.

**Hardware**: 2× RTX 4090 48G (PCIe 3.0, no NVLink, no P2P) + a Pascal-era utility GPU
**Model**: Qwen3.8-27B, W4A16 (groupwise-int compressed-tensors)
**Stacks**: vLLM 0.28 (TP2) and NInfer (community sm_89 fork, single GPU)

## Headline results

| workload | throughput | vs theoretical ceiling |
|---|---:|---:|
| Single-stream decode (MTP3 speculative) | **139 tok/s** | — |
| Single-stream decode (no speculation) | 52 tok/s | **91% of bandwidth roofline** |
| Batch-8 decode, single GPU (NInfer) | **412 tok/s** | **91% of bandwidth roofline** |
| Batch-64 decode, TP2 (vLLM) | 630 tok/s | ~19% (compute-bound region) |
| Batch-128 decode, TP2 (vLLM, 512-token prompts) | **1,120 tok/s** | 28% (compute wall ≈ 4,000) |

## The three findings (each with full data in `findings/`)

1. **[MTP speculative decoding inverts at batch ≥4](findings/mtp-concurrency-law.md)** —
   +167% single-stream, **-24% at batch 8**. Every community MTP speedup number you've
   seen is single-stream; batched serving should turn it off.

2. **[One env var made our TP2 8× slower](findings/nccl-shm-trap.md)** —
   `NCCL_SHM_DISABLE=1` (set during an unrelated GPU install) forces allreduce onto TCP
   sockets. Full forensics: how a benchmark archive + script archaeology caught it
   after days of misattribution (we even wrongly blamed a VM for a while — the
   exonerating evidence is in the finding).

3. **[Where the ceilings actually are](findings/roofline-analysis.md)** —
   bandwidth wall 57.5 tok/s single-stream (we hit 91%), compute wall ≈ 4,000 tok/s at
   large batch (vLLM sits at 28%, and we show all the official fusion flags move it
   ±0.5%, i.e. not at all). Includes a claim audit: every published higher number uses
   smaller models, speculation, or total-token (prefill-inclusive) accounting.

## Repository layout

```
data/           raw benchmark JSONs (vLLM bench serve output, reproducible seeds)
findings/       the three findings, each self-contained with tables and method
kernel/         W4A16 SmallT kernel (dynamic smem + tiered tile + double buffering)
                + KERNEL_NOTES.md documenting the three patches and their traps
diagnostics/    cudaEvent layer-trace instrumentation (per-layer, per-graph-replay,
                per-step-segment timing) — how we got honest numbers
reproduction/   one-shot scripts: concurrency sweeps, ceiling sprint matrix,
                production deploy, @reboot orchestration
pascal-revival/ running embedding/reranking utility models on a Pascal (sm_60) GPU
                with torch cu126 — includes the reranker chat-template format that
                took three wrong attempts to get right
```

## Measurement methodology (read this before disputing a number)

- **Short-output vLLM bench numbers under-report 2-3×**: chunked prefill interleaves
  with decode; we extract steady-state full-batch windows from server-side throughput
  logs instead. All headline numbers above are steady-state.
- Step-time ironclad evidence for the 91% claim: 30 consecutive decode steps,
  median 19.40 ms, p25-p75 = 19.37-19.44 ms (theory: 17.4 ms).
- Correctness verified alongside every perf number (arithmetic smoke tests);
  one kernel variant that scored +26% on microbench produced all-zero outputs —
  caught by the smoke test, documented in KERNEL_NOTES.md.

## Honest limitations

- Kernel work reaches state-of-practice parity (Marlin-class), not beyond
- The high-batch gap to the compute wall is GDN state-update + engine internals —
  an open engineering problem, not solved here
- NInfer work is on a community fork; single-GPU, max 8 lanes by design

## License & attribution

Apache-2.0. Benchmark archive technique inspired by hard-won lessons; the NCCL
forensics stand on [arXiv 2603.22774](https://arxiv.org/abs/2603.22774)
(CPU-induced slowdowns in multi-GPU LLM inference).
