# Finding 2: One Env Var Made TP2 8× Slower (NCCL_SHM_DISABLE=1)

## The crime scene
Same model, same script except NCCL env, same load window:

| config | c64 decode | TPOT |
|---|---:|---:|
| `NCCL_SHM_DISABLE=1` (the trap) | 82-184 tok/s | 225-356 ms |
| SHM allowed | **630 tok/s** | 50.25 ms |
| + fixed measurement (input 512) | **1120 tok/s @ c128** | 88 ms |

## How the trap was set
Adding a GPU to the bus (a Pascal card for utility models) triggered an NCCL init
hang; the fix at the time was the "NCCL three disables" (P2P/SHM/IB). P2P disable is
harmless on PCIe 3.0 (no P2P hardware anyway), **SHM disable forces allreduce onto TCP
sockets** — 64 layers × 2 allreduces/step at socket latency = 8× decode collapse.

## Forensics that found it
1. Historic bench archive showed 663 tok/s (9-14) vs 82 (9-17) — same binary
2. Script archaeology (session transcripts + file mtimes): the three-disables line
   appeared between the two dates, coinciding with the GPU install
3. Controlled A/B: unset only SHM → immediate 3× recovery, then full 8× after
   isolating a concurrent measurement artifact
4. Cross-check: [arXiv 2603.22774](https://arxiv.org/abs/2603.22774) — CPU-side paths
   stall synchronous NCCL collectives; GPU sits idle

## Checklist for anyone
- TP on PCIe without NVLink: never disable SHM transport
- After ANY hardware change, re-run your benchmark archive, not just a smoke test
- vLLM TP degradation? `nccl-shm` is the first thing to check, before blaming kernels
