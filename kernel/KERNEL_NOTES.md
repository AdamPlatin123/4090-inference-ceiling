# W4A16 SmallT Kernel Notes (sm_89)

Three modifications to the q4_small_t_mma kernel (vs upstream ninfer-3090/4090 fork),
each verified with microbenchmarks + end-to-end + arithmetic correctness (12×12=144 etc.):

## 1. Dynamic shared memory (48KB → 99KB ceiling unlocked)
`Q4SmallTSharedBytes` template computes sizes for the launch side; three launch sites
set `cudaFuncAttributeMaxDynamicSharedMemorySize` via magic-static one-shot. Kernel-side
union stays intact, just `extern __shared__` + reinterpret.

## 2. Tiered kTileK (single-buffer deep chain for T≤16)
`kTileK = TileCols <= 16 ? 128 : 64` — deeper K chain amortizes q4→bf16 dequant.
**Trap**: scaling the tile alone produces all-zero outputs. The scale layout must change
in lockstep (three places): `scales[bufs][rows][warps * (kTileK/64)]`, staging loop
bounds `kWarps * (kTileK/64)` in 16B chunks, and the mma consumer's `k_split * (kTileK/64) + half` index.
Result: T=16 bandwidth 468→590 GB/s (+26%).

## 3. Double-buffered pipeline for T>16 (kTileK=64)
Ping-pong staging: `stage(g+2)` overlaps `compute(g)` via per-group cp.async commit/wait
pairing. TileCols≤16 keeps single buffer (double-buffering halves occupancy there and
HURTS: 590→546 measured).

## Why not HMMA harder?
At T≥16 the kernel is compute-bound at 96% FP32 (dequant + fmaf on CUDA cores, 79 of
82.6 TF). The remaining 2× headroom needs a dequant→tensor-core data-path rewrite
(what Marlin does upstream), not parameter tuning — we measured kStages/kWarpsPerBlock/
launch_bounds variants at ±0.5%.

## End-to-end effect
c8 (batch-8 decode) steady state 294 → 315 tok/s (+7.1%) on Qwen3.8-27B @ 4090.
Single-lane path (T=1, GEMV) unaffected — it was already at 67% of peak bandwidth.
