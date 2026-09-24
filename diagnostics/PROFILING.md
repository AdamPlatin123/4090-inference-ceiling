# Honest Measurement Instrumentation

Three-layer cudaEvent instrumentation for NInfer-class engines (nsys-free):
per-layer, per-graph-replay, per-step-segment (submit/wait/consume).
Gate with NINFER_LAYER_TRACE=1. Exposes two measurement traps:

1. **Prefill contamination**: short-output bench (256 tok) mixes chunked-prefill
   windows into decode; steady-state is 2-3× higher than the blended number.
2. **Context-length drift**: KV growth crosses graph envelope tiers, decaying step
   time (19.4 → 25 ms observed from 2k → 3k context). Always report the KV window.

vLLM side: extract "throughput" log lines at full batch (e.g. `batch 8.00`),
ignore windows containing prefill tokens, and re-run same-seed for repeatability.
