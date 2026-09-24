# Reviving a Pascal GPU (sm_60) for Utility Models

vLLM/torch cu130 refuses sm_60 ("no kernel image"). Fix: separate venv with
torch cu126 (arch list includes sm_60) + transformers + a ~120-line HTTP shim
(`/v1/embeddings`, `/rerank`).

## Reranker format — three wrong attempts documented
The correct inference format for Qwen3-VL-Reranker class models:
1. `AutoModelForImageTextToText` (NOT AutoModelForCausalLM — config class rejected)
2. The `additional_chat_templates/reranker.jinja` template with **role="query"**
   and **role="document"** messages (plain concatenation or prompt prefixes
   produce garbage scores that look plausible — verify with a known-good triad)
3. Score = softmax over last-position logits at true/false token ids
   (in the model's 1_LogitScore/config.json)

Naive pair-encoding scores "lunch" 0.89 vs "PyTorch" 0.96 — subtle nonsense.
Correct format: PyTorch 0.71 vs lunch 0.27.

fp16 only (Pascal bf16 unsupported). ~220ms/embedding, ~330ms/rerank(3 docs)
on a P100-class card — plenty for retrieval-side duty.
