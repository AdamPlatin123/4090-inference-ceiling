#!/usr/bin/env python3
"""emb/rerank 轻量服务 (PH402/GPU1, Pascal sm_60, torch cu126)
兼容端点: POST /v1/embeddings (OpenAI), POST /rerank (vLLM/Cohere风格)
模型: qwen3vl-emb-2b (embedding) + qwen3vl-rer-2b (rerank, 非w4用bf16)
pooling: LAST token (对齐 vLLM runner pooling 默认)
"""
import json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

DEV = "cuda"
BASE = os.path.expanduser("~/ai-models")
EMB_MODEL = f"{BASE}/qwen3vl-emb-2b"
RER_MODEL = f"{BASE}/qwen3vl-rer-2b"  # bf16原版(PH402 32G放得下, 无需w4)

import os as _os
MODE = _os.environ.get("MODE", "both")  # emb | rer | both

emb_model = emb_tok = rer_model = rer_tok = None
if MODE in ("emb", "both"):
    print("loading embedding model...", flush=True)
    emb_tok = AutoTokenizer.from_pretrained(EMB_MODEL, trust_remote_code=True)
    emb_model = AutoModel.from_pretrained(EMB_MODEL, torch_dtype=torch.float16,
                                          trust_remote_code=True, device_map=DEV)
    emb_model.eval()
if MODE in ("rer", "both"):
    print("loading rerank model...", flush=True)
    rer_tok = AutoTokenizer.from_pretrained(RER_MODEL, trust_remote_code=True)
    rer_model = AutoModelForImageTextToText.from_pretrained(RER_MODEL, torch_dtype=torch.float16,
                                                     trust_remote_code=True, device_map=DEV)
    rer_model.eval()
print("models ready", flush=True)

_lock = threading.Lock()

@torch.inference_mode()
def embed(texts):
    with _lock:
        out = []
        for t in texts:
            inputs = emb_tok(t, return_tensors="pt", truncation=True,
                             max_length=8192).to(DEV)
            hidden = emb_model(**inputs).last_hidden_state  # [1, L, H]
            vec = hidden[0, -1]  # LAST token pooling
            vec = torch.nn.functional.normalize(vec, dim=-1)
            out.append(vec.float().cpu().tolist())
        return out

@torch.inference_mode()
def rerank(query, docs):
    with _lock:
        scores = []
        for d in docs:
            chat = [
                {"role": "system",
                 "content": "Given a search query, retrieve relevant candidates that answer the query."},
                {"role": "query", "content": [{"type": "text", "text": query}]},
                {"role": "document", "content": [{"type": "text", "text": d}]},
            ]
            text = rer_tok.apply_chat_template(chat, chat_template="reranker",
                                               tokenize=False, add_generation_prompt=True)
            inputs = rer_tok(text, return_tensors="pt", truncation=True,
                             max_length=8192).to(DEV)
            logits = rer_model(**inputs).logits[0, -1]
            two = torch.tensor([logits[9693], logits[2152]])  # true/false
            scores.append(float(torch.softmax(two, dim=-1)[0]))  # P(true)
        return scores

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "qwen3vl-emb"}, {"id": "qwen3vl-rer"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            if self.path == "/v1/embeddings":
                inp = req.get("input", "")
                texts = inp if isinstance(inp, list) else [inp]
                vecs = embed(texts)
                self._send(200, {"data": [{"object": "embedding", "index": i,
                                           "embedding": v} for i, v in enumerate(vecs)],
                                 "model": req.get("model", "qwen3vl-emb")})
            elif self.path == "/rerank":
                q = req.get("query", "")
                docs = [d if isinstance(d, str) else d.get("text", "")
                        for d in req.get("documents", [])]
                scores = rerank(q, docs)
                order = sorted(range(len(docs)), key=lambda i: -scores[i])
                self._send(200, {"model": req.get("model", "qwen3vl-rer"),
                                 "results": [{"index": i, "document": {"text": docs[i]},
                                              "relevance_score": scores[i]} for i in order]})
            else:
                self._send(404, {"error": "unknown path"})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"MODE={MODE} port={port}", flush=True)
    print(f"serving on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
