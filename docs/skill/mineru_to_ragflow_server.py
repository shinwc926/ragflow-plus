#!/usr/bin/env python3
"""
minerU ➜ Embeddings ➜ RagFlow Plus (Python bridge)

1) Reads MinerU outputs (layout JSON or TSV/JSONL of parsed text) from a folder.
2) Reconstructs page-ordered text segments with metadata (page, block type, bbox).
3) Chunks text using a layout-aware greedy splitter.
4) Embeds chunks using either:
   - An OpenAI-compatible /v1/embeddings HTTP endpoint (vLLM, TGI, etc.), or
   - A local SentenceTransformers model (fallback option).
5) Ingests the chunks into RagFlow Plus in one of two ways:
   - "file_upload": pack as JSONL and upload via ragflow_sdk (treated as a doc), or
   - "direct_vectors": call a (configurable) batch upsert endpoint for segments with vectors.

*notice
- MinerU layout file(s) contain fields like: page_num, type, bbox, text. If your keys differ,
  edit `_normalize_mineru_item()`.
- Embedding endpoint is OpenAI-compatible. If your server expects a different JSON shape,
  adapt `_embed_via_http()`.
- RagFlow Plus ingestion: this script supports two modes (see ENV/CLI config). If your
  deployment exposes a specific batch upsert API, set `RAGFLOW_DIRECT_UPSERT_PATH` accordingly.

Usage
-----
python mineru_to_ragflow_plus.py \
  --mineru-dir /path/to/mineru_outputs \
  --kb-id <ragflow_kb_id> \
  --base-url http://localhost:9380 \
  --api-key ragflow-xxxx \
  --embedding-mode http \
  --embedding-url http://127.0.0.1:8000/v1/embeddings \
  --embedding-model /data/models/Qwen3-emb-8B-naive \
  --ingest-mode direct_vectors

Dependencies
------------
- pip install ragflow-sdk sentence-transformers requests tiktoken orjson tqdm

"""
from __future__ import annotations
import os
import json
import orjson
import math
import time
import uuid
import glob
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

try:
    import tiktoken  # for token estimation
except Exception:
    tiktoken = None

# Optional local embedding fallback
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# RagFlow SDK (for file upload mode)
try:
    from ragflow_sdk import RAGFlow
except Exception:
    RAGFlow = None

##############################
# Configuration via env vars #
##############################

DEF_EMBEDDING_BATCH = int(os.getenv("EMBED_BATCH", "64"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

# Ingestion mode: "file_upload" or "direct_vectors"
INGEST_MODE = os.getenv("RAGFLOW_INGEST_MODE", "direct_vectors")

# For file_upload mode
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", "")

# For direct_vectors mode
RAGFLOW_DIRECT_BASE = os.getenv("RAGFLOW_DIRECT_BASE", RAGFLOW_BASE_URL)
# Example path you may need to change to your deployment (leave leading slash)
RAGFLOW_DIRECT_UPSERT_PATH = os.getenv(
    "RAGFLOW_DIRECT_UPSERT_PATH",
    "/v1/kb/{kb_id}/segments:batch_upsert"  # <-- adjust to your API
)

# Embedding config
EMBED_MODE = os.getenv("EMBED_MODE", "http")  # "http" | "local"
EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:8000/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "/data/models/Qwen3-emb-8B-naive")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")  # if your server checks auth
# Some servers require different JSON field names — adapt here if needed
EMBED_INPUT_FIELD = os.getenv("EMBED_INPUT_FIELD", "input")  # e.g., "input" or "texts" or "content"
EMBED_MODEL_FIELD = os.getenv("EMBED_MODEL_FIELD", "model")
EMBED_OUTPUT_PATH = os.getenv("EMBED_OUTPUT_PATH", "./_artifacts/embeddings.jsonl")

# MinerU input directory
MINERU_DIR = os.getenv("MINERU_DIR", "./mineru_outputs")

@dataclass
class Segment:
    id: str
    page: int
    text: str
    type: str
    bbox: Optional[Tuple[float, float, float, float]]
    meta: Dict[str, Any]

@dataclass
class Chunk:
    id: str
    text: str
    tokens: int
    page: int
    meta: Dict[str, Any]

#minerU 파싱ㅎ하는
def _load_json_any(path: str) -> Any:
    with open(path, "rb") as f:
        try:
            return orjson.loads(f.read())
        except Exception:
            return json.load(f)


def _normalize_mineru_item(item: Dict[str, Any]) -> Optional[Segment]:
    """Map your MinerU keys here.
    Expected fields (customize as needed):
    - page_num -> int
    - text -> str
    - type -> str (e.g., "paragraph", "title", "table_cell", ...)
    - bbox -> [x1,y1,x2,y2]
    """
    text = (item.get("text") or "").strip()
    if not text:
        return None
    page = int(item.get("page_num") or item.get("page") or 1)
    typ = str(item.get("type") or "unknown")
    bbox = item.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        bbox = tuple(float(x) for x in bbox)
    else:
        bbox = None

    seg_id = item.get("id") or str(uuid.uuid4())
    meta = {k: v for k, v in item.items() if k not in {"page_num", "page", "text", "type", "bbox", "id"}}
    return Segment(id=seg_id, page=page, text=text, type=typ, bbox=bbox, meta=meta)


def read_mineru_segments(mineru_dir: str) -> List[Segment]:
    paths = []
    paths += glob.glob(os.path.join(mineru_dir, "**/*.json"), recursive=True)
    paths += glob.glob(os.path.join(mineru_dir, "**/*.jsonl"), recursive=True)

    segments: List[Segment] = []
    for p in paths:
        if p.endswith(".jsonl"):
            with open(p, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = orjson.loads(line)
                    seg = _normalize_mineru_item(obj)
                    if seg:
                        segments.append(seg)
        else:
            data = _load_json_any(p)
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
            elif isinstance(data, list):
                items = data
            else:
                continue
            for it in items:
                seg = _normalize_mineru_item(it)
                if seg:
                    segments.append(seg)

    # order by page, top-left (y then x) if bbox exists
    def sort_key(s: Segment):
        if s.bbox:
            x1, y1, x2, y2 = s.bbox
            return (s.page, y1, x1)
        return (s.page, 1e9, 1e9)

    segments.sort(key=sort_key)
    return segments

#청킹

class TokenEstimator:
    def __init__(self, model_hint: str = "cl100k_base"):
        self.encoder = None
        if tiktoken is not None:
            try:
                self.encoder = tiktoken.get_encoding(model_hint)
            except Exception:
                self.encoder = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not self.encoder:
            # simple fallback heuristic
            return max(1, math.ceil(len(text) / 4))
        return len(self.encoder.encode(text))


def chunk_segments(segments: List[Segment], max_tokens=512, overlap=64) -> List[Chunk]:
    est = TokenEstimator()
    chunks: List[Chunk] = []

    current: List[Segment] = []
    cur_tokens = 0

    def flush():
        nonlocal current, cur_tokens
        if not current:
            return
        text = "\n".join(s.text for s in current)
        tokens = est.count(text)
        page = current[0].page
        meta = {
            "segment_ids": [s.id for s in current],
            "types": [s.type for s in current],
        }
        chunks.append(Chunk(id=str(uuid.uuid4()), text=text, tokens=tokens, page=page, meta=meta))
        # create overlap by keeping tail tokens
        if overlap > 0:
            # keep last segment if it fits; otherwise empty
            tail = []
            tok = 0
            for s in reversed(current):
                st = est.count(s.text)
                if tok + st <= overlap:
                    tail.append(s)
                    tok += st
                else:
                    break
            current = list(reversed(tail))
            cur_tokens = sum(est.count(s.text) for s in current)
        else:
            current = []
            cur_tokens = 0

    for seg in segments:
        st = est.count(seg.text)
        if st > max_tokens:
            # hard split long single segment
            words = seg.text.split()
            buf = []
            btok = 0
            for w in words:
                wt = est.count(w + " ")
                if btok + wt > max_tokens:
                    text = " ".join(buf).strip()
                    chunks.append(Chunk(id=str(uuid.uuid4()), text=text, tokens=est.count(text), page=seg.page, meta={"segment_ids": [seg.id], "types": [seg.type]}))
                    buf, btok = [w], wt
                else:
                    buf.append(w)
                    btok += wt
            if buf:
                text = " ".join(buf).strip()
                chunks.append(Chunk(id=str(uuid.uuid4()), text=text, tokens=est.count(text), page=seg.page, meta={"segment_ids": [seg.id], "types": [seg.type]}))
            continue

        if cur_tokens + st > max_tokens:
            flush()
        current.append(seg)
        cur_tokens += st
    flush()
    return chunks

#임베딩 수행

def _embed_via_http(texts: List[str], *, url: str, model: str, api_key: str = "", input_field: str = "input", model_field: str = "model") -> List[List[float]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        input_field: texts,
        model_field: model,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    data = r.json()
    # OpenAI-compatible shape: { data: [{embedding: [...]}, ...] }
    if "data" in data:
        return [item["embedding"] for item in data["data"]]
    # Some servers return { embeddings: [...] }
    if "embeddings" in data:
        return data["embeddings"]
    # Some return directly list[list[float]]
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data
    raise RuntimeError(f"Unknown embeddings response shape: {list(data.keys())}")


def _embed_via_local_model(texts: List[str], model_path: str) -> List[List[float]]:
    if not SentenceTransformer:
        raise RuntimeError("sentence-transformers is not installed.")
    model = SentenceTransformer(model_path)
    return model.encode(texts, batch_size=DEF_EMBEDDING_BATCH, show_progress_bar=True, convert_to_numpy=False)


def embed_chunks(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    batch = []
    batch_ids = []

    def flush_batch():
        if not batch:
            return
        if EMBED_MODE == "http":
            vecs = _embed_via_http(batch, url=EMBED_URL, model=EMBED_MODEL, api_key=EMBED_API_KEY, input_field=EMBED_INPUT_FIELD, model_field=EMBED_MODEL_FIELD)
        else:
            vecs = _embed_via_local_model(batch, EMBED_MODEL)
        for cid, vec in zip(batch_ids, vecs):
            results.append({"id": cid, "embedding": vec})
        batch.clear(); batch_ids.clear()

    for ch in tqdm(chunks, desc="Embedding", unit="chunk"):
        batch.append(ch.text)
        batch_ids.append(ch.id)
        if len(batch) >= DEF_EMBEDDING_BATCH:
            flush_batch()
    flush_batch()

    # persist as JSONL artifact (optional)
    os.makedirs(os.path.dirname(EMBED_OUTPUT_PATH), exist_ok=True)
    with open(EMBED_OUTPUT_PATH, "wb") as f:
        for e in results:
            f.write(orjson.dumps(e) + b"\n")
    return results

# 벡터 저장

def _ingest_file_upload_jsonl(kb_id: str, jsonl_path: str, display_name: Optional[str] = None):
    if not RAGFlow:
        raise RuntimeError("ragflow-sdk is not installed.")
    rf = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)
    display_name = display_name or os.path.basename(jsonl_path)
    with open(jsonl_path, "rb") as f:
        blob = f.read()
    # SDKs differ; this follows earlier docs: rf.upload_document(kb_id, display_name=..., blob=...)
    return rf.upload_document(kb_id=kb_id, display_name=display_name, blob=blob)


def _ingest_direct_vectors(kb_id: str, 
                           payload: List[Dict[str, Any]],
                           extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    url = RAGFLOW_DIRECT_BASE.rstrip("/") + RAGFLOW_DIRECT_UPSERT_PATH.format(kb_id=kb_id)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    if extra_headers:
        headers.update(extra_headers)
    body = {"vectors": payload}
    r = requests.post(url, headers=headers, data=json.dumps(body))
    r.raise_for_status()
    return r.json()


def to_ragflow_vectors(chunks: List[Chunk], embeddings: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    out = []
    for ch in chunks:
        emb = embeddings.get(ch.id)
        if emb is None:
            continue
        vec = {
            "id": ch.id,
            "text": ch.text,
            "vector": emb,
            "metadata": {
                "page": ch.page,
                **ch.meta,
            }
        }
        out.append(vec)
    return out


# 명령어

import argparse

def main():
    p = argparse.ArgumentParser(description="MinerU ➜ Embeddings ➜ RagFlow Plus")
    p.add_argument("--mineru-dir", default=MINERU_DIR)
    p.add_argument("--kb-id", required=True)
    p.add_argument("--ingest-mode", default=INGEST_MODE, choices=["file_upload", "direct_vectors"]) 
    p.add_argument("--base-url", default=RAGFLOW_BASE_URL)
    p.add_argument("--api-key", default=RAGFLOW_API_KEY)

    # Embedding
    p.add_argument("--embedding-mode", default=EMBED_MODE, choices=["http", "local"]) 
    p.add_argument("--embedding-url", default=EMBED_URL)
    p.add_argument("--embedding-model", default=EMBED_MODEL)
    p.add_argument("--embedding-api-key", default=EMBED_API_KEY)
    p.add_argument("--embedding-input-field", default=EMBED_INPUT_FIELD)
    p.add_argument("--embedding-model-field", default=EMBED_MODEL_FIELD)

    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_CHUNK)
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_TOKENS)

    args = p.parse_args()

    global RAGFLOW_BASE_URL, RAGFLOW_API_KEY, INGEST_MODE
    global EMBED_MODE, EMBED_URL, EMBED_MODEL, EMBED_API_KEY, EMBED_INPUT_FIELD, EMBED_MODEL_FIELD

    RAGFLOW_BASE_URL = args.base_url
    RAGFLOW_API_KEY = args.api_key
    INGEST_MODE = args.ingest_mode

    EMBED_MODE = args.embedding_mode
    EMBED_URL = args.embedding_url
    EMBED_MODEL = args.embedding_model
    EMBED_API_KEY = args.embedding_api_key
    EMBED_INPUT_FIELD = args.embedding_input_field
    EMBED_MODEL_FIELD = args.embedding_model_field

    print(f"[1/4] Reading MinerU outputs from: {args.mineru_dir}")
    segments = read_mineru_segments(args.mineru_dir)
    print(f"  Loaded {len(segments)} segments")

    print(f"[2/4] Chunking (max_tokens={args.max_tokens}, overlap={args.overlap})")
    chunks = chunk_segments(segments, max_tokens=args.max_tokens, overlap=args.overlap)
    print(f"  Produced {len(chunks)} chunks")

    print(f"[3/4] Embedding with mode={EMBED_MODE}")
    embs = embed_chunks(chunks)
    emb_map = {e["id"]: e["embedding"] for e in embs}

    print(f"[4/4] Ingesting to RagFlow Plus (mode={INGEST_MODE})")
    if INGEST_MODE == "file_upload":
        # Write JSONL with text + metadata; RagFlow will embed server-side if configured.
        out_path = EMBED_OUTPUT_PATH.replace("embeddings.jsonl", "segments.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            for ch in chunks:
                rec = {
                    "id": ch.id,
                    "text": ch.text,
                    "metadata": {"page": ch.page, **ch.meta},
                }
                f.write(orjson.dumps(rec) + b"\n")
        resp = _ingest_file_upload_jsonl(args.kb_id, out_path, display_name=os.path.basename(out_path))
        print("Ingest (file_upload) response:", resp)
    else:
        vectors = to_ragflow_vectors(chunks, emb_map)
        resp = _ingest_direct_vectors(args.kb_id, vectors)
        print("Ingest (direct_vectors) response:", resp)


if __name__ == "__main__":
    main()

