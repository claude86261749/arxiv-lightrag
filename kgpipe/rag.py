"""Stages 3-5: chunk, extract and embed with LightRAG, run in-process.

The server's upload route resolves `LIGHTRAG_PARSER` and enqueues the file as
`pending_parse`; this module does the same through the Python API so that the
resolution stage can reach the vector store and LightRAG's summary function in
the same process. Every model call goes through counting wrappers.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

DEFAULT_TYPES_GUIDANCE = """Classify each entity using one of the following types. If no type fits, use `Other`.

- Method: A named technique, algorithm, training objective, loss or procedure (e.g. Contrastive Learning, Beam Search, Residual Quantization)
- Model: A specific named model or system architecture, including model families (e.g. SASRec, BERT, OneRec)
- Dataset: A named dataset, benchmark or corpus (e.g. MovieLens-1M, MS MARCO, BEIR)
- Metric: An evaluation measure (e.g. NDCG@10, Recall, Mean Reciprocal Rank)
- Task: A problem setting or application (e.g. Sequential Recommendation, Dense Retrieval, Query Rewriting)
- Concept: A technical idea or phenomenon that is not a method or task (e.g. Semantic ID, Cold Start, Popularity Bias)
- Tool: Software, library, platform or hardware used as a tool (e.g. FAISS, PyTorch, Elasticsearch)

Do not extract people, institutions, companies acting as authors, conferences, or venues.
Do not extract generic placeholders such as "Model", "Method", "Baseline", "Results", "Experiment", "This Paper" or "Proposed Approach"; name the specific thing instead.
Do not extract anonymous labels that only mean something inside one paper (e.g. "Model A", "Mixture B", "Setting 2", "Variant (iii)"); describe what they stand for in the description of a named entity instead.
Entity names: use the singular form and the common short name (write "Transformer", not "Transformers" or "Transformer Architecture"). For acronyms, use the acronym as the name when the paper mainly uses it, and put the expansion in the description."""


_http = None


async def batch_embed(texts: list[str], model: str, dim: int, key: str):
    """One batchEmbedContents request per call, one vector per text, L2-normalised.

    Replaces LightRAG's gemini binding: with gemini-embedding-2 the SDK's list
    call returns a single vector for the whole batch (checked on LightRAG 1.5.7,
    google-genai 2.28), which fails every vector-store flush.
    """
    import httpx
    import numpy as np

    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=120)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    body = {"requests": [{"model": f"models/{model}", "content": {"parts": [{"text": t}]},
                          "outputDimensionality": dim} for t in texts]}
    for attempt in range(8):
        r = await _http.post(url, headers={"x-goog-api-key": key}, json=body)
        if r.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(min(60, 2 ** (attempt + 1)))
            continue
        r.raise_for_status()
        v = np.array([e["values"] for e in r.json()["embeddings"]], dtype=np.float32)
        if v.shape[0] != len(texts):
            raise ValueError(f"embedding count {v.shape[0]} != {len(texts)}")
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    raise RuntimeError(f"embedding kept failing: {r.status_code} {r.text[:200]}")


@dataclass
class Counters:
    llm_calls: dict = field(default_factory=dict)
    llm_tokens: dict = field(default_factory=dict)
    embed_requests: int = 0
    embed_texts: int = 0
    embed_chars: int = 0
    started: float = field(default_factory=time.time)

    def llm(self, kind: str, usage: dict | None):
        self.llm_calls[kind] = self.llm_calls.get(kind, 0) + 1
        t = self.llm_tokens.setdefault(kind, {"prompt": 0, "completion": 0, "thinking": 0})
        if usage:
            p, c = usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0
            t["prompt"] += p
            t["completion"] += c
            # LightRAG's tracker leaves thinking out of completion_tokens; Gemini bills it as output.
            t["thinking"] += max(0, (usage.get("total_tokens", 0) or 0) - p - c)

    def as_dict(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_tokens": self.llm_tokens,
                "embed_requests": self.embed_requests, "embed_texts": self.embed_texts,
                "embed_chars": self.embed_chars, "seconds": round(time.time() - self.started, 1)}


def classify_prompt(prompt: str, system_prompt: str | None) -> str:
    head = prompt[:400]
    if "Based on the last extraction task" in head:
        return "glean"
    if "Extract entities" in head or "---Input Text---" in prompt and "entity" in (system_prompt or "").lower()[:400]:
        return "extract"
    if "summar" in (system_prompt or "").lower() or "summar" in head.lower():
        return "summary"
    return "other"


class _Tracker:
    def __init__(self):
        self.last = None

    def add_usage(self, counts):
        self.last = counts


@dataclass
class RagSettings:
    working_dir: Path
    llm_model: str = "gemini-3.8-flash"
    embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768
    chunk_size: int = 1200
    max_gleaning: int = 1
    max_async: int = 8
    max_parallel_insert: int = 4
    embedding_batch_num: int = 32
    embedding_max_async: int = 8
    force_summary_on_merge: int = 8
    types_guidance: str = DEFAULT_TYPES_GUIDANCE


def configure_env(s: RagSettings) -> None:
    """Env read by LightRAG's parser and chunker config (same names as the server)."""
    os.environ["LIGHTRAG_PARSER"] = "md:native-P"
    os.environ["CHUNK_P_SIZE"] = str(s.chunk_size)
    os.environ["NATIVE_MD_IMAGE_DOWNLOAD_ENABLED"] = "false"
    os.environ["INPUT_DIR"] = str(s.working_dir / "inputs")
    os.environ.setdefault("LOG_DIR", str(s.working_dir))


ACTIVE: dict = {"counters": None}


def use_counters(c: Counters) -> None:
    """Route subsequent model-call counts to `c` (one Counters object per stage)."""
    ACTIVE["counters"] = c


async def build_rag(s: RagSettings, counters: Counters):
    configure_env(s)
    use_counters(counters)
    from lightrag import LightRAG
    from lightrag.llm.gemini import gemini_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    key = os.environ.get("GEMINI_API_KEY") or os.environ["AI_STUDIO_KEY"]

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        kind = classify_prompt(prompt, system_prompt)
        kwargs.pop("hashing_kv", None)
        kwargs.pop("model_name", None)
        tracker = _Tracker()
        out = await gemini_complete_if_cache(s.llm_model, prompt, system_prompt=system_prompt,
                                             history_messages=history_messages, api_key=key,
                                             token_tracker=tracker, **kwargs)
        ACTIVE["counters"].llm(kind, tracker.last)
        return out

    async def embed_func(texts, **kwargs):
        c = ACTIVE["counters"]
        c.embed_requests += 1
        c.embed_texts += len(texts)
        c.embed_chars += sum(len(t) for t in texts)
        return await batch_embed(texts, s.embed_model, s.embed_dim, key)

    s.working_dir.mkdir(parents=True, exist_ok=True)
    (s.working_dir / "inputs").mkdir(exist_ok=True)
    rag = LightRAG(
        working_dir=str(s.working_dir),
        llm_model_func=llm_func,
        llm_model_name=s.llm_model,
        llm_model_max_async=s.max_async,
        embedding_func=EmbeddingFunc(embedding_dim=s.embed_dim, func=embed_func,
                                     max_token_size=2048, model_name=s.embed_model),
        embedding_batch_num=s.embedding_batch_num,
        embedding_func_max_async=s.embedding_max_async,
        entity_extract_max_gleaning=s.max_gleaning,
        force_llm_summary_on_merge=s.force_summary_on_merge,
        max_parallel_insert=s.max_parallel_insert,
        addon_params={"language": "English", "entity_types_guidance": s.types_guidance},
    )
    await rag.initialize_storages()
    return rag


async def ingest_files(rag, files: list[Path], input_dir: Path, track_id: str) -> None:
    """Enqueue cleaned Markdown files the way /documents/upload does, then process."""
    from lightrag.parser.routing import resolve_chunk_options

    names = []
    for f in files:
        shutil.copy(f, input_dir / f.name)
        names.append(f.name)
    await rag.apipeline_enqueue_documents(
        [""] * len(names), file_paths=names, track_id=track_id, docs_format="pending_parse",
        parse_engine="native", process_options="P",
        chunk_options=resolve_chunk_options(rag.addon_params, process_options="P"))
    await rag.apipeline_process_enqueue_documents()


async def delete_docs(rag, doc_ids: list[str]) -> None:
    for d in doc_ids:
        await rag.adelete_by_doc_id(d)


def doc_id_for(filename: str) -> str:
    from lightrag.utils import compute_mdhash_id
    return compute_mdhash_id(filename, prefix="doc-")

