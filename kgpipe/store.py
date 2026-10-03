"""Backend-neutral reads from LightRAG storage (file JSON or Postgres).

The pilot read LightRAG's JSON files directly; from the 200-paper phase the KV,
vector and doc-status stores live in Postgres, so everything goes through the
storage APIs instead.
"""
from __future__ import annotations

from pathlib import Path


def _batches(xs: list, n: int = 500):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


async def processed_docs(rag) -> dict[str, dict]:
    """doc_id -> {arxiv_id, file_path, chunks_count, chunks_list} for processed documents."""
    from lightrag.base import DocStatus
    docs = await rag.doc_status.get_docs_by_statuses([DocStatus.PROCESSED])
    out = {}
    for doc_id, st in docs.items():
        fp = st.file_path or ""
        out[doc_id] = {"arxiv_id": Path(fp).stem if fp else doc_id, "file_path": fp,
                       "chunks_count": st.chunks_count or 0, "chunks_list": list(st.chunks_list or [])}
    return out


async def kv_get(kv, ids: list[str]) -> dict[str, dict]:
    out = {}
    for b in _batches(list(ids)):
        for k, v in zip(b, await kv.get_by_ids(b)):
            if v:
                out[k] = v
    return out


async def chunk_records(rag, ids: list[str]) -> dict[str, dict]:
    return await kv_get(rag.text_chunks, ids)


async def full_entity_names(rag, doc_ids: list[str]) -> dict[str, list[str]]:
    rows = await kv_get(rag.full_entities, doc_ids)
    return {d: list(r.get("entity_names", [])) for d, r in rows.items()}


async def llm_cache_entries(rag, chunk_ids: list[str]) -> list[dict]:
    """Cached LLM calls made for these chunks (extraction and gleaning), via each chunk's llm_cache_list."""
    chunks = await chunk_records(rag, chunk_ids)
    keys, owner = [], {}
    for cid, ch in chunks.items():
        for k in ch.get("llm_cache_list") or []:
            keys.append(k)
            owner[k] = cid
    rows = await kv_get(rag.llm_response_cache, keys)
    out = []
    for k, v in rows.items():
        v = dict(v)
        v.setdefault("chunk_id", owner[k])
        out.append(v)
    return out


async def vectors(vdb, ids: list[str]) -> dict[str, list[float]]:
    out = {}
    for b in _batches(list(ids)):
        out.update(await vdb.get_vectors_by_ids(b))
    return out
