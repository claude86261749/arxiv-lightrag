"""Stage 8: measurements, review sheets and gates for one batch."""
from __future__ import annotations

import csv
import json
import random
import re

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from .resolve import AliasTable, entity_vectors, load_entities, norm_key

ENT_ROW = re.compile(r"^\s*\(?\s*entity<\|#\|>([^<\n]+)<\|#\|>", re.M)
REL_ROW = re.compile(r"^\s*\(?\s*relation<\|#\|>", re.M)


async def extraction_rows(rag, chunk_ids: list[str]) -> dict:
    """Per chunk: entity names / relation rows from the extract call and the gleaning call."""
    from .store import llm_cache_entries
    per_chunk: dict[str, dict] = {c: {"extract_ents": [], "glean_ents": [], "extract_rels": 0, "glean_rels": 0}
                                  for c in chunk_ids}
    for v in await llm_cache_entries(rag, chunk_ids):
        if v.get("cache_type") != "extract":
            continue
        prompt = v.get("original_prompt", "") or ""
        kind = "glean" if "Based on the last extraction task" in prompt[:400] else "extract"
        ret = v.get("return", "") or ""
        rec = per_chunk.setdefault(v.get("chunk_id", "?"), {"extract_ents": [], "glean_ents": [],
                                                             "extract_rels": 0, "glean_rels": 0})
        rec[f"{kind}_ents"] += [m.strip() for m in ENT_ROW.findall(ret)]
        rec[f"{kind}_rels"] += len(REL_ROW.findall(ret))
    return {"chunks": per_chunk}


def gleaning_effect(rows: dict) -> dict:
    ext = sum(len(set(n.lower() for n in r["extract_ents"])) for r in rows["chunks"].values())
    new = sum(len(set(n.lower() for n in r["glean_ents"]) - set(n.lower() for n in r["extract_ents"]))
              for r in rows["chunks"].values())
    ext_rel = sum(r["extract_rels"] for r in rows["chunks"].values())
    gl_rel = sum(r["glean_rels"] for r in rows["chunks"].values())
    return {"extract_entity_names": ext, "glean_new_entity_names": new,
            "glean_entity_add_rate": round(new / ext, 3) if ext else None,
            "extract_relation_rows": ext_rel, "glean_relation_rows": gl_rel,
            "glean_relation_add_rate": round(gl_rel / ext_rel, 3) if ext_rel else None}


async def missed_duplicate_sheet(rag, path: Path, n: int = 100) -> int:
    """The n most similar entity pairs that are still separate (top-5 neighbours, computed in blocks)."""
    ents = await load_entities(rag)
    names = sorted(ents)
    if len(names) < 2:
        return 0
    v = await entity_vectors(rag, names)
    pairs = set()
    for start in range(0, len(names), 1024):
        sims = v[start:start + 1024] @ v.T
        for r in range(sims.shape[0]):
            i = start + r
            sims[r, i] = -1
            for j in np.argpartition(-sims[r], 5)[:5]:
                a, b = sorted((names[i], names[int(j)]))
                pairs.add((round(float(sims[r, j]), 4), a, b))
    pairs = sorted(pairs, reverse=True)[:n]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["similarity", "name_a", "type_a", "name_b", "type_b", "desc_a", "desc_b", "same? (y/n)"])
        for s_, a, b in pairs:
            w.writerow([f"{s_:.3f}", a, ents[a].type, b, ents[b].type,
                        ents[a].description[:200], ents[b].description[:200], ""])
    return len(pairs)


def merge_review_sheet(merges: list[dict], path: Path, n: int = 100, seed: int = 0) -> int:
    rows = [m for m in merges if "canonical" in m]
    sample = rows if len(rows) <= n else random.Random(seed).sample(rows, n)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["canonical", "aliases", "method", "cluster_id", "reason", "correct? (y/n)"])
        for m in sample:
            w.writerow([m["canonical"], " | ".join(m["aliases"]), m["method"], m["cluster_id"], m["reason"], ""])
    return len(sample)


async def measure(rag, run_dir: Path, manifest, counters_by_stage: dict, resolve_stats: dict,
                  graph_report: dict, batch_doc_ids: list[str] | None = None,
                  names_before: set[str] | None = None) -> dict:
    """Parameters for this batch (the documents ingested in this invocation), plus corpus totals."""
    from .store import full_entity_names, processed_docs
    all_docs = await processed_docs(rag)
    batch = [d for d in (batch_doc_ids or list(all_docs)) if d in all_docs]
    chunk_ids = [c for d in batch for c in all_docs[d]["chunks_list"]]
    rows = await extraction_rows(rag, chunk_ids)
    n_papers = len(batch)
    n_chunks = sum(all_docs[d]["chunks_count"] for d in batch)
    chunks = rows["chunks"]
    ent_rows = sum(len(r["extract_ents"]) + len(r["glean_ents"]) for r in chunks.values())
    rel_rows = sum(r["extract_rels"] + r["glean_rels"] for r in chunks.values())
    batch_names = set()
    for names in (await full_entity_names(rag, batch)).values():
        batch_names |= set(names)
    new_names = batch_names - (names_before or set())
    llm_calls = Counter()
    for st in counters_by_stage.values():
        for k, v in st.get("llm_calls", {}).items():
            llm_calls[k] += v
    ents = await load_entities(rag)
    types = Counter(e.type for e in ents.values())
    merged_names = resolve_stats.get("code_merged_names", 0) + resolve_stats.get("llm_merged_names", 0) \
        + resolve_stats.get("alias_merges", 0)
    clusters = resolve_stats.get("clusters", 0)
    focus = resolve_stats.get("focus_names") or len(new_names)
    params = {
        "c_chunks_per_paper": round(n_chunks / n_papers, 2) if n_papers else None,
        "e_entity_rows_per_chunk": round(ent_rows / n_chunks, 2) if n_chunks else None,
        "r_relation_rows_per_chunk": round(rel_rows / n_chunks, 2) if n_chunks else None,
        "u_new_names_per_entity_row": round(len(new_names) / ent_rows, 3) if ent_rows else None,
        "batch_names_already_in_graph": round(1 - len(new_names) / len(batch_names), 3) if batch_names else None,
        "p_new_names_reaching_llm": round(resolve_stats.get("names_in_clusters", 0) / focus, 3) if focus else None,
        "k_names_per_cluster": round(resolve_stats.get("names_in_clusters", 0) / clusters, 2) if clusters else None,
        "q_second_look_share": round(resolve_stats.get("second_looks", 0) / clusters, 3) if clusters else None,
        "m_texts_reembedded_per_merged_name": round(resolve_stats.get("embed_texts_during_merges", 0) / merged_names, 2) if merged_names else None,
        "summary_calls_per_paper": round(llm_calls.get("summary", 0) / n_papers, 2) if n_papers else None,
        "h_implied": round(7 * llm_calls.get("summary", 0) / (ent_rows + rel_rows), 3) if ent_rows + rel_rows else None,
    }
    other_share = (types.get("other", 0) + types.get("unknown", 0)) / max(1, len(ents))
    flags = {k: v["cleanup_flags"] for k, v in manifest.papers.items() if v.get("cleanup_flags")}
    hubs = graph_report.get("hubs", [])
    return {
        "papers": n_papers, "chunks": n_chunks, "corpus_papers": len(all_docs),
        "entities_final": len(ents), "type_distribution": dict(types.most_common()),
        "params": params,
        "gleaning": gleaning_effect(rows),
        "llm_calls_by_type": dict(llm_calls),
        "llm_calls_per_paper": {k: round(v / n_papers, 2) for k, v in llm_calls.items()} if n_papers else {},
        "gates": {
            "type_coverage_other_share": round(other_share, 3),
            "type_coverage_pass": other_share < 0.15,
            "cleanup_unreviewed_flags": flags,
            "hubs_flagged_several": [h["name"] for h in hubs if h.get("verdict") == "several"],
        },
    }
