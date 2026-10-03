"""Stage 8: measurements, review sheets and gates for one batch."""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .resolve import AliasTable, entity_vectors, load_entities, norm_key

ENT_ROW = re.compile(r"^\s*\(?\s*entity<\|#\|>([^<\n]+)<\|#\|>", re.M)
REL_ROW = re.compile(r"^\s*\(?\s*relation<\|#\|>", re.M)


def extraction_rows(working_dir: Path) -> dict:
    """Per chunk: entity names / relation rows from the extract call and the gleaning call."""
    cache = json.loads((working_dir / "kv_store_llm_response_cache.json").read_text())
    per_chunk: dict[str, dict] = defaultdict(lambda: {"extract_ents": [], "glean_ents": [],
                                                      "extract_rels": 0, "glean_rels": 0})
    summaries = 0
    for v in cache.values():
        ct = v.get("cache_type")
        if ct == "summary":
            summaries += 1
        if ct != "extract":
            continue
        prompt = v.get("original_prompt", "")
        kind = "glean" if "Based on the last extraction task" in prompt[:400] else "extract"
        ret = v.get("return", "") or ""
        rec = per_chunk[v.get("chunk_id", "?")]
        rec[f"{kind}_ents"] += [m.strip() for m in ENT_ROW.findall(ret)]
        rec[f"{kind}_rels"] += len(REL_ROW.findall(ret))
    return {"chunks": dict(per_chunk), "summary_cache_entries": summaries}


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
    ents = await load_entities(rag)
    names = sorted(ents)
    if len(names) < 2:
        return 0
    v = await entity_vectors(rag, names)
    sims = v @ v.T
    pairs = []
    for i in range(len(names)):
        sims[i, i] = -1
        for j in sims[i].argsort()[-5:]:
            a, b = sorted((names[i], names[int(j)]))
            pairs.append((float(sims[i, j]), a, b))
    pairs = sorted(set(pairs), reverse=True)[:n]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["similarity", "name_a", "type_a", "name_b", "type_b", "desc_a", "desc_b", "same? (y/n)"])
        for s, a, b in pairs:
            w.writerow([f"{s:.3f}", a, ents[a].type, b, ents[b].type,
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
                  graph_report: dict) -> dict:
    wd = Path(rag.working_dir)
    rows = extraction_rows(wd)
    status = json.loads((wd / "kv_store_doc_status.json").read_text())
    docs = {k: v for k, v in status.items() if str(v.get("status", "")).lower().endswith("processed")}
    n_papers = len(docs)
    n_chunks = sum(v.get("chunks_count", 0) for v in docs.values())
    chunks = rows["chunks"]
    ent_rows = sum(len(r["extract_ents"]) + len(r["glean_ents"]) for r in chunks.values())
    rel_rows = sum(r["extract_rels"] + r["glean_rels"] for r in chunks.values())
    distinct_names = resolve_stats.get("entities_before") or 0
    llm_calls = Counter()
    for st in counters_by_stage.values():
        for k, v in st.get("llm_calls", {}).items():
            llm_calls[k] += v
    ents = await load_entities(rag)
    types = Counter(e.type for e in ents.values())
    merged_names = resolve_stats.get("code_merged_names", 0) + resolve_stats.get("llm_merged_names", 0) \
        + resolve_stats.get("alias_merges", 0)
    clusters = resolve_stats.get("clusters", 0)
    c = n_chunks / n_papers if n_papers else 0
    params = {
        "c_chunks_per_paper": round(c, 2),
        "e_entity_rows_per_chunk": round(ent_rows / n_chunks, 2) if n_chunks else None,
        "r_relation_rows_per_chunk": round(rel_rows / n_chunks, 2) if n_chunks else None,
        "u_new_names_per_entity_row": round(distinct_names / ent_rows, 3) if ent_rows else None,
        "p_new_names_reaching_llm": round(resolve_stats.get("names_in_clusters", 0) / distinct_names, 3) if distinct_names else None,
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
        "papers": n_papers, "chunks": n_chunks,
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
