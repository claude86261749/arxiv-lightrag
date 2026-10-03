"""Stage 7: graph cleanup after resolution."""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .resolve import SEP, load_entities, norm_key

HUB_SYSTEM = """You check nodes of a knowledge graph built from research papers in one arXiv category.
A node was formed by merging description fragments written while reading different papers.
Decide whether the fragments describe ONE concept or SEVERAL distinct concepts that share a name.
Different aspects, uses or results of one concept still count as one concept. Answer "several" only if
the fragments clearly refer to different things (e.g. two different methods that happen to share an acronym)."""

HUB_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["one", "several"]},
        "senses": {"type": "ARRAY", "items": {"type": "STRING"}},
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "senses", "reason"],
}


def load_stoplist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {norm_key(l.strip()) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")}


async def clean_graph(rag, llm, stoplist_path: Path, hubs: int = 50, summary_min_fragments: int = 8,
                      seed: int = 0, workers: int = 8) -> dict:
    report: dict = {}
    g = rag.chunk_entity_relation_graph
    stop = load_stoplist(stoplist_path)

    # Generic names.
    ents = await load_entities(rag)
    removed = []
    for n in list(ents):
        if norm_key(n) in stop:
            await rag.adelete_by_entity(n)
            removed.append({"name": n, "degree": ents[n].degree})
            ents.pop(n)
    report["stoplist_removed"] = removed

    # Self-loops left by merging.
    loops = []
    for n in list(ents):
        if await g.has_edge(n, n):
            await rag.adelete_by_relation(n, n)
            loops.append(n)
    report["self_loops_removed"] = loops

    # Paper-local labels ("Model A", "Mixture B") collide by exact name across papers.
    # Rename those found in exactly one paper to carry the arXiv ID.
    local = re.compile(r"^(model|mixture|setting|variant|config(uration)?|system|method|baseline|"
                       r"dataset|split|run|group|case|condition|arm)\s+([a-z]|\d{1,2}|[ivx]{1,4})$", re.I)
    renamed = []
    for n, e in list(ents.items()):
        papers = {Path(f).stem for f in e.file_paths}
        if local.match(n) and len(papers) == 1:
            new = f"{n} ({papers.pop()})"
            if not await g.has_node(new):
                await rag.aedit_entity(n, {"entity_name": new}, allow_rename=True)
                renamed.append({"from": n, "to": new})
                ents[new] = ents.pop(n)
    report["paper_local_renamed"] = renamed

    # Condense merged nodes with many description fragments.
    from lightrag.operate import _handle_entity_relation_summary
    cfg = rag._build_global_config()
    # Summaries run concurrently (bounded by LightRAG's own LLM limiter); edits are applied in order.
    import asyncio
    todo = [(n, e) for n, e in ents.items() if len(e.fragments) >= summary_min_fragments]
    sem = asyncio.Semaphore(workers)

    async def summarise(n, e):
        async with sem:
            out, _ = await _handle_entity_relation_summary("Entity", n, e.fragments, SEP, cfg, rag.llm_response_cache)
            return out
    summaries = await asyncio.gather(*(summarise(n, e) for n, e in todo))
    condensed = []
    for (n, e), summary in zip(todo, summaries):
        await rag.aedit_entity(n, {"description": summary}, allow_rename=False)
        condensed.append({"name": n, "fragments": len(e.fragments)})
        e.description = summary
    report["condensed"] = condensed

    # Low evidence: one chunk, no relations. Flagged, kept.
    ents = await load_entities(rag)
    report["low_evidence"] = sorted(n for n, e in ents.items() if len(set(e.source_ids)) <= 1 and e.degree == 0)

    # Hub review: LLM reads sampled fragments of the highest-degree nodes and only reports.
    rng = random.Random(seed)
    top = sorted(ents.values(), key=lambda e: -e.degree)[:hubs]
    # Fragments are gone after condensing, so sample from relation descriptions + node description.
    async def fragments_for(e):
        frags = list(e.fragments)
        edges = await g.get_node_edges(e.name) or []
        for a, b in edges[:200]:
            ed = await g.get_edge(a, b) or {}
            frags += [d for d in ed.get("description", "").split(SEP) if d.strip()]
        return frags

    items = []
    for e in top:
        frags = await fragments_for(e)
        rng.shuffle(frags)
        items.append((e, frags[:12]))

    def check(item):
        e, frags = item
        prompt = (f"Node: {e.name} (type {e.type}, degree {e.degree}, papers {e.papers})\n\nFragments:\n"
                  + "\n".join(f"- {f[:500]}" for f in frags))
        out = llm.json_call("hub_check", prompt, HUB_SCHEMA, HUB_SYSTEM)
        return {"name": e.name, "type": e.type, "degree": e.degree, **out}

    with ThreadPoolExecutor(workers) as ex:
        report["hubs"] = list(ex.map(check, items))
    return report
