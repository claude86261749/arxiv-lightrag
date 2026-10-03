"""After-run repairs and per-batch parameters for a multi-batch run.

    python -m kgpipe.repair <run_dir> <workspace> <batch_size>

1. Alias rows lost when a run is interrupted before saving its alias table: an extracted
   name (from full_entities) that is no longer a node and has no alias row is mapped to the
   surviving node with the same normalised key (code merges use that key). Others are listed.
2. Re-exports, so the paper-to-concept table uses the repaired aliases.
3. Per-batch parameters (c, e, r, u, gleaning) from storage, with batches = consecutive
   groups of `batch_size` papers in arXiv-ID order (how intake chose them).
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from .export import export
from .intake import Manifest
from .quality import extraction_rows, gleaning_effect
from .rag import Counters, RagSettings, build_rag
from .resolve import AliasTable, load_entities, norm_key
from .store import full_entity_names, processed_docs


async def main(run: Path, workspace: str, batch_size: int) -> dict:
    rag = await build_rag(RagSettings(working_dir=run / "rag", storage="postgres", workspace=workspace), Counters())
    out: dict = {}
    try:
        ents = await load_entities(rag)
        alias = AliasTable(run / "alias_table.csv")
        canon = alias.canonical_of()
        docs = await processed_docs(rag)
        names_by_doc = await full_entity_names(rag, list(docs))
        by_key = defaultdict(list)
        for n in ents:
            by_key[norm_key(n)].append(n)
        recovered, unresolved = 0, []
        for n in sorted({x for v in names_by_doc.values() for x in v}):
            if n in ents or n in canon:
                continue
            cands = by_key.get(norm_key(n), [])
            if cands:
                target = max(cands, key=lambda c: (ents[c].degree, c))
                alias.add(alias=n, canonical=target, verdict="same", cluster_id="recovered",
                          method="code-recovered", reason="identical normalised key (alias row lost on interrupt)")
                recovered += 1
            else:
                unresolved.append(n)
        alias.save()
        out["alias_rows_recovered"] = recovered
        out["names_unresolved"] = len(unresolved)
        out["names_unresolved_sample"] = unresolved[:40]

        manifest = Manifest(run / "manifest.json")
        out["export"] = await export(rag, run, manifest, alias)

        # Per-batch parameters.
        order = sorted(docs, key=lambda d: docs[d]["arxiv_id"])
        seen_names: set[str] = set()
        batches = []
        for bi in range(0, len(order), batch_size):
            ids = order[bi:bi + batch_size]
            chunk_ids = [c for d in ids for c in docs[d]["chunks_list"]]
            rows = await extraction_rows(rag, chunk_ids)
            ch = rows["chunks"]
            ent_rows = sum(len(r["extract_ents"]) + len(r["glean_ents"]) for r in ch.values())
            rel_rows = sum(r["extract_rels"] + r["glean_rels"] for r in ch.values())
            names = set().union(*(set(names_by_doc.get(d, [])) for d in ids))
            new = names - seen_names
            seen_names |= names
            n_chunks = sum(docs[d]["chunks_count"] for d in ids)
            batches.append({
                "papers": len(ids), "chunks": n_chunks,
                "c": round(n_chunks / len(ids), 2),
                "e": round(ent_rows / n_chunks, 2), "r": round(rel_rows / n_chunks, 2),
                "u_new_names_per_entity_row": round(len(new) / ent_rows, 3),
                "names_already_seen": round(1 - len(new) / len(names), 3),
                "gleaning": gleaning_effect(rows),
            })
        out["batches"] = batches
        out["corpus"] = {"papers": len(docs), "entities": len(ents),
                         "edges": len(await rag.chunk_entity_relation_graph.get_all_edges())
                         if hasattr(rag.chunk_entity_relation_graph, "get_all_edges") else None}
    finally:
        await rag.finalize_storages()
    (run / "phase_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = asyncio.run(main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])))
    print(json.dumps({k: v for k, v in r.items() if k != "names_unresolved_sample"}, indent=1))
