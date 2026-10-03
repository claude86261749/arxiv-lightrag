"""Stage 9 exports: files that analysis can use without the running services."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .resolve import AliasTable, load_entities
from .store import chunk_records, full_entity_names, processed_docs, vectors


async def export(rag, run_dir: Path, manifest, alias: AliasTable) -> dict:
    from lightrag.utils import compute_mdhash_id

    wd = Path(rag.working_dir)
    out = run_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    graphml = next(wd.rglob("graph_chunk_entity_relation.graphml"))
    shutil.copy(graphml, out / "graph.graphml")
    ents = await load_entities(rag)
    docs = await processed_docs(rag)

    # Entity vectors.
    names = sorted(ents)
    ids = [compute_mdhash_id(n, prefix="ent-") for n in names]
    got = await vectors(rag.entities_vdb, ids)
    keep = [(n, i) for n, i in zip(names, ids) if i in got]
    np.savez_compressed(out / "vectors_entities.npz",
                        vectors=np.array([got[i] for _, i in keep], dtype=np.float32),
                        meta=np.array([json.dumps({"__id__": i, "entity_name": n,
                                                   "file_path": "<SEP>".join(ents[n].file_paths)}) for n, i in keep]))
    # Chunk vectors and texts.
    chunk_ids = [c for d in docs.values() for c in d["chunks_list"]]
    cv = await vectors(rag.chunks_vdb, chunk_ids)
    recs = await chunk_records(rag, chunk_ids)
    ck = [c for c in chunk_ids if c in cv]
    np.savez_compressed(out / "vectors_chunks.npz", vectors=np.array([cv[c] for c in ck], dtype=np.float32),
                        meta=np.array([json.dumps({"__id__": c, "full_doc_id": recs.get(c, {}).get("full_doc_id"),
                                                   "file_path": recs.get(c, {}).get("file_path")}) for c in ck]))
    with (out / "chunks.jsonl").open("w") as f:
        for c in chunk_ids:
            r = recs.get(c, {})
            f.write(json.dumps({"id": c, "arxiv_id": Path(r.get("file_path") or "").stem,
                                "order": r.get("chunk_order_index"), "tokens": r.get("tokens"),
                                "heading": r.get("heading"), "content": r.get("content")}, ensure_ascii=False) + "\n")
    shutil.copy(alias.path, out / "alias_table.csv")

    # Paper-to-concept table from full_entities (per document), mapped through aliases.
    canon = alias.canonical_of()

    def resolve_name(n: str) -> str:
        seen = set()
        while n in canon and n not in seen:
            seen.add(n)
            n = canon[n]
        return n

    rows, unmapped, stale = [], 0, 0
    for doc_id, names_ in (await full_entity_names(rag, list(docs))).items():
        arxiv_id = docs[doc_id]["arxiv_id"]
        for n in sorted(set(names_)):
            if n in canon:
                stale += 1
            c = resolve_name(n)
            if c not in ents:
                unmapped += 1
                continue
            rows.append((arxiv_id, c, ents[c].type, n))
    rows = sorted(set(rows))
    with (out / "paper_concepts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "concept", "type", "extracted_name"])
        w.writerows(rows)
    shutil.copy(manifest.path, out / "manifest.json")
    if (run_dir / "references").exists():
        shutil.copytree(run_dir / "references", out / "references", dirs_exist_ok=True)
    return {"paper_concept_rows": len(rows), "names_dropped_not_in_graph": unmapped,
            "full_entities_rows_holding_alias_names": stale, "entity_vectors": len(keep), "chunk_vectors": len(ck)}
