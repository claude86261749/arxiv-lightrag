"""Stage 9 exports: files that analysis can use without the running services."""
from __future__ import annotations

import base64
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .resolve import AliasTable, load_entities


def _vdb_matrix(path: Path) -> tuple[list[dict], np.ndarray]:
    d = json.loads(path.read_text())
    data = d.get("data", [])
    dim = d.get("embedding_dim")
    mat = np.frombuffer(base64.b64decode(d["matrix"]), dtype=np.float32).reshape(len(data), dim) \
        if d.get("matrix") else np.zeros((0, dim or 0), np.float32)
    return data, mat


async def export(rag, run_dir: Path, manifest, alias: AliasTable) -> dict:
    wd = Path(rag.working_dir)
    out = run_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(wd / "graph_chunk_entity_relation.graphml", out / "graph.graphml")
    for ns in ("entities", "relationships", "chunks"):
        p = wd / f"vdb_{ns}.json"
        if p.exists():
            data, mat = _vdb_matrix(p)
            keep = ["__id__", "entity_name", "src_id", "tgt_id", "full_doc_id", "file_path"]
            np.savez_compressed(out / f"vectors_{ns}.npz", vectors=mat,
                                meta=np.array([json.dumps({k: r.get(k) for k in keep if k in r}) for r in data]))
    shutil.copy(alias.path, out / "alias_table.csv")
    # Paper-to-concept table from full_entities (per document), mapped through aliases.
    status = json.loads((wd / "kv_store_doc_status.json").read_text())
    full = json.loads((wd / "kv_store_full_entities.json").read_text())
    canon = alias.canonical_of()
    ents = await load_entities(rag)

    def resolve_name(n: str) -> str:
        seen = set()
        while n in canon and n not in seen:
            seen.add(n)
            n = canon[n]
        return n

    rows, unmapped, stale = [], 0, 0
    for doc_id, row in full.items():
        fp = status.get(doc_id, {}).get("file_path", "")
        arxiv_id = Path(fp).stem if fp else doc_id
        for n in sorted(set(row.get("entity_names", []))):
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
            "full_entities_rows_holding_alias_names": stale}
