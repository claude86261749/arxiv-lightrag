"""Pilot analysis helpers: threshold calibration and appendix comparison.

    python -m kgpipe.analyze threshold runs/pilot
    python -m kgpipe.analyze appendix runs/pilot runs/pilot-appx
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def threshold(run: Path) -> dict:
    """LLM verdicts by vector-similarity band, for candidate pairs that reached the LLM."""
    pairs = json.loads((run / "candidate_pairs.json").read_text())
    verdict = {}
    for r in csv.DictReader((run / "alias_table.csv").open()):
        verdict.setdefault(frozenset((r["alias"], r["canonical"])), r["verdict"])
    for m in json.loads((run / "merges.json").read_text()):
        if m.get("method") == "llm":
            names = [m["canonical"], *m["aliases"]]
            for a in names:
                for b in names:
                    if a != b:
                        verdict[frozenset((a, b))] = "same"
    bands = defaultdict(Counter)
    by_reason = defaultdict(Counter)
    for p in pairs:
        key = frozenset(p["pair"])
        v = verdict.get(key, "kept separate, no explicit verdict")
        sims = [float(x.split(":")[1]) for x in p["reasons"] if x.startswith("vector:")]
        kinds = sorted({x.split(":")[0] for x in p["reasons"]})
        by_reason["+".join(kinds)][v] += 1
        if sims:
            s = max(sims)
            band = f"{int(s * 100) // 2 * 2 / 100:.2f}"
            bands[band][v] += 1
    return {"by_similarity_band": {b: dict(c) for b, c in sorted(bands.items())},
            "by_reason": {k: dict(v) for k, v in by_reason.items()}}


def appendix(base: Path, appx: Path) -> dict:
    """Entity names per paper with appendices cut (base) vs kept (appx)."""
    def names(run):
        st = json.loads((run / "rag" / "kv_store_doc_status.json").read_text())
        full = json.loads((run / "rag" / "kv_store_full_entities.json").read_text())
        out = {}
        for doc, row in full.items():
            pid = Path(st.get(doc, {}).get("file_path", doc)).stem
            out[pid] = {n.lower() for n in row.get("entity_names", [])}
        chunks = {Path(v.get("file_path", k)).stem: v.get("chunks_count") for k, v in st.items()}
        return out, chunks
    b, bc = names(base)
    a, ac = names(appx)
    rows = []
    for pid in sorted(a):
        if pid not in b:
            continue
        only_appx = a[pid] - b[pid]
        rows.append({"paper": pid, "chunks_cut": bc.get(pid), "chunks_kept": ac.get(pid),
                     "names_cut": len(b[pid]), "names_kept": len(a[pid]),
                     "names_only_with_appendix": len(only_appx),
                     "sample_only_with_appendix": sorted(only_appx)[:15]})
    return {"papers": rows,
            "total_extra_chunks": sum((r["chunks_kept"] or 0) - (r["chunks_cut"] or 0) for r in rows),
            "total_names_cut": sum(r["names_cut"] for r in rows),
            "total_names_only_with_appendix": sum(r["names_only_with_appendix"] for r in rows)}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "threshold":
        print(json.dumps(threshold(Path(sys.argv[2])), indent=1))
    elif cmd == "appendix":
        print(json.dumps(appendix(Path(sys.argv[2]), Path(sys.argv[3])), indent=1))
