"""Build the data for the pilot quality explorer page.

    python -m kgpipe.viz runs/pilot2 out.json

Entity map (t-SNE of entity vectors), nearest neighbours, graph edges, merged
aliases, and search probes: fixed questions embedded with the same model and
matched against entity and chunk vectors.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from .rag import batch_embed

PROBES = [
    "turning items into discrete token codes for generative recommendation",
    "metrics for judging ranking quality",
    "compressing retrieved evidence before it goes to the language model",
    "language models preferring text written by language models",
    "finding the right skills or tools for a coding agent",
    "matching a text query to the right video",
    "deciding at serving time which users get a different recommender",
    "reducing wasted accelerator time in large training jobs",
    "comparing news crawls of the web",
    "letting an LLM search a small shop's product catalog",
    "recommendation baseline built on self-attention over the item sequence",
    "do multilingual encoders give the same semantic IDs across languages",
    "ad retrieval in a short-video app",
    "parsing patent claims into structured graphs",
]


def build(run: Path, dim: int = 768, model: str = "gemini-embedding-2") -> dict:
    exp = run / "export"
    ev = np.load(exp / "vectors_entities.npz")
    vecs = ev["vectors"].astype(np.float32)
    meta = [json.loads(m) for m in ev["meta"]]
    names = [m["entity_name"] for m in meta]
    g = nx.read_graphml(exp / "graph.graphml")
    manifest = json.loads((run / "manifest.json").read_text())
    if not (exp / "chunks.jsonl").exists():
        raise SystemExit("export/chunks.jsonl missing: re-run stage 9 (export) for this run")
    titles = {k: v.get("title") for k, v in manifest.items()}

    idx = {n: i for i, n in enumerate(names)}
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    nn = np.argsort(-sims, axis=1)[:, :8]

    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca", random_state=0).fit_transform(vecs)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))

    aliases = defaultdict(list)
    for r in csv.DictReader((run / "alias_table.csv").open()):
        if r["verdict"] == "same":
            aliases[r["canonical"]].append(r["alias"])
    # resolve chains so aliases land on the final canonical name
    final = {}
    for c, al in aliases.items():
        t = c
        seen = set()
        while t not in idx and t not in seen:
            seen.add(t)
            t = next((k for k, v in aliases.items() if t in v), t)
        final.setdefault(t, []).extend(al)

    ents = []
    for i, n in enumerate(names):
        d = g.nodes[n] if n in g else {}
        papers = sorted({Path(p).stem for p in d.get("file_path", "").split("<SEP>") if p})
        desc = d.get("description", "").split("<SEP>")[0].strip()
        ents.append({
            "n": n, "t": (d.get("entity_type") or "unknown").lower(), "d": desc[:260],
            "p": papers, "deg": g.degree(n) if n in g else 0,
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "nn": [[int(j), round(float(sims[i, j]), 3)] for j in nn[i]],
            "e": sorted(idx[m] for m in g.neighbors(n) if m in idx) if n in g else [],
            "a": sorted(set(final.get(n, []))),
        })

    # Neighbour agreement: how often top-5 neighbours share a type / a paper.
    same_type = np.mean([np.mean([ents[j]["t"] == e["t"] for j, _ in e["nn"][:5]]) for e in ents])
    same_paper = np.mean([np.mean([bool(set(ents[j]["p"]) & set(e["p"])) for j, _ in e["nn"][:5]]) for e in ents])

    # Search probes against entities and chunks.
    cv = np.load(exp / "vectors_chunks.npz")
    cvecs = cv["vectors"].astype(np.float32)
    cvecs /= np.linalg.norm(cvecs, axis=1, keepdims=True)
    cmeta = [json.loads(m) for m in cv["meta"]]
    chunks = {}
    for line in (exp / "chunks.jsonl").open():
        r = json.loads(line)
        chunks[r["id"]] = r
    key = os.environ.get("GEMINI_API_KEY") or os.environ["AI_STUDIO_KEY"]
    qv = asyncio.run(batch_embed(PROBES, model, dim, key))
    probes = []
    for q, v in zip(PROBES, qv):
        es = vecs @ v
        cs = cvecs @ v
        top_c = []
        for j in np.argsort(-cs)[:3]:
            cid = cmeta[j]["__id__"]
            ch = chunks.get(cid, {})
            head = ch.get("heading") or {}
            pid = Path(cmeta[j].get("file_path", "")).stem
            text = " ".join(ch.get("content", "").split())
            top_c.append({"paper": pid, "title": titles.get(pid), "sim": round(float(cs[j]), 3),
                          "section": " → ".join([*head.get("parent_headings", []), head.get("heading", "")][1:]).strip(" →"),
                          "text": text[:420]})
        probes.append({"q": q, "ents": [[int(j), round(float(es[j]), 3)] for j in np.argsort(-es)[:8]],
                       "chunks": top_c})

    papers = sorted({p for e in ents for p in e["p"]})
    return {"entities": ents, "papers": {p: titles.get(p) for p in papers}, "probes": probes,
            "stats": {"entities": len(ents), "edges": g.number_of_edges(), "papers": len(papers),
                      "nn_same_type": round(float(same_type), 3), "nn_shared_paper": round(float(same_paper), 3),
                      "model": model, "dim": dim}}


if __name__ == "__main__":
    data = build(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(json.dumps(data["stats"]))
    for p in data["probes"]:
        print("\n" + p["q"])
        print("   ents:", [(data["entities"][j]["n"], s) for j, s in p["ents"][:5]])
        print("   chunks:", [(c["paper"], c["sim"], (c["title"] or "")[:40]) for c in p["chunks"]])
