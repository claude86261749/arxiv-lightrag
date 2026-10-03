# arxiv-lightrag

A concept graph, with embeddings, built from arXiv papers in one category.
LightRAG 1.5.7 does extraction; the code here adds text cleanup before it and
entity resolution, graph cleanup, quality checks and exports after it.
Models: `gemini-3.8-flash` for text generation, `gemini-embedding-2` (768 dims) for vectors.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...        # or AI_STUDIO_KEY
```

## Run

```
# Optional: build a corpus of N papers from arXiv's HTML versions
.venv/bin/python -c "from pathlib import Path; from kgpipe.fetch import fetch_corpus; fetch_corpus('cs.IR', 20, Path('data/corpus/cs.IR'))"

# All nine stages on a batch
.venv/bin/python -m kgpipe.pipeline --corpus data/corpus/cs.IR --run runs/pilot --limit 20
```

Options: `--gleaning 0|1`, `--keep-appendix`, `--embed-dim`, `--nn-threshold` (default 0.85),
`--hubs` (default 50), `--stoplist`, `--stop-after ingest`.
Rerunning on the same `--run` directory skips papers whose Markdown is unchanged and
replaces those that changed.

## Stages and modules

| # | Stage | Module |
|---|---|---|
| 1 | Intake: manifest keyed by arXiv ID + content hash, arXiv metadata | `kgpipe/intake.py` |
| 2 | Clean text: references and citing sentences to a side file, then cut references, acknowledgements, repeated lines, appendices | `kgpipe/clean.py` |
| 3–5 | Chunk (native Markdown parser + paragraph chunker, 1,200 tokens), extract, embed | `kgpipe/rag.py` |
| 6 | Resolve entities: normalised key, nearest neighbours, acronyms → code merges and LLM merge decisions → alias table | `kgpipe/resolve.py` |
| 7 | Clean graph: stoplist, self-loops, condense long descriptions, low-evidence flags, hub check | `kgpipe/graph_clean.py` |
| 8 | Quality: parameters, gleaning effect, review sheets, gates | `kgpipe/quality.py` |
| 9 | Exports: GraphML, vectors, alias table, paper-to-concept table, manifest, reference side files | `kgpipe/export.py` |

Run outputs (`runs/<name>/`): `report.json`, `quality.json`, `merges.json`, `alias_table.csv`,
`graph_cleanup.json`, `review_merges.csv`, `review_missed_duplicates.csv`, `export/`,
`references/`, `cleaned/`, `lightrag.log`, and LightRAG's storage under `rag/`.

## Differences from the design draft

- **LightRAG runs in-process, not as a server.** Ingestion calls the same enqueue path as
  `/documents/upload` (`pending_parse`, engine `native`, chunker `P`), so chunking is the
  same. In-process access is needed for stage 6 (entity vectors have no HTTP route) and
  stage 7 (merging concatenates descriptions; condensing them needs LightRAG's summary
  function). `lightrag-server` can serve the same working directory afterwards.
- **Embeddings bypass LightRAG's gemini binding.** With `gemini-embedding-2` the binding's
  batched call returns one vector per batch instead of one per text, which fails every
  vector-store write. `kgpipe/rag.py` calls `batchEmbedContents` directly.
- **Type guidance** is passed as `addon_params["entity_types_guidance"]` (the server only
  accepts it through a YAML prompt file).

## Current state

The 200-paper phase is paused on the remote machine. See
`reports/phase200-frozen/RESUME.md` for the state, the backups and the one command
that resumes it; batch A's outputs are in `reports/phase200-frozen/batchA/`.
