#!/bin/bash
# Resume the 200-paper phase from the state frozen on 3 Oct 2026 (see RESUME.md).
# Batch A (100 papers) is complete. Batch B is re-run: intake retries every paper not
# marked indexed, deletes its partial LightRAG document and ingests it again; extraction
# results already in the LLM cache (Postgres) are reused, so finished calls are not paid again.
set -o pipefail
cd ~/arxiv-lightrag
source .env
step() { echo "=== $(date -u +%H:%M:%S) $*"; }
sudo systemctl start postgresql
until pg_isready -h localhost -q; do sleep 1; done
step "postgres up"

common="--corpus data/corpus/cs.IR --run runs/phase200 --storage postgres --workspace phase200 --max-async 24 --parallel-insert 6 --llm-workers 24"
step "batch B"
.venv/bin/python -m kgpipe.pipeline $common --limit 200 > runs/phase200/batchB.out 2>&1 || { echo BATCH_B_FAILED; tail -30 runs/phase200/batchB.out; exit 1; }
grep -E '^\[' runs/phase200/batchB.out | cut -c1-300
mkdir -p runs/phase200/batchB && cp runs/phase200/{report.json,quality.json,merges.json,graph_cleanup.json,review_merges.csv,review_missed_duplicates.csv,candidate_pairs.json,clusters.json} runs/phase200/batchB/ 2>/dev/null

step "repair aliases, re-export, per-batch parameters"
.venv/bin/python -m kgpipe.repair runs/phase200 phase200 100 > runs/phase200/repair.out 2>&1 || tail -20 runs/phase200/repair.out

step "explorer data"
.venv/bin/python -m kgpipe.viz runs/phase200 runs/phase200/viz.json > runs/phase200/viz.out 2>&1 || tail -20 runs/phase200/viz.out
step DONE
