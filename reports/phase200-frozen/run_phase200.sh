#!/bin/bash
# 200-paper phase on the remote machine: env, corpus download, batch A (100), batch B (+100), explorer data.
set -o pipefail
cd ~/arxiv-lightrag
source .env
mkdir -p runs/phase200
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

if [ ! -x .venv/bin/python ]; then
  step venv
  python3.11 -m venv .venv && .venv/bin/pip install -q --upgrade pip && \
  .venv/bin/pip install -q -r requirements.txt asyncpg pgvector || { echo VENV_FAILED; exit 1; }
fi
step "python deps ok: $(.venv/bin/python -c 'import lightrag, asyncpg; print(lightrag.__version__)')"

step "fetch corpus by ID"
.venv/bin/python -m kgpipe.fetch ids config/corpora/cs.IR-200.json data/corpus/cs.IR > runs/phase200/fetch.out 2>&1
echo "fetched $(ls data/corpus/cs.IR/*.md | wc -l) papers; failures: $(grep -c '^failed' runs/phase200/fetch.out)"

common="--corpus data/corpus/cs.IR --run runs/phase200 --storage postgres --workspace phase200 --max-async 24 --parallel-insert 6 --llm-workers 24"
step "batch A"
.venv/bin/python -m kgpipe.pipeline $common --limit 100 > runs/phase200/batchA.out 2>&1 || { echo BATCH_A_FAILED; tail -30 runs/phase200/batchA.out; exit 1; }
grep -E '^\[' runs/phase200/batchA.out | cut -c1-300
mkdir -p runs/phase200/batchA && cp runs/phase200/{report.json,quality.json,merges.json,graph_cleanup.json,review_merges.csv,review_missed_duplicates.csv,candidate_pairs.json,clusters.json} runs/phase200/batchA/ 2>/dev/null

step "batch B"
.venv/bin/python -m kgpipe.pipeline $common --limit 200 > runs/phase200/batchB.out 2>&1 || { echo BATCH_B_FAILED; tail -30 runs/phase200/batchB.out; exit 1; }
grep -E '^\[' runs/phase200/batchB.out | cut -c1-300
mkdir -p runs/phase200/batchB && cp runs/phase200/{report.json,quality.json,merges.json,graph_cleanup.json,review_merges.csv,review_missed_duplicates.csv,candidate_pairs.json,clusters.json} runs/phase200/batchB/ 2>/dev/null

step "explorer data"
.venv/bin/python -m kgpipe.viz runs/phase200 runs/phase200/viz.json > runs/phase200/viz.out 2>&1 || tail -20 runs/phase200/viz.out
step DONE
