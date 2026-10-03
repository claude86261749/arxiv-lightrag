# Frozen: 200-paper phase (3 Oct 2026, 16:54 UTC)

Code: commit 682477df724b56c07bea3f6ebe365c59f8582355 of claude86261749/arxiv-lightrag, branch claude/epic-goodall-dglc5e
(unpacked in ~/arxiv-lightrag; VERSION holds the hash).

## State
- Corpus: data/corpus/cs.IR, 200 papers (list: config/corpora/cs.IR-200.json).
- Postgres 16 + pgvector 0.8.0, database `lightrag`, workspace `phase200`
  (KV, vectors, doc status, LLM cache). Graph: runs/phase200/rag/phase200/*.graphml.
- Batch A (first 100 papers by arXiv ID): indexed, resolved, cleaned, exported.
  Outputs in runs/phase200/batchA/ and runs/phase200/export/.
- Batch B (next 100): interrupted during ingestion. In LightRAG 23 processed,
  77 part-way; all 100 are 'cleaned' in the manifest, so they are retried.
- Known gap: alias rows for ~267 code merges from the first batch-A attempt were not
  saved; resume step 'repair' rebuilds them from normalised keys.
- API key: ~/arxiv-lightrag/.env (mode 600).

## Backups (backups/)
- lightrag-<time>.dump : pg_dump -Fc of database lightrag
- phase200-<time>.tar.gz : runs/phase200 and data/corpus
- SHA256SUMS

## Resume
    bash ~/arxiv-lightrag/resume_phase200.sh > ~/arxiv-lightrag/run_phase200.resume.log 2>&1

Starts Postgres, runs batch B, repairs aliases and re-exports, rebuilds explorer data.

## Restore on another machine
    createdb lightrag && psql -d lightrag -c 'CREATE EXTENSION vector'
    pg_restore -d lightrag backups/lightrag-<time>.dump
    tar xzf backups/phase200-<time>.tar.gz
