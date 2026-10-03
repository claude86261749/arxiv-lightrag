"""Performance helpers for stages 6-7.

LightRAG's merge/edit/delete calls end with the graph store's index_done_callback,
which for NetworkX rewrites the whole GraphML file. At 9,000 nodes that is ~1.5 s per
call, twice per merge, and it grows with the graph. The graph lives in memory and is
authoritative inside this process, so during resolution and cleanup we persist it
every `every` calls and once at the end instead.
"""
from __future__ import annotations

import time


class DeferredGraphWrites:
    def __init__(self, rag, every: int = 250):
        self.g = rag.chunk_entity_relation_graph
        self.every = every
        self.calls = 0
        self.writes = 0
        self.write_seconds = 0.0
        self._orig = None

    async def _write(self):
        t = time.time()
        await self._orig()
        self.writes += 1
        self.write_seconds += time.time() - t

    async def _deferred(self, *a, **k):
        self.calls += 1
        if self.calls % self.every == 0:
            await self._write()
        return True

    async def __aenter__(self):
        self._orig = self.g.index_done_callback
        self.g.index_done_callback = self._deferred  # instance attribute shadows the method
        return self

    async def __aexit__(self, *exc):
        del self.g.index_done_callback
        await self._write()
        return False

    def stats(self) -> dict:
        return {"graph_commit_calls": self.calls, "graph_writes": self.writes,
                "graph_write_seconds": round(self.write_seconds, 1)}
