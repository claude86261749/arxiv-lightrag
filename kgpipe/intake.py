"""Stages 1-2: intake into a manifest, then cleanup to files on disk."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .clean import clean_markdown


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Manifest:
    """One record per paper, keyed by arXiv ID. Stored as JSON next to the run."""

    def __init__(self, path: Path):
        self.path = path
        self.papers: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.papers, indent=1, sort_keys=True))
        tmp.replace(self.path)


def intake(corpus_dir: Path, manifest: Manifest, limit: int | None = None) -> list[str]:
    """Record each Markdown file; return IDs that are new or whose content changed."""
    meta_path = corpus_dir / "_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    changed = []
    for p in sorted(corpus_dir.glob("*.md"))[:limit]:
        arxiv_id = p.stem
        text = p.read_text()
        h = sha256(text)
        rec = manifest.papers.get(arxiv_id)
        if rec and rec["content_hash"] == h and rec.get("status") != "failed":
            continue  # unchanged and indexed; failed papers are retried (replaced) on the next run
        m = meta.get(arxiv_id, {})
        manifest.papers[arxiv_id] = {
            **(rec or {}),
            "arxiv_id": arxiv_id,
            "source_path": str(p),
            "content_hash": h,
            "previous_hash": rec["content_hash"] if rec else None,
            "title": m.get("title"),
            "authors": m.get("authors"),
            "submitted": m.get("submitted"),
            "categories": m.get("categories"),
            "status": "new" if not rec else "changed",
        }
        changed.append(arxiv_id)
    manifest.save()
    return changed


def clean_papers(manifest: Manifest, ids: list[str], out_dir: Path, drop_appendix: bool) -> None:
    (out_dir / "cleaned").mkdir(parents=True, exist_ok=True)
    (out_dir / "references").mkdir(parents=True, exist_ok=True)
    for arxiv_id in ids:
        rec = manifest.papers[arxiv_id]
        r = clean_markdown(Path(rec["source_path"]).read_text(), drop_appendix=drop_appendix)
        cleaned = out_dir / "cleaned" / f"{arxiv_id}.md"
        cleaned.write_text(r.text)
        known = set(manifest.papers)
        for e in r.references:
            e["in_corpus"] = e["arxiv_id"] in known if e["arxiv_id"] else False
        refs = out_dir / "references" / f"{arxiv_id}.json"
        refs.write_text(json.dumps({"arxiv_id": arxiv_id, "entries": r.references}, indent=1))
        rec.update({
            "cleaned_path": str(cleaned),
            "references_path": str(refs),
            "cleaned_hash": sha256(r.text),
            "tokens_before": r.tokens_before,
            "tokens_after": r.tokens_after,
            "removed": r.removed,
            "reference_entries": len(r.references),
            "cleanup_flags": r.flags,
            "drop_appendix": drop_appendix,
            "status": "cleaned",
        })
    manifest.save()
