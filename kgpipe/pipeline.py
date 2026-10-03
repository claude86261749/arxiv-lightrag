"""Run the nine stages on one batch of papers.

    python -m kgpipe.pipeline --corpus data/corpus/cs.IR --run runs/pilot --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from .export import export
from .gemini import Gemini
from .graph_clean import clean_graph
from .intake import Manifest, clean_papers, intake
from .quality import measure, merge_review_sheet, missed_duplicate_sheet
from .rag import Counters, RagSettings, build_rag, delete_docs, doc_id_for, ingest_files, use_counters
from .resolve import AliasTable, resolve


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(run_dir / "lightrag.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg = logging.getLogger("lightrag")
    lg.handlers = [h]
    lg.setLevel(logging.INFO)
    lg.propagate = False


async def run(args) -> dict:
    run_dir = Path(args.run)
    setup_logging(run_dir)
    t0 = time.time()
    report: dict = {"args": vars(args), "stages": {}}
    manifest = Manifest(run_dir / "manifest.json")

    def stage(name, **data):
        report["stages"][name] = {**data, "elapsed_s": round(time.time() - t0, 1)}
        (run_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        print(f"[{time.time() - t0:7.1f}s] {name}: " + json.dumps(data, default=str)[:300], flush=True)

    # 1-2. Intake and cleanup.
    changed = intake(Path(args.corpus), manifest, args.limit)
    clean_papers(manifest, changed, run_dir, drop_appendix=not args.keep_appendix)
    stage("intake_clean", new_or_changed=len(changed),
          tokens_before=sum(manifest.papers[i]["tokens_before"] for i in changed),
          tokens_after=sum(manifest.papers[i]["tokens_after"] for i in changed),
          flagged=[i for i in changed if manifest.papers[i]["cleanup_flags"]])

    settings = RagSettings(working_dir=run_dir / "rag", max_gleaning=args.gleaning,
                           embed_dim=args.embed_dim, llm_model=args.llm_model)
    counters_by_stage: dict = {}
    c_ingest = Counters()
    rag = await build_rag(settings, c_ingest)
    try:
        # 3-5. Chunk, extract, embed (replace changed documents first).
        replace = [doc_id_for(f"{i}.md") for i in changed if manifest.papers[i].get("previous_hash")]
        if replace:
            await delete_docs(rag, replace)
        files = [Path(manifest.papers[i]["cleaned_path"]) for i in changed]
        if files:
            await ingest_files(rag, files, settings.working_dir / "inputs", f"batch_{int(t0)}")
        failed = []
        for i in changed:
            st = await rag.doc_status.get_by_id(doc_id_for(f"{i}.md"))
            ok = st and str(st.get("status")).lower().endswith("processed")
            manifest.papers[i]["status"] = "indexed" if ok else "failed"
            manifest.papers[i]["doc_id"] = doc_id_for(f"{i}.md")
            manifest.papers[i]["chunks"] = st.get("chunks_count") if st else None
            if not ok:
                failed.append((i, st.get("error_msg") if st else "missing"))
        manifest.save()
        counters_by_stage["ingest"] = c_ingest.as_dict()
        stage("ingest", failed=failed, **c_ingest.as_dict())
        if args.stop_after == "ingest":
            return report

        # 6. Entity resolution.
        llm = Gemini(args.llm_model)
        c_res = Counters()
        use_counters(c_res)
        alias = AliasTable(run_dir / "alias_table.csv")
        rs, merges, pairs = await resolve(rag, llm, alias, run_dir / "resolve_state.json", c_res,
                                          threshold=args.nn_threshold, strong=args.nn_strong, log_dir=run_dir)
        (run_dir / "merges.json").write_text(json.dumps(merges, indent=1))
        (run_dir / "candidate_pairs.json").write_text(json.dumps(
            [{"pair": sorted(p), "reasons": sorted(r)} for p, r in pairs.items()], indent=1))
        counters_by_stage["resolve"] = {**c_res.as_dict(), "llm_calls": {**c_res.llm_calls, **llm.calls},
                                        "direct_llm_tokens": dict(llm.tokens)}
        stage("resolve", **asdict(rs), llm_calls=llm.calls, llm_tokens=dict(llm.tokens))

        # 7. Graph cleanup.
        llm7 = Gemini(args.llm_model)
        c_gc = Counters()
        use_counters(c_gc)
        gr = await clean_graph(rag, llm7, Path(args.stoplist), hubs=args.hubs)
        (run_dir / "graph_cleanup.json").write_text(json.dumps(gr, indent=1))
        counters_by_stage["graph_clean"] = {**c_gc.as_dict(), "llm_calls": {**c_gc.llm_calls, **llm7.calls},
                                            "direct_llm_tokens": dict(llm7.tokens)}
        stage("graph_clean", stoplist_removed=len(gr["stoplist_removed"]),
              self_loops=len(gr["self_loops_removed"]), condensed=len(gr["condensed"]),
              low_evidence=len(gr["low_evidence"]), paper_local_renamed=len(gr["paper_local_renamed"]), hubs_several=sum(h["verdict"] == "several" for h in gr["hubs"]),
              llm_calls=llm7.calls)

        # 8. Quality report and review sheets.
        merge_review_sheet(merges, run_dir / "review_merges.csv")
        missed = await missed_duplicate_sheet(rag, run_dir / "review_missed_duplicates.csv")
        q = await measure(rag, run_dir, manifest, counters_by_stage, asdict(rs), gr)
        q["review_sheets"] = {"merges": "review_merges.csv", "missed_duplicates": f"review_missed_duplicates.csv ({missed} pairs)"}
        q["counters_by_stage"] = counters_by_stage
        (run_dir / "quality.json").write_text(json.dumps(q, indent=1))
        stage("quality", **{k: q[k] for k in ("papers", "chunks", "entities_final", "params")})

        # 9. Exports.
        ex = await export(rag, run_dir, manifest, alias)
        stage("export", **ex)
    finally:
        await rag.finalize_storages()
    report["elapsed_s"] = round(time.time() - t0, 1)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gleaning", type=int, default=1)
    ap.add_argument("--keep-appendix", action="store_true")
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--llm-model", default="gemini-3.8-flash")
    ap.add_argument("--nn-threshold", type=float, default=0.85)
    ap.add_argument("--nn-strong", type=float, default=0.90,
                    help="vector-only candidates need this similarity; lexical ones need --nn-threshold")
    ap.add_argument("--hubs", type=int, default=50)
    ap.add_argument("--stoplist", default="config/stoplist.txt")
    ap.add_argument("--stop-after", choices=["ingest"], default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
