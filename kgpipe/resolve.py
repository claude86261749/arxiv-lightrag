"""Stage 6: entity resolution.

1. Candidates from three local checks (normalised key, nearest neighbours on the
   stored entity vectors, acronyms). No model calls.
2. Same key + same type -> merged in code.
3. Remaining candidates -> clusters of <= 8 names -> one LLM call per cluster that
   answers with groups: same / related / different / unsure. Unsure clusters get
   one second look with fuller context.
4. Apply with LightRAG's merge_entities / create_relation; record every decision
   in the alias table so later batches resolve known names in code.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEP = "<SEP>"
STOP_INITIALS = {"of", "and", "for", "the", "a", "an", "in", "on", "to", "with", "via", "by"}


def singular(word: str) -> str:
    w = word
    if len(w) <= 3 or not w.isalpha():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "ics")):
        return w[:-1]
    return w


def norm_key(name: str) -> str:
    # "+", "%" and "#" carry meaning in names (AgentCF vs AgentCF++, ETT vs ETT%).
    name = name.replace("+", " plus ").replace("%", " pct ").replace("#", " num ")
    words = re.findall(r"[a-z0-9]+", name.lower())
    if not words:
        return name.lower()
    words[-1] = singular(words[-1])
    return "".join(words)


def initials(name: str, drop_stopwords: bool = True) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", name)
             if not (drop_stopwords and w.lower() in STOP_INITIALS)]
    return "".join(w[0] for w in words).lower()


def word_set(name: str) -> set[str]:
    return {singular(w) for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 1}


def acronym_like(a: str, b: str) -> bool:
    """One name is all caps and its letters open the other's initials (LLM / Large Language Model)."""
    for x, y in ((a, b), (b, a)):
        cx = re.sub(r"[^A-Za-z0-9]", "", x)
        if 2 <= len(cx) <= 10 and cx.upper() == cx and any(
                initials(y, d).startswith(cx.lower()[:2]) for d in (True, False)):
            return True
    return False


ACRO_DEF = re.compile(r"([A-Z][\w-]*(?:\s+[\w-]+){1,7})\s+\(([A-Z][A-Za-z0-9-]{1,11})\)")


@dataclass
class Entity:
    name: str
    type: str
    description: str
    source_ids: list[str]
    file_paths: list[str]
    degree: int = 0
    papers: int = 0

    @property
    def fragments(self) -> list[str]:
        return [d for d in self.description.split(SEP) if d.strip()]


@dataclass
class ResolveStats:
    entities_before: int = 0
    focus_names: int = 0
    candidate_pairs: dict = field(default_factory=dict)
    code_merges: int = 0
    code_merged_names: int = 0
    alias_merges: int = 0
    clusters: int = 0
    names_in_clusters: int = 0
    second_looks: int = 0
    llm_same_groups: int = 0
    llm_merged_names: int = 0
    related: int = 0
    different_pairs: int = 0
    unsure_final: int = 0
    rejected_groups: int = 0
    entities_after: int = 0
    embed_texts_during_merges: int = 0


class AliasTable:
    FIELDS = ["alias", "canonical", "verdict", "date", "cluster_id", "method", "reason"]

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        if path.exists():
            with path.open() as f:
                self.rows = list(csv.DictReader(f))

    def add(self, **row):
        row.setdefault("date", dt.date.today().isoformat())
        self.rows.append({k: row.get(k, "") for k in self.FIELDS})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    def canonical_of(self) -> dict[str, str]:
        return {r["alias"]: r["canonical"] for r in self.rows if r["verdict"] == "same"}

    def decided_pairs(self) -> set[frozenset]:
        out = set()
        for r in self.rows:
            if r["verdict"] in ("different", "related", "unsure"):
                out.add(frozenset((r["alias"], r["canonical"])))
        return out


async def load_entities(rag) -> dict[str, Entity]:
    g = rag.chunk_entity_relation_graph
    names = await g.get_all_labels()
    ents = {}
    for n in names:
        d = await g.get_node(n)
        if not d:
            continue
        ents[n] = Entity(
            name=n, type=(d.get("entity_type") or "other").lower(),
            description=d.get("description", ""),
            source_ids=[s for s in d.get("source_id", "").split(SEP) if s],
            file_paths=[s for s in d.get("file_path", "").split(SEP) if s],
            degree=await g.node_degree(n))
    return ents


async def paper_concepts(rag) -> dict[str, list[str]]:
    """doc_id -> extracted entity names, from the per-document full_entities store."""
    from .store import full_entity_names, processed_docs
    docs = await processed_docs(rag)
    return await full_entity_names(rag, list(docs))


def paper_counts(ents: dict[str, Entity], concepts: dict[str, list[str]], canon: dict[str, str]) -> None:
    cnt: dict[str, set] = defaultdict(set)
    for doc, names in concepts.items():
        for n in names:
            cnt[canon.get(n, n)].add(doc)
    for e in ents.values():
        e.papers = len(cnt.get(e.name, ())) or len(set(e.file_paths))


async def entity_vectors(rag, names: list[str]) -> np.ndarray:
    from lightrag.utils import compute_mdhash_id
    ids = [compute_mdhash_id(n, prefix="ent-") for n in names]
    got = await rag.entities_vdb.get_vectors_by_ids(ids)
    dim = rag.embedding_func.embedding_dim
    mat = np.zeros((len(names), dim), dtype=np.float32)
    for i, k in enumerate(ids):
        v = got.get(k)
        if v is not None:
            mat[i] = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1
    return mat / n


def find_candidates(ents: dict[str, Entity], vecs: np.ndarray, names: list[str], focus: set[str],
                    threshold: float, top_k: int, strong: float = 0.90) -> dict[frozenset, set]:
    """pair -> set of reasons. Only pairs touching a `focus` (new/changed) name."""
    pairs: dict[frozenset, set] = defaultdict(set)
    idx = {n: i for i, n in enumerate(names)}
    by_key = defaultdict(list)
    for n in names:
        by_key[norm_key(n)].append(n)
    for group in by_key.values():
        if len(group) > 1 and focus & set(group):
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    pairs[frozenset((a, b))].add("key")
    focus_idx = [idx[n] for n in names if n in focus]
    for start in range(0, len(focus_idx), 512):
        rows = focus_idx[start:start + 512]
        sims = vecs[rows] @ vecs.T
        for r, i in enumerate(rows):
            sims[r, i] = -1
            order = np.argsort(-sims[r])[:top_k]
            for j in order:
                if sims[r, j] < threshold:
                    break
                a, b = names[i], names[j]
                # Pilot: below `strong`, only 4-8% of vector pairs were duplicates unless the
                # names share a word or look like an acronym pair. Lexical support also lifts
                # the same-type restriction (types are assigned per chunk and drift).
                lexical = bool(word_set(a) & word_set(b)) or acronym_like(a, b)
                same_type = ents[a].type == ents[b].type or {"other", "unknown"} & {ents[a].type, ents[b].type}
                if (sims[r, j] >= strong and same_type) or lexical:
                    pairs[frozenset((a, b))].add(f"vector:{sims[r, j]:.3f}")
    # Acronyms: initials, or "Full Name (ACR)" in a description.
    lower = {n.lower(): n for n in names}
    by_init = defaultdict(list)
    for n in names:
        if len(n.split()) >= 2:
            by_init[initials(n)].append(n)
    for n in names:
        if len(n.split()) >= 2:
            by_init[initials(n, False)].append(n)
    for n in names:
        compact = re.sub(r"[^A-Za-z0-9]", "", n)
        if 2 <= len(compact) <= 8 and compact.upper() == compact and compact.isalpha():
            for full in set(by_init.get(compact.lower(), [])):
                if n in focus or full in focus:
                    pairs[frozenset((n, full))].add("acronym")
    for n in names:
        if n not in focus:
            continue
        for m in ACRO_DEF.finditer(ents[n].description):
            full, acr = m.group(1).strip(), m.group(2)
            # keep only the trailing words whose initials match the acronym
            words = full.split()
            for k in range(len(words)):
                cand = " ".join(words[k:])
                if initials(cand) == acr.lower() and cand.lower() in lower and acr.lower() in lower:
                    a, b = lower[cand.lower()], lower[acr.lower()]
                    if a != b:
                        pairs[frozenset((a, b))].add("acronym-def")
                    break
    return pairs


def pick_canonical(group: list[str], ents: dict[str, Entity]) -> str:
    return max(group, key=lambda n: (ents[n].papers, ents[n].degree, -len(n), n))


def clusters_from_pairs(pairs: list[frozenset], max_size: int, weights: dict[frozenset, float]) -> list[list[str]]:
    """Clusters of <= max_size names such that every candidate pair shares at least one cluster.

    Small connected components are one cluster. Larger ones are covered by ego clusters:
    a name plus up to max_size-1 of its not-yet-covered neighbours, strongest first.
    (The pilot's greedy split of large components dropped pairs between sub-clusters.)
    """
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for p in pairs:
        a, b = tuple(p)
        adj[a][b] = adj[b][a] = weights.get(p, 0.0)
    seen, comps = set(), []
    for n in sorted(adj):
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    out = []
    for comp in comps:
        if len(comp) <= max_size:
            out.append(sorted(comp))
            continue
        uncovered = {frozenset((a, b)) for a in comp for b in adj[a]}
        while uncovered:
            deg = defaultdict(int)
            for p in uncovered:
                for x in p:
                    deg[x] += 1
            seed = max(deg, key=lambda x: (deg[x], x))
            nb = sorted((y for y in adj[seed] if frozenset((seed, y)) in uncovered),
                        key=lambda y: (-adj[seed][y], y))[:max_size - 1]
            cl = [seed] + nb
            out.append(sorted(cl))
            for i, a in enumerate(cl):
                for b in cl[i + 1:]:
                    uncovered.discard(frozenset((a, b)))
    return [c for c in out if len(c) > 1]


def source_sentences(ent: Entity, chunks: dict, n: int, max_chars: int = 300) -> list[str]:
    out = []
    pat = re.compile(re.escape(ent.name), re.I)
    for cid in ent.source_ids:
        text = chunks.get(cid, {}).get("content", "")
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if pat.search(sent):
                out.append(" ".join(sent.split())[:max_chars])
                break
        if len(out) >= n:
            break
    return out


MERGE_SYSTEM = """You decide whether names in a knowledge graph built from research papers (one arXiv category) refer to the same thing.

Definitions, applied to each pair of names:
- same: any sentence about one name is true of the other. Spelling variants, plural/singular, acronym vs expansion, and the same thing named with or without a generic suffix are "same".
- related: one is a variant, version, extension, component or subtype of the other (e.g. a model and its larger release, a method and a modified method, a general task and a specific task). Keep separate.
- different: unrelated or merely similar-sounding things, or two distinct methods/models/datasets that happen to share words.
- unsure: you cannot tell from the information given.

Rule above all others: when in doubt, keep names separate. A missed duplicate can be merged later; a wrong merge mixes two concepts and is hard to undo.
Two different named models, datasets or metrics are never "same" just because they serve the same purpose. Different metrics are different (NDCG@100 vs Recall@100), and so are different cutoffs of one metric (NDCG@10 vs NDCG@5).
An artifact about a thing (a results matrix, table, split or run) is not the same as the thing (the dataset or domain it reports on).
Only use names exactly as given. Every name may appear in at most one "same" group. Choose as canonical the clearest common short name among the members."""

MERGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "same": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "canonical": {"type": "STRING"}, "members": {"type": "ARRAY", "items": {"type": "STRING"}},
            "reason": {"type": "STRING"}}, "required": ["canonical", "members", "reason"]}},
        "related": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "from": {"type": "STRING"}, "to": {"type": "STRING"}, "relation": {"type": "STRING"}},
            "required": ["from", "to", "relation"]}},
        "different": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "a": {"type": "STRING"}, "b": {"type": "STRING"}}, "required": ["a", "b"]}},
        "unsure": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["same", "related", "different", "unsure"],
}


def cluster_payload(cid: str, names: list[str], ents: dict[str, Entity], chunks: dict,
                    desc_chars: int, n_examples: int) -> str:
    cands = []
    for n in names:
        e = ents[n]
        desc = " | ".join(e.fragments)
        cands.append({"name": n, "type": e.type, "papers": e.papers,
                      "description": desc[:desc_chars],
                      "examples": source_sentences(e, chunks, n_examples)})
    return json.dumps({"cluster_id": cid, "candidates": cands}, ensure_ascii=False, indent=1)


def validate_decision(dec: dict, names: list[str]) -> tuple[list[dict], list[dict], list[tuple], list[str], int]:
    valid = set(names)
    used, same, rejected = set(), [], 0
    for g in dec.get("same", []):
        members = [m for m in dict.fromkeys(g.get("members", [])) if m in valid and m not in used]
        canon = g.get("canonical")
        if len(members) < 2:
            rejected += 1 if g.get("members") else 0
            continue
        if canon not in members:
            canon = None  # chosen later from members
        used |= set(members)
        same.append({"canonical": canon, "members": members, "reason": g.get("reason", "")})
    related = [r for r in dec.get("related", []) if r.get("from") in valid and r.get("to") in valid
               and r["from"] != r["to"]]
    different = [(d["a"], d["b"]) for d in dec.get("different", []) if d.get("a") in valid and d.get("b") in valid]
    unsure = [u for u in dec.get("unsure", []) if u in valid]
    return same, related, different, unsure, rejected


async def resolve(rag, llm, alias: AliasTable, state_path: Path, counters, threshold: float = 0.85,
                  strong: float = 0.90,
                  top_k: int = 10, cluster_max: int = 8, workers: int = 8, log_dir: Path | None = None) -> tuple[ResolveStats, list[dict], dict]:
    st = ResolveStats()
    state = json.loads(state_path.read_text()) if state_path.exists() else {"seen": {}}
    ents = await load_entities(rag)
    st.entities_before = len(ents)
    concepts = await paper_concepts(rag)
    canon_map = alias.canonical_of()
    paper_counts(ents, concepts, canon_map)
    merges_log: list[dict] = []
    embed_before = counters.embed_texts

    async def do_merge(members: list[str], target: str, method: str, cid: str, reason: str):
        members = [m for m in members if m in ents]
        if target not in members or len(members) < 2:
            return
        others = [m for m in members if m != target]
        await rag.amerge_entities(source_entities=members, target_entity=target)
        merged = ents[target]
        for o in others:
            alias.add(alias=o, canonical=target, verdict="same", cluster_id=cid, method=method, reason=reason)
            e = ents.pop(o)
            merged.source_ids += e.source_ids
            merged.description += SEP + e.description
            merged.papers = max(merged.papers, e.papers)
        merges_log.append({"canonical": target, "aliases": others, "method": method, "cluster_id": cid,
                           "reason": reason, "types": sorted({merged.type})})

    # 0. Known aliases from earlier batches.
    for n in list(ents):
        c = canon_map.get(n)
        if c and c != n and c in ents and n in ents:
            await do_merge([c, n], c, "alias", "alias-table", "known alias")
            st.alias_merges += 1

    focus = {n for n, e in ents.items() if state["seen"].get(n) != len(e.fragments)}
    st.focus_names = len(focus)
    names = sorted(ents)
    vecs = await entity_vectors(rag, names)
    pairs = find_candidates(ents, vecs, names, focus, threshold, top_k, strong)
    decided = alias.decided_pairs()
    pairs = {p: r for p, r in pairs.items() if p not in decided}
    reasons_count = defaultdict(int)
    for r in pairs.values():
        for x in r:
            reasons_count[x.split(":")[0]] += 1
    st.candidate_pairs = {"total": len(pairs), **reasons_count}

    # 2. Same normalised key and same type: merge in code.
    by_key = defaultdict(list)
    for n in names:
        by_key[(norm_key(n), ents[n].type)].append(n)
    for (k, t), group in sorted(by_key.items()):
        if len(group) > 1:
            target = pick_canonical(group, ents)
            await do_merge(group, target, "code", f"key:{k}", "identical normalised key and type")
            st.code_merges += 1
            st.code_merged_names += len(group) - 1
    pairs = {p: r for p, r in pairs.items() if all(x in ents for x in p) and len(p) == 2}

    # 3. LLM decisions on the rest.
    weights = {p: max([float(x.split(":")[1]) for x in r if x.startswith("vector:")] or [1.0]) for p, r in pairs.items()}
    clusters = clusters_from_pairs(list(pairs), cluster_max, weights)
    st.clusters = len(clusters)
    if log_dir:
        (log_dir / "clusters.json").write_text(json.dumps(clusters, indent=0))
    from .store import chunk_records
    need = sorted({cid for c in clusters for n in c for cid in ents[n].source_ids[:12]})
    chunks = await chunk_records(rag, need)
    st.names_in_clusters = sum(len(c) for c in clusters)

    def decide(item):
        i, names_ = item
        cid = f"c-{i:04d}"
        dec = llm.json_call("merge_decision", cluster_payload(cid, names_, ents, chunks, 600, 2),
                            MERGE_SCHEMA, MERGE_SYSTEM)
        same, related, different, unsure, rej = validate_decision(dec, names_)
        second = False
        if unsure:
            second = True
            dec2 = llm.json_call("second_look", cluster_payload(cid, names_, ents, chunks, 4000, 5),
                                 MERGE_SCHEMA, MERGE_SYSTEM + "\n\nThis is a second look with fuller context. If still unsure, say unsure.")
            same, related, different, unsure, rej2 = validate_decision(dec2, names_)
            rej += rej2
        return cid, names_, same, related, different, unsure, rej, second

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(decide, enumerate(clusters)))

    # Union "same" groups across clusters (a pair can now appear in several clusters),
    # then apply each resulting group once.
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    group_info: dict[str, dict] = {}
    # When clusters disagree, keep separate: a pair any cluster called different is never unioned.
    vetoed = {frozenset(p) for r in results for p in r[4]}
    for cid, names_, same, related, different, unsure, rej, second in results:
        st.second_looks += int(second)
        st.rejected_groups += rej
        for g in same:
            members = [m for m in g["members"] if m in ents]
            members = [members[0]] + [m for m in members[1:]
                                      if not any(frozenset((m, o)) in vetoed for o in members if o != m)]
            for m in members[1:]:
                ra, rb = find(members[0]), find(m)
                if ra != rb:
                    parent[rb] = ra
            for m in members:
                group_info.setdefault(m, {"cid": cid, "reason": g["reason"], "canonical": g["canonical"]})
    groups: dict[str, list[str]] = defaultdict(list)
    for m in group_info:
        groups[find(m)].append(m)
    for members in groups.values():
        members = sorted(set(members))
        if len(members) < 2:
            continue
        votes = [group_info[m]["canonical"] for m in members if group_info[m]["canonical"] in members]
        target = max(set(votes), key=votes.count) if votes else pick_canonical(members, ents)
        info = group_info[members[0]]
        await do_merge(members, target, "llm", info["cid"], info["reason"])
        st.llm_same_groups += 1
        st.llm_merged_names += len(members) - 1
    for cid, names_, same, related, different, unsure, rej, second in results:
        for r in related:
            a, b = r["from"], r["to"]
            if a in ents and b in ents and not await rag.chunk_entity_relation_graph.has_edge(a, b):
                try:
                    await rag.acreate_relation(a, b, {"description": f"{a} is {r['relation']} {b}.",
                                                      "keywords": r["relation"], "weight": 1.0})
                    st.related += 1
                except Exception as e:  # recorded, not fatal
                    merges_log.append({"error": f"create_relation {a}->{b}: {e}"})
            alias.add(alias=a, canonical=b, verdict="related", cluster_id=cid, method="llm", reason=r["relation"])
        for a, b in different:
            alias.add(alias=a, canonical=b, verdict="different", cluster_id=cid, method="llm")
            st.different_pairs += 1
        for u in unsure:
            st.unsure_final += 1
            for other in names_:
                if other != u:
                    alias.add(alias=u, canonical=other, verdict="unsure", cluster_id=cid, method="llm")

    st.entities_after = len(ents)
    st.embed_texts_during_merges = counters.embed_texts - embed_before
    state["seen"] = {n: len(e.fragments) for n, e in ents.items()}
    state_path.write_text(json.dumps(state))
    alias.save()
    return st, merges_log, pairs
