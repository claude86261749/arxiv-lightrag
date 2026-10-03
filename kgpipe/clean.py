"""Stage 2: rule-based text cleanup.

Saves references and citing sentences to a side file, then removes references,
acknowledgements-type sections, repeated lines and (optionally) appendices.
Deterministic: same input and flags give the same output.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
REF_HEAD = re.compile(r"^(\d+(\.\d+)*\.?\s+)?(references|bibliography|literature cited|works cited)\b", re.I)
DROP_HEAD = re.compile(
    r"^(\d+(\.\d+)*\.?\s+)?(acknowledge?ments?|funding|financial (support|disclosure)|"
    r"author(s'?)? contributions?|contributions? statement|credit authorship.*)\b", re.I)
APPX_HEAD = re.compile(r"^(appendix|appendices|supplementary (material|information)|"
                       r"technical appendix)\b|^[A-Z](\.\d+)*\.?\s+\S", re.I)
APPX_LETTER = re.compile(r"^[A-Z](\.\d+)*\.?\s+\S")  # "A Details", "B.1 Setup" (case-sensitive)
ARXIV_ID = re.compile(r"(?:arxiv[:\s]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(v\d+)?", re.I)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;)])")
BIB_LINE = re.compile(r"^\s*[-*]\s+\[(bib\.bib\d+)\]\s*(.*)$")
NUM_LINE = re.compile(r"^\s*(?:[-*]\s+)?\[(\d+)\]\s+(.*)$")
CITE_LINK = re.compile(r"\[([^\]]*)\]\(#(bib\.bib\d+)\)")
CITE_NUM = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
PAGE_NO = re.compile(r"^\s*(page\s+)?\d{1,3}(\s+of\s+\d+)?\s*$", re.I)
ARXIV_STAMP = re.compile(r"^\s*arXiv:\d{4}\.\d{4,5}v\d+\s+\[[^\]]+\]\s+\d{1,2}\s+\w+\s+\d{4}\s*$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")
ABBREV = re.compile(r"\b(et al|e\.g|i\.e|cf|Fig|Figs|Eq|Eqs|Sec|Tab|vs|resp|approx)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking citation links or abbreviations."""
    links: list[str] = []
    prot = CITE_LINK.sub(lambda m: links.append(m.group(0)) or f"\x00{len(links) - 1}\x00", text)
    parts: list[str] = []
    for piece in SENT_SPLIT.split(prot):
        if parts and ABBREV.search(parts[-1]):
            parts[-1] += " " + piece
        else:
            parts.append(piece)
    return [re.sub("\x00(\\d+)\x00", lambda m: links[int(m.group(1))], x) for x in parts]


def ntokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


@dataclass
class Section:
    level: int          # 0 = preamble before first heading
    title: str
    start: int          # line index of heading (or 0)
    end: int            # exclusive
    path: list[str] = field(default_factory=list)


def sections(lines: list[str]) -> list[Section]:
    heads = []
    in_code = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(ln)
        if m:
            heads.append((i, len(m.group(1)), m.group(2).strip()))
    out = [Section(0, "", 0, heads[0][0] if heads else len(lines))]
    stack: list[tuple[int, str]] = []
    for k, (i, lvl, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, title))
        out.append(Section(lvl, title, i, end, [t for _, t in stack]))
    return out


def subtree_end(secs: list[Section], idx: int) -> int:
    """Line index where section idx and its subsections end."""
    lvl = secs[idx].level
    for s in secs[idx + 1:]:
        if s.level <= lvl:
            return s.start
    return secs[-1].end


def parse_references(lines: list[str]) -> list[dict]:
    entries, cur = [], None
    for ln in lines:
        m = BIB_LINE.match(ln) or NUM_LINE.match(ln)
        if m:
            cur = {"key": m.group(1), "raw": m.group(2).strip()}
            entries.append(cur)
        elif re.match(r"^\s*[-*]\s+\S", ln):
            cur = {"key": str(len(entries) + 1), "raw": ln.strip()[2:].strip()}
            entries.append(cur)
        elif ln.strip() and cur is not None and not HEADING.match(ln):
            cur["raw"] += " " + ln.strip()
        elif ln.strip() and cur is None and not HEADING.match(ln):
            cur = {"key": str(len(entries) + 1), "raw": ln.strip()}  # unbulleted list
            entries.append(cur)
        elif not ln.strip() and cur is not None and not entries[-1]["key"].startswith("bib."):
            cur = None if not NUM_LINE.match(ln) else cur
    for e in entries:
        a = ARXIV_ID.search(e["raw"])
        d = DOI.search(e["raw"])
        e["arxiv_id"] = a.group(1) if a else None
        e["doi"] = d.group(1) if d else None
        e["citing"] = []
    return entries


def _expand_nums(s: str) -> list[str]:
    out = []
    for part in re.split(r"\s*,\s*", s):
        m = re.match(r"(\d+)\s*[–-]\s*(\d+)", part)
        if m and int(m.group(2)) - int(m.group(1)) < 50:
            out += [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]
        elif part.strip().isdigit():
            out.append(part.strip())
    return out


def strip_cite_links(text: str) -> str:
    return CITE_LINK.sub(lambda m: m.group(1), text)


@dataclass
class CleanResult:
    text: str
    references: list[dict]
    tokens_before: int
    tokens_after: int
    removed: dict
    flags: list[str]


def clean_markdown(md: str, drop_appendix: bool = True,
                   max_removed: float = 0.5, min_removed: float = 0.02) -> CleanResult:
    md = re.sub(r"(\w)-\n(\w)", r"\1\2", md)  # rule: rejoin hyphenated line breaks
    lines = md.split("\n")
    secs = sections(lines)
    drop = [False] * len(lines)
    removed: Counter = Counter()
    refs: list[dict] = []

    # References section(s)
    ref_idx = [i for i, s in enumerate(secs) if s.level and REF_HEAD.match(s.title)]
    ref_level = None
    for i in ref_idx:
        end = subtree_end(secs, i)
        refs += parse_references(lines[secs[i].start + 1:end])
        for j in range(secs[i].start, end):
            drop[j] = True
        removed["references"] += end - secs[i].start
        ref_level = secs[i].level if ref_level is None else ref_level

    # Acknowledgements, funding, contributions
    for i, s in enumerate(secs):
        if s.level and DROP_HEAD.match(s.title):
            end = subtree_end(secs, i)
            for j in range(s.start, end):
                drop[j] = True
            removed["acknowledgements"] += end - s.start

    # Appendices: same-level sections after the references, or appendix-titled ones.
    appx_lines: list[int] = []
    if drop_appendix:
        after_refs = secs[ref_idx[0]].start if ref_idx else None
        in_appx = False
        for i, s in enumerate(secs):
            if not s.level or drop[s.start]:
                continue
            top = ref_level or min((x.level for x in secs if x.level > 1), default=2)
            is_appx = (s.level <= top and (
                (after_refs is not None and s.start > after_refs) or
                re.match(r"^(appendix|appendices|supplementary)", s.title, re.I) or
                (in_appx and APPX_LETTER.match(s.title))))
            if s.level <= top and re.match(r"^(appendix|appendices)", s.title, re.I):
                in_appx = True
            if is_appx:
                end = subtree_end(secs, i)
                for j in range(s.start, end):
                    if not drop[j]:
                        appx_lines.append(j)
                    drop[j] = True
                removed["appendix"] += end - s.start

    # Citing sentences (from text that will be kept, before stripping links)
    by_key = {e["key"]: e for e in refs}
    for s in secs:
        if not s.level:
            continue
        body_end = next((x.start for x in secs if x.start > s.start), s.end)
        body = " ".join(ln for j, ln in enumerate(lines[s.start + 1:body_end], s.start + 1)
                        if not drop[j] and ln.strip())
        for sent in split_sentences(body):
            keys = [m.group(2) for m in CITE_LINK.finditer(sent)]
            if not keys and refs and not refs[0]["key"].startswith("bib."):
                keys = [k for m in CITE_NUM.finditer(sent) for k in _expand_nums(m.group(1))]
            for k in dict.fromkeys(keys):
                if k in by_key and len(by_key[k]["citing"]) < 10:
                    by_key[k]["citing"].append({"sentence": strip_cite_links(sent).strip()[:600],
                                                "section": " > ".join(s.path)})

    # Repeated lines: running headers/footers recur throughout the file (not just in one
    # table), page numbers form an increasing run, arXiv stamps match a fixed pattern.
    live = [j for j in range(len(lines)) if not drop[j] and lines[j].strip()]
    pos: dict[str, list[int]] = {}
    for j in live:
        pos.setdefault(lines[j].strip(), []).append(j)
    span = (live[-1] - live[0]) if live else 0
    pages = [j for j in live if PAGE_NO.match(lines[j])]
    nums = [int(re.search(r"\d+", lines[j]).group()) for j in pages]
    page_run = len(nums) >= 5 and sum(b == a + 1 for a, b in zip(nums, nums[1:])) >= 0.8 * (len(nums) - 1)
    for j in live:
        t = lines[j].strip()
        if HEADING.match(t):
            continue
        ps = pos[t]
        header = (len(ps) >= 4 and len(t) >= 15 and re.search(r"[A-Za-z]{3}", t) and "|" not in t
                  and not t.startswith(("-", "*")) and span and (ps[-1] - ps[0]) > 0.5 * span)
        if ARXIV_STAMP.match(t) or header or (page_run and j in pages):
            drop[j] = True
            removed["repeated_lines"] += 1

    kept = "\n".join(strip_cite_links(ln) for j, ln in enumerate(lines) if not drop[j])
    kept = re.sub(r"\n{3,}", "\n\n", kept).strip() + "\n"
    before, after = ntokens(md), ntokens(kept)
    # The flag is about heading detection, so appendices (dropped by choice) are left out.
    appx_tokens = ntokens("\n".join(lines[j] for j in appx_lines))
    base = before - appx_tokens
    frac = 1 - after / base if base else 0
    flags = []
    if frac > max_removed:
        flags.append(f"removed {frac:.0%} of tokens (> {max_removed:.0%})")
    if frac < min_removed:
        flags.append(f"removed {frac:.1%} of tokens (< {min_removed:.0%})")
    if not refs:
        flags.append("no reference entries found")
    removed = {k: v for k, v in removed.items()}
    removed["appendix_tokens"] = appx_tokens
    return CleanResult(kept, refs, before, after, removed, flags)
