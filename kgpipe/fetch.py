"""Corpus acquisition for pilots: arXiv listing + arXiv HTML -> Markdown.

Outside the pipeline proper (the design takes Markdown as input). This exists so
there is a reproducible corpus to run on. The converter keeps heading levels,
in-text citations as links to `#bib.bibN`, and the reference list as
`- [bib.bibN] ...` lines, so stage 2 can recover citing sentences.
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString
from markdownify import MarkdownConverter

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
UA = {"User-Agent": "kgpipe-pilot/0.1 (research pipeline)"}


def list_category(category: str, max_results: int = 200) -> list[dict]:
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": f"cat:{category}", "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": max_results}
    r = requests.get(url, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).findall(f"{ATOM}entry"):
        arxiv_id = e.findtext(f"{ATOM}id").rsplit("/abs/", 1)[1]
        out.append({
            "arxiv_id": re.sub(r"v\d+$", "", arxiv_id),
            "version": (re.search(r"v(\d+)$", arxiv_id) or [None, "1"])[1],
            "title": " ".join(e.findtext(f"{ATOM}title").split()),
            "authors": [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")],
            "submitted": e.findtext(f"{ATOM}published"),
            "primary_category": e.find(f"{ARXIV_NS}primary_category").get("term"),
            "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
        })
    return out


class _Conv(MarkdownConverter):
    def convert_a(self, el, text, *args, **kwargs):
        href = el.get("href") or ""
        if href.startswith("#bib."):
            return f"[{text.strip()}]({href})"
        return text  # drop other links (internal refs, urls) but keep their text

    def convert_img(self, el, text, *args, **kwargs):
        return ""


def _prepare(soup: BeautifulSoup) -> BeautifulSoup:
    art = soup.select_one("article.ltx_document") or soup.body
    for sel in [".ltx_authors", ".ltx_note", ".ltx_page_footer", "nav", "header",
                "footer", ".ltx_dates", ".ltx_role_footnote", "figure img", "svg",
                ".ltx_TOC", "button", ".ltx_classification", ".ltx_keywords",
                ".ltx_tag_item", ".ltx_bib_cited"]:
        for t in art.select(sel):
            t.decompose()
    for m in art.select("math"):
        alt = (m.get("alttext") or "").strip()
        m.replace_with(NavigableString(f" ${alt}$ " if alt else ""))
    # Abstract heading
    abs_ = art.select_one(".ltx_abstract")
    if abs_ is not None:
        h = abs_.select_one("h6, .ltx_title_abstract")
        if h is not None:
            h.name = "h2"
    # Bibliography -> one line per entry with a stable key.
    bib = art.select_one("section.ltx_bibliography")
    if bib is not None:
        items = []
        for li in bib.select("li.ltx_bibitem"):
            key = li.get("id", "")
            tag = li.select_one(".ltx_tag")
            label = tag.get_text(" ", strip=True) if tag else ""
            if tag:
                tag.decompose()
            body = " ".join(li.get_text(" ", strip=True).split())
            items.append(f"- [{key}] {label}. {body}" if label else f"- [{key}] {body}")
        new = soup.new_tag("section")
        h = soup.new_tag("h2")
        h.string = "References"
        new.append(h)
        pre = soup.new_tag("p")
        pre.string = "\n".join(items)
        new.append(pre)
        bib.replace_with(new)
    # Tables: flatten into pipe-separated rows; LaTeXML tables convert badly otherwise.
    for tb in art.select("table, .ltx_tabular"):
        if tb.find_parent(["table"]) or tb.find_parent(class_="ltx_tabular"):
            continue  # nested; flattened with its outermost table
        rows = []
        for tr in tb.select("tr, .ltx_tr"):
            if tr.find_parent(["tr"]) or tr.find_parent(class_="ltx_tr"):
                continue
            cells = [" ".join(c.get_text(" ", strip=True).split())
                     for c in tr.select("th, td, .ltx_td, .ltx_th")
                     if not (c.find_parent(["td", "th"]) or c.find_parent(class_="ltx_td"))]
            if any(cells):
                rows.append(" | ".join(cells))
        p = soup.new_tag("p")
        p.string = "\n".join(rows)
        tb.replace_with(p)
    return art


def html_to_markdown(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("h1.ltx_title_document")
    title = " ".join(title_el.get_text(" ", strip=True).split()) if title_el else ""
    art = _prepare(soup)
    md = _Conv(heading_style="ATX", bullets="-", escape_asterisks=False,
               escape_underscores=False, escape_misc=False).convert_soup(art)
    # Bibliography lines were put in one <p>; markdownify keeps newlines inside text.
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    return title, md.strip() + "\n"


def fetch_corpus(category: str, n: int, out_dir: Path, min_words: int = 2500,
                 pool: int = 200, sleep: float = 3.0) -> list[dict]:
    """Download n papers whose primary category is `category` and that have arXiv HTML."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    skipped = []
    for p in list_category(category, pool):
        if len(meta) >= n:
            break
        if p["primary_category"] != category or p["arxiv_id"] in meta:
            continue
        time.sleep(sleep)
        r = requests.get(f"https://arxiv.org/html/{p['arxiv_id']}v{p['version']}", headers=UA, timeout=60)
        if r.status_code != 200 or "ltx_document" not in r.text:
            skipped.append((p["arxiv_id"], f"no html ({r.status_code})"))
            continue
        (out_dir / "_html").mkdir(exist_ok=True)
        (out_dir / "_html" / f"{p['arxiv_id']}.html").write_text(r.text)
        _, md = html_to_markdown(r.text)
        words = len(md.split())
        if words < min_words or "## References" not in md:
            skipped.append((p["arxiv_id"], f"too short or no references ({words} words)"))
            continue
        (out_dir / f"{p['arxiv_id']}.md").write_text(md)
        meta[p["arxiv_id"]] = p
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"fetched {p['arxiv_id']} ({words} words) {p['title'][:70]}")
    for s in skipped:
        print("skipped", *s)
    return list(meta.values())


def reconvert(out_dir: Path) -> None:
    """Re-run the HTML -> Markdown conversion on cached HTML (after converter changes)."""
    for h in sorted((out_dir / "_html").glob("*.html")):
        _, md = html_to_markdown(h.read_text())
        (out_dir / f"{h.stem}.md").write_text(md)
