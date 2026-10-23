"""Verify every bibliography entry against the record its identifier points to.

The PIMALUOS submission was rejected because references had real-looking DOIs that resolved to different papers.
This check makes that impossible to miss: each entry must carry a DOI, an arXiv ID or an ISBN, and the title,
first author, year and (for articles) journal must match the record that identifier resolves to.

Run with:  python -m planarena.tools.check_refs refs.bib [--out docs/reference_check.md]
Exit code 1 if any entry fails or cannot be verified. Passing this check does not replace reading the source:
every cited claim also needs a row in docs/citation_log.csv (claim, source, supporting passage or page).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from xml.etree import ElementTree

UA = "PlanArena-reference-check/1.0"


# ----------------------------------------------------------------------------- BibTeX
def parse_bib(text: str) -> list[dict]:
    """Minimal BibTeX parser: @type{key, field = {value} | "value" | number, ...} with nested braces."""
    entries, i = [], 0
    while True:
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.S).search(text, i)
        if not m:
            return entries
        typ, key, j = m.group(1).lower(), m.group(2), m.end()
        depth, k = 1, j
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        body, i = text[j:k - 1], k
        if typ in ("comment", "preamble", "string"):
            continue
        fields, p = {}, 0
        while True:
            fm = re.compile(r"\s*(\w[\w-]*)\s*=\s*", re.S).match(body, p)
            if not fm:
                break
            name, p = fm.group(1).lower(), fm.end()
            if p < len(body) and body[p] == "{":
                depth, q = 1, p + 1
                while q < len(body) and depth:
                    depth += {"{": 1, "}": -1}.get(body[q], 0)
                    q += 1
                val, p = body[p + 1:q - 1], q
            elif p < len(body) and body[p] == '"':
                q = body.index('"', p + 1)
                val, p = body[p + 1:q], q + 1
            else:
                q = re.compile(r"[^,]*").match(body, p).end()
                val, p = body[p:q].strip(), q
            fields[name] = re.sub(r"\s+", " ", val.replace("{", "").replace("}", "")).strip()
            p = body.find(",", p)
            if p < 0:
                break
            p += 1
        entries.append({"type": typ, "key": key, **fields})


# ----------------------------------------------------------------------------- comparison helpers
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def first_family(authors: str) -> str:
    """Family name of the first author, without lowercase particles ("te Brömmelstroet" -> "brommelstroet")."""
    first = re.split(r"\s+and\s+", authors or "")[0].strip()
    family = first.split(",")[0] if "," in first else (first.split()[-1] if first else "")
    words = family.split()
    while len(words) > 1 and words[0].islower():
        words = words[1:]
    return norm(" ".join(words))


# ----------------------------------------------------------------------------- resolvers
def _get(url: str, timeout: float = 30.0, tries: int = 4) -> bytes:
    """GET with retries: registries rate-limit, so a transient failure must not read as an unresolvable identifier."""
    import time

    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == tries - 1:
                raise
        except Exception:
            if k == tries - 1:
                raise
        time.sleep(2 * (k + 1))


def crossref(doi: str) -> dict | None:
    try:
        m = json.loads(_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    except Exception:
        return None
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(k) or {}).get("date-parts") or [[None]]
        if parts[0][0]:
            year = int(parts[0][0])
            break
    return {"title": (m.get("title") or [""])[0], "first_family": norm(((m.get("author") or [{}])[0]).get("family", "")),
            "first_editor": norm(((m.get("editor") or [{}])[0]).get("family", "")),
            "year": year, "container": (m.get("container-title") or [""])[0]}


def arxiv(aid: str) -> dict | None:
    try:
        xml = _get("http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(aid))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ElementTree.fromstring(xml).find("a:entry", ns)
        if e is None or e.find("a:title", ns) is None:
            return None
        name = e.find("a:author/a:name", ns).text
        return {"title": e.find("a:title", ns).text, "first_family": norm(name.split()[-1]),
                "year": int(e.find("a:published", ns).text[:4]), "container": "arXiv"}
    except Exception:
        return None


def openlibrary(isbn: str) -> dict | None:
    try:
        m = json.loads(_get(f"https://openlibrary.org/isbn/{re.sub(r'[^0-9Xx]', '', isbn)}.json"))
    except Exception:
        return None
    year = re.search(r"(\d{4})", m.get("publish_date", "") or "")
    return {"title": m.get("title", ""), "first_family": None, "year": int(year.group(1)) if year else None, "container": None}


# ----------------------------------------------------------------------------- the check
def check_entry(e: dict, resolve=None) -> dict:
    """Return {key, status: ok | mismatch | unverifiable, details}. ``resolve`` maps (kind, id) -> record (testable)."""
    resolve = resolve or {"doi": crossref, "arxiv": arxiv, "isbn": openlibrary}
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", e.get("doi", "")).strip()
    aid = e.get("eprint") if "arxiv" in (e.get("archiveprefix", "") + e.get("journal", "")).lower() else e.get("arxiv", "")
    if not aid:
        m = re.search(r"arxiv\.org/abs/([\w.\-/]+)", e.get("url", ""))
        aid = m.group(1) if m else ""
    for kind, ident in (("doi", doi), ("arxiv", aid), ("isbn", e.get("isbn", ""))):
        if not ident:
            continue
        rec = resolve[kind](ident)
        if rec is None:
            return {"key": e["key"], "status": "unverifiable", "details": f"{kind} {ident} did not resolve"}
        problems = []
        if similar(e.get("title", ""), rec["title"]) < 0.85:
            problems.append(f"title differs: bib '{e.get('title', '')[:80]}' vs record '{rec['title'][:80]}'")
        if not e.get("author") and e.get("editor"):  # an edited book: compare editors (its record may list a foreword author)
            if rec.get("first_editor") and first_family(e["editor"]) != rec["first_editor"]:
                problems.append(f"first editor differs: bib '{first_family(e['editor'])}' vs record '{rec['first_editor']}'")
        elif rec.get("first_family") and first_family(e.get("author", "")) != rec["first_family"]:
            problems.append(f"first author differs: bib '{first_family(e.get('author', ''))}' vs record '{rec['first_family']}'")
        if rec.get("year") and e.get("year", "").isdigit() and abs(int(e["year"]) - rec["year"]) > 1:
            problems.append(f"year differs: bib {e['year']} vs record {rec['year']}")
        venue = e.get("journal") or e.get("booktitle")
        if kind == "doi" and venue and rec.get("container") and similar(venue, rec["container"]) < 0.7:
            problems.append(f"venue differs: bib '{venue}' vs record '{rec['container']}'")
        return {"key": e["key"], "status": "mismatch" if problems else "ok", "details": "; ".join(problems) or f"{kind} {ident}"}
    return {"key": e["key"], "status": "unverifiable", "details": "no DOI, arXiv ID or ISBN"}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    entries = parse_bib(open(argv[0], encoding="utf-8").read())
    rows = [check_entry(e) for e in entries]
    bad = [r for r in rows if r["status"] != "ok"]
    lines = ["# Reference check", "", f"{len(rows)} entries: {len(rows) - len(bad)} ok, {len(bad)} failing.", "",
             "| Key | Status | Details |", "|---|---|---|"] + [f"| {r['key']} | {r['status']} | {r['details']} |" for r in rows]
    report = "\n".join(lines) + "\n"
    if "--out" in argv:
        open(argv[argv.index("--out") + 1], "w", encoding="utf-8").write(report)
    print(report)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
