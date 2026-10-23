"""Trace a municipal land-management requirements catalogue to PlanArena capabilities.

The catalogue is practitioner domain knowledge: a city's enterprise permitting/planning system
specification (about 2,100 numbered requirements in 14 functional areas). It is used as a document,
with no human-participant research. This script keeps it de-identified: outputs carry only area codes,
requirement numbers, theme tags and counts, never requirement text or the issuing agency's name.

Theme tagging is a transparent first-pass keyword screen. It shows where the catalogue's demand lies;
it does not certify that any single requirement is met. The thesis (Chapter 3) replaces it with
hand coding checked against a gold sample.

Run with:  python -m planarena.requirements /path/to/catalogue.xlsx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from planarena.config import ROOT

AREAS = {
    "A": "System administration", "B": "General operations", "C": "Customer service & online portal",
    "D": "Application submission & intake", "E": "Application processing", "F": "Financials",
    "G": "Advanced (long-range) planning", "H": "Current planning", "I": "Hearings", "J": "Permitting",
    "K": "Plan review", "L": "Inspections", "M": "Investigations & enforcement", "N": "Licensing & registration",
}

# theme: (regex, PlanArena feature, status)
THEMES = {
    "map_gis": (r"\bgis\b|\bmaps?\b|mapping|geograph|spatial|parcel|\bapns?\b|polygon|aerial|imagery",
                "Parcel digital twin, map, address/BBL search, block and lasso selection", "implemented"),
    "development_potential": (r"development potential|cumulative|quantif|densit|intensit|capacity|build[- ]?out",
                              "Impact engine: zoning capacity, soft sites, homes/jobs/value deltas with ranges",
                              "implemented"),
    "plan_study": (r"plan amendment|planning study|stud(y|ies)|amendment|rezon|general plan|specific plan",
                   "Studies: saved scenarios with description, study area and comparison", "implemented"),
    "status_tracking": (r"status|checklist|due date|deadline|milestone",
                        "Study status workflow, checklist and hearing date", "implemented"),
    "history_audit": (r"audit|histor|\blogs?\b|version",
                      "Hash-chained, tamper-evident audit log; per-study history", "implemented"),
    "reporting": (r"report|export|download",
                  "Staff report (HTML), CSV/GeoJSON export, provenance JSON with source SHA-256", "implemented"),
    "data_upload": (r"upload|import|shapefile|electronic(ally)? submi",
                    "Plan import (CSV, GeoJSON, zipped shapefile, GeoPackage) with review before apply",
                    "implemented"),
    "notice_radius": (r"radius|mailing|nearby (propert|owner|resident)|surrounding propert",
                      "Notice-radius parcel list (no mailing)", "partial"),
    "zoning_rules": (r"zoning|\bzones?\b|setback|floor area|land use|code compliance",
                     "Simplified zoning rules-as-code screen and warnings (not a legal determination)",
                     "partial"),
    "hearings": (r"hearing|agenda|commission|council|public notice",
                 "Hearing date and staff report; agendas and minutes are not handled", "partial"),
    "public_input": (r"public (comment|input)|stakeholder|communit|testimon|social media",
                     "Modelled stakeholder vote and residents' referendum; no comment intake", "partial"),
    "fees_payments": (r"\bfees?\b|payment|invoice|refund|bond|escrow|deposit|revenue",
                      "Out of scope for PlanArena (thesis agent roster: fees and bonds)", "out of scope"),
    "inspections": (r"inspect", "Out of scope (thesis agent roster: inspection scheduling and routing)",
                    "out of scope"),
    "enforcement": (r"complain|violation|enforce|citation", "Out of scope (thesis agent roster: enforcement triage)",
                    "out of scope"),
    "licensing": (r"licen[cs]e|registration|renewal", "Out of scope (thesis agent roster: licensing)", "out of scope"),
    "plan_sheets": (r"sheet|markup|stamp|drawing|redline", "Out of scope (thesis agent roster: drawing extraction)",
                    "out of scope"),
}
COMPILED = {k: re.compile(v[0].replace("(", "(?:").replace("(?:?:", "(?:"), re.I) for k, v in THEMES.items()}


def load(path: Path) -> pd.DataFrame:
    rows = []
    for sheet in pd.ExcelFile(path).sheet_names:
        code = sheet[:1]
        if code not in AREAS or not sheet[1:2] == ".":
            continue
        d = pd.read_excel(path, sheet_name=sheet, header=0)
        d.columns = [str(c).strip() for c in d.columns]
        req = next(c for c in d.columns if c.lower().startswith("requirement"))
        num = pd.to_numeric(d[d.columns[0]], errors="coerce")
        d = d[num.notna() & d[req].notna()]
        for n, text in zip(pd.to_numeric(d[d.columns[0]]), d[req].astype(str)):
            rows.append({"id": f"{code}{int(n)}", "area": code, "text": text})
    return pd.DataFrame(rows)


def tag(df: pd.DataFrame) -> pd.DataFrame:
    for k, rx in COMPILED.items():
        df[k] = df.text.str.contains(rx)
    df["themes"] = df[list(THEMES)].apply(lambda r: ";".join(k for k in THEMES if r[k]), axis=1)
    pa = [k for k, v in THEMES.items() if v[2] in ("implemented", "partial")]
    df["planarena_related"] = df[pa].any(axis=1)
    return df


def write(df: pd.DataFrame, out_md: Path, out_csv: Path) -> None:
    df[["id", "area", "themes", "planarena_related"]].to_csv(out_csv, index=False)  # no requirement text
    n = len(df)
    lines = [
        "# Requirements traceability (generated)",
        "",
        "<!-- GENERATED by python -m planarena.requirements. Do not edit by hand. -->",
        "",
        f"Source: a de-identified municipal land-management system requirements catalogue, {n:,} numbered "
        f"requirements in {df.area.nunique()} functional areas. It is used as practitioner domain knowledge "
        "(document analysis). There are no human participants, so no IRB review is involved. Outputs keep "
        "only area codes, requirement numbers and theme tags.",
        "",
        "Tagging is a transparent keyword screen (`planarena/requirements.py`). It shows where demand lies; "
        "it does not certify that a given requirement is met.",
        "",
        "## Requirements by area",
        "",
        "| Area | Requirements | Related to PlanArena themes |",
        "|---|---:|---:|",
    ]
    for code, name in AREAS.items():
        a = df[df.area == code]
        if len(a):
            lines.append(f"| {code}. {name} | {len(a)} | {int(a.planarena_related.sum())} ({a.planarena_related.mean():.0%}) |")
    lines.append(f"| **All** | **{n}** | **{int(df.planarena_related.sum())} ({df.planarena_related.mean():.0%})** |")
    lines += ["", "## Themes and PlanArena coverage", "",
              "| Theme | Requirements | Top areas | PlanArena feature | Status |", "|---|---:|---|---|---|"]
    for k, (_, feature, status) in THEMES.items():
        hit = df[df[k]]
        top = ", ".join(f"{c} {v}" for c, v in hit.area.value_counts().head(3).items())
        lines.append(f"| {k.replace('_', ' ')} | {len(hit)} | {top} | {feature} | {status} |")
    coding_file = Path(__file__).with_name("requirements_coding.yaml")
    if coding_file.exists():
        import yaml

        cod = yaml.safe_load(coding_file.read_text())
        items = cod["items"]
        counts = pd.Series([i["status"] for i in items]).value_counts()
        order = ["met", "partial", "not met", "out of scope"]
        lines += ["", f"## Hand-coded: area {cod['area']}, requirement by requirement ({cod['coded_on']})", "",
                  "Each requirement was read against PlanArena. Every item coded **met** is backed by an automated "
                  "acceptance test in `tests/test_requirements.py`. Needs are paraphrased in our own words.", "",
                  " · ".join(f"**{k}** {int(counts.get(k, 0))}" for k in order) + f" (of {len(items)})", "",
                  "| ID | Need (paraphrase) | Status | PlanArena feature | Gap |", "|---|---|---|---|---|"]
        for i in sorted(items, key=lambda x: (order.index(x["status"]), int(x["id"][1:]))):
            lines.append(f"| {i['id']} | {i['need']} | {i['status']} | {i.get('feature', '')} | {i.get('gap', '')} |")
    lines += ["", "## Beyond the catalogue", "",
              "The catalogue specifies records, workflow and compliance. It asks for development-potential "
              "quantification (area G) but not for these PlanArena features:",
              "",
              "* multi-indicator social, economic and environmental impact with sensitivity ranges",
              "* a graph neural network land-use suitability model with spatial cross-validation",
              "* an optimal land-use plan (best response) and a Pareto front (NSGA-III)",
              "* a stakeholder vote under nine social-choice rules, a residents' referendum and a parcel-by-parcel ballot, compared with bargaining solutions (Nash, Kalai-Smorodinsky) and price of anarchy",
              "* a game-like mobile interface", ""]
    out_md.write_text("\n".join(lines))


def main(path: str) -> None:
    df = tag(load(Path(path)))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    write(df, ROOT / "docs" / "REQUIREMENTS_TRACEABILITY.md", ROOT / "results" / "traceability.csv")
    print(f"{len(df)} requirements; {int(df.planarena_related.sum())} related to PlanArena themes")


if __name__ == "__main__":
    main(sys.argv[1])
