"""Per-city ingestion report: what each official layer covers, and how the twin compares with Census and LODES.

Written for the planner to review before a city is used: parcel count against the source, the share of parcels with
an existing land use, a plan designation and a zoning class, the 16-class distributions side by side, how often the
existing use already matches the plan and zoning, and homes and jobs against the federal counts.

Run:  python -I -m planarena.cities.ingestion "<city>"      -> docs/ingestion/<city>.md
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import yaml

from planarena.cities.build import CITIES, resolve
from planarena.cities.registry import city_dir
from planarena.config import ROOT
from planarena.indicators.engine import load_engine
from planarena.taxonomy import CLASSES, INDEX

OUT = ROOT / "docs" / "ingestion"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report(target: str) -> str:
    geoid = resolve(target)
    out = city_dir(geoid)
    df = pd.read_parquet(out / "parcels.parquet")
    meta = json.loads((out / "twin_meta.json").read_text())
    q = meta.get("data_quality", {})
    n = len(df)
    e = load_engine(out)
    st = e.baseline_state
    lines = [f"# Ingestion report: {CITIES[geoid][0]}", "",
             f"Source: {meta.get('source', '')} ({meta.get('source_url', '')}).", "",
             "## Coverage", "", "| Layer | Parcels | Share |", "|---|---:|---:|",
             f"| Parcels in the twin | {n:,} | |"]
    for col, label in (("lu16", "Existing land use (baseline)"), ("plan16", "Adopted plan designation"),
                       ("zone16", "Zoning district use class")):
        k = int((df[col] >= 0).sum())
        lines.append(f"| {label} | {k:,} | {_pct(k / n) if k else 'no layer'} |")
    lines.append(f"| Floor area recorded | {int((df.bldg_area > 0).sum()):,} | {_pct((df.bldg_area > 0).mean())} |")
    homes, jobs = float(st["homes"].sum()), float(st["jobs"].sum())
    hu, lodes = q.get("homes_census_2020"), q.get("jobs_lodes")
    lines += ["", "## Homes and jobs", "", "| Measure | Twin | Federal count | Ratio |", "|---|---:|---:|---:|"]
    if hu:
        lines.append(f"| Homes | {homes:,.0f} | {hu:,} (Census 2020 housing units) | {homes / hu:.2f} |")
    if lodes:
        lines.append(f"| Jobs (fitted to LODES, so not a check) | {jobs:,.0f} | {lodes:,} (LODES workplace jobs) | "
                     f"{jobs / lodes:.2f} |")
    rates = yaml.safe_load((out / "calibrated.yaml").read_text()).get("jobs_sqft_per_worker", {})
    lines += ["", "Fitted floor area per worker (sq ft): " + ", ".join(
        f"{k.replace('_', ' ')} {v['value']:,.0f}" for k, v in rates.items() if isinstance(v, dict)) + "."]
    if rates.get("replaced_by_cbecs"):
        lines += [f"Replaced by the CBECS value because the fit was more than 4x away: {rates['replaced_by_cbecs']}."]
    grp = np.array([c.group for c in CLASSES])
    lines += ["", "## Land-use classes", "",
              "Plan and zoning layers often distinguish fewer classes than the existing-use layer (a plan may "
              "say 'public facilities' where the parcel is a school), so agreement is shown for the class and for "
              "its analysis group.", "",
              "| Class | Existing | Plan | Zoning | Plan agrees: class | Plan agrees: group |",
              "|---|---:|---:|---:|---:|---:|"]
    for k in CLASSES:
        i = INDEX[k.key]
        ex, pl, zn = int((df.lu16 == i).sum()), int((df.plan16 == i).sum()), int((df.zone16 == i).sum())
        both = (df.lu16 == i) & (df.plan16 >= 0)
        agree = _pct(float((df.plan16[both] == i).mean())) if both.any() else ""
        agree_g = _pct(float((grp[df.plan16[both].to_numpy()] == k.group).mean())) if both.any() else ""
        lines.append(f"| {k.label} | {ex:,} | {pl:,} | {zn:,} | {agree} | {agree_g} |")
    lines.append(f"| Unknown | {int((df.lu16 < 0).sum()):,} | {int((df.plan16 < 0).sum()):,} | "
                 f"{int((df.zone16 < 0).sum()):,} | | |")
    has = (df.lu16 >= 0) & (df.plan16 >= 0)
    if has.any():
        g = lambda c: np.array([CLASSES[x].group if x >= 0 else "" for x in c])
        lines += ["", f"Existing use already matches the plan's class on {_pct(float((df.lu16[has] == df.plan16[has]).mean()))}"
                  f" of parcels, and its analysis group on {_pct(float((g(df.lu16[has]) == g(df.plan16[has])).mean()))}."]
    if (df.zone16 >= 0).any():
        allowed_now = ((df.allowed.to_numpy() >> df.use.clip(lower=0).to_numpy()) & 1).astype(bool) & (df.use >= 0).to_numpy()
        lines += [f"The existing use is allowed by the zoning screen on {_pct(float(allowed_now.mean()))} of parcels "
                  "(legal non-conforming uses and planned developments account for the rest)."]
    else:
        lines += ["The city has no zoning layer: every use is allowed."]
    dig = meta.get("plan_digitised")
    if dig:
        lines += ["", "## Adopted plan digitised from PDF maps", "",
                  "Each district plan's land-use map, registered to the district's parcels (METHODS section 9). A map "
                  "is used if its median parcel-to-shape distance is at most 10 ft and the plan agrees with the existing "
                  "use clearly more often than the same map shifted 300 ft.", "",
                  "| District | Used | PDF pages | Median / 90th pct. error (ft) | Parcels covered | Agrees with existing use "
                  "| Same, map shifted 300 ft |", "|---|---|---|---|---:|---:|---:|"]
        for d, v in sorted(dig.items()):
            if v.get("used"):
                lines.append(f"| {d} | yes | {', '.join(map(str, v['pages']))} | {v['median_error_ft']} / "
                             f"{v['p90_error_ft']} | {_pct(v['coverage'])} | {_pct(v['agreement_with_existing'])} | "
                             f"{_pct(v['agreement_if_shifted_300ft'])} |")
            else:
                lines.append(f"| {d} | no: {v.get('reason', '')} | | | | | |")
    lines += ["", "## Data quality", "", "| Check | Value |", "|---|---:|"]
    lines += [f"| {k.replace('_', ' ')} | {v:,} |" if isinstance(v, (int, float)) else f"| {k.replace('_', ' ')} | {v} |"
              for k, v in q.items()]
    lines += ["", "## What each column is", "", "| Column | Status |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in meta.get("columns", {}).items()]
    return "\n".join(lines) + "\n"


def summary() -> str:
    """One table across the built cities: each layer's coverage, homes against the Census, and what is estimated."""
    rows = ["# Ingestion summary: the ten PlanArena cities", "",
            "Generated by `python -I -m planarena.cities.ingestion --summary` from each built twin. Each city's own "
            "report is in this folder.", "",
            "| City | Parcels | Existing use | Adopted plan | Zoning class | Floor area recorded | Homes / Census 2020 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for geoid, (label, adapter) in CITIES.items():
        out = city_dir(geoid)
        if adapter is None or not (out / "parcels.parquet").exists():
            rows.append(f"| {label} | not built | | | | | |")
            continue
        df = pd.read_parquet(out / "parcels.parquet")
        for c in ("lu16", "plan16", "zone16"):
            if c not in df:
                df[c] = -1
        q = json.loads((out / "twin_meta.json").read_text()).get("data_quality", {})
        share = lambda c: _pct(float((df[c] >= 0).mean())) if (df[c] >= 0).any() else "none"
        hu = q.get("homes_census_2020")
        rows.append(f"| {label} | {len(df):,} | {share('lu16')} | {share('plan16')} | {share('zone16')} | "
                    f"{_pct(float((df.bldg_area > 0).mean()))} | {df.units_res.sum() / hu:.2f} |" if hu else
                    f"| {label} | {len(df):,} | {share('lu16')} | {share('plan16')} | {share('zone16')} | "
                    f"{_pct(float((df.bldg_area > 0).mean()))} | |")
    return "\n".join(rows) + "\n"


def layers_summary() -> dict:
    """Per city: parcels, each official layer's coverage, homes against the Census, and the digitised plan's checks.
    Written to results/validation/cities_layers.json for the report and the paper (the twins are not committed)."""
    out = {}
    for geoid, (label, adapter) in CITIES.items():
        d = city_dir(geoid)
        if adapter is None or not (d / "parcels.parquet").exists():
            continue
        df = pd.read_parquet(d / "parcels.parquet", columns=None)
        meta = json.loads((d / "twin_meta.json").read_text())
        q = meta.get("data_quality", {})
        share = lambda c: float((df[c] >= 0).mean()) if c in df else 0.0
        out[geoid] = {"city": label, "parcels": int(len(df)), "existing": share("lu16"), "plan": share("plan16"),
                      "zoning": share("zone16"), "floor_area": float((df.bldg_area > 0).mean()),
                      "homes_twin": float(df.units_res.sum()), "homes_census_2020": q.get("homes_census_2020"),
                      "plan_matches_existing_group": None}
        has = (df.lu16 >= 0) & (df.plan16 >= 0) if "plan16" in df else None
        if has is not None and has.any():
            grp = np.array([c.group for c in CLASSES])
            out[geoid]["plan_matches_existing_class"] = float((df.lu16[has] == df.plan16[has]).mean())
            out[geoid]["plan_matches_existing_group"] = float((grp[df.lu16[has].to_numpy()] == grp[df.plan16[has].to_numpy()]).mean())
        if meta.get("plan_digitised"):
            out[geoid]["plan_digitised"] = meta["plan_digitised"]
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "README.md").write_text(summary())
        (ROOT / "results" / "validation" / "cities_layers.json").write_text(json.dumps(layers_summary(), indent=1))
        print(summary())
        raise SystemExit(0)
    target = " ".join(sys.argv[1:])
    text = report(target)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (CITIES[resolve(target)][0].split(",")[0].lower().replace(" ", "_") + ".md")
    path.write_text(text)
    print(text)
    print(f"written to {path}")
