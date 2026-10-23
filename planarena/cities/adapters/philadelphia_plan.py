"""Philadelphia's adopted plan as a parcel layer, digitised from the Philadelphia2035 district plans (PDF).

Philadelphia publishes its comprehensive plan's land-use recommendations only inside the 18 district plans (17
documents; the Far Northeast plan covers two districts), as "Proposed Land Use" maps drawn parcel by parcel in the
City Planning Commission's land-use categories. No GIS layer of them is published. Each map is digitised with
planarena.sources.pdfplan (vector shapes, legend colours, registration to the district's parcels) and every parcel of
the district takes the category drawn on it. The result is flagged "digitised from the plan PDF" with each district's
registration error.

Sources: the district plans linked from https://www.phila2035.org/plan (Philadelphia City Planning Commission) and
the Planning Districts layer (City of Philadelphia).

Run:  python -I -m planarena.cities.adapters.philadelphia_plan
"""
from __future__ import annotations

import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.registry import city_dir
from planarena.config import DATA
from planarena.national.common import download
from planarena.sources import crosswalk, pdfplan
from planarena.sources.arcgis import fetch_layer
from planarena.taxonomy import INDEX

RAW = DATA / "raw" / "philadelphia"
DISTRICTS_URL = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Planning_Districts/FeatureServer/0"
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
# district -> Google Drive id of its district plan (as linked from phila2035.org, October 2026)
PLANS = {"Central": "16azD7ae4P6UGh6ALBWRDKZK2T-w9ONAF", "Central Northeast": "1OGonMNh7ujgm6lgvhhchMlwuH5-F8cov",
         "Lower Far Northeast": "1A1lyw4lbfXRtZQ3ev7cBzgUmvGvfNJza", "Upper Far Northeast": "1A1lyw4lbfXRtZQ3ev7cBzgUmvGvfNJza",
         "Lower North": "1FfNt4ABjSCg76VjUUwtwPNWs-l8caa2S", "Lower Northeast": "1od8WrgnblFHGfLohBxrf508PLiDorLno",
         "Lower Northwest": "1JzzHHBn7R8zTOZ-Wf6LWNcLpz-pB77pJ", "Lower South": "1E2lC1GRyq_ceEiDHWl-aZ0-ZOFq_AAnl",
         "Lower Southwest": "1ACSPjtP6xKHm60V8z-hclrYsgcHCkobY", "North": "1CCwaSu5je-KmCffB5J4TsCCQlytFQJaW",
         "North Delaware": "1n2Qo9RgOnT4f9c2hxpfatPKdwGiiOPXy", "River Wards": "1O-1cvLHsB3qDZjwcRCLH-gUhpzN_6Ghj",
         "South": "1c5sVjAUwj7toKIxmhzrbYUIF2TZCA09J", "University Southwest": "1_Y_5CH37Y8zQUiikt7PPNQ8obfz5R-I1",
         "Upper North": "1CkYtX664g-NgefOw1tioIxaKId4KnBI7", "Upper Northwest": "1NUnAh8k_e1VhzZOFgvu_6p4i3gba0fL7",
         "West": "16Sl3O_EAZ10Zb3DozMbF9eNZg1Ws6nfp", "West Park": "1pgz4_0OMjnYxAacXwHAoC9vL0heTexpY"}
OUT = city_dir("4260000") / "plan_digitised.parquet"
REPORT = city_dir("4260000") / "plan_digitised.json"
MAX_MEDIAN_FT = 10.0  # a map whose shapes sit further than this from the parcels (median) is not used


def _agreement(labels: np.ndarray, existing: np.ndarray) -> float:
    """Share of the district's parcels whose planned class (16 classes) equals the existing class: most land keeps its
    use in a plan, so a map that lines up with the lots agrees far more than the same map shifted."""
    plan = np.array([INDEX.get(crosswalk.philadelphia_plan(l), -1) if l else -1 for l in labels])
    ok = (plan >= 0) & (existing >= 0)
    return float((plan[ok] == existing[ok]).mean()) if ok.any() else 0.0


def _existing_chapter(pdf) -> set[int]:
    """Pages of the existing-conditions land-use map. In every district plan that chapter opens with the "Citywide
    Land Use" legend page and the existing land-use map follows within three pages; its legend is the plan map's."""
    texts = pdfplan.page_texts(pdf)
    first = min((i for i, t in texts.items() if i > 10 and "citywide land use" in t.lower()), default=None)
    return set(range(first, first + 4)) if first else set()


def build(only: list[str] | None = None) -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    lots = gpd.read_parquet(city_dir("4260000") / "parcels.geoparquet")
    twin = pd.read_parquet(city_dir("4260000") / "parcels.parquet", columns=["bbl", "lu16"])
    existing = twin.lu16.to_numpy()
    rp = lots.geometry.representative_point()
    dist = fetch_layer(DISTRICTS_URL, RAW / "planning_districts.parquet", ["objectid", "dist_name", "abbrev"]).to_crs(lots.crs)
    label = np.full(len(lots), "", dtype=object)
    district_of = np.full(len(lots), "", dtype=object)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    prev = pd.read_parquet(OUT) if OUT.exists() else None
    if prev is not None and len(prev) == len(lots):
        label[:] = prev.plan_label.fillna("").to_numpy()
        district_of[:] = prev.plan_district.fillna("").to_numpy()
    by_file: dict[str, list[str]] = {}
    for name, fid in PLANS.items():
        by_file.setdefault(fid, []).append(name)
    for fid, names in by_file.items():
        if only and not set(names) & set(only):
            continue
        pdf = download(DRIVE.format(fid), RAW / "plans" / f"{'_'.join(n.lower().replace(' ', '_') for n in names)}.pdf")
        area = shapely.union_all(list(dist[dist.dist_name.isin(names)].geometry))
        inside = rp.within(area).to_numpy()
        log(f"{' + '.join(names)}: {inside.sum():,} parcels, {pdf.stat().st_size / 1e6:.0f} MB plan")
        r = pdfplan.digitise(pdf, area, lots[inside], log=log, skip_pages=_existing_chapter(pdf))
        if r is None or r["transform"]["median_match_ft"] > MAX_MEDIAN_FT:
            report[" + ".join(names)] = {"used": False, "reason": "no Proposed Land Use map registered"
                                         if r is None else f"median error {r['transform']['median_match_ft']:.1f} ft"}
            log("  not used")
            continue
        idx = np.nonzero(inside)[0]
        agree = _agreement(r["labels"], existing[idx])
        control = _agreement(r["control_labels"], existing[idx])
        if agree <= control + 0.05:  # the map does not line up with the lots better than a map 300 ft off
            report[" + ".join(names)] = {"used": False, "reason": f"agreement with existing use {agree:.0%} vs "
                                         f"{control:.0%} for the map shifted 300 ft"}
            log("  not used")
            continue
        label[idx] = r["labels"]
        district_of[idx] = " + ".join(names)
        report[" + ".join(names)] = {"used": True, "pages": r["pages"], "coverage": round(r["coverage"], 4),
                                     "median_error_ft": round(r["transform"]["median_match_ft"], 2),
                                     "p90_error_ft": round(r["transform"]["p90_match_ft"], 1),
                                     "scale_ft_per_pt": round(r["transform"]["s"], 3),
                                     "rotation_deg": round(r["transform"]["rotation_deg"], 2),
                                     "shapes_visible": r["shapes_visible"], "shapes_drawn": r["shapes_drawn"],
                                     "agreement_with_existing": round(agree, 3),
                                     "agreement_if_shifted_300ft": round(control, 3),
                                     "legend": r["legend"], "colours": r["colours"]}
        log(f"  used pages {r['pages']}: coverage {r['coverage']:.1%}, median error {r['transform']['median_match_ft']:.1f} ft")
        pd.DataFrame({"bbl": twin.bbl, "plan_label": label, "plan_district": district_of}).to_parquet(OUT)
        REPORT.write_text(json.dumps(report, indent=2))
    pd.DataFrame({"bbl": twin.bbl, "plan_label": label, "plan_district": district_of}).to_parquet(OUT)
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"done: {int((label != '').sum()):,} of {len(lots):,} parcels with a plan category")
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(build(sys.argv[1:] or None), indent=1)[:4000])
