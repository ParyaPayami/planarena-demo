"""Export a city, or one district of it, and optionally a PlanArena plan, as the data an ArcGIS Urban model imports.

ArcGIS Urban imports types from its spreadsheet template or from hosted tables, and boundaries and parcels from
hosted feature layers. Fields are matched in its import dialog, and types are referenced by Custom ID (Esri, "Preparing
your data", ArcGIS Urban documentation). This module writes:

  space_use_types.csv  the nine analysis groups (Custom ID = group key), with floor area per job and per home
                       derived from the city's twin
  zoning_types.csv     one row per zoning district code: allowed space-use types (JSON list of Custom IDs) and the
                       median maximum FAR of its parcels
  zoning.geojson       zoning boundaries: parcels dissolved by district code (ZoningType = Custom ID)
  land_use_types.csv   one row per adopted-plan designation: its class and allowed space-use type
  land_use.geojson     land-use boundaries: parcels dissolved by designation (LandUseType = Custom ID)
  parcels.geojson      parcels (CustomID = parcel id) with existing use, plan, zoning, floor area, homes and jobs;
                       with a plan file from the Studio, also the scenario use and FAR
  README.md            import order and field mapping

Upload the GeoJSON files to ArcGIS Online as hosted feature layers (and the CSV files as hosted tables, or copy them
into Urban's spreadsheet template), then import them in Urban's data manager in the README's order. This module only
writes files; nothing is sent anywhere.

Run:  python -I -m planarena.integrations.urban_export "Dallas" [--district 14] [--plan plan.csv] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from planarena.cities.build import CITIES, resolve
from planarena.cities.registry import city_dir
from planarena.config import LAND_USES, ROOT, USE_KEYS
from planarena.indicators.engine import load_engine
from planarena.taxonomy import CLASSES, TO_GROUP

OUT = ROOT / "exports" / "urban"


def _label(cls: np.ndarray) -> np.ndarray:
    return np.array([CLASSES[c].label if c >= 0 else "" for c in cls], dtype=object)


def _group(cls: np.ndarray) -> np.ndarray:
    return np.array([USE_KEYS[TO_GROUP[c]] if c >= 0 else "" for c in cls], dtype=object)


def _allowed(mask: int) -> str:
    return json.dumps([k for i, k in enumerate(USE_KEYS) if (int(mask) >> i) & 1])


def space_use_types(p: pd.DataFrame, st: dict) -> pd.DataFrame:
    """The nine groups, with floor area per job and per home as the city's twin has them (derived)."""
    rows = []
    for i, u in enumerate(LAND_USES):
        m = (p.use == i).to_numpy()
        area, jobs, homes = p.bldg_area[m].sum(), st["jobs"][m].sum(), st["homes"][m].sum()
        rows.append({"CustomID": u.key, "Label": u.label,
                     "AreaPerJob_sqft": round(area / jobs, 1) if jobs > 0 else None,
                     "AreaPerHome_sqft": round(p.res_area[m].sum() / homes, 1) if homes > 0 and u.key in ("res", "mix")
                     else None})
    return pd.DataFrame(rows)


def export(target: str, out: Path | None = None, district: str | None = None, plan: Path | None = None) -> Path:
    geoid = resolve(target)
    src = city_dir(geoid)
    out = Path(out or OUT / CITIES[geoid][0].split(",")[0].lower().replace(" ", "_"))
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(src / "parcels.parquet")
    geo = gpd.read_parquet(src / "parcels.geoparquet")
    st = load_engine(src).baseline_state
    keep = np.ones(len(p), bool) if district is None else (p.cd.astype(str) == str(district)).to_numpy()
    if not keep.any():
        raise SystemExit(f"no parcels in district {district}")
    space_use_types(p, st).to_csv(out / "space_use_types.csv", index=False)

    p, geo = p[keep].reset_index(drop=True), geo[keep].reset_index(drop=True)
    homes, jobs = st["homes"][keep], st["jobs"][keep]
    lots = gpd.GeoDataFrame({
        "CustomID": p.bbl.astype(str), "Address": p.address.astype(str),
        "ExistingUse": _label(p.lu16.to_numpy()), "ExistingGroup": _group(p.lu16.to_numpy()),
        "PlanDesignation": p.plan_designation.astype(str), "PlanUse": _label(p.plan16.to_numpy()),
        "ZoningType": p.zone.astype(str), "ZoningUse": _label(p.zone16.to_numpy()),
        "LotArea_sqft": p.lot_area.round(0), "FloorArea_sqft": p.bldg_area.round(0), "Floors": p.floors,
        "Homes": np.round(homes, 1), "Jobs": np.round(jobs, 1), "YearBuilt": p.year_built,
    }, geometry=geo.geometry.values, crs=geo.crs).to_crs(4326)
    if plan is not None:  # a plan exported from the Studio (columns bbl, land_use, far)
        ch = pd.read_csv(plan, dtype={"bbl": str})
        use = lots.CustomID.map(dict(zip(ch.bbl, ch.land_use)))
        far = lots.CustomID.map(dict(zip(ch.bbl, pd.to_numeric(ch.get("far"), errors="coerce"))))
        lots["ScenarioUse"] = use.fillna(lots.ExistingGroup)
        lots["ScenarioFAR"] = far
        lots["Changed"] = use.notna()
    lots.to_file(out / "parcels.geojson", driver="GeoJSON")

    zoned = p.zone16.to_numpy() >= 0
    if zoned.any():
        z = lots[zoned].dissolve(by="ZoningType", as_index=False)[["ZoningType", "geometry"]]
        z.to_file(out / "zoning.geojson", driver="GeoJSON")
        g = p[zoned].groupby("zone")
        pd.DataFrame({"CustomID": g.size().index, "Label": g.size().index,
                      "ZoningUse": [CLASSES[c].label for c in g.zone16.agg(lambda s: s.mode().iloc[0])],
                      "AllowedSpaceUseTypes": [_allowed(m) for m in g.allowed.agg(lambda s: s.mode().iloc[0])],
                      "MaxFAR": g.max_far.median().round(2).to_numpy(),
                      "Parcels": g.size().to_numpy()}).to_csv(out / "zoning_types.csv", index=False)
    planned = p.plan16.to_numpy() >= 0
    if planned.any():
        lu = lots[planned].dissolve(by="PlanDesignation", as_index=False)[["PlanDesignation", "geometry"]]
        lu.rename(columns={"PlanDesignation": "LandUseType"}).to_file(out / "land_use.geojson", driver="GeoJSON")
        g = p[planned].groupby("plan_designation")
        cls = g.plan16.agg(lambda s: s.mode().iloc[0])
        pd.DataFrame({"CustomID": cls.index, "Label": cls.index, "PlanUse": [CLASSES[c].label for c in cls],
                      "AllowedSpaceUseTypes": [json.dumps([USE_KEYS[TO_GROUP[c]]]) for c in cls],
                      "Parcels": g.size().to_numpy()}).to_csv(out / "land_use_types.csv", index=False)
    (out / "README.md").write_text(_readme(CITIES[geoid][0], district, zoned.any(), planned.any(), plan is not None))
    return out


def _readme(city: str, district, zoned: bool, planned: bool, scenario: bool) -> str:
    steps = ["1. **Space-use types:** `space_use_types.csv` (Custom ID = `CustomID`)."]
    if zoned:
        steps += ["2. **Zoning types:** `zoning_types.csv` (Custom ID = `CustomID`; allowed space-use types = "
                  "`AllowedSpaceUseTypes`, a JSON list of space-use Custom IDs; max FAR = `MaxFAR`).",
                  "3. **Zoning boundaries:** `zoning.geojson` (zoning type = `ZoningType`)."]
    if planned:
        steps += ["4. **Land-use types:** `land_use_types.csv` (Custom ID = `CustomID`; allowed space-use types = "
                  "`AllowedSpaceUseTypes`).", "5. **Land-use boundaries:** `land_use.geojson` (land-use type = `LandUseType`)."]
    steps += ["6. **Parcels:** `parcels.geojson` (Custom ID = `CustomID`). The other fields can be imported as parcel "
              "parameters." + (" `ScenarioUse` and `ScenarioFAR` carry the PlanArena plan; `Changed` marks its parcels."
                               if scenario else "")]
    return "\n".join([
        f"# ArcGIS Urban import: {city}" + (f", district {district}" if district else ""), "",
        "Written by `python -I -m planarena.integrations.urban_export`. Upload the GeoJSON files to ArcGIS Online as "
        "hosted feature layers, and the CSV files as hosted tables (or copy them into Urban's spreadsheet template "
        "without changing its column headings). Then import them in Urban's data manager in this order, matching "
        "fields as listed:", "", *steps, "",
        "**Notes**",
        "- Space-use types are PlanArena's nine analysis groups.",
        "- Floor area per job and per home are derived from the city's twin.",
        "- `MaxFAR` is the median of the zone's parcels. It comes from the Zoning Resolution in New York and from the "
        "context FAR (the 90th percentile of built FAR in the block group) elsewhere; see METHODS section 9.",
        "- Boundaries are parcels dissolved by code, so streets are not part of any zone.", ""])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("city")
    ap.add_argument("--district")
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = export(a.city, a.out, a.district, a.plan)
    print("written:", ", ".join(sorted(f.name for f in out.iterdir())), "in", out)


if __name__ == "__main__":
    main()
