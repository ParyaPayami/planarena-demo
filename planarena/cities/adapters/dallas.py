"""Adapter: City of Dallas, TX, from the city's own existing land use, plan and zoning, with the appraisal roll.

Sources (City of Dallas GIS; Dallas Central Appraisal District):
- Existing Land Use Layer (2025 survey, 2026 release): a polygon per parcel with the city's land-use category,
  vacancy, and the appraisal district's building and property classes.
- forwardDallas! 2.0 (adopted 2024): the comprehensive plan's placetype for each parcel (the plan layer).
- Base zoning districts (Dallas Development Code, Chapter 51A).
- Council districts (game levels).
- DCAD 2026 certified data files: building area, stories, year built, units and values per account. Accounts are
  joined to parcels by GIS parcel id, so condominium units are summed to their parcel. Only building, unit and value
  columns are read; owner columns never are.
Parcels outside Dallas County (parts of Collin, Denton, Kaufman and Rockwall) have no DCAD record: their floor area
comes from Overture footprints (estimated, flagged).

Run:  python -I -m planarena.cities.adapters.dallas
"""
from __future__ import annotations

import json
import re
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.common import finish_twin, local_crs, overture_buildings, plausible_floor_area
from planarena.cities.registry import city_dir, register
from planarena.config import DATA, USE_INDEX as U, USE_KEYS, WEB_CRS
from planarena.national import census, places
from planarena.national.common import download
from planarena.sources import crosswalk
from planarena.sources.arcgis import fetch_layer
from planarena.taxonomy import CLASSES, INDEX as TAX_INDEX, group_of
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "4819000"
RAW = DATA / "raw" / "dallas"
GIS = "https://services2.arcgis.com/rwnOSbfKSwyTBcwN/arcgis/rest/services"
LAND_USE_URL = f"{GIS}/Existing_Land_Use_Layer_2025/FeatureServer/5"
PLAN_URL = f"{GIS}/COD_FU_Placetypes_20240925/FeatureServer/0"
ZONING_URL = f"{GIS}/Dallas_Zoning/FeatureServer/15"
COUNCIL_URL = f"{GIS}/Dallasbaselayers/FeatureServer/1"
DCAD_URL = ("https://www.dallascad.org/ViewPDFs.aspx?type=3&id=%5C%5CDCAD.ORG%5CWEB%5CWEBDATA%5CWEBFORMS%5C"
            "DATA%20PRODUCTS%5CDCAD2026_CURRENT.ZIP")
APARTMENT = ("APARTMENT", "RESIDENTIAL MULTI-FAMILY", "SENIOR CITIZEN HOUSING", "RETIREMENT", "REAL, RESIDENTIAL")
MIX_RES = 0.7  # assumption, as for the other cities

ALL = (1 << len(USE_KEYS)) - 1
bit = lambda *keys: sum(1 << U[k] for k in keys)
RES = bit("res", "pub", "opn", "vac")
# zoning district family -> allowed analysis groups and the district's own use class (Dallas Development Code
# Chapter 51A, simplified). Planned developments (PD) and conservation districts (CD) follow their own ordinances
# and are not screened.
ZONING = [("R-", RES, "sf"), ("A(", RES, "sf"), ("D(", RES, "mf_small"), ("TH-", RES, "sf"), ("CH", RES, "sf"),
          ("MF-", bit("res", "pub", "opn", "prk", "vac"), "mf"), ("MH", RES, "mf"),
          ("WR-", bit("res", "mix", "pub", "opn", "prk", "vac"), "mf"),
          ("MU-", bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "mix"),
          ("WMU-", bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "mix"),
          ("CA-", bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "mix"),
          ("UC-", bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "mix"),
          ("NO(", bit("mix", "com", "pub", "opn", "prk", "vac"), "off"), ("LO-", bit("mix", "com", "pub", "opn", "prk", "vac"), "off"),
          ("MO-", bit("mix", "com", "pub", "opn", "prk", "vac"), "off"), ("GO(", bit("mix", "com", "pub", "opn", "prk", "vac"), "off"),
          ("O-", bit("mix", "com", "pub", "opn", "prk", "vac"), "off"),
          ("NS(", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"), ("CR", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"),
          ("RR", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"), ("CS", bit("com", "ind", "pub", "opn", "prk", "vac"), "ret"),
          ("GR", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"), ("MC-", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"),
          ("LI", bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
          ("IR", bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
          ("IM", bit("ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
          ("P(", bit("prk", "opn", "vac"), "prk")]


def zone_rule(zone: str) -> tuple[int, str]:
    z = zone.upper().strip() if isinstance(zone, str) else ""
    for prefix, mask, cls in ZONING:
        if z.startswith(prefix) and (not prefix[-1].isalpha() or not z[len(prefix):len(prefix) + 1].isalpha()):
            return mask, cls
    return ALL, "unknown"


def _dcad(zpath) -> dict:
    """Building, unit and value columns of the DCAD files (owner columns are never read)."""
    z = zipfile.ZipFile(zpath)
    read = lambda name, cols: pd.read_csv(z.open(name), usecols=cols, dtype=str, encoding="latin-1")
    return {"accounts": read("ACCOUNT_INFO.CSV", ["ACCOUNT_NUM", "DIVISION_CD", "GIS_PARCEL_ID"]),
            "res": read("RES_DETAIL.CSV", ["ACCOUNT_NUM", "BLDG_CLASS_DESC", "YR_BUILT", "TOT_LIVING_AREA_SF",
                                           "NUM_KITCHENS", "NUM_STORIES_DESC"]),
            "com": read("COM_DETAIL.CSV", ["ACCOUNT_NUM", "BLDG_CLASS_DESC", "YEAR_BUILT", "GROSS_BLDG_AREA",
                                           "NUM_STORIES", "NUM_UNITS"]),
            "values": read("ACCOUNT_APPRL_YEAR.CSV", ["ACCOUNT_NUM", "IMPR_VAL", "LAND_VAL", "TOT_VAL"])}


def _stories(desc) -> float:
    m = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4}
    d = desc.upper() if isinstance(desc, str) else ""
    for w, v in m.items():
        if w in d:
            return float(v)
    x = re.search(r"\d+(\.\d+)?", d)
    return float(x.group()) if x else np.nan


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    crs = local_crs(pl["lat"], pl["lon"])
    log("downloading the city's land use, plan, zoning and council districts")
    g = fetch_layer(LAND_USE_URL, RAW / "city" / "existing_land_use_2026.parquet",
                    ["OBJECTID", "ACCT", "GIS_ACCT", "BLDG_CL", "PROP_CL", "Address", "COUNTY", "Year",
                     "Existing_Land_Use", "Vacant", "Source"], workers=4)
    plan = fetch_layer(PLAN_URL, RAW / "city" / "forwarddallas_placetypes_2024.parquet", ["OBJECTID", "FUTUREPT"], workers=4)
    zon = fetch_layer(ZONING_URL, RAW / "city" / "base_zoning.parquet", ["OBJECTID", "ZONE_DIST", "LONG_ZONE_DIST", "PD_NUM"])
    council = fetch_layer(COUNCIL_URL, RAW / "city" / "council_districts.parquet", ["*"])
    d = _dcad(download(DCAD_URL, RAW / "dcad" / "DCAD2026_CURRENT.zip"))
    n_records = len(g)
    p = g[g.geometry.notna()].to_crs(crs).reset_index(drop=True)
    p["geometry"] = p.geometry.make_valid().map(_polygonal)
    p = p[~p.geometry.is_empty].reset_index(drop=True)
    n = len(p)
    rp = p.geometry.representative_point()

    # DCAD accounts -> parcels by GIS parcel id
    acc = d["accounts"]
    lot_of = pd.Series(np.arange(n), index=p.GIS_ACCT.astype(str).str.strip()).groupby(level=0).first()
    acc["lot"] = acc.GIS_PARCEL_ID.astype(str).str.strip().map(lot_of)
    acc = acc.dropna(subset=["lot"]).astype({"lot": int}).set_index("ACCOUNT_NUM").lot
    res, com, val = d["res"], d["com"], d["values"]
    for t in (res, com, val):
        t["lot"] = t.ACCOUNT_NUM.map(acc).fillna(-1).astype(int)
    num = lambda s: pd.to_numeric(s, errors="coerce")
    res["sf"], res["yr"], res["st"] = num(res.TOT_LIVING_AREA_SF).fillna(0), num(res.YR_BUILT), res.NUM_STORIES_DESC.map(_stories)
    res["units"] = np.where(res.BLDG_CLASS_DESC.astype(str).str.upper().eq("LAND ONLY") | (res.sf <= 0), 0,
                            num(res.NUM_KITCHENS).fillna(1).clip(lower=1))
    com["gba"], com["yr"], com["st"] = num(com.GROSS_BLDG_AREA).fillna(0), num(com.YEAR_BUILT), num(com.NUM_STORIES)
    # NUM_UNITS also counts hotel rooms, storage units and parking spaces: homes only on apartment building classes
    apt = com.BLDG_CLASS_DESC.astype(str).str.upper().map(lambda s: any(w in s for w in APARTMENT))
    com["units"] = np.where(apt, num(com.NUM_UNITS).fillna(0), 0)
    val["tot"], val["land"] = num(val.TOT_VAL).fillna(0), num(val.LAND_VAL).fillna(0)
    by = lambda t, c, how="sum": t[t.lot >= 0].groupby("lot")[c].agg(how).reindex(range(n))
    units = (by(res, "units").fillna(0) + by(com, "units").fillna(0)).to_numpy()
    fa_obs = (by(res, "sf").fillna(0) + by(com, "gba").fillna(0)).to_numpy()
    stories = pd.concat([by(res, "st", "max"), by(com, "st", "max")], axis=1).max(axis=1).to_numpy()
    year = pd.concat([by(res, "yr", "max"), by(com, "yr", "max")], axis=1).max(axis=1).to_numpy()
    value, land = by(val, "tot").fillna(0).to_numpy(), by(val, "land").fillna(0).to_numpy()
    log(f"{n:,} parcels; DCAD: {int((res.lot >= 0).sum() + (com.lot >= 0).sum()):,} building records placed, "
        f"{int(units.sum()):,} homes")

    # three layers: existing use (city survey refined by DCAD), plan (forwardDallas placetype), zoning (base district)
    lu16 = np.array([TAX_INDEX.get(crosswalk.dallas(lu, v, bc, u), -1)
                     for lu, v, bc, u in zip(p.Existing_Land_Use, p.Vacant, p.BLDG_CL, units)], dtype="int16")
    use_a = np.array([group_of(c) for c in lu16])
    pv = plan.to_crs(crs)
    i, k = shapely.STRtree(pv.geometry.values).query(rp.values, predicate="within")
    hits = pd.DataFrame({"i": i, "k": k, "area": pv.geometry.area.to_numpy()[k]}).sort_values(["i", "area"]).drop_duplicates("i")
    placetype = pd.Series(pv.FUTUREPT.astype(str).to_numpy()[hits.k.to_numpy()], index=hits.i.to_numpy()).reindex(range(n)).fillna("")
    plan16 = np.array([TAX_INDEX.get(crosswalk.dallas_plan(t), -1) for t in placetype], dtype="int16")
    zv = zon.to_crs(crs)
    i, k = shapely.STRtree(zv.geometry.values).query(rp.values, predicate="within")
    zh = pd.Series(k, index=i).groupby(level=0).first()
    zone = pd.Series(zv.ZONE_DIST.astype(str).to_numpy()[zh.to_numpy()], index=zh.index).reindex(range(n)).fillna("NA")
    rules = zone.map(zone_rule)
    log(f"existing use {int((lu16 >= 0).sum()):,}, plan {int((plan16 >= 0).sum()):,}, zoning {int((zone != 'NA').sum()):,} parcels")

    log("Overture buildings (parcels without an appraisal record)")
    fa_ov, fl_ov = overture_buildings(GEOID, p.geometry.values, crs)
    built = np.array([c >= 0 and CLASSES[c].built for c in lu16])
    gap = (fa_obs <= 0) & (fa_ov > 0) & built
    lot = p.geometry.area.to_numpy()
    fa, n_fa_dropped = plausible_floor_area(np.where(gap, fa_ov, fa_obs), lot)
    floors = np.where(~np.isnan(stories) & (stories > 0), stories, np.where(~np.isnan(fl_ov), fl_ov, np.where(fa > 0, 1, 0)))

    bnd = places.boundary(GEOID)
    blocks = census.blocks(GEOID, bnd.bounds)
    inside = blocks.representative_point().within(bnd).to_numpy()
    blocks = blocks[(blocks.AREALAND > 0).to_numpy() & inside].to_crs(crs).reset_index(drop=True)
    bt = shapely.STRtree(blocks.geometry.values)
    i, k = bt.query(rp.values, predicate="within")
    blk = np.full(n, -1)
    blk[i] = k
    miss = blk < 0
    if miss.any():
        blk[miss] = bt.nearest(rp.values[miss])
    jobs_block = census.lodes_jobs("48", "TX").reindex(blocks.GEOID).fillna(0).to_numpy()
    cv = council.to_crs(crs)
    dcol = next(c for c in cv.columns if c.upper() in ("DISTRICT", "COUNCIL", "DIST", "COUNCIL_DISTRICT", "DISTRICT_NUM"))
    i, k = shapely.STRtree(cv.geometry.values).query(rp.values, predicate="within")
    cd = np.zeros(n, dtype="int32")
    cd[i] = pd.to_numeric(cv[dcol], errors="coerce").fillna(0).astype(int).to_numpy()[k]
    if (cd == 0).any():
        cd[cd == 0] = pd.to_numeric(cv[dcol], errors="coerce").fillna(0).astype(int).to_numpy()[
            shapely.STRtree(cv.geometry.values).nearest(rp.values[cd == 0])]

    k16 = lambda key: lu16 == TAX_INDEX[key]
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], MIX_RES * fa, 0.0))
    nonres = fa - res_area
    mix = use_a == U["mix"]
    office = np.where(k16("off"), nonres, np.where(mix, 0.5 * nonres, 0.0))
    retail = np.where(k16("ret"), nonres, np.where(mix, 0.3 * nonres, 0.0))
    other = np.where(k16("hot"), nonres, np.where(mix, 0.2 * nonres, 0.0))
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    df = pd.DataFrame({
        "bbl": np.arange(n, dtype="int64"), "address": p.Address.fillna("").astype(str).str.split().str.join(" ").to_numpy(),
        "cd": cd, "block": blk.astype("int32"), "zone": zone.to_numpy(), "overlay": "", "special": "",
        "bldg_class": p.BLDG_CL.fillna("").astype(str).to_numpy(), "use": use_a.astype("int16"), "lu16": lu16,
        "lu_code": p.Existing_Land_Use.fillna("").astype(str).to_numpy(), "plan16": plan16,
        "plan_designation": placetype.to_numpy(), "zone16": np.array([TAX_INDEX.get(r[1], -1) for r in rules], dtype="int16"),
        "lot_area": lot, "bldg_area": fa, "res_area": res_area, "com_area": office + retail + other,
        "office_area": office, "retail_area": retail, "other_area": other,
        "factory_area": np.where(k16("ind"), nonres, 0.0), "storage_area": np.where(k16("whs"), nonres, 0.0),
        "garage_area": 0.0, "num_bldgs": (fa > 0).astype(float), "floors": floors, "units_res": units,
        "units_total": units, "assess_land": land, "assess_tot": value, "year_built": np.nan_to_num(year),
        "built_far": np.divide(fa, lot, out=np.zeros(n), where=lot > 0), "historic": False, "landmark": False,
        "x": rp.x.to_numpy(), "y": rp.y.to_numpy(), "lon": ll.x.to_numpy(), "lat": ll.y.to_numpy(),
        "height_m": floors * FLOOR_HEIGHT_M, "census_block": blocks.GEOID.to_numpy()[blk],
    })
    bg = df.census_block.str[:12]
    has = df.bldg_area > 0
    ctx = df[has].groupby(bg[has]).built_far.quantile(0.9)
    df["max_far"] = np.maximum(bg.map(ctx).fillna(df.built_far[has].quantile(0.9)).to_numpy(), df.built_far.to_numpy())
    df["resid_far"] = df["comm_far"] = df["facil_far"] = df["max_far"]
    df["allowed"] = np.array([r[0] for r in rules], dtype="int64")
    hu = float(blocks.HU100.sum())
    meta = finish_twin(df, p.geometry.values, crs, out, "TX", float(blocks.POP100.sum() / max(hu, 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "City of Dallas Existing Land Use (2026), forwardDallas! 2.0 placetypes (2024), base zoning, "
                  "and Dallas Central Appraisal District 2026 certified data",
        "source_url": LAND_USE_URL,
        "data_quality": {"missing_land_use": int((lu16 < 0).sum()), "missing_plan": int((plan16 < 0).sum()),
                         "missing_zoning_class": int((df.zone16 < 0).sum()), "records": int(n_records),
                         "parcels_without_dcad_record": int((by(val, "tot").isna()).sum()),
                         "floor_area_from_overture": int(gap.sum()), "floor_area_implausible_dropped": n_fa_dropped,
                         "homes_dcad": int(units.sum()), "homes_census_2020": int(hu),
                         "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (City of Dallas parcels)",
                    "land use": "observed (City of Dallas existing land use; DCAD building classes and units refine it)",
                    "adopted plan": "observed (forwardDallas! 2.0 placetypes, adopted 2024)",
                    "zoning": "observed base district; simplified rules (district family sets allowed uses; context FAR)",
                    "floor area": "observed (DCAD living and gross building area); estimated from Overture footprints "
                                  "where there is no DCAD record (parcels outside Dallas County, exempt property)",
                    "floors and height": "observed (DCAD stories), else Overture", "homes": "observed (DCAD)",
                    "jobs": "observed by block (LODES), modelled by floor area",
                    "assessed value": "observed (DCAD market value)", "year built": "observed",
                    "landmarks and historic districts": "missing", "mixed-use split": "assumed 70% residential"},
        "missing_indicators": [], "zoning_rules": "Dallas Development Code district families (simplified)",
        "levels": {str(c): f"Council District {c}" for c in sorted(df.cd.unique())},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {n:,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "class_counts", "data_quality", "build_seconds")}, indent=1))
