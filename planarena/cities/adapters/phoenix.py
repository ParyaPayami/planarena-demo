"""Adapter: City of Phoenix, AZ, from Maricopa County Assessor parcels, MAG land use and the city's plan and zoning.

Sources:
- Maricopa County parcels as published by the City of Phoenix (County Parcels feature layer): the assessor's
  property use code, its group and description for every parcel. Owner fields are never requested.
- MAG (Maricopa Association of Governments) Existing Land Use 2026: the regional planning agency's land use, used
  where the assessor's group describes ownership or tax status rather than use (public, exempt, miscellaneous).
- City of Phoenix: General Plan land use with the city's General Plan code table, zoning (general zone), and the 15
  urban villages (game levels).
The published parcels carry no building area or unit counts: floor area and floors come from Overture building
footprints and heights, and apartment homes from Census 2020 block housing units (both estimated, flagged).

Run:  python -I -m planarena.cities.adapters.phoenix
"""
from __future__ import annotations

import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.common import (finish_twin, local_crs, overture_buildings, plausible_floor_area,
                                     round_by_block)
from planarena.cities.registry import city_dir, register
from planarena.config import DATA, USE_INDEX as U, USE_KEYS, WEB_CRS
from planarena.national import census, places
from planarena.national.common import download
from planarena.sources import crosswalk
from planarena.sources.arcgis import fetch_layer
from planarena.taxonomy import INDEX as TAX_INDEX, group_of
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "0455000"
RAW = DATA / "raw" / "phoenix"
PHX = "https://maps.phoenix.gov/pub/rest/services/Public"
PARCEL_URL = f"{PHX}/COUNTY_PARCELS/MapServer/3"
PLAN_URL = f"{PHX}/GeneralPlan/MapServer/0"
PLAN_CODES_URL = "https://www.arcgis.com/sharing/rest/content/items/d0ccaf3cc9cf4743ba05dc21c3a6aece/data"
ZONING_URL = f"{PHX}/Zoning/MapServer/0"
VILLAGE_URL = f"{PHX}/Villages/MapServer/0"
MAG_URL = "https://services1.arcgis.com/MdyCMZnX1raZ7TS3/arcgis/rest/services/EXLU_2026/FeatureServer/0"
PARCEL_FIELDS = ["OBJECTID", "APN", "ADDRESS", "PROPERTY_USE_CODE", "DESCRIPTION", "PROPERTY_USE_GROUP",
                 "GROUP_DESCRIPTION", "PARCEL_GROUP_CLASS", "PARCEL_GROUP_SUBCLASS"]
MIX_RES = 0.7  # assumption, as for the other cities

ALL = (1 << len(USE_KEYS)) - 1
bit = lambda *keys: sum(1 << U[k] for k in keys)
ALLOWED = {"sf": bit("res", "pub", "opn", "vac"), "mf": bit("res", "pub", "opn", "prk", "vac"),
           "ret": bit("mix", "com", "pub", "opn", "prk", "vac"),
           "ind": bit("com", "ind", "tra", "pub", "opn", "prk", "vac"),
           "mix": bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "opn": bit("opn", "vac")}


def _smallest_under(points, polys) -> pd.Series:
    """Index of the smallest polygon containing each point (overlapping layers), by point position."""
    i, k = shapely.STRtree(polys.geometry.values).query(points.values, predicate="within")
    h = pd.DataFrame({"i": i, "k": k, "a": polys.geometry.area.to_numpy()[k]}).sort_values(["i", "a"]).drop_duplicates("i")
    return pd.Series(h.k.to_numpy(), index=h.i.to_numpy())


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    crs = local_crs(pl["lat"], pl["lon"])
    bnd = places.boundary(GEOID)
    x0, y0, x1, y1 = bnd.bounds
    box = {"geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope", "inSR": "4326",
           "spatialRel": "esriSpatialRelIntersects"}
    log("downloading parcels, MAG land use, the General Plan, zoning and villages")
    g = fetch_layer(PARCEL_URL, RAW / "maricopa" / "parcels_phoenix_bbox.parquet", PARCEL_FIELDS, workers=4, extra=box)
    mag = fetch_layer(MAG_URL, RAW / "mag" / "exlu_2026_phoenix_bbox.parquet", ["OBJECTID", "LONG_DISPLAY_NAME"], extra=box)
    plan = fetch_layer(PLAN_URL, RAW / "city" / "general_plan.parquet", ["OBJECTID", "NEWCODE"])
    codes = pd.read_excel(download(PLAN_CODES_URL, RAW / "city" / "general_plan_codes.xlsx"))
    zon = fetch_layer(ZONING_URL, RAW / "city" / "zoning.parquet", ["OBJECTID", "ZONING", "GEN_ZONE"])
    vil = fetch_layer(VILLAGE_URL, RAW / "city" / "villages.parquet", ["OBJECTID", "NAME"])
    g = g[g.geometry.notna()].to_crs(crs)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    g = g[~g.geometry.is_empty]
    city = gpd.GeoSeries([bnd], crs=4326).to_crs(crs).iloc[0]
    g = g[g.geometry.representative_point().within(city)].reset_index(drop=True)
    n_records = len(g)
    # condominium units drawn on one polygon become one parcel
    g["wkb"] = shapely.to_wkb(shapely.normalize(shapely.set_precision(g.geometry.values, 0.01)))
    one = g.drop_duplicates("wkb").reset_index(drop=True)
    stacked = g.groupby("wkb", sort=False).size().reindex(one.wkb).to_numpy()
    p = gpd.GeoDataFrame(one.drop(columns="wkb"), geometry="geometry", crs=crs)
    n = len(p)
    rp = p.geometry.representative_point()
    mv = mag.to_crs(crs)
    k = _smallest_under(rp, mv)
    mag_use = pd.Series(mv.LONG_DISPLAY_NAME.astype(str).to_numpy()[k.to_numpy()], index=k.index).reindex(range(n))
    grp = p.PROPERTY_USE_GROUP.fillna("").astype(str).str.strip()
    cls = pd.Series([crosswalk.phoenix(gp, d, m) for gp, d, m in zip(grp, p.DESCRIPTION, mag_use)])
    condo = grp.eq("07").to_numpy()
    cls[condo] = np.where(stacked[condo] >= 5, "mf", np.where(stacked[condo] >= 2, "mf_small", "sf"))
    lu16 = np.array([TAX_INDEX.get(c, -1) for c in cls], dtype="int16")
    use_a = np.array([group_of(c) for c in lu16])

    pv = plan.to_crs(crs)
    k = _smallest_under(rp, pv)
    code = pd.Series(pd.to_numeric(pv.NEWCODE, errors="coerce").to_numpy()[k.to_numpy()], index=k.index).reindex(range(n))
    label = dict(zip(pd.to_numeric(codes.NEWCODE, errors="coerce"), codes.DESCRIPTION.astype(str).str.strip()))
    designation = code.map(label).fillna("")
    plan16 = np.array([TAX_INDEX.get(crosswalk.phoenix_plan(d), -1) for d in designation], dtype="int16")
    zv = zon.to_crs(crs)
    k = _smallest_under(rp, zv)
    zone = pd.Series(zv.ZONING.astype(str).to_numpy()[k.to_numpy()], index=k.index).reindex(range(n)).fillna("NA")
    gen = pd.Series(zv.GEN_ZONE.astype(str).to_numpy()[k.to_numpy()], index=k.index).reindex(range(n)).fillna("")
    zcls = gen.map(lambda z: crosswalk.PHOENIX_ZONE.get(z, "unknown"))
    zone16 = np.array([TAX_INDEX.get(c, -1) for c in zcls], dtype="int16")
    log(f"{n_records:,} parcels in the city -> {n:,} lots; existing use {int((lu16 >= 0).sum()):,}, "
        f"plan {int((plan16 >= 0).sum()):,}, zoning class {int((zone16 >= 0).sum()):,}")

    log("Overture buildings (floor area and floors)")
    fa_ov, fl_ov = overture_buildings(GEOID, p.geometry.values, crs)
    lot = p.geometry.area.to_numpy()
    fa, n_fa_dropped = plausible_floor_area(fa_ov, lot)
    floors = np.where(fa > 0, np.nan_to_num(fl_ov, nan=1.0), 0.0)

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
    jobs_block = census.lodes_jobs("04", "AZ").reindex(blocks.GEOID).fillna(0).to_numpy()
    # homes: houses and townhouses one each, condominium units by count; apartments, mobile home parks and parcels
    # with several houses take the rest of their block's 2020 housing units by floor area (estimated)
    observed = np.where(grp.isin(["01", "85"]).to_numpy(), 1.0, np.where(condo, stacked, 0.0))
    need = np.isin(lu16, [TAX_INDEX["mf"], TAX_INDEX["mf_small"], TAX_INDEX["mix"]]) & ~condo
    w = np.where(need, np.where(fa > 0, fa, lot * 0.25), 0.0)
    left = np.clip(blocks.HU100.to_numpy() - pd.Series(observed).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy(), 0, None)
    tot = pd.Series(w).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy()
    est = np.divide(w * left[blk], tot[blk], out=np.zeros(n), where=tot[blk] > 0)
    units = observed + round_by_block(est, blk)
    log(f"homes: {observed.sum():,.0f} observed, {units.sum() - observed.sum():,.0f} estimated; Census 2020: {int(blocks.HU100.sum()):,}")

    vv = vil.to_crs(crs)
    i, k = shapely.STRtree(vv.geometry.values).query(rp.values, predicate="within")
    cd = np.zeros(n, dtype="int32")
    cd[i] = k + 1
    if (cd == 0).any():
        cd[cd == 0] = shapely.STRtree(vv.geometry.values).nearest(rp.values[cd == 0]) + 1
    k16 = lambda key: lu16 == TAX_INDEX[key]
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], MIX_RES * fa, 0.0))
    nonres = fa - res_area
    mix = use_a == U["mix"]
    office = np.where(k16("off"), nonres, np.where(mix, 0.5 * nonres, 0.0))
    retail = np.where(k16("ret"), nonres, np.where(mix, 0.3 * nonres, 0.0))
    other = np.where(k16("hot"), nonres, np.where(mix, 0.2 * nonres, 0.0))
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    df = pd.DataFrame({
        "bbl": np.arange(n, dtype="int64"), "address": p.ADDRESS.fillna("").astype(str).str.split().str.join(" ").to_numpy(),
        "cd": cd, "block": blk.astype("int32"), "zone": zone.to_numpy(), "overlay": "", "special": "",
        "bldg_class": p.PROPERTY_USE_CODE.fillna("").astype(str).to_numpy(), "use": use_a.astype("int16"),
        "lu16": lu16, "lu_code": p.PROPERTY_USE_CODE.fillna("").astype(str).to_numpy(), "plan16": plan16,
        "plan_designation": designation.to_numpy(), "zone16": zone16,
        "lot_area": lot, "bldg_area": fa, "res_area": res_area, "com_area": office + retail + other,
        "office_area": office, "retail_area": retail, "other_area": other,
        "factory_area": np.where(k16("ind"), nonres, 0.0), "storage_area": np.where(k16("whs"), nonres, 0.0),
        "garage_area": 0.0, "num_bldgs": (fa > 0).astype(float), "floors": floors, "units_res": units,
        "units_total": units, "assess_land": 0.0, "assess_tot": 0.0, "year_built": 0.0,
        "built_far": np.divide(fa, lot, out=np.zeros(n), where=lot > 0), "historic": False, "landmark": False,
        "x": rp.x.to_numpy(), "y": rp.y.to_numpy(), "lon": ll.x.to_numpy(), "lat": ll.y.to_numpy(),
        "height_m": floors * FLOOR_HEIGHT_M, "census_block": blocks.GEOID.to_numpy()[blk],
    })
    bg = df.census_block.str[:12]
    has = df.bldg_area > 0
    ctx = df[has].groupby(bg[has]).built_far.quantile(0.9)
    df["max_far"] = np.maximum(bg.map(ctx).fillna(df.built_far[has].quantile(0.9)).to_numpy(), df.built_far.to_numpy())
    df["resid_far"] = df["comm_far"] = df["facil_far"] = df["max_far"]
    df["allowed"] = zcls.map(lambda c: ALLOWED.get(c, ALL)).to_numpy().astype("int64")
    hu = float(blocks.HU100.sum())
    meta = finish_twin(df, p.geometry.values, crs, out, "AZ", float(blocks.POP100.sum() / max(hu, 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "Maricopa County Assessor parcels (property use codes), MAG Existing Land Use 2026, City of Phoenix "
                  "General Plan and zoning",
        "source_url": PARCEL_URL,
        "data_quality": {"missing_land_use": int((lu16 < 0).sum()), "missing_plan": int((plan16 < 0).sum()),
                         "missing_zoning_class": int((zone16 < 0).sum()), "records": int(n_records),
                         "stacked_records_merged": int(n_records - n), "floor_area_implausible_dropped": n_fa_dropped,
                         "homes_observed": int(observed.sum()), "homes_estimated": int(units.sum() - observed.sum()),
                         "homes_census_2020": int(hu), "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (Maricopa County parcels)",
                    "land use": "observed (assessor property use group; MAG 2026 land use for public and exempt parcels)",
                    "adopted plan": "observed (City of Phoenix General Plan land use, with the city's code table)",
                    "zoning": "observed district; simplified rules (general zone sets allowed uses; context FAR)",
                    "floor area": "estimated (Overture building footprints x floors; the published parcels have none)",
                    "floors and height": "estimated (Overture building floors or heights)",
                    "homes": "observed for houses, townhouses and condominiums; apartments estimated from Census 2020 "
                             "block housing units",
                    "jobs": "observed by block (LODES), modelled by floor area",
                    "assessed value": "missing (not in the published parcels)", "year built": "missing",
                    "landmarks and historic districts": "missing", "mixed-use split": "assumed 70% residential"},
        "missing_indicators": ["value"], "zoning_rules": "City of Phoenix general zones (simplified)",
        "levels": {str(i + 1): f"{nm} Village" for i, nm in enumerate(vv.NAME.astype(str).str.title())},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {n:,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "class_counts", "data_quality", "build_seconds")}, indent=1))
