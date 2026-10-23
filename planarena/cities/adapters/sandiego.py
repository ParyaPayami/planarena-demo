"""Adapter: City of San Diego, CA, from SANDAG's land use, the county assessor's parcels and the city's plan and zoning.

Sources (SANDAG regional GIS; SanGIS parcels from the San Diego County Assessor):
- SANDAG Land Use 2025: the regional planning agency's annual land use and housing inventory (104 codes). Each parcel
  takes the smallest land-use polygon containing its representative point.
- Parcels in the City of San Diego (situs jurisdiction SD): assessor land use, unit count, living area, roll values and
  community. Owner-occupancy, deed and legal-description fields are never requested.
- City of San Diego: General Plan land use (community plan designations with density ranges), base zoning (Land
  Development Code), and the community planning areas (game levels).
Non-residential floor area is not in the parcel layer: it comes from Overture building footprints and heights
(estimated, flagged); floors likewise.

Run:  python -I -m planarena.cities.adapters.sandiego
"""
from __future__ import annotations

import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.common import finish_twin, local_crs, overture_buildings, plausible_floor_area
from planarena.cities.registry import city_dir, register
from planarena.config import DATA, USE_INDEX as U, USE_KEYS, WEB_CRS
from planarena.national import census, places
from planarena.sources import crosswalk
from planarena.sources.arcgis import fetch_layer
from planarena.taxonomy import CLASSES, INDEX as TAX_INDEX, group_of
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "0666000"
RAW = DATA / "raw" / "sandiego"
SANDAG = "https://geo.sandag.org/server/rest/services/Hosted"
LAND_USE_URL = f"{SANDAG}/Land_Use_2025/FeatureServer/0"
PARCEL_URL = f"{SANDAG}/Parcels/FeatureServer/0"
PLAN_URL = f"{SANDAG}/General_Plan_Land_Use_SD/FeatureServer/0"
ZONING_URL = f"{SANDAG}/Zoning_Base_SD/FeatureServer/0"
COMMUNITY_URL = f"{SANDAG}/Community_Plan_SD/FeatureServer/0"
PARCEL_FIELDS = ["objectid", "apn", "situs_address", "situs_street", "situs_suffix", "asr_landuse", "nucleus_use_cd",
                 "unitqty", "total_lvg_area", "asr_land", "asr_total", "year_effective", "situs_community"]
MIX_RES = 0.7  # assumption, as for the other cities

ALL = (1 << len(USE_KEYS)) - 1
bit = lambda *keys: sum(1 << U[k] for k in keys)
ALLOWED = {"sf": bit("res", "pub", "opn", "vac"), "mf_small": bit("res", "pub", "opn", "vac"),
           "mf": bit("res", "pub", "opn", "prk", "vac"), "ret": bit("mix", "com", "pub", "opn", "prk", "vac"),
           "off": bit("mix", "com", "pub", "opn", "prk", "vac"), "hot": bit("com", "pub", "opn", "prk", "vac"),
           "prk": bit("prk", "opn", "vac"), "mix": bit("res", "mix", "com", "pub", "opn", "prk", "vac"),
           "ind": bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "opn": bit("opn", "pub", "vac")}


def _smallest_under(points, polys) -> pd.Series:
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
    log("downloading parcels, SANDAG land use, the General Plan, zoning and community plan areas")
    g = fetch_layer(PARCEL_URL, RAW / "parcels" / "parcels_sd.parquet", PARCEL_FIELDS, where="situs_juris='SD'", workers=4)
    lu = fetch_layer(LAND_USE_URL, RAW / "sandag" / "land_use_2025_sd_bbox.parquet", ["objectid", "lu", "description"], extra=box)
    plan = fetch_layer(PLAN_URL, RAW / "city" / "general_plan_land_use.parquet",
                       ["objectid", "plan_desc", "plan_name", "density_low", "density_hi", "gp_lu_desc"])
    zon = fetch_layer(ZONING_URL, RAW / "city" / "zoning_base.parquet", ["objectid", "zone_name"])
    cpa = fetch_layer(COMMUNITY_URL, RAW / "city" / "community_plans.parquet", ["objectid", "cpcode", "cpname"])
    n_records = len(g)
    g = g[g.geometry.notna()].to_crs(crs)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    g = g[~g.geometry.is_empty].reset_index(drop=True)
    num = lambda c: pd.to_numeric(g[c], errors="coerce").fillna(0)
    g["units"], g["lvg"], g["tot"], g["land"] = num("unitqty"), num("total_lvg_area"), num("asr_total"), num("asr_land")
    g["yr"] = pd.to_numeric(g.year_effective, errors="coerce")
    g["yr"] = np.where(g.yr < 100, np.where(g.yr <= 26, 2000 + g.yr, 1900 + g.yr), g.yr)  # two-digit effective year
    # condominium units drawn on one polygon become one parcel (units, living area and values summed)
    g["wkb"] = shapely.to_wkb(shapely.normalize(shapely.set_precision(g.geometry.values, 0.01)))
    one = g.drop_duplicates("wkb").reset_index(drop=True)
    agg = g.groupby("wkb", sort=False).agg(units=("units", "sum"), lvg=("lvg", "sum"), tot=("tot", "sum"),
                                            land=("land", "sum"), yr=("yr", "max"), records=("apn", "size")).reindex(one.wkb)
    p = gpd.GeoDataFrame(one[["apn", "situs_address", "situs_street", "situs_suffix", "situs_community"]].reset_index(drop=True)
                         .assign(**{c: agg[c].to_numpy() for c in agg.columns}), geometry=one.geometry.values, crs=crs)
    n = len(p)
    rp = p.geometry.representative_point()

    lv = lu.to_crs(crs)
    k = _smallest_under(rp, lv)
    lu_code = pd.Series(pd.to_numeric(lv.lu, errors="coerce").to_numpy()[k.to_numpy()], index=k.index).reindex(range(n))
    lu16 = np.array([TAX_INDEX.get(crosswalk.sandag(c, u), -1) for c, u in zip(lu_code, p.units)], dtype="int16")
    use_a = np.array([group_of(c) for c in lu16])
    pv = plan.to_crs(crs)
    k = _smallest_under(rp, pv)
    take = lambda col: pd.Series(pv[col].to_numpy()[k.to_numpy()], index=k.index).reindex(range(n))
    gp_cat, gp_desc, gp_hi = take("gp_lu_desc"), take("plan_desc").fillna(""), take("density_hi")
    plan16 = np.array([TAX_INDEX.get(crosswalk.san_diego_plan(c, d), -1) for c, d in zip(gp_cat, gp_hi)], dtype="int16")
    zv = zon.to_crs(crs)
    k = _smallest_under(rp, zv)
    zone = pd.Series(zv.zone_name.astype(str).to_numpy()[k.to_numpy()], index=k.index).reindex(range(n)).fillna("NA")
    zcls = zone.map(crosswalk.san_diego_zone)
    zone16 = np.array([TAX_INDEX.get(c, -1) for c in zcls], dtype="int16")
    log(f"{n_records:,} parcel records -> {n:,} lots; existing use {int((lu16 >= 0).sum()):,}, "
        f"plan {int((plan16 >= 0).sum()):,}, zoning class {int((zone16 >= 0).sum()):,}")

    log("Overture buildings (non-residential floor area and floors)")
    fa_ov, fl_ov = overture_buildings(GEOID, p.geometry.values, crs)
    lot = p.geometry.area.to_numpy()
    lvg = p.lvg.to_numpy().astype(float)
    built = np.array([c >= 0 and CLASSES[c].built for c in lu16])
    res_like = np.isin(use_a, [U["res"]])
    # residential: the assessor's living area; everything else (and homes without one): Overture footprints x floors
    fa_obs = np.where(res_like & (lvg > 0), lvg, 0.0)
    gap = (fa_obs <= 0) & (fa_ov > 0) & built
    fa, n_fa_dropped = plausible_floor_area(np.where(gap, fa_ov, fa_obs), lot)
    floors = np.where(fa > 0, np.nan_to_num(fl_ov, nan=1.0), 0.0)
    # the assessor's unit count also covers hotel rooms, storage units and other non-dwelling units: homes are
    # counted on residential and mixed-use parcels only
    units = np.where(np.isin(use_a, [U["res"], U["mix"]]), p.units.to_numpy().astype(float), 0.0)

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
    jobs_block = census.lodes_jobs("06", "CA").reindex(blocks.GEOID).fillna(0).to_numpy()
    cv = cpa.to_crs(crs).reset_index(drop=True)
    i, k = shapely.STRtree(cv.geometry.values).query(rp.values, predicate="within")
    cd = np.zeros(n, dtype="int32")
    cd[i] = k + 1
    if (cd == 0).any():
        cd[cd == 0] = shapely.STRtree(cv.geometry.values).nearest(rp.values[cd == 0]) + 1

    k16 = lambda key: lu16 == TAX_INDEX[key]
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], MIX_RES * fa, 0.0))
    nonres = fa - res_area
    mix = use_a == U["mix"]
    office = np.where(k16("off"), nonres, np.where(mix, 0.5 * nonres, 0.0))
    retail = np.where(k16("ret"), nonres, np.where(mix, 0.3 * nonres, 0.0))
    other = np.where(k16("hot"), nonres, np.where(mix, 0.2 * nonres, 0.0))
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    addr = (p.situs_address.fillna("").astype(str) + " " + p.situs_street.fillna("").astype(str) + " "
            + p.situs_suffix.fillna("").astype(str)).str.split().str.join(" ")
    df = pd.DataFrame({
        "bbl": np.arange(n, dtype="int64"), "address": addr.to_numpy(), "cd": cd, "block": blk.astype("int32"),
        "zone": zone.to_numpy(), "overlay": "", "special": "", "bldg_class": "", "use": use_a.astype("int16"),
        "lu16": lu16, "lu_code": lu_code.fillna(-1).astype(int).astype(str).to_numpy(), "plan16": plan16,
        "plan_designation": gp_desc.astype(str).to_numpy(), "zone16": zone16,
        "lot_area": lot, "bldg_area": fa, "res_area": res_area, "com_area": office + retail + other,
        "office_area": office, "retail_area": retail, "other_area": other,
        "factory_area": np.where(k16("ind"), nonres, 0.0), "storage_area": np.where(k16("whs"), nonres, 0.0),
        "garage_area": 0.0, "num_bldgs": (fa > 0).astype(float), "floors": floors, "units_res": units,
        "units_total": units, "assess_land": p.land.to_numpy(), "assess_tot": p.tot.to_numpy(),
        "year_built": np.nan_to_num(p.yr.to_numpy().astype(float)),
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
    meta = finish_twin(df, p.geometry.values, crs, out, "CA", float(blocks.POP100.sum() / max(hu, 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "SANDAG Land Use 2025, San Diego County Assessor parcels (SanGIS), City of San Diego General Plan and zoning",
        "source_url": LAND_USE_URL,
        "data_quality": {"missing_land_use": int((lu16 < 0).sum()), "missing_plan": int((plan16 < 0).sum()),
                         "missing_zoning_class": int((zone16 < 0).sum()), "records": int(n_records),
                         "stacked_records_merged": int(n_records - n), "floor_area_from_overture": int(gap.sum()),
                         "floor_area_implausible_dropped": n_fa_dropped, "homes_assessor": int(units.sum()),
                         "homes_census_2020": int(hu), "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (San Diego County Assessor parcels)",
                    "land use": "observed (SANDAG Land Use 2025; assessor unit counts split housing)",
                    "adopted plan": "observed (City of San Diego General Plan and community plan land use)",
                    "zoning": "observed base zone; simplified rules (zone family sets allowed uses; context FAR)",
                    "floor area": "observed living area for homes; estimated from Overture footprints for other "
                                  "buildings (the parcel layer has no commercial floor area)",
                    "floors and height": "estimated (Overture building floors or heights)",
                    "homes": "observed (assessor unit counts on residential and mixed-use parcels)",
                    "jobs": "observed by block (LODES), modelled by floor area",
                    "assessed value": "observed (Proposition 13 roll value: below market for long-held property)",
                    "year built": "observed (assessor effective year)", "landmarks and historic districts": "missing",
                    "mixed-use split": "assumed 70% residential"},
        "missing_indicators": [], "zoning_rules": "San Diego Land Development Code base zone families (simplified)",
        "levels": {str(i + 1): str(nm).title() for i, nm in enumerate(cv.cpname)},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {n:,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "class_counts", "data_quality", "build_seconds")}, indent=1))
