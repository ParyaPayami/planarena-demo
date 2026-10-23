"""Adapter: City of San Antonio, TX, from the Bexar Appraisal District parcels and the city's plan and zoning.

Sources:
- Bexar Appraisal District (BCAD) parcels, December 2025 export, as published by the San Antonio River Authority:
  the Texas property class of every parcel (the existing use), building square feet, units, year built and market
  value. Owner, agent, exemption and deed fields are never requested (exemptions carry personal information).
- City of San Antonio: Future Land Use (the comprehensive plan's land-use categories of every adopted area plan),
  base zoning (Unified Development Code), and the 2022 council districts (game levels).
The appraisal classes do not split commercial property (retail, office, hotel are one class, F1) and its code table
for property use is not published, so commercial parcels are shown as retail and services and the report says so;
nothing is estimated in their place. Parcels whose use the appraisal district does not record (State_cd missing) are
unknown.

Run:  python -I -m planarena.cities.adapters.sanantonio
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
from planarena.sources import crosswalk
from planarena.sources.arcgis import _ids, fetch_layer
from planarena.taxonomy import CLASSES, INDEX as TAX_INDEX, group_of
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "4865000"
RAW = DATA / "raw" / "sanantonio"
BCAD_URL = "https://gis.sara-tx.org/ags1/rest/services/FW_Bexar/BCAD_Parcels_PROD/FeatureServer/0"
COSA = "https://services.arcgis.com/g1fRTDLeMgspWrYp/arcgis/rest/services"
PLAN_URL = f"{COSA}/Future_Land_Use/FeatureServer/4"
ZONING_URL = f"{COSA}/COSA_Zoning/FeatureServer/12"
COUNCIL_URL = f"{COSA}/RedistrictedCouncilDistricts2022/FeatureServer/0"
FIELDS = ["OBJECTID", "Prop_id", "Prop_yr", "Geo_id", "Yr_blt", "Sq_ft", "Land_acres", "State_cd", "Market_val",
          "Land_hstd_val", "Land_non_hstd_val", "City", "Situs", "Property_use_cd", "Num_Units"]
MIX_RES = 0.7  # assumption, as for the other cities

ALL = (1 << len(USE_KEYS)) - 1
bit = lambda *keys: sum(1 << U[k] for k in keys)
ALLOWED = {"sf": bit("res", "pub", "opn", "vac"), "mf_small": bit("res", "pub", "opn", "vac"),
           "mf": bit("res", "pub", "opn", "prk", "vac"), "ret": bit("mix", "com", "pub", "opn", "prk", "vac"),
           "off": bit("mix", "com", "pub", "opn", "prk", "vac"), "mix": bit("res", "mix", "com", "pub", "opn", "prk", "vac"),
           "ind": bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "opn": bit("opn", "res", "pub", "vac"),
           "tra": bit("tra", "prk", "opn", "vac")}


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    crs = local_crs(pl["lat"], pl["lon"])
    log("downloading BCAD parcels, the city's future land use, zoning and council districts")
    g = fetch_layer(BCAD_URL, RAW / "bcad" / "bcad_parcels_2025_cosa.parquet", FIELDS,
                    where="City='CITY OF SAN ANTONIO'", workers=4)
    plan = fetch_layer(PLAN_URL, RAW / "city" / "future_land_use.parquet", ["OBJECTID", "PlanName", "LandUse"], workers=4)
    zon = fetch_layer(ZONING_URL, RAW / "city" / "zoning.parquet", ["OBJECTID", "Base", "Zoning", "SpecDistrict"],
                      where="Base <> 'OCL'", workers=4)
    council = fetch_layer(COUNCIL_URL, RAW / "city" / "council_districts_2022.parquet", ["*"])
    # totally exempt property (public, charitable, religious): only the ids are requested, never the exemptions field
    ids_file = RAW / "bcad" / "exempt_ids.json"
    if not ids_file.exists():
        ids_file.write_text(json.dumps([i for c in _ids(BCAD_URL, "City='CITY OF SAN ANTONIO' AND Exemptions LIKE 'EX%'")[2] for i in c]))
    exempt_ids = set(json.loads(ids_file.read_text()))
    n_records = len(g)
    g = g[g.geometry.notna()].to_crs(crs)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    p = g[~g.geometry.is_empty].reset_index(drop=True)
    n = len(p)
    num = lambda c: pd.to_numeric(p[c], errors="coerce")
    sc = p.State_cd.fillna("").astype(str).str.strip().str.upper()
    units_rec = num("Num_Units").fillna(0)
    lu16 = np.array([TAX_INDEX.get(crosswalk.san_antonio(s, u), -1) for s, u in zip(sc, units_rec)], dtype="int16")
    # exempt buildings are institutions (schools, churches, government, hospitals: civic until facility registries
    # split them); exempt land without a building keeps its class (vacant)
    exempt = p.OBJECTID.isin(exempt_ids).to_numpy()
    exempt_built = exempt & (num("Sq_ft").fillna(0).to_numpy() > 0) & ~np.isin(lu16, [TAX_INDEX["sf"], TAX_INDEX["mf_small"], TAX_INDEX["mf"]])
    lu16[exempt_built] = TAX_INDEX["civ"]
    use_a = np.array([group_of(c) for c in lu16])
    rp = p.geometry.representative_point()

    pv = plan.to_crs(crs)
    i, k = shapely.STRtree(pv.geometry.values).query(rp.values, predicate="within")
    hits = pd.DataFrame({"i": i, "k": k, "area": pv.geometry.area.to_numpy()[k]}).sort_values(["i", "area"]).drop_duplicates("i")
    designation = pd.Series(pv.LandUse.astype(str).to_numpy()[hits.k.to_numpy()], index=hits.i.to_numpy()).reindex(range(n)).fillna("")
    plan16 = np.array([TAX_INDEX.get(crosswalk.san_antonio_plan(d), -1) for d in designation], dtype="int16")
    zv = zon.to_crs(crs)
    i, k = shapely.STRtree(zv.geometry.values).query(rp.values, predicate="within")
    zh = pd.DataFrame({"i": i, "k": k, "area": zv.geometry.area.to_numpy()[k]}).sort_values(["i", "area"]).drop_duplicates("i")
    base = pd.Series(zv.Base.astype(str).str.strip().to_numpy()[zh.k.to_numpy()], index=zh.i.to_numpy()).reindex(range(n)).fillna("NA")
    zfull = pd.Series(zv.Zoning.astype(str).to_numpy()[zh.k.to_numpy()], index=zh.i.to_numpy()).reindex(range(n)).fillna("NA")
    zone_cls = base.map(crosswalk.san_antonio_zone)
    zone16 = np.array([TAX_INDEX.get(c, -1) for c in zone_cls], dtype="int16")
    log(f"{n_records:,} BCAD parcels; existing use {int((lu16 >= 0).sum()):,}, plan {int((plan16 >= 0).sum()):,}, "
        f"zoning class {int((zone16 >= 0).sum()):,}")

    lot = p.geometry.area.to_numpy()
    fa_obs = num("Sq_ft").fillna(0).to_numpy().astype(float)
    log("Overture buildings (exempt and unrecorded buildings)")
    fa_ov, fl_ov = overture_buildings(GEOID, p.geometry.values, crs)
    built = np.array([c >= 0 and CLASSES[c].built for c in lu16])
    gap = (fa_obs <= 0) & (fa_ov > 0) & built
    fa, n_fa_dropped = plausible_floor_area(np.where(gap, fa_ov, fa_obs), lot)
    floors = np.where(fa > 0, np.nan_to_num(fl_ov, nan=1.0), 0.0)

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
    # homes: houses one each; duplexes and apartments by the appraisal district's unit count where it gives one;
    # apartment parcels without a count get the rest of their block's 2020 housing units by floor area (estimated)
    plex = np.where(sc.eq("B2"), np.maximum(units_rec, 2), units_rec)
    observed = np.where(np.isin(lu16, [TAX_INDEX["sf"]]), 1.0,
                        np.where(np.isin(lu16, [TAX_INDEX["mf_small"], TAX_INDEX["mf"]]), plex, 0.0))
    need = np.isin(lu16, [TAX_INDEX["mf_small"], TAX_INDEX["mf"]]) & (observed <= 0)
    w = np.where(need, np.where(fa > 0, fa, lot * 0.25), 0.0)
    left = np.clip(blocks.HU100.to_numpy() - pd.Series(observed).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy(), 0, None)
    tot = pd.Series(w).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy()
    est = np.divide(w * left[blk], tot[blk], out=np.zeros(n), where=tot[blk] > 0)
    units = observed + round_by_block(est, blk)
    log(f"homes: {observed.sum():,.0f} observed, {units.sum() - observed.sum():,.0f} estimated; Census 2020: {int(blocks.HU100.sum()):,}")

    cv = council.to_crs(crs)
    dcol = next(c for c in cv.columns if c.upper() in ("DISTRICT", "COUNCIL", "DIST", "COUNCIL_DISTRICT", "DISTRICT_NUM", "DISTRICTS"))
    i, k = shapely.STRtree(cv.geometry.values).query(rp.values, predicate="within")
    cd = np.zeros(n, dtype="int32")
    cd[i] = pd.to_numeric(cv[dcol], errors="coerce").fillna(0).astype(int).to_numpy()[k]
    if (cd == 0).any():
        cd[cd == 0] = pd.to_numeric(cv[dcol], errors="coerce").fillna(0).astype(int).to_numpy()[
            shapely.STRtree(cv.geometry.values).nearest(rp.values[cd == 0])]

    k16 = lambda key: lu16 == TAX_INDEX[key]
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], MIX_RES * fa, 0.0))
    nonres = fa - res_area
    retail = np.where(k16("ret"), nonres, 0.0)
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    land = (num("Land_hstd_val").fillna(0) + num("Land_non_hstd_val").fillna(0)).to_numpy()
    df = pd.DataFrame({
        "bbl": num("Prop_id").fillna(0).astype("int64").to_numpy(), "address": p.Situs.fillna("").astype(str).str.split().str.join(" ").to_numpy(),
        "cd": cd, "block": blk.astype("int32"), "zone": zfull.to_numpy(), "overlay": "", "special": "",
        "bldg_class": sc.to_numpy(), "use": use_a.astype("int16"), "lu16": lu16, "lu_code": sc.to_numpy(),
        "plan16": plan16, "plan_designation": designation.to_numpy(), "zone16": zone16,
        "lot_area": lot, "bldg_area": fa, "res_area": res_area, "com_area": retail, "office_area": 0.0,
        "retail_area": retail, "other_area": 0.0, "factory_area": np.where(k16("ind"), nonres, 0.0),
        "storage_area": 0.0, "garage_area": 0.0, "num_bldgs": (fa > 0).astype(float), "floors": floors,
        "units_res": units, "units_total": units, "assess_land": land, "assess_tot": num("Market_val").fillna(0).to_numpy(),
        "year_built": num("Yr_blt").fillna(0).to_numpy(), "built_far": np.divide(fa, lot, out=np.zeros(n), where=lot > 0),
        "historic": False, "landmark": False, "x": rp.x.to_numpy(), "y": rp.y.to_numpy(), "lon": ll.x.to_numpy(),
        "lat": ll.y.to_numpy(), "height_m": floors * FLOOR_HEIGHT_M, "census_block": blocks.GEOID.to_numpy()[blk],
    })
    bg = df.census_block.str[:12]
    has = df.bldg_area > 0
    ctx = df[has].groupby(bg[has]).built_far.quantile(0.9)
    df["max_far"] = np.maximum(bg.map(ctx).fillna(df.built_far[has].quantile(0.9)).to_numpy(), df.built_far.to_numpy())
    df["resid_far"] = df["comm_far"] = df["facil_far"] = df["max_far"]
    df["allowed"] = zone_cls.map(lambda c: ALLOWED.get(c, ALL)).to_numpy().astype("int64")
    hu = float(blocks.HU100.sum())
    meta = finish_twin(df, p.geometry.values, crs, out, "TX", float(blocks.POP100.sum() / max(hu, 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "Bexar Appraisal District parcels (December 2025), City of San Antonio Future Land Use and zoning",
        "source_url": BCAD_URL,
        "data_quality": {"missing_land_use": int((lu16 < 0).sum()), "missing_plan": int((plan16 < 0).sum()),
                         "missing_zoning_class": int((zone16 < 0).sum()), "records": int(n_records),
                         "commercial_not_split": int(k16("ret").sum()), "exempt_parcels": int(exempt.sum()),
                         "exempt_buildings_as_civic": int(exempt_built.sum()),
                         "floor_area_from_overture": int(gap.sum()), "floor_area_implausible_dropped": n_fa_dropped,
                         "homes_observed": int(observed.sum()), "homes_estimated": int(units.sum() - observed.sum()),
                         "homes_census_2020": int(hu), "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (BCAD parcels)",
                    "land use": "observed (BCAD property class); commercial is one class in the source (shown as "
                                "retail and services: office, hotel and medical are not split); exempt buildings "
                                "shown as civic (schools and hospitals not yet split)",
                    "adopted plan": "observed (City of San Antonio Future Land Use, all adopted area plans)",
                    "zoning": "observed base district; simplified rules (district family sets allowed uses; context FAR)",
                    "floor area": "observed (BCAD square feet); estimated from Overture footprints where none is recorded",
                    "floors and height": "estimated (Overture building floors or heights)",
                    "homes": "observed for houses, duplexes and apartments with a unit count; other apartments "
                             "estimated from Census 2020 block housing units",
                    "jobs": "observed by block (LODES), modelled by floor area",
                    "assessed value": "observed (BCAD market value)", "year built": "observed",
                    "landmarks and historic districts": "missing"},
        "missing_indicators": [], "zoning_rules": "San Antonio Unified Development Code district families (simplified)",
        "levels": {str(c): f"Council District {c}" for c in sorted(df.cd.unique())},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {n:,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "class_counts", "data_quality", "build_seconds")}, indent=1))
