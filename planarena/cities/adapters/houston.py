"""Adapter: City of Houston, TX, from the city's parcel land-use layer.

Source: City of Houston "Land Use (Grouped)" (Planning and Development; derived by city staff from Harris Central
Appraisal District records): parcel polygons inside the city with the appraisal district's land-use code and Texas
property class, the city's land-use group, building area, year built, market and land values, council district and
FEMA flood zone. Owner and mailing fields are never requested.

Houston has no zoning (land use is shaped by deed restrictions and the subdivision rules of Chapter 42, which are not
modelled) and no adopted future land-use map, so both layers are missing and every use is allowed (stated in the app).
Homes: houses count one each, duplexes to fourplexes by their property class (B2-B4), condominium records one each;
the rest of each census block's 2020 housing units go to its apartment and mixed-use parcels by floor area
(estimated, flagged), because the layer has no unit counts.

Run:  python -I -m planarena.cities.adapters.houston
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
from planarena.sources.arcgis import fetch_layer
from planarena.taxonomy import INDEX as TAX_INDEX, group_of
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "4835000"
RAW = DATA / "raw" / "houston"
LAND_USE_URL = "https://mycity2.houstontx.gov/gisweb02/rest/services/HoustonMap/Landuse/MapServer/0"
FIELDS = ["OBJECTID", "TAX_ID", "TAX_YEAR", "SITE_ADDR_1", "STATE_CLASS", "LANDUSE_CD", "LANDUSE_DSCR", "GROUP_DSCR",
          "YR_IMPR", "TOTAL_BUILDING_AREA", "TOTAL_LAND_AREA", "LAND_VALUE", "TOTAL_MARKET_VALUE", "STACKED",
          "COUNCIL_DISTRICT", "FLOOD_ZONE", "FLOOD_SFHA", "PDLandMark"]
PLEX = {"B2": 2, "B3": 3, "B4": 4}
MIX_RES = 0.7  # assumption, as for the other cities


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    crs = local_crs(pl["lat"], pl["lon"])
    log("downloading the City of Houston land-use parcels")
    g = fetch_layer(LAND_USE_URL, RAW / "land_use" / "landuse_grouped_city.parquet", FIELDS,
                    where="COUNCIL_DISTRICT IS NOT NULL", workers=4)
    n_records = len(g)
    g = g[g.geometry.notna()].to_crs(crs)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    g = g[~g.geometry.is_empty].reset_index(drop=True)
    num = lambda c: pd.to_numeric(g[c], errors="coerce")
    g["cls"] = [crosswalk.houston(c, gr, sc) for c, gr, sc in zip(g.LANDUSE_CD, g.GROUP_DSCR, g.STATE_CLASS)]
    sc = g.STATE_CLASS.fillna("").astype(str).str.strip().str.upper()
    # observed homes: houses one each, duplexes to fourplexes by class, condominium records one each
    g["homes"] = np.where(g.LANDUSE_CD.astype(str).eq("1006"), 1,
                          np.where(sc.isin(list(PLEX)), sc.map(PLEX).fillna(0), np.where(g.cls == "sf", 1, 0))).astype(float)
    g["fa"] = num("TOTAL_BUILDING_AREA").fillna(0)
    g["mv"], g["lv"], g["yr"] = num("TOTAL_MARKET_VALUE").fillna(0), num("LAND_VALUE").fillna(0), num("YR_IMPR")
    # stacked records (condominium units drawn on one polygon) become one parcel
    g["wkb"] = shapely.to_wkb(shapely.normalize(shapely.set_precision(g.geometry.values, 0.01)))
    one = g.drop_duplicates("wkb")
    agg = g.groupby("wkb", sort=False).agg(
        homes=("homes", "sum"), fa=("fa", "sum"), mv=("mv", "sum"), lv=("lv", "sum"), yr=("yr", "max"),
        records=("OBJECTID", "size")).reindex(one.wkb)
    cls_mode = g.groupby("wkb", sort=False).cls.agg(lambda x: x.value_counts().index[0]).reindex(one.wkb)
    p = gpd.GeoDataFrame({"address": one.SITE_ADDR_1.fillna("").astype(str).str.split().str.join(" ").to_numpy(),
                          "code": one.LANDUSE_CD.fillna("").astype(str).to_numpy(), "cls": cls_mode.to_numpy(),
                          "state_class": one.STATE_CLASS.fillna("").astype(str).to_numpy(),
                          "district": one.COUNCIL_DISTRICT.fillna("").astype(str).to_numpy(),
                          "flood": one.FLOOD_SFHA.fillna("").astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"]).to_numpy(),
                          "landmark": one.PDLandMark.notna().to_numpy() & one.PDLandMark.fillna("").astype(str).str.strip().ne("").to_numpy(),
                          **{c: agg[c].to_numpy() for c in agg.columns}},
                         geometry=one.geometry.values, crs=crs).reset_index(drop=True)
    n = len(p)
    # condominium polygons: their class follows the number of units stacked on them
    condo = p.code.eq("1006").to_numpy()
    p.loc[condo, "cls"] = np.where(p.homes[condo] >= 5, "mf", np.where(p.homes[condo] >= 2, "mf_small", "sf"))
    lu16 = np.array([TAX_INDEX.get(c, -1) for c in p.cls], dtype="int16")
    use_a = np.array([group_of(c) for c in lu16])
    log(f"{n_records:,} records -> {n:,} parcels; {int((lu16 >= 0).sum()):,} with an official land use")

    rp = p.geometry.representative_point()
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

    lot = p.geometry.area.to_numpy()
    fa, n_fa_dropped = plausible_floor_area(p.fa.to_numpy().astype(float), lot)
    observed = p.homes.to_numpy().astype(float)
    # apartments, mobile home parks and mixed use: the rest of each block's 2020 housing units, by floor area (or lot
    # area where none is recorded); estimated
    apt = (np.isin(lu16, [TAX_INDEX["mf"], TAX_INDEX["mix"]]) & ~condo) | np.isin(p.code, ["4318", "4213"])
    w = np.where(apt, np.where(fa > 0, fa, lot * 0.25), 0.0)
    left = np.clip(blocks.HU100.to_numpy() - pd.Series(observed).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy(), 0, None)
    tot = pd.Series(w).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy()
    est = np.divide(w * left[blk], tot[blk], out=np.zeros(n), where=tot[blk] > 0)
    units = observed + round_by_block(est, blk)
    log(f"homes: {observed.sum():,.0f} observed, {units.sum() - observed.sum():,.0f} estimated on apartment parcels; "
        f"Census 2020: {int(blocks.HU100.sum()):,}")

    k16 = lambda key: lu16 == TAX_INDEX[key]
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], MIX_RES * fa, 0.0))
    nonres = fa - res_area
    mix = use_a == U["mix"]
    office = np.where(k16("off"), nonres, np.where(mix, 0.5 * nonres, 0.0))
    retail = np.where(k16("ret"), nonres, np.where(mix, 0.3 * nonres, 0.0))
    other = np.where(k16("hot"), nonres, np.where(mix, 0.2 * nonres, 0.0))
    # stories are not recorded: the most floors of the parcel's Overture buildings (estimated), else 1 where built
    log("Overture buildings (floors)")
    _, fl_ov = overture_buildings(GEOID, p.geometry.values, crs)
    floors = np.where(fa > 0, np.nan_to_num(fl_ov, nan=1.0), 0.0)
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    letters = sorted(d for d in set(p.district) if d)
    cd = p.district.map({d: i + 1 for i, d in enumerate(letters)}).fillna(0).astype("int32").to_numpy()
    df = pd.DataFrame({
        "bbl": np.arange(n, dtype="int64"), "address": p.address.to_numpy(), "cd": cd, "block": blk.astype("int32"),
        "zone": "none", "overlay": np.where(p.flood, "flood", ""), "special": "",
        "bldg_class": p.state_class.to_numpy(), "use": use_a.astype("int16"), "lu16": lu16,
        "lu_code": p.code.to_numpy(), "lot_area": lot, "bldg_area": fa, "res_area": res_area,
        "com_area": office + retail + other, "office_area": office, "retail_area": retail, "other_area": other,
        "factory_area": np.where(k16("ind"), nonres, 0.0), "storage_area": np.where(k16("whs"), nonres, 0.0),
        "garage_area": 0.0, "num_bldgs": (fa > 0).astype(float), "floors": floors, "units_res": units,
        "units_total": units, "assess_land": p.lv.to_numpy(), "assess_tot": p.mv.to_numpy(),
        "year_built": p.yr.fillna(0).to_numpy(), "built_far": np.divide(fa, lot, out=np.zeros(n), where=lot > 0),
        "historic": False, "landmark": p.landmark.to_numpy(),
        "x": rp.x.to_numpy(), "y": rp.y.to_numpy(), "lon": ll.x.to_numpy(), "lat": ll.y.to_numpy(),
        "height_m": floors * FLOOR_HEIGHT_M, "census_block": blocks.GEOID.to_numpy()[blk],
    })
    bg = df.census_block.str[:12]
    built = df.bldg_area > 0
    ctx = df[built].groupby(bg[built]).built_far.quantile(0.9)
    df["max_far"] = np.maximum(bg.map(ctx).fillna(df.built_far[built].quantile(0.9)).to_numpy(), df.built_far.to_numpy())
    df["resid_far"] = df["comm_far"] = df["facil_far"] = df["max_far"]
    df["allowed"] = (1 << len(USE_KEYS)) - 1  # no zoning
    hu = float(blocks.HU100.sum())
    meta = finish_twin(df, p.geometry.values, crs, out, "TX", float(blocks.POP100.sum() / max(hu, 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "City of Houston Land Use (Grouped), from Harris Central Appraisal District records",
        "source_url": LAND_USE_URL,
        "data_quality": {"missing_land_use": int((lu16 < 0).sum()), "missing_plan": n, "plan_layer": "none (no adopted land-use map)",
                         "zoning_layer": "none (Houston has no zoning)", "records": int(n_records),
                         "stacked_records_merged": int(n_records - n), "floor_area_implausible_dropped": n_fa_dropped,
                         "homes_observed": int(observed.sum()), "homes_estimated": int(units.sum() - observed.sum()),
                         "homes_census_2020": int(hu), "parcels_in_flood_hazard_area": int(p.flood.sum()),
                         "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (City of Houston parcels from HCAD)",
                    "land use": "observed (HCAD land-use code and property class, City of Houston groups)",
                    "adopted plan": "missing (Houston has no adopted land-use map)",
                    "zoning": "none: Houston has no zoning; deed restrictions are not modelled",
                    "floor area": "observed (HCAD building area)",
                    "floors and height": "estimated (Overture building floors or heights; stories are not recorded)",
                    "homes": "observed for houses, duplexes to fourplexes and condominiums; apartments estimated from "
                             "Census 2020 block housing units",
                    "jobs": "observed by block (LODES), modelled by floor area",
                    "assessed value": "observed (HCAD market value)", "year built": "observed",
                    "flood hazard": "observed (FEMA special flood hazard area flag)",
                    "landmarks and historic districts": "observed (city landmark flag)",
                    "mixed-use split": "assumed 70% residential"},
        "missing_indicators": [], "zoning_rules": "none (Houston has no zoning)",
        "levels": {str(i + 1): f"Council District {d}" for i, d in enumerate(letters)},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {n:,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "class_counts", "data_quality", "build_seconds")}, indent=1))
