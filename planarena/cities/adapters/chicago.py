"""Adapter: City of Chicago, IL, from CMAP's Land Use Inventory, Cook County parcels and assessor records.

Sources:
- CMAP (Chicago Metropolitan Agency for Planning) Land Use Inventory 2023: the regional planning agency's land-use
  polygons (neighbouring parcels with the same use are merged). Each parcel takes the smallest inventory polygon that
  contains its representative point, as in Philadelphia.
- Cook County 2025 parcels (base parcels in the City of Chicago).
- Cook County Assessor (Cook County Open Data):
  - Parcel Universe 2026: class and location of every property identification number (PIN);
  - residential improvement characteristics 2026: building square feet and apartments of class 2 buildings;
  - condominium unit characteristics 2026: units, and parking and common-area flags;
  - commercial valuation 2024 (the city's last reassessment): building square feet, units and property type of
    classes 3 (apartments of 7+ units) and 5 (commercial and industrial);
  - assessed values 2025 (board of review).
  Owner and taxpayer columns are never requested.
- City of Chicago: current zoning districts and the 77 community areas (game levels).
- Exempt property (schools, universities, public housing, government) has no building record: its floor area is
  estimated from Overture building footprints and heights, flagged. Chicago has no citywide adopted land-use plan, so
  the plan layer is missing (stated in the app).

Run:  python -I -m planarena.cities.adapters.chicago
"""
from __future__ import annotations

import json
import re
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
from planarena.sources.socrata import fetch, fetch_geojson
from planarena.taxonomy import CLASSES, INDEX as TAX_INDEX, group_of
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "1714000"
RAW = DATA / "raw" / "chicago"
LUI_URL = "https://services5.arcgis.com/LcMXE3TFhi1BSaCY/arcgis/rest/services/LUI_2023_view/FeatureServer/1"
PARCEL_URL = "https://gis.cookcountyil.gov/traditional/rest/services/parcelHistorical/MapServer/2025"
COOK = "datacatalog.cookcountyil.gov"
CHICAGO = "data.cityofchicago.org"
TOWNSHIPS = "('70','71','72','73','74','75','76','77')"  # the eight city townships
TOWNSHIP_NAMES = ("('Hyde Park','Jefferson','Lake','Lake View','North Chicago','Rogers Park','South Chicago',"
                  "'West Chicago')")


def sources() -> dict:
    """Every input, downloaded once and cached under data/raw/chicago (with SOURCE.json provenance)."""
    s = {}
    s["lui"] = fetch_layer(LUI_URL, RAW / "cmap" / "lui2023_cook.parquet",
                           ["OBJECTID", "COUNTY", "LANDUSE", "LANDUSE2", "FAC_NAME", "OS_MGMT", "MODIFIER"],
                           where="COUNTY='031' AND LANDUSE <> '6000'")  # 6000: the street network, not parcels
    # base parcels, and condominium parcels (the land under a condominium building carries the condominium's PIN)
    s["parcels"] = fetch_layer(PARCEL_URL, RAW / "cook" / "parcels2025_chicago_base_condo.parquet",
                               ["OBJECTID", "Name", "PIN10", "AssessorBLDGclass", "PARCELTYPE", "Ward"],
                               where="MUNICIPALITY='CITY OF CHICAGO' AND PARCELTYPE IN ('BaseParcel','CondominiumParcel')",
                               workers=4)
    s["universe"] = fetch(COOK, "pabr-t5kh", RAW / "cook" / "parcel_universe_2026.parquet",
                          ["pin", "pin10", "year", "class", "lon", "lat", "chicago_community_area_num"],
                          where="year='2026' AND cook_municipality_name='CITY OF CHICAGO'")
    s["res"] = fetch(COOK, "x54s-btds", RAW / "cook" / "residential_chars_2026.parquet",
                     ["pin", "card", "class", "char_bldg_sf", "char_apts", "char_yrblt"],
                     where=f"year=2026 AND township_code in {TOWNSHIPS}")
    s["condo"] = fetch(COOK, "3r7i-mrz4", RAW / "cook" / "condo_chars_2026.parquet",
                       ["pin", "pin10", "card", "class", "char_unit_sf", "char_building_sf", "char_building_pins",
                        "char_yrblt", "bldg_is_mixed_use", "is_parking_space", "is_common_area"],
                       where=f"year=2026 AND township_code in {TOWNSHIPS}")
    s["commercial"] = fetch(COOK, "csik-bsws", RAW / "cook" / "commercial_valuation_2024.parquet",
                            ["keypin", "pins", "class_es", "tot_units", "bldgsf", "stories", "yearbuilt",
                             "property_type_use", "category"],
                            where=f"year='2024' AND township in {TOWNSHIP_NAMES}")
    s["values"] = fetch(COOK, "uzyt-m557", RAW / "cook" / "assessed_values_2025.parquet",
                        ["pin", "class", "board_tot", "certified_tot", "board_land", "certified_land"],
                        where=f"year='2025' AND township_code in {TOWNSHIPS}")
    s["zoning"] = fetch_geojson(CHICAGO, "dj47-wfun", RAW / "city" / "zoning_districts.parquet")
    s["areas"] = fetch_geojson(CHICAGO, "igwz-8jzy", RAW / "city" / "community_areas.parquet")
    return s


ALL = (1 << len(USE_KEYS)) - 1
bit = lambda *keys: sum(1 << U[k] for k in keys)
# zoning district family -> allowed analysis groups and the district's own use class (Chicago Zoning Ordinance,
# Title 17: residential RS/RT/RM, business B, commercial C, downtown D, manufacturing M, planned manufacturing PMD,
# parks POS, transportation T; simplified: B1, B3, C1 and C2 allow homes above the ground floor only, so mixed use).
# Planned developments (PD) follow their own ordinance and are not screened.
ZONING = [
    ("RS", bit("res", "pub", "opn", "vac"), "sf"), ("RT", bit("res", "pub", "opn", "vac"), "mf_small"),
    ("RM", bit("res", "pub", "opn", "prk", "vac"), "mf"),
    ("B2", bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "mix"),
    ("B1", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"), ("B3", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"),
    ("C1", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"), ("C2", bit("mix", "com", "pub", "opn", "prk", "vac"), "ret"),
    ("C3", bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "ret"),
    ("DC", bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "off"),
    ("DX", bit("res", "mix", "com", "pub", "opn", "prk", "vac"), "mix"),
    ("DR", bit("res", "mix", "pub", "opn", "prk", "vac"), "mf"),
    ("DS", bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
    ("PMD", bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
    ("M1", bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
    ("M2", bit("com", "ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
    ("M3", bit("ind", "tra", "pub", "opn", "prk", "vac"), "ind"),
    ("POS", bit("opn", "pub", "vac"), "opn"), ("T", bit("tra", "prk", "opn", "vac"), "tra"),
]
SF_CLASSES = {"202", "203", "204", "205", "206", "207", "208", "209", "210", "234", "278", "295"}
APTS = {"Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6}
LEVEL_OF_ASSESSMENT = {"5": 0.25}  # Cook County: commercial and industrial 25%; residential and others 10%


def zone_rule(zone: str) -> tuple[int, str]:
    z = zone.upper().strip() if isinstance(zone, str) else ""
    for prefix, mask, cls in ZONING:
        if re.match(prefix + r"(?![A-Z])", z):
            return mask, cls
    return ALL, "unknown"


def _digits(pin) -> str:
    return re.sub(r"\D", "", str(pin)) if pin is not None else ""


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    crs = local_crs(pl["lat"], pl["lon"])
    log("downloading sources")
    s = sources()
    g = s["parcels"][s["parcels"].geometry.notna()].to_crs(crs)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    g = g[~g.geometry.is_empty]
    base = g[g.PARCELTYPE.eq("BaseParcel")]
    condo = g[g.PARCELTYPE.eq("CondominiumParcel")]
    # a condominium parcel is a lot of its own unless it lies on a base parcel
    on_base = shapely.STRtree(base.geometry.values).query(condo.geometry.representative_point().values, predicate="within")[0]
    condo = condo[~np.isin(np.arange(len(condo)), on_base)].drop_duplicates("PIN10")
    p = pd.concat([base, condo], ignore_index=True)
    p["pin10"] = p.PIN10.map(_digits).str[:10]
    rp = p.geometry.representative_point()
    log(f"{len(base):,} base parcels and {len(condo):,} condominium parcels")

    # official land use: the smallest CMAP inventory polygon containing each parcel's representative point
    lui = s["lui"].to_crs(crs)
    i, k = shapely.STRtree(lui.geometry.values).query(rp.values, predicate="within")
    hits = pd.DataFrame({"i": i, "k": k, "area": lui.geometry.area.to_numpy()[k]}).sort_values(["i", "area"])
    under = hits.drop_duplicates("i").set_index("i").k
    p["lui"] = pd.Series(lui.LANDUSE.astype(str).to_numpy()[under.to_numpy()], index=under.index).reindex(p.index)

    # every assessor PIN -> its base parcel: by PIN10, else by its point
    lot_of = pd.Series(np.arange(len(p)), index=p.pin10).groupby(level=0).first()
    uni = s["universe"].copy()
    uni["pin"] = uni.pin.map(_digits)
    uni["lot"] = uni.pin.str[:10].map(lot_of).fillna(-1).astype(int)
    for c in ("lon", "lat"):
        uni[c] = pd.to_numeric(uni[c], errors="coerce")
    need = (uni.lot < 0) & uni.lon.notna() & uni.lat.notna()
    pts = gpd.GeoSeries(gpd.points_from_xy(uni.lon[need], uni.lat[need]), crs=4326).to_crs(crs)
    i, k = shapely.STRtree(p.geometry.values).query(pts.values, predicate="within")
    hit = pd.Series(k, index=i).groupby(level=0).first()
    uni.loc[uni.index[need.to_numpy()][hit.index], "lot"] = hit.to_numpy()
    lot_pin = uni.set_index("pin").lot
    lot_of_pin = lambda pins: pd.Series(pins).map(lot_pin).fillna(pd.Series(pins).str[:10].map(lot_of)).fillna(-1).astype(int).to_numpy()
    log(f"{len(uni):,} PINs: {int((uni.lot >= 0).sum()):,} placed ({int(need.sum()):,} needed a point)")

    # class 2 buildings: square feet and homes (single-family types 1 per building; 2-6 flats by apartments)
    res = s["res"].copy()
    res["pin"] = res.pin.map(_digits)
    res["sf"] = pd.to_numeric(res.char_bldg_sf, errors="coerce").fillna(0)
    res["yr"] = pd.to_numeric(res.char_yrblt, errors="coerce")
    res["units"] = np.where(res["class"].isin(["211", "212"]), res.char_apts.map(APTS).fillna(2),
                            np.where(res["class"].isin(SF_CLASSES), 1, 0))
    res["lot"] = lot_of_pin(res.pin)
    # condominium units (parking spaces and common areas are not homes); building area once per building
    con = s["condo"].copy()
    con["pin"] = con.pin.map(_digits)
    con["lot"] = lot_of_pin(con.pin)
    con["home"] = ~(con.is_parking_space.astype(str).str.lower().eq("true") | con.is_common_area.astype(str).str.lower().eq("true"))
    con["bsf"] = pd.to_numeric(con.char_building_sf, errors="coerce")
    con["usf"] = pd.to_numeric(con.char_unit_sf, errors="coerce").fillna(0)
    con["yr"] = pd.to_numeric(con.char_yrblt, errors="coerce")
    # commercial valuation (classes 3 and 5): one key PIN per project
    com = s["commercial"].copy()
    com["lot"] = lot_of_pin(com.keypin.map(_digits))
    for c in ("tot_units", "bldgsf", "stories", "yearbuilt"):
        com[c] = pd.to_numeric(com[c], errors="coerce")
    av = s["values"].copy()
    av["pin"] = av.pin.map(_digits)
    av["lot"] = lot_of_pin(av.pin)
    av["tot"] = pd.to_numeric(av.board_tot, errors="coerce").fillna(pd.to_numeric(av.certified_tot, errors="coerce")).fillna(0)
    av["land"] = pd.to_numeric(av.board_land, errors="coerce").fillna(pd.to_numeric(av.certified_land, errors="coerce")).fillna(0)
    level = av["class"].astype(str).str[:1].map(LEVEL_OF_ASSESSMENT).fillna(0.10)

    n = len(p)
    by = lambda df, col, how="sum": df[df.lot >= 0].groupby("lot")[col].agg(how).reindex(range(n))
    units = (by(res, "units").fillna(0) + by(con[con.home], "home", "size").fillna(0) + by(com, "tot_units").fillna(0)).to_numpy()
    fa_obs = (by(res, "sf").fillna(0) + by(con, "bsf", "max").fillna(by(con, "usf").fillna(0)) + by(com, "bldgsf").fillna(0)).to_numpy()
    year = pd.concat([by(res, "yr", "max"), by(con, "yr", "max"), by(com, "yearbuilt", "max")], axis=1).max(axis=1).to_numpy()
    stories = by(com, "stories", "max").to_numpy()
    ptype = com[com.lot >= 0].assign(pt=com.property_type_use.fillna("")).groupby("lot").pt.agg(
        lambda x: x.value_counts().index[0]).reindex(range(n)).fillna("")
    mkt = av.assign(v=av.tot / level, l=av.land / level)
    value, land = by(mkt, "v").fillna(0).to_numpy(), by(mkt, "l").fillna(0).to_numpy()
    cls_mode = uni[uni.lot >= 0].groupby("lot")["class"].agg(lambda x: x.value_counts().index[0]).reindex(range(n)).fillna("")

    p["lu16"] = np.array([TAX_INDEX.get(crosswalk.cmap(c, u, t), -1) for c, u, t in zip(p.lui, units, ptype)], dtype="int16")
    use_a = np.array([group_of(c) for c in p.lu16])
    log(f"{int((p.lu16 >= 0).sum()):,} of {n:,} parcels with an official land use")

    # exempt and unrecorded buildings: floor area from Overture footprints x floors (estimated, flagged)
    log("Overture buildings (for exempt property)")
    fa_ov, fl_ov = overture_buildings(GEOID, p.geometry.values, crs)
    built = np.array([c >= 0 and CLASSES[c].built for c in p.lu16])
    gap = (fa_obs <= 0) & (fa_ov > 0) & built
    lot = p.geometry.area.to_numpy()
    fa, n_fa_dropped = plausible_floor_area(np.where(gap, fa_ov, fa_obs), lot)
    floors = np.where(~np.isnan(stories) & (stories > 0), stories, np.where(~np.isnan(fl_ov), fl_ov, np.where(fa > 0, 1, 0)))
    log(f"{int(gap.sum()):,} built parcels without an assessor building record: floor area from Overture footprints")

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
    jobs_block = census.lodes_jobs("17", "IL").reindex(blocks.GEOID).fillna(0).to_numpy()
    # homes on residential parcels with no assessor home count (public housing and other exempt property): the rest
    # of each block's 2020 housing units, by floor area (estimated, flagged)
    home_like = np.isin(use_a, [U["res"], U["mix"]])
    empty = home_like & (units <= 0) & (fa > 0)
    left = np.clip(blocks.HU100.to_numpy() - pd.Series(units).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy(), 0, None)
    w = np.where(empty, fa, 0.0)
    tot = pd.Series(w).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy()
    est = np.divide(w * left[blk], tot[blk], out=np.zeros(n), where=tot[blk] > 0)
    units_all = units + np.round(est)
    log(f"homes: {units.sum():,.0f} from the assessor, {np.round(est).sum():,.0f} estimated on {int((est >= 0.5).sum()):,} "
        f"exempt or unrecorded residential parcels; Census 2020: {int(blocks.HU100.sum()):,}")

    areas = s["areas"].to_crs(crs)
    i, k = shapely.STRtree(areas.geometry.values).query(rp.values, predicate="within")
    cd = np.zeros(n, dtype="int32")
    cd[i] = pd.to_numeric(areas.area_numbe, errors="coerce").fillna(0).astype(int).to_numpy()[k]
    if (cd == 0).any():
        cd[cd == 0] = pd.to_numeric(areas.area_numbe, errors="coerce").fillna(0).astype(int).to_numpy()[
            shapely.STRtree(areas.geometry.values).nearest(rp.values[cd == 0])]
    zn = s["zoning"].to_crs(crs)
    i, k = shapely.STRtree(zn.geometry.values).query(rp.values, predicate="within")
    zhit = pd.Series(k, index=i).groupby(level=0).first()
    zone = pd.Series(zn.zone_class.astype(str).to_numpy()[zhit.to_numpy()], index=zhit.index).reindex(range(n)).fillna("NA")
    rules = zone.map(zone_rule)

    lu16 = p.lu16.to_numpy().astype(int)
    k16 = lambda key: lu16 == TAX_INDEX[key]
    MIX_RES = 0.7  # assumption, as for the other cities
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], MIX_RES * fa, 0.0))
    nonres = fa - res_area
    mix = use_a == U["mix"]
    office = np.where(k16("off"), nonres, np.where(mix, 0.5 * nonres, 0.0))
    retail = np.where(k16("ret"), nonres, np.where(mix, 0.3 * nonres, 0.0))
    other = np.where(k16("hot"), nonres, np.where(mix, 0.2 * nonres, 0.0))
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    df = pd.DataFrame({
        "bbl": pd.to_numeric(p.pin10, errors="coerce").fillna(0).astype("int64").to_numpy(), "address": "",
        "cd": cd, "block": blk.astype("int32"), "zone": zone.to_numpy(), "overlay": "", "special": "",
        "bldg_class": cls_mode.astype(str).to_numpy(), "use": use_a.astype("int16"), "lu16": lu16.astype("int16"),
        "lu_code": p.lui.fillna("").astype(str).to_numpy(),
        "zone16": np.array([TAX_INDEX.get(r[1], -1) for r in rules], dtype="int16"),
        "lot_area": lot, "bldg_area": fa, "res_area": res_area, "com_area": office + retail + other,
        "office_area": office, "retail_area": retail, "other_area": other,
        "factory_area": np.where(k16("ind"), nonres, 0.0), "storage_area": np.where(k16("whs"), nonres, 0.0),
        "garage_area": 0.0, "num_bldgs": (fa > 0).astype(float), "floors": floors, "units_res": units_all,
        "units_total": units_all, "assess_land": land, "assess_tot": value, "year_built": np.nan_to_num(year),
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
    names = dict(zip(pd.to_numeric(areas.area_numbe, errors="coerce").fillna(0).astype(int), areas.community.str.title()))
    hu = float(blocks.HU100.sum())
    meta = finish_twin(df, p.geometry.values, crs, out, "IL", float(blocks.POP100.sum() / max(hu, 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "CMAP Land Use Inventory 2023, Cook County parcels 2025 and Cook County Assessor records",
        "source_url": "https://www.cmap.illinois.gov/data/land-use/inventory",
        "data_quality": {"missing_land_use": int((df.lu16 < 0).sum()), "missing_plan": n, "plan_layer": "none citywide",
                         "missing_zoning_class": int((df.zone16 < 0).sum()), "parcels": n, "assessor_pins": int(len(uni)),
                         "pins_not_in_a_parcel": int((uni.lot < 0).sum()),
                         "floor_area_from_overture": int(gap.sum()), "floor_area_implausible_dropped": n_fa_dropped,
                         "homes_assessor": int(units.sum()), "homes_estimated": int(np.round(est).sum()),
                         "homes_census_2020": int(hu), "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (Cook County base parcels 2025)",
                    "land use": "observed (CMAP Land Use Inventory 2023; assessor units and property types refine it)",
                    "adopted plan": "missing (Chicago has no citywide adopted land-use plan)",
                    "zoning": "observed district; simplified rules (district family sets allowed uses; context FAR)",
                    "floor area": "observed (assessor building square feet); estimated from Overture footprints and "
                                  "floors where the assessor has no building record (mostly exempt property)",
                    "floors and height": "observed stories where the commercial record has them, else Overture",
                    "homes": "observed (assessor apartments, condominium units and units); estimated from Census 2020 "
                             "block housing units on residential parcels with no record (mostly public housing)",
                    "jobs": "observed by block (LODES), modelled by floor area",
                    "assessed value": "derived (board of review assessed value / level of assessment: 10% residential, "
                                      "25% commercial and industrial)",
                    "year built": "observed", "landmarks and historic districts": "missing",
                    "mixed-use split": "assumed 70% residential"},
        "missing_indicators": [], "zoning_rules": "Chicago Zoning Ordinance district families (simplified)",
        "levels": {str(d): names.get(int(d), f"Area {d}") for d in sorted(df.cd.unique())},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {n:,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "class_counts", "data_quality", "build_seconds")}, indent=1))
