"""Adapter: City of Los Angeles, CA, from SCAG's 2020 Annual Land Use parcels and the LA County Assessor roll.

Sources:
- SCAG (Southern California Association of Governments) 2020 Annual Land Use, the regional planning agency's parcel
  layer. Each parcel carries three separate official layers in one code list: existing land use (ALU20), the city's
  General Plan designation (GP19_CITY, translated by SCAG to GP19_SCAG) and zoning (ZN19_CITY, translated to
  ZN19_SCAG), both as compiled by SCAG in 2019, plus the specific plan's designation where one governs (SP19_SCAG).
- LA County Assessor, Parcel Data roll 2025: use code, units, building square feet, year built and roll values per
  assessor identification number (AIN). Each AIN's point is placed in the SCAG parcel that contains it, so condominium
  units and several AINs on one parcel are summed. The table has no owner fields; none are requested.
- City of Los Angeles Department of City Planning: the 36 Community Plan Areas (game levels).

Floors are not in either source: they are derived as floor area over the building footprint (SCAG BF_SQFT), flagged.
Zoning: the zone family (Los Angeles Municipal Code, Chapter 1) sets the allowed uses; capacity uses the context FAR
because the height district (which sets FAR) is not in the compiled zone field. Both flagged in the UI.

Run:  python -I -m planarena.cities.adapters.losangeles
"""
from __future__ import annotations

import json
import re
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.common import finish_twin, local_crs, plausible_floor_area
from planarena.cities.registry import city_dir, register
from planarena.config import DATA, USE_INDEX as U, USE_KEYS, WEB_CRS
from planarena.national import census, places
from planarena.sources import crosswalk
from planarena.sources.arcgis import fetch_layer, fetch_table
from planarena.taxonomy import CLASSES, INDEX as TAX_INDEX, group_of
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "0644000"
RAW = DATA / "raw" / "losangeles"
SCAG_URL = "https://rdp.scag.ca.gov/mapping/rest/services/Housing/2020_Annual_Land_Use_NAD83/MapServer/0"
SCAG_FIELDS = ["OBJECTID", "PID20", "APN20", "APN20_P", "STACK", "APN_DUP", "MULTIPART", "ACRES", "GEOID20", "ALU20",
               "ALU20_SRC", "GP19_CITY", "GP19_SCAG", "SP19_CITY", "SP19_SCAG", "ZN19_CITY", "ZN19_SCAG", "BF_SQFT",
               "PUB_OWN", "PUB_TYPE", "YEAR"]
ASSESSOR_URL = ("https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/Parcel_Data_2021_Table/"
                "FeatureServer/0")
ASSESSOR_FIELDS = ["OBJECTID", "AIN", "RollYear", "UseType", "UseCode", "YearBuilt", "EffectiveYearBuilt", "SQFTmain",
                   "Units", "Bedrooms", "totBuildingDataLines", "Roll_LandValue", "Roll_ImpValue", "Roll_TotalValue",
                   "PropertyLocation", "CENTER_LAT", "CENTER_LON"]
ROLL_YEAR = "2025"
CPA_URL = "https://services5.arcgis.com/7nsPwEMP38bSkCjy/arcgis/rest/services/Community_Plan_Areas/FeatureServer/0"
MIX_RES = 0.7  # assumption, as for the other cities: neither source splits a mixed-use building's floor area
MAX_FLOORS = 80

ALL = (1 << len(USE_KEYS)) - 1
bit = lambda *keys: sum(1 << U[k] for k in keys)
# zone family -> allowed analysis groups (LAMC Chapter 1 use regulations, simplified; residential uses are permitted in
# the C and CM zones). Longest prefix first.
ZONING = [("RAS", bit("res", "mix", "com", "pub", "opn", "prk", "vac")),
          ("RMP", bit("res", "pub", "opn", "vac")), ("RW", bit("res", "pub", "opn", "vac")),
          ("RZ", bit("res", "pub", "opn", "vac")), ("RD", bit("res", "pub", "opn", "vac")),
          ("RE", bit("res", "pub", "opn", "vac")), ("RS", bit("res", "pub", "opn", "vac")),
          ("RU", bit("res", "pub", "opn", "vac")), ("RA", bit("res", "pub", "opn", "vac")),
          ("R1", bit("res", "pub", "opn", "vac")), ("R2", bit("res", "pub", "opn", "vac")),
          ("R3", bit("res", "pub", "opn", "prk", "vac")), ("R4", bit("res", "pub", "opn", "prk", "vac")),
          ("R5", bit("res", "pub", "opn", "prk", "vac")),
          ("CM", bit("res", "mix", "com", "ind", "pub", "opn", "prk", "vac")),
          ("CR", bit("res", "mix", "com", "pub", "opn", "prk", "vac")), ("CW", bit("res", "mix", "com", "pub", "opn", "prk", "vac")),
          ("C1", bit("res", "mix", "com", "pub", "opn", "prk", "vac")), ("C2", bit("res", "mix", "com", "pub", "opn", "prk", "vac")),
          ("C4", bit("res", "mix", "com", "pub", "opn", "prk", "vac")), ("C5", bit("res", "mix", "com", "pub", "opn", "prk", "vac")),
          ("MR", bit("com", "ind", "tra", "pub", "opn", "prk", "vac")),
          ("M1", bit("com", "ind", "tra", "pub", "opn", "prk", "vac")), ("M2", bit("com", "ind", "tra", "pub", "opn", "prk", "vac")),
          ("M3", bit("ind", "tra", "pub", "opn", "prk", "vac")),
          ("PF", bit("pub", "tra", "opn", "prk", "vac")), ("PB", bit("prk", "opn", "vac")), ("P", bit("prk", "opn", "vac")),
          ("OS", bit("opn", "pub", "vac")), ("A1", bit("res", "opn", "pub", "vac")), ("A2", bit("res", "opn", "pub", "vac"))]


def allowed(zone: str) -> int:
    z = zone.upper().strip() if isinstance(zone, str) else ""
    for prefix, mask in ZONING:
        if re.match(prefix + r"(?![A-Z])", z):  # 'P' is the parking zone, not PVSP (Playa Vista specific plan)
            return mask
    return ALL  # specific plans (Downtown, LAX, Playa Vista, ...) and unknown zones: not screened


def _cls(codes, units=None, sp=None) -> np.ndarray:
    n = len(codes)
    units = [None] * n if units is None else units
    sp = [None] * n if sp is None else sp
    return np.array([TAX_INDEX.get(crosswalk.scag(c, u, s), -1) for c, u, s in zip(codes, units, sp)], dtype="int16")


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    crs = local_crs(pl["lat"], pl["lon"])
    log("downloading SCAG 2020 land use parcels (City of Los Angeles)")
    g = fetch_layer(SCAG_URL, RAW / "scag" / "scag_alu2020_la.parquet", SCAG_FIELDS, where="CITY='Los Angeles'", workers=4)
    n_scag = len(g)
    g = g[g.geometry.notna()].to_crs(crs)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    g = g[~g.geometry.is_empty]
    # stacked parcels (SCAG STACK > 1: several records drawn on one polygon, e.g. condominium lots) become one parcel
    g["wkb"] = shapely.to_wkb(shapely.normalize(shapely.set_precision(g.geometry.values, 0.01)))
    one = g.drop_duplicates("wkb")
    agg = g.groupby("wkb", sort=False).agg(footprint=("BF_SQFT", "max"), stacked=("APN20", "size")).reindex(one.wkb)
    p = gpd.GeoDataFrame({"apn": one.APN20.to_numpy(), "alu": one.ALU20.to_numpy(), "gp_city": one.GP19_CITY.to_numpy(),
                          "gp": one.GP19_SCAG.to_numpy(), "sp_city": one.SP19_CITY.to_numpy(), "sp": one.SP19_SCAG.to_numpy(),
                          "zn_city": one.ZN19_CITY.to_numpy(), "zn": one.ZN19_SCAG.to_numpy(),
                          "footprint": agg.footprint.to_numpy(), "stacked": agg.stacked.to_numpy()},
                         geometry=one.geometry.values, crs=crs)
    log(f"{n_scag:,} SCAG records -> {len(p):,} parcels")

    log("downloading the LA County Assessor roll")
    a = fetch_table(ASSESSOR_URL, RAW / "assessor" / f"assessor_{ROLL_YEAR}_la.parquet", ASSESSOR_FIELDS,
                    where=f"RollYear='{ROLL_YEAR}' AND TaxRateArea_CITY='LOS ANGELES'")
    for c in ("SQFTmain", "Units", "Roll_LandValue", "Roll_TotalValue", "CENTER_LAT", "CENTER_LON"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a["street"] = a.PropertyLocation.fillna("").str.split("  ").str[0].str.split().str.join(" ")
    a["year"] = pd.to_numeric(a.YearBuilt, errors="coerce").where(lambda y: y.between(1800, 2026))
    # each AIN: first by its own number (SCAG APN20), else the parcel containing its point
    lot_of = pd.Series(np.arange(len(p)), index=p.apn.astype(str)).groupby(level=0).first()
    a["lot"] = a.AIN.astype(str).map(lot_of).fillna(-1).astype(int)
    has_pt = (a.lot < 0) & a.CENTER_LAT.notna() & a.CENTER_LON.notna() & (a.CENTER_LAT != 0)
    pts = gpd.GeoSeries(gpd.points_from_xy(a.CENTER_LON[has_pt], a.CENTER_LAT[has_pt]), crs=4326).to_crs(crs)
    i, k = shapely.STRtree(p.geometry.values).query(pts.values, predicate="within")
    hit = pd.Series(k, index=i).groupby(level=0).first()
    a.loc[a.index[has_pt.to_numpy()][hit.index], "lot"] = hit.to_numpy()
    placed = a[a.lot >= 0]
    log(f"{len(a):,} AINs: {int((a.lot >= 0).sum()):,} placed ({int(has_pt.sum()):,} by point)")
    # homes: units of residential use codes (0xxx), and of other codes only on residential or mixed parcels (below)
    placed = placed.assign(res_units=np.where(placed.UseCode.fillna("").str[:1] == "0", placed.Units.fillna(0), 0.0),
                           other_units=np.where(placed.UseCode.fillna("").str[:1] != "0", placed.Units.fillna(0), 0.0))
    s = placed.groupby("lot").agg(fa=("SQFTmain", "sum"), res_units=("res_units", "sum"), other_units=("other_units", "sum"),
                                  year=("year", "max"), land=("Roll_LandValue", "sum"), value=("Roll_TotalValue", "sum"),
                                  address=("street", "first"), ains=("AIN", "size"),
                                  use_code=("UseCode", lambda x: x.mode().iloc[0] if x.notna().any() else ""))
    p = p.join(s, how="left")
    units0 = p.res_units.fillna(0).to_numpy()

    # three official layers -> 16 classes
    p["lu16"] = _cls(p.alu, units0)
    p["plan16"] = np.array([TAX_INDEX.get(crosswalk.los_angeles_plan(d, c, sp), -1)
                            for d, c, sp in zip(p.gp_city, p.gp, p.sp)], dtype="int16")
    p["zone16"] = np.array([TAX_INDEX.get(crosswalk.los_angeles_zone(z, c, sp), -1)
                            for z, c, sp in zip(p.zn_city, p.zn, p.sp)], dtype="int16")
    use_a = np.array([group_of(c) for c in p.lu16])
    home_like = np.isin(use_a, [U["res"], U["mix"]])
    units = units0 + np.where(home_like, p.other_units.fillna(0).to_numpy(), 0.0)
    log(f"{int((p.lu16 >= 0).sum()):,} parcels with an existing land use, {int((p.plan16 >= 0).sum()):,} with a plan "
        f"designation, {int((p.zone16 >= 0).sum()):,} with a zoning class")

    bnd = places.boundary(GEOID)
    blocks = census.blocks(GEOID, bnd.bounds)
    inside = blocks.representative_point().within(bnd).to_numpy()
    blocks = blocks[(blocks.AREALAND > 0).to_numpy() & inside].to_crs(crs).reset_index(drop=True)
    rp = p.geometry.representative_point()
    bt = shapely.STRtree(blocks.geometry.values)
    i, k = bt.query(rp.values, predicate="within")
    blk = np.full(len(p), -1)
    blk[i] = k
    miss = blk < 0
    if miss.any():
        blk[miss] = bt.nearest(rp.values[miss])
    jobs_block = census.lodes_jobs("06", "CA").reindex(blocks.GEOID).fillna(0).to_numpy()
    cpa = fetch_layer(CPA_URL, RAW / "planning" / "community_plan_areas.parquet", ["OBJECTID", "CPA_NUM", "NAME_ALF"]).to_crs(crs)
    i, k = shapely.STRtree(cpa.geometry.values).query(rp.values, predicate="within")
    cd = np.zeros(len(p), dtype="int32")
    cd[i] = cpa.CPA_NUM.to_numpy()[k]
    if (cd == 0).any():
        cd[cd == 0] = cpa.CPA_NUM.to_numpy()[shapely.STRtree(cpa.geometry.values).nearest(rp.values[cd == 0])]

    lot = p.geometry.area.to_numpy()  # SCAG gives no recorded lot area; the parcel polygon's
    fa, n_fa_dropped = plausible_floor_area(p.fa.fillna(0).to_numpy().astype(float), lot)
    footprint = p.footprint.fillna(0).to_numpy()
    # exempt property (schools, government) often has no building area on the roll: where a built class has a mapped
    # building footprint but no recorded floor area, floor area = footprint x the class's median floor-area-to-
    # footprint ratio among parcels that have both (estimated, counted in data_quality)
    lu16 = p.lu16.to_numpy().astype(int)
    both = (fa > 0) & (footprint > 0)
    ratio = pd.Series(fa[both] / footprint[both]).groupby(lu16[both]).median()
    gap = (fa <= 0) & (footprint > 0) & np.array([c >= 0 and CLASSES[c].built for c in lu16])
    fa = np.where(gap, footprint * pd.Series(lu16).map(ratio).fillna(1.0).to_numpy(), fa)
    log(f"{int(gap.sum()):,} built parcels without recorded floor area: estimated from the building footprint")
    floors = np.where((fa > 0) & (footprint > 0), np.clip(np.round(fa / np.maximum(footprint, 1)), 1, MAX_FLOORS),
                      np.where(fa > 0, 1, 0)).astype(float)
    k16 = lambda key: lu16 == TAX_INDEX[key]
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], MIX_RES * fa, 0.0))
    nonres = fa - res_area
    mix = use_a == U["mix"]
    office = np.where(k16("off"), nonres, np.where(mix, 0.5 * nonres, 0.0))
    retail = np.where(k16("ret"), nonres, np.where(mix, 0.3 * nonres, 0.0))
    other = np.where(k16("hot"), nonres, np.where(mix, 0.2 * nonres, 0.0))
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    zone = p.zn_city.fillna("").astype(str).str.strip().replace("", "NA")
    df = pd.DataFrame({
        "bbl": np.arange(len(p), dtype="int64"), "address": p.address.fillna("").astype(str).to_numpy(),
        "cd": cd, "block": blk.astype("int32"), "zone": zone.to_numpy(), "overlay": "",
        "special": p.sp_city.fillna("").astype(str).to_numpy(), "bldg_class": p.use_code.fillna("").astype(str).to_numpy(),
        "use": use_a.astype("int16"), "lu16": lu16.astype("int16"), "lu_code": p.alu.fillna("").astype(str).to_numpy(),
        "plan16": p.plan16.to_numpy(), "plan_designation": p.gp_city.fillna("").astype(str).to_numpy(),
        "zone16": p.zone16.to_numpy(),
        "lot_area": lot, "bldg_area": fa, "res_area": res_area, "com_area": office + retail + other,
        "office_area": office, "retail_area": retail, "other_area": other,
        "factory_area": np.where(k16("ind"), nonres, 0.0), "storage_area": np.where(k16("whs"), nonres, 0.0),
        "garage_area": 0.0, "num_bldgs": (fa > 0).astype(float), "floors": floors, "units_res": units,
        "units_total": units, "assess_land": p.land.fillna(0).to_numpy(), "assess_tot": p.value.fillna(0).to_numpy(),
        "year_built": p.year.fillna(0).to_numpy(),
        "built_far": np.divide(fa, lot, out=np.zeros(len(p)), where=lot > 0), "historic": False, "landmark": False,
        "x": rp.x.to_numpy(), "y": rp.y.to_numpy(), "lon": ll.x.to_numpy(), "lat": ll.y.to_numpy(),
        "height_m": floors * FLOOR_HEIGHT_M, "census_block": blocks.GEOID.to_numpy()[blk],
    })
    bg = df.census_block.str[:12]
    built = df.bldg_area > 0
    ctx = df[built].groupby(bg[built]).built_far.quantile(0.9)
    df["max_far"] = np.maximum(bg.map(ctx).fillna(df.built_far[built].quantile(0.9)).to_numpy(), df.built_far.to_numpy())
    df["resid_far"] = df["comm_far"] = df["facil_far"] = df["max_far"]
    df["allowed"] = zone.map(allowed).to_numpy().astype("int64")
    names = dict(zip(cpa.CPA_NUM.astype(int), cpa.NAME_ALF))
    hu = float(blocks.HU100.sum())
    meta = finish_twin(df, p.geometry.values, crs, out, "CA", float(blocks.POP100.sum() / max(hu, 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "SCAG 2020 Annual Land Use (existing use, General Plan and zoning as compiled in 2019) and "
                  f"LA County Assessor roll {ROLL_YEAR}",
        "source_url": "https://scag.ca.gov/data-tools-land-use",
        "data_quality": {"missing_land_use": int((df.lu16 < 0).sum()), "missing_plan": int((df.plan16 < 0).sum()),
                         "missing_zoning_class": int((df.zone16 < 0).sum()), "scag_records": int(n_scag),
                         "stacked_records_merged": int(n_scag - len(p)), "assessor_ains": int(len(a)),
                         "ains_not_in_a_parcel": int((a.lot < 0).sum()),
                         "parcels_without_assessor_record": int(p.ains.isna().sum()),
                         "floor_area_implausible_dropped": n_fa_dropped,
                         "floor_area_estimated_from_footprint": int(gap.sum()), "homes_assessor": int(units.sum()),
                         "homes_census_2020": int(hu), "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (SCAG parcels, from the county assessor)",
                    "land use": "observed (SCAG 2020 existing land use, translated to 16 classes)",
                    "adopted plan": "observed (City General Plan land use, as compiled by SCAG in 2019)",
                    "zoning": "observed zone; simplified rules (zone family sets allowed uses; context FAR for capacity)",
                    "floor area": "observed (assessor building square feet, summed over the parcel's AINs); estimated from the "
                                  "building footprint where the roll has none (mostly exempt public property)",
                    "plan and zoning classes": "SCAG's translation, except residential designations and zones, "
                                               "classed by the city's own density categories",
                    "floors and height": "derived (floor area / building footprint)",
                    "homes": "observed (assessor units)", "jobs": "observed by block (LODES), modelled by floor area",
                    "assessed value": "observed (Proposition 13 roll value: below market for long-held property)",
                    "year built": "observed", "landmarks and historic districts": "missing",
                    "mixed-use split": "assumed 70% residential"},
        "missing_indicators": [], "zoning_rules": "Los Angeles Municipal Code zone families (simplified)",
        "levels": {str(d): names.get(int(d), f"Area {d}") for d in sorted(df.cd.unique())},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {len(df):,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "class_counts", "data_quality", "build_seconds")}, indent=1))
