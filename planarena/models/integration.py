"""Does combining the layers improve predictions and scenario assessments? Component ablation across cities.

The question (PAPER_OUTLINE, contribution 1): PlanArena combines what exists, what zoning permits, what the adopted
plan intends and the neighbourhood context. Is each layer worth having? Tested on observed history:

Panel per city: one row per lot at t0, every attribute measured at t0 or before (plans and zoning adopted before t0):
  developed     a building completed in (t0, t1], or floor area +25% and +2,000 sq ft
  homes_added   max(0, homes at t1 - homes at t0)
  lu16, zone16, plan16   the three official layers (16 classes; -1 where the city has no layer)
  lot_area, fa0, units0, year0, land_share0, x, y, district

Feature blocks (all portable across cities: classes, ratios and local context, no city-specific codes):
  E  existing   analysis group of the existing use, log lot area, log floor area, built FAR, age, homes,
                land-value share of the assessed value
  C  context    share of lots within 400 m built in the five years before t0; neighbour group composition
  Z  zoning     zoning class group, whether the existing use departs from it, homes step from existing to zoning
  P  plan       plan class group, whether the plan differs from the existing use, homes step from existing to plan

Within a city: LightGBM, 5-fold spatial-block cross-fitting (2 km tiles); each block set is scored by AUC, average
precision and precision in the top 1% and 5%, with bootstrap intervals over tiles for the gain of adding a block.
Across cities: trained on the other cities' panels, tested on the city (portable blocks only).
Scenario assessment by district: observed homes added against zoning build-out, plan build-out, the same increments
weighted by the cross-fitted likelihood, and the five-year trend.

Run:  python -I -m planarena.models.integration [geoid ...]
"""
from __future__ import annotations

import json
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.registry import city_dir
from planarena.config import DATA, FT_PER_M, ROOT, USE_INDEX as U, USE_KEYS
from planarena.models.common import SEED, spatial_folds
from planarena.sources.arcgis import fetch_layer
from planarena.taxonomy import CLASSES, INDEX as TAX_INDEX

WORK = DATA / "integration"
VACANT = TAX_INDEX["vac"]
OUT = ROOT / "results" / "validation" / "integration_cities.json"
TILE_M = 2000
RADIUS_M = 400
BLOCKS = {"E": "existing", "C": "context", "Z": "zoning", "P": "plan"}
SETS = ["E", "E+C", "E+Z", "E+P", "E+Z+P", "E+C+Z", "E+C+P", "E+C+Z+P"]
# homes intensity of each class (0 none, 1 single-family, 2 small multi-family, 3 multi-family or mixed use)
HOMES_RANK = np.array([{"sf": 1, "mf_small": 2, "mf": 3, "mix": 3}.get(c.key, 0) for c in CLASSES])
GROUP_OF = np.array([U[c.group] for c in CLASSES])
N_GROUPS = len(USE_KEYS)


# ----------------------------------------------------------------------------- panels
def _place(points_lon, points_lat, lots: gpd.GeoSeries) -> np.ndarray:
    """Lot index of each point (-1 outside every lot)."""
    ok = np.isfinite(points_lon) & np.isfinite(points_lat) & (points_lat != 0)
    pts = gpd.GeoSeries(gpd.points_from_xy(points_lon[ok], points_lat[ok]), crs=4326).to_crs(lots.crs)
    i, k = shapely.STRtree(lots.values).query(pts.values, predicate="within")
    out = np.full(len(points_lon), -1)
    hit = pd.Series(k, index=i).groupby(level=0).first()
    out[np.nonzero(ok)[0][hit.index]] = hit.to_numpy()
    return out


def panel_los_angeles(t0: int, t1: int) -> pd.DataFrame:
    """Los Angeles between the January assessor rolls of t0 and t1 (2021, 2023, 2025); existing use from SCAG 2020,
    General Plan and zoning as compiled by SCAG in 2019 (all before 2021)."""
    from planarena.cities.adapters.losangeles import RAW
    from planarena.sources.crosswalk import la_county_use

    codes = json.loads((RAW / "assessor" / "use_codes_2025.json").read_text())  # [code, desc1, desc2, count]
    use_class = {c: TAX_INDEX.get(la_county_use(c, d1, d2), -1) for c, d1, d2, _ in codes}
    out = city_dir("0644000")
    df = pd.read_parquet(out / "parcels.parquet", columns=["lu16", "plan16", "zone16", "cd", "x", "y", "lot_area", "lu_code"])
    df["underway0"] = df.pop("lu_code").astype(str).eq("1700")  # SCAG: under construction in 2020
    lots = gpd.read_parquet(out / "parcels.geoparquet").geometry
    rolls = {}
    for yr in (t0, t1):
        a = pd.read_parquet(RAW / "assessor" / f"assessor_{yr}_la.parquet")
        for c in ("SQFTmain", "Units", "Roll_LandValue", "Roll_TotalValue", "CENTER_LAT", "CENTER_LON"):
            a[c] = pd.to_numeric(a[c], errors="coerce")
        a["yb"] = pd.to_numeric(a.YearBuilt, errors="coerce").where(lambda y: y.between(1800, 2026))
        a["lot"] = _place(a.CENTER_LON.to_numpy(), a.CENTER_LAT.to_numpy(), lots)
        a["homes"] = np.where(a.UseCode.fillna("").str[:1] == "0", a.Units.fillna(0), 0.0)
        a["cls"] = a.UseCode.map(use_class).fillna(-1).astype(int)
        a = a[a.lot >= 0]
        r = a.groupby("lot").agg(fa=("SQFTmain", "sum"), homes=("homes", "sum"), yb=("yb", "max"),
                                 land=("Roll_LandValue", "sum"), tot=("Roll_TotalValue", "sum")).reindex(range(len(df)))
        known = a[a.cls >= 0]  # the lot's class on this roll: its most common use
        r["cls"] = known.groupby("lot").cls.agg(lambda x: x.value_counts().index[0]).reindex(range(len(df)))
        rolls[yr] = r
    r0, r1 = rolls[t0], rolls[t1]
    fa0, fa1 = r0.fa.fillna(0).to_numpy(), r1.fa.fillna(0).to_numpy()
    df["fa0"], df["units0"], df["year0"] = fa0, r0.homes.fillna(0).to_numpy(), r0.yb.to_numpy()
    df["land_share0"] = np.divide(r0.land.to_numpy(), r0.tot.to_numpy(), out=np.full(len(df), np.nan),
                                  where=r0.tot.to_numpy() > 0)
    # a building already on the t0 roll with a year built of t0 or later is not new in (t0, t1]
    newb = (r1.yb.to_numpy() >= t0) & ~(r0.yb.to_numpy() >= t0)
    df["developed"] = (newb | ((fa1 > 1.25 * fa0) & (fa1 - fa0 > 2000))).astype(bool)
    df["homes_added"] = np.clip(r1.homes.fillna(0).to_numpy() - df.units0.to_numpy(), 0, None)
    df["dest16"] = r1.cls.fillna(-1).astype("int16").to_numpy()
    df["observed"] = r0.fa.notna().to_numpy() & r1.fa.notna().to_numpy()  # lots on both rolls
    df["t0"], df["t1"] = t0, t1
    return df


def _dallas_lots():
    """The Dallas twin's lots with the city's GIS account of each (the adapter's filters, replayed in order)."""
    from planarena.cities.adapters.dallas import RAW
    from planarena.twin.build import _polygonal

    out = city_dir("4819000")
    df = pd.read_parquet(out / "parcels.parquet", columns=["lu16", "plan16", "zone16", "cd", "x", "y", "lot_area"])
    lots = gpd.read_parquet(out / "parcels.geoparquet")
    g = gpd.read_parquet(RAW / "city" / "existing_land_use_2026.parquet")
    g = g[g.geometry.notna()].to_crs(lots.crs).reset_index(drop=True)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    g = g[~g.geometry.is_empty].reset_index(drop=True)
    assert len(g) == len(df), "Dallas lots do not line up with the raw layer"
    d = shapely.distance(g.geometry.representative_point().values, lots.geometry.representative_point().values)
    assert float(np.nanmax(d)) < 1.0, "Dallas lots do not line up with the raw layer"
    df["acct"] = g.GIS_ACCT.astype(str).str.strip().to_numpy()
    return df


def panel_dallas(t0: int, t1: int) -> pd.DataFrame:
    """Dallas between the DCAD certified files of t0 and t1 (2022, 2024, 2026). Existing use at t0 from DCAD's own
    codes (the city's existing land-use survey is 2025, after t0); zoning and the forwardDallas! plan (adopted
    2024) are the current layers, flagged."""
    import zipfile

    from planarena.cities.adapters.dallas import RAW
    from planarena.sources.crosswalk import dcad

    df = _dallas_lots()
    lot_of = pd.Series(np.arange(len(df)), index=df.acct).groupby(level=0).first()
    num = lambda x: pd.to_numeric(x, errors="coerce")
    years = {}
    for yr in (t0, t1):
        z = zipfile.ZipFile(RAW / "dcad" / f"DCAD{yr}_CURRENT.zip")
        read = lambda name, cols: pd.read_csv(z.open(name), usecols=cols, dtype=str, encoding="latin-1")
        acc = read("ACCOUNT_INFO.CSV", ["ACCOUNT_NUM", "GIS_PARCEL_ID"])
        acc["lot"] = acc.GIS_PARCEL_ID.astype(str).str.strip().map(lot_of)
        acct_lot = acc.dropna(subset=["lot"]).set_index("ACCOUNT_NUM").lot.astype(int)
        res = read("RES_DETAIL.CSV", ["ACCOUNT_NUM", "BLDG_CLASS_DESC", "YR_BUILT", "TOT_LIVING_AREA_SF", "NUM_KITCHENS"])
        com = read("COM_DETAIL.CSV", ["ACCOUNT_NUM", "BLDG_CLASS_DESC", "YEAR_BUILT", "GROSS_BLDG_AREA", "NUM_UNITS"])
        val = read("ACCOUNT_APPRL_YEAR.CSV", ["ACCOUNT_NUM", "LAND_VAL", "TOT_VAL"])
        land = read("LAND.CSV", ["ACCOUNT_NUM", "SPTD_CD"])
        for t in (res, com, val, land):
            t["lot"] = t.ACCOUNT_NUM.map(acct_lot)
        res, com, val, land = (t.dropna(subset=["lot"]).astype({"lot": int}) for t in (res, com, val, land))
        res["sf"], res["yr"] = num(res.TOT_LIVING_AREA_SF).fillna(0), num(res.YR_BUILT)
        res["units"] = np.where(res.BLDG_CLASS_DESC.astype(str).str.upper().eq("LAND ONLY") | (res.sf <= 0), 0,
                                num(res.NUM_KITCHENS).fillna(1).clip(lower=1))
        com["gba"], com["yr"] = num(com.GROSS_BLDG_AREA).fillna(0), num(com.YEAR_BUILT)
        apt = com.BLDG_CLASS_DESC.astype(str).str.upper().map(lambda x: any(w in x for w in
                                                                               ("APARTMENT", "RESIDENTIAL MULTI-FAMILY", "SENIOR CITIZEN HOUSING", "RETIREMENT")))
        com["units"] = np.where(apt, num(com.NUM_UNITS).fillna(0), 0)
        by = lambda t, c, how="sum": t.groupby("lot")[c].agg(how).reindex(range(len(df)))
        r = pd.DataFrame({"fa": by(res, "sf").fillna(0) + by(com, "gba").fillna(0),
                          "homes": by(res, "units").fillna(0) + by(com, "units").fillna(0),
                          "yb": pd.concat([by(res, "yr", "max"), by(com, "yr", "max")], axis=1).max(axis=1),
                          "land": by(val.assign(v=num(val.LAND_VAL)), "v"), "tot": by(val.assign(v=num(val.TOT_VAL)), "v")})
        r["known"] = val.groupby("lot").size().reindex(range(len(df))).notna()
        mode = lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None
        sptd = land.groupby("lot").SPTD_CD.agg(mode).reindex(range(len(df)))
        bcl = com.groupby("lot").BLDG_CLASS_DESC.agg(mode).reindex(range(len(df)))
        r["cls"] = [TAX_INDEX.get(dcad(sc, bc, u), -1) for sc, bc, u in zip(sptd, bcl, r.homes)]
        years[yr] = r
    r0, r1 = years[t0], years[t1]
    fa0, fa1 = r0.fa.to_numpy(), r1.fa.to_numpy()
    df["lu16"] = r0.cls.to_numpy().astype("int16")
    df["fa0"], df["units0"], df["year0"] = fa0, r0.homes.to_numpy(), r0.yb.to_numpy()
    df["land_share0"] = np.divide(r0.land.to_numpy(), r0.tot.to_numpy(), out=np.full(len(df), np.nan),
                                  where=np.nan_to_num(r0.tot.to_numpy()) > 0)
    # a building already on the t0 roll with a year built of t0 or later is not new in (t0, t1]
    newb = (r1.yb.to_numpy() >= t0) & ~(r0.yb.to_numpy() >= t0)
    df["developed"] = (newb | ((fa1 > 1.25 * fa0) & (fa1 - fa0 > 2000))).astype(bool)
    df["homes_added"] = np.clip(r1.homes.to_numpy() - df.units0.to_numpy(), 0, None)
    df["dest16"] = r1.cls.to_numpy().astype("int16")
    df["observed"] = r0.known.to_numpy() & r1.known.to_numpy()
    df["t0"], df["t1"] = t0, t1
    return df.drop(columns="acct")


def _under(points: gpd.GeoSeries, polys: gpd.GeoDataFrame) -> pd.Series:
    """Index of the smallest polygon containing each point (by point position)."""
    i, k = shapely.STRtree(polys.geometry.values).query(points.values, predicate="within")
    h = pd.DataFrame({"i": i, "k": k, "a": polys.geometry.area.to_numpy()[k]}).sort_values(["i", "a"]).drop_duplicates("i")
    return pd.Series(h.k.to_numpy(), index=h.i.to_numpy())


def _landuse_change_panel(geoid: str, layers: dict, t0: int, t1: int, flags) -> pd.DataFrame:
    """Lots of the city's twin with the regional agency's land use at t0 and t1 (and the vintage before t0, for
    recent change). developed = the lot's use at t1 is a built use of another analysis group, or more homes."""
    out = city_dir(geoid)
    df = pd.read_parquet(out / "parcels.parquet", columns=["plan16", "zone16", "cd", "x", "y", "lot_area"])
    lots = gpd.read_parquet(out / "parcels.geoparquet")
    rp = lots.geometry.representative_point()
    cls = {}
    raw = {}
    for yr, (load, field, cw) in layers.items():
        lu = load().to_crs(lots.crs)
        k = _under(rp, lu)
        code = pd.Series(lu[field].to_numpy()[k.to_numpy()], index=k.index).reindex(range(len(df)))
        raw[yr] = code
        cls[yr] = np.array([TAX_INDEX.get(cw(c), -1) if pd.notna(c) else -1 for c in code], dtype="int16")
    c0, c1 = cls[t0], cls[t1]
    built = np.array([c.built for c in CLASSES])
    g = lambda c: np.where(c >= 0, GROUP_OF[np.clip(c, 0, None)], -1)
    changed = lambda a, b: (a >= 0) & (b >= 0) & built[np.clip(b, 0, None)] & (
        (g(a) != g(b)) | (HOMES_RANK[np.clip(b, 0, None)] > HOMES_RANK[np.clip(a, 0, None)]))
    df["lu16"] = c0
    df["developed"] = changed(c0, c1)
    df["dest16"] = c1
    prior = [y for y in layers if y < t0]
    df["recent0"] = changed(cls[max(prior)], c0).astype(float) if prior else np.nan
    for c in ("fa0", "units0", "year0", "land_share0"):
        df[c] = np.nan
    df["homes_added"] = np.nan
    df["observed"] = (c0 >= 0) & (c1 >= 0)
    df["underway0"] = raw[t0].astype(str).str.upper().str.contains("DEVELOPING|UNDER CONSTRUCTION|^950", regex=True).to_numpy()
    df["t0"], df["t1"] = t0, t1
    zone_after, plan_after = flags(lots, rp, t0)
    df["zone_after_t0"], df["plan_after_t0"] = zone_after, plan_after
    return df


def panel_phoenix(t0: int, t1: int) -> pd.DataFrame:
    """Phoenix between MAG Existing Land Use vintages (2022, 2024, 2026); General Plan land use (2015, amended) and
    zoning are current layers: lots under a plan polygon added after t0, or a zoning case of t0 or later, are
    flagged."""
    from planarena.cities.adapters.phoenix import PLAN_URL, RAW, ZONING_URL
    from planarena.national import places
    from planarena.sources.arcgis import fetch_layer
    from planarena.sources.crosswalk import MAG_LAND_USE

    x0, y0, x1, y1 = places.boundary("0455000").bounds
    box = {"geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope", "inSR": "4326",
           "spatialRel": "esriSpatialRelIntersects"}
    url = lambda yr: f"https://services1.arcgis.com/MdyCMZnX1raZ7TS3/arcgis/rest/services/EXLU_{yr}/FeatureServer/0"
    load = lambda yr: (lambda: fetch_layer(url(yr), RAW / "mag" / f"exlu_{yr}_phoenix_bbox.parquet",
                                           ["OBJECTID", "LONG_DISPLAY_NAME"], extra=box))
    layers = {yr: (load(yr), "LONG_DISPLAY_NAME", lambda c: MAG_LAND_USE.get(str(c).strip(), "unknown"))
              for yr in (2022, 2024, 2026) if yr <= t1}

    def flags(lots, rp, t0):
        z = fetch_layer(ZONING_URL, RAW / "city" / "zoning_dates.parquet", ["OBJECTID", "ZONING", "ZYEAR"]).to_crs(lots.crs)
        yy = pd.to_numeric(z.ZYEAR, errors="coerce")
        zyear = np.where(yy <= 30, 2000 + yy, 1900 + yy)
        k = _under(rp, z)
        za = pd.Series(zyear[k.to_numpy()] >= t0, index=k.index).reindex(range(len(lots))).fillna(False).to_numpy()
        gp = fetch_layer(PLAN_URL, RAW / "city" / "general_plan_dates.parquet", ["OBJECTID", "NEWCODE", "INSERT_DATE"]).to_crs(lots.crs)
        ins = pd.to_datetime(pd.to_numeric(gp.INSERT_DATE, errors="coerce"), unit="ms").dt.year.to_numpy()
        k = _under(rp, gp)
        pa = pd.Series(ins[k.to_numpy()] >= t0, index=k.index).reindex(range(len(lots))).fillna(False).to_numpy()
        return za, pa

    return _landuse_change_panel("0455000", layers, t0, t1, flags)


def panel_san_diego(t0: int, t1: int) -> pd.DataFrame:
    """San Diego between SANDAG Land Use vintages (2023, 2024, 2025); zoning implemented on or after t0 flagged
    (the General Plan layer carries no amendment dates)."""
    from planarena.cities.adapters.sandiego import RAW, SANDAG, ZONING_URL
    from planarena.national import places
    from planarena.sources.arcgis import fetch_layer
    from planarena.sources.crosswalk import sandag

    x0, y0, x1, y1 = places.boundary("0666000").bounds
    box = {"geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope", "inSR": "4326",
           "spatialRel": "esriSpatialRelIntersects"}
    load = lambda yr: (lambda: fetch_layer(f"{SANDAG}/Land_Use_{yr}/FeatureServer/0",
                                           RAW / "sandag" / f"land_use_{yr}_sd_bbox.parquet", ["objectid", "lu"], extra=box))
    layers = {yr: (load(yr), "lu", sandag) for yr in (2023, 2024, 2025) if yr <= t1}

    def flags(lots, rp, t0):
        z = fetch_layer(ZONING_URL, RAW / "city" / "zoning_dates.parquet", ["objectid", "zone_name", "imp_date"]).to_crs(lots.crs)
        yr = pd.to_datetime(pd.to_numeric(z.imp_date, errors="coerce"), unit="ms").dt.year.to_numpy()
        k = _under(rp, z)
        za = pd.Series(yr[k.to_numpy()] >= t0, index=k.index).reindex(range(len(lots))).fillna(False).to_numpy()
        return za, np.zeros(len(lots), dtype=bool)

    return _landuse_change_panel("0666000", layers, t0, t1, flags)


def _pluto(year: int) -> gpd.GeoDataFrame:
    """Citywide MapPLUTO of a January (21v1, 23v1) or the current twin (25v3.1), in the panel's terms."""
    from planarena.sources import crosswalk

    if year == 2025:
        out = city_dir("3651000")
        t = pd.read_parquet(out / "parcels.parquet", columns=["bbl", "lu16", "zone", "units_res", "bldg_area",
                                                              "lot_area", "year_built", "assess_land", "assess_tot", "cd"])
        g = gpd.read_parquet(out / "parcels.geoparquet")
        return gpd.GeoDataFrame(t.rename(columns={"units_res": "units", "bldg_area": "fa", "year_built": "yb",
                                                  "assess_land": "land", "assess_tot": "tot"}), geometry=g.geometry.values, crs=g.crs)
    v = {2021: "21v1", 2023: "23v1"}[year]
    cols = ["BBL", "LandUse", "BldgClass", "UnitsRes", "BldgArea", "LotArea", "YearBuilt", "AssessLand", "AssessTot",
            "ZoneDist1", "CD"]
    g = gpd.read_file(DATA / "raw" / f"pluto_{v}" / "extracted" / "MapPLUTO.shp", columns=cols, engine="pyogrio")
    g = g[g.geometry.notna()].reset_index(drop=True)
    num = lambda c: pd.to_numeric(g[c], errors="coerce")
    return gpd.GeoDataFrame({
        "bbl": num("BBL").fillna(0).astype("int64"),
        "lu16": [TAX_INDEX.get(crosswalk.nyc(a, b, u), -1) for a, b, u in zip(g.LandUse, g.BldgClass, g.UnitsRes)],
        "zone": g.ZoneDist1.fillna("").astype(str), "units": num("UnitsRes").fillna(0), "fa": num("BldgArea").fillna(0),
        "lot_area": num("LotArea").fillna(0), "yb": num("YearBuilt").where(lambda y: y.between(1800, 2026)),
        "land": num("AssessLand"), "tot": num("AssessTot"), "cd": num("CD").fillna(0).astype(int)},
        geometry=g.geometry.values, crs=g.crs)


def panel_new_york(t0: int, t1: int) -> pd.DataFrame:
    """New York City between MapPLUTO releases (21v1, 23v1, 25v3.1): zoning is each release's own; no citywide
    adopted land-use plan. t0 lots are followed to the t1 lot with the same BBL, else the one containing them."""
    from planarena.sources import crosswalk

    a, b = _pluto(t0), _pluto(t1)
    b = b.to_crs(a.crs)
    rp = a.geometry.representative_point()
    pos = pd.Series(np.arange(len(b)), index=b.bbl).groupby(level=0).first()
    j = a.bbl.map(pos)
    miss = j.isna().to_numpy()
    k = _under(rp[miss].reset_index(drop=True), b)
    jj = j.to_numpy().copy()
    jj[np.nonzero(miss)[0][k.index]] = k.to_numpy()
    ok = ~np.isnan(jj)
    jj = np.where(ok, jj, 0).astype(int)
    share = pd.Series(1.0, index=np.arange(len(a)))[ok].groupby(jj[ok]).transform("count").reindex(range(len(a))).to_numpy()
    yb1, fa1, u1 = b.yb.to_numpy()[jj], b.fa.to_numpy()[jj], b.units.to_numpy()[jj]
    fa0 = a.fa.to_numpy()
    df = pd.DataFrame({"lu16": a.lu16.to_numpy().astype("int16"), "plan16": np.int16(-1),
                       "zone16": np.array([TAX_INDEX.get(crosswalk.nyc_zone(z), -1) for z in a.zone], dtype="int16"),
                       "cd": a.cd.to_numpy(), "x": rp.x.to_numpy(), "y": rp.y.to_numpy(),
                       "lot_area": np.where(a.lot_area > 0, a.lot_area, a.geometry.area).astype(float)})
    df["fa0"], df["units0"], df["year0"] = fa0, a.units.to_numpy(), a.yb.to_numpy()
    df["land_share0"] = np.divide(a.land.to_numpy(), a.tot.to_numpy(), out=np.full(len(df), np.nan),
                                  where=np.nan_to_num(a.tot.to_numpy()) > 0)
    df["developed"] = ok & (((yb1 >= t0) & ~(a.yb.to_numpy() >= t0)) | ((fa1 > 1.25 * fa0) & (fa1 - fa0 > 2000)))
    df["homes_added"] = np.where(ok, np.clip(u1 / share - df.units0.to_numpy(), 0, None), 0.0)
    df["dest16"] = np.where(ok, b.lu16.to_numpy()[jj], -1).astype("int16")
    df["observed"] = ok
    df["t0"], df["t1"] = t0, t1
    return df


def panel_milwaukee(t0: int, t1: int) -> pd.DataFrame:
    """Milwaukee between MPROP assessment years (2021, 2023, 2025), joined by tax key on today's polygons; zoning
    from each year's MPROP; no adopted land-use map as data. MPROP floor area is not recorded consistently over time,
    so developed = a building built in (t0, t1] (year built), as in the cross-city test."""
    from planarena.cities.mke_history import snapshot_dir

    a = pd.read_parquet(snapshot_dir(t0) / "parcels.parquet")
    b = pd.read_parquet(snapshot_dir(t1) / "parcels.parquet", columns=["bbl", "year_built", "units_res", "lu16"])
    j = a.bbl.map(pd.Series(np.arange(len(b)), index=b.bbl).groupby(level=0).first())
    ok = j.notna().to_numpy()
    jj = np.where(ok, j.fillna(0), 0).astype(int)
    yb0 = pd.to_numeric(a.year_built, errors="coerce").where(lambda y: y.between(1800, 2026)).to_numpy()
    yb1 = pd.to_numeric(b.year_built, errors="coerce").where(lambda y: y.between(1800, 2026)).to_numpy()[jj]
    df = pd.DataFrame({"lu16": a.lu16.to_numpy().astype("int16"), "plan16": np.int16(-1), "zone16": a.zone16.to_numpy().astype("int16"),
                       "cd": a.cd.to_numpy(), "x": a.x.to_numpy(), "y": a.y.to_numpy(), "lot_area": a.lot_area.to_numpy(),
                       "fa0": a.bldg_area.to_numpy(), "units0": a.units_res.to_numpy(), "year0": yb0})
    df["land_share0"] = np.divide(a.assess_land.to_numpy(), a.assess_tot.to_numpy(), out=np.full(len(df), np.nan),
                                  where=a.assess_tot.to_numpy() > 0)
    df["developed"] = ok & (yb1 >= t0) & ~(yb0 >= t0)
    df["homes_added"] = np.where(ok, np.clip(b.units_res.to_numpy()[jj] - df.units0.to_numpy(), 0, None), 0.0)
    df["dest16"] = np.where(ok, b.lu16.to_numpy()[jj], -1).astype("int16")
    df["observed"] = ok
    df["t0"], df["t1"] = t0, t1
    return df


def panel_chicago(t0: int, t1: int) -> pd.DataFrame:
    """Chicago: development = a new-construction building permit issued in [t0, t1) (City of Chicago building
    permits; issue, not completion). Existing use from CMAP's inventory before t0 (2020 for t0 2021, 2023 for t0
    2023); buildings, homes and values from the Cook County Assessor of year t0 (class 2 buildings only); zoning is
    the current layer (districts created on or after t0 flagged); no citywide plan."""
    import re

    from planarena.cities.adapters.chicago import COOK, CHICAGO, LUI_URL, RAW, TOWNSHIPS
    from planarena.sources.crosswalk import cmap
    from planarena.sources.socrata import fetch, fetch_geojson

    out = city_dir("1714000")
    df = pd.read_parquet(out / "parcels.parquet", columns=["bbl", "zone16", "cd", "x", "y", "lot_area"])
    lots = gpd.read_parquet(out / "parcels.geoparquet")
    pin10 = df.pop("bbl").astype("int64").astype(str).str.zfill(10)
    lot_of = pd.Series(np.arange(len(df)), index=pin10.to_numpy()).groupby(level=0).first()
    digits = lambda x: re.sub(r"\D", "", str(x))
    # existing use: the CMAP inventory before t0
    lui_url = (LUI_URL if t0 >= 2023 else
               "https://services5.arcgis.com/LcMXE3TFhi1BSaCY/arcgis/rest/services/LUI20_geodatabase_v1_CMAP/FeatureServer/0")
    lui_file = RAW / "cmap" / ("lui2023_cook.parquet" if t0 >= 2023 else "lui2020_cook.parquet")
    lui = (fetch_layer(lui_url, lui_file, ["OBJECTID", "LANDUSE"], where="COUNTY='031' AND LANDUSE <> '6000'") if t0 >= 2023 else
           fetch_layer(lui_url, lui_file, ["OBJECTID", "LANDUSE"], where="FIRST_COUNTY='031' AND LANDUSE <> '6000'")).to_crs(lots.crs)
    rp = lots.geometry.representative_point()
    k = _under(rp, lui)
    code = pd.Series(lui.LANDUSE.astype(str).to_numpy()[k.to_numpy()], index=k.index).reindex(range(len(df)))
    # class 2 buildings and assessed values of year t0
    res = fetch(COOK, "x54s-btds", RAW / "cook" / f"residential_chars_{t0}.parquet",
                ["pin", "card", "class", "char_bldg_sf", "char_apts", "char_yrblt"],
                where=f"year={t0} AND township_code in {TOWNSHIPS}")
    av = fetch(COOK, "uzyt-m557", RAW / "cook" / f"assessed_values_{t0}.parquet",
               ["pin", "class", "board_tot", "certified_tot", "board_land", "certified_land"],
               where=f"year='{t0}' AND township_code in {TOWNSHIPS}")
    for t in (res, av):
        t["lot"] = t.pin.map(digits).str[:10].map(lot_of)
    res, av = res.dropna(subset=["lot"]).astype({"lot": int}), av.dropna(subset=["lot"]).astype({"lot": int})
    apts = {"Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6}
    res["sf"], res["yr"] = pd.to_numeric(res.char_bldg_sf, errors="coerce").fillna(0), pd.to_numeric(res.char_yrblt, errors="coerce")
    res["units"] = np.where(res["class"].isin(["211", "212"]), res.char_apts.map(apts).fillna(2), 1)
    by = lambda t, c, how="sum": t.groupby("lot")[c].agg(how).reindex(range(len(df)))
    df["fa0"], df["units0"], df["year0"] = by(res, "sf").to_numpy(), by(res, "units").to_numpy(), by(res, "yr", "max").to_numpy()
    tot = pd.to_numeric(av.board_tot, errors="coerce").fillna(pd.to_numeric(av.certified_tot, errors="coerce"))
    land = pd.to_numeric(av.board_land, errors="coerce").fillna(pd.to_numeric(av.certified_land, errors="coerce"))
    a2 = av.assign(t=tot, l=land)
    df["land_share0"] = np.divide(by(a2, "l").to_numpy(), by(a2, "t").to_numpy(), out=np.full(len(df), np.nan),
                                  where=np.nan_to_num(by(a2, "t").to_numpy()) > 0)
    df["lu16"] = np.array([TAX_INDEX.get(cmap(c, u), -1) if isinstance(c, str) else -1
                           for c, u in zip(code, df.units0.fillna(0))], dtype="int16")
    df["plan16"] = np.int16(-1)
    # new-construction permits: by their first PIN, else by their point
    pm = fetch(CHICAGO, "ydr8-5enu", RAW / "city" / "permits_new_construction.parquet",
               ["id", "permit_type", "issue_date", "pin_list", "latitude", "longitude", "reported_cost"],
               where="permit_type='PERMIT - NEW CONSTRUCTION' AND issue_date >= '2014-01-01'")
    pm["year"] = pd.to_datetime(pm.issue_date, errors="coerce").dt.year
    pm["lot"] = pm.pin_list.fillna("").astype(str).str.split(r"[|,; ]").str[0].map(digits).str[:10].map(lot_of)
    miss = pm.lot.isna().to_numpy()
    pts = _place(pd.to_numeric(pm.longitude, errors="coerce").to_numpy(), pd.to_numeric(pm.latitude, errors="coerce").to_numpy(), lots.geometry)
    pm.loc[miss, "lot"] = np.where(pts[miss] >= 0, pts[miss], np.nan)
    pm = pm.dropna(subset=["lot"]).astype({"lot": int})
    hit = lambda y0, y1: np.isin(np.arange(len(df)), pm.lot[(pm.year >= y0) & (pm.year < y1)].to_numpy())
    df["developed"] = hit(t0, t1)
    df["recent0"] = hit(t0 - 5, t0).astype(float)
    df["homes_added"] = np.nan
    df["dest16"] = np.int16(-1)
    df["observed"] = df.lu16.to_numpy() >= 0
    z = fetch_geojson(CHICAGO, "dj47-wfun", RAW / "city" / "zoning_districts.parquet").to_crs(lots.crs)
    created = pd.to_datetime(z.create_dat, errors="coerce").dt.year.to_numpy()
    k = _under(rp, z)
    df["zone_after_t0"] = pd.Series(created[k.to_numpy()] >= t0, index=k.index).reindex(range(len(df))).fillna(False).to_numpy()
    df["plan_after_t0"] = False
    df["t0"], df["t1"] = t0, t1
    return df


# city -> panel builder and windows: the first two are the temporal holdout (train, test), the last the full window
PANELS = {"0644000": (panel_los_angeles, [(2021, 2023), (2023, 2025), (2021, 2025)]),
          "4819000": (panel_dallas, [(2022, 2024), (2024, 2026), (2022, 2026)]),
          "0455000": (panel_phoenix, [(2022, 2024), (2024, 2026), (2022, 2026)]),
          "0666000": (panel_san_diego, [(2023, 2024), (2024, 2025), (2023, 2025)]),
          "3651000": (panel_new_york, [(2021, 2023), (2023, 2025), (2021, 2025)]),
          "5553000": (panel_milwaukee, [(2021, 2023), (2023, 2025), (2021, 2025)]),
          "1714000": (panel_chicago, [(2021, 2023), (2023, 2025), (2021, 2025)])}


def panel(geoid: str, t0: int, t1: int) -> pd.DataFrame:
    f = WORK / f"panel_{geoid}_{t0}_{t1}.parquet"
    if not f.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        PANELS[geoid][0](t0, t1).to_parquet(f)
    p = pd.read_parquet(f)
    keep = p.observed & ~p.underway0 if "underway0" in p else p.observed  # construction under way at t0: left out
    return p[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- features
def _onehot(cls: np.ndarray) -> np.ndarray:
    g = np.where(cls >= 0, GROUP_OF[np.clip(cls, 0, None)], -1)
    m = np.zeros((len(cls), N_GROUPS + 1), dtype="float32")
    m[np.arange(len(cls)), np.where(g >= 0, g, N_GROUPS)] = 1
    return m


def _context(p: pd.DataFrame) -> np.ndarray:
    """Recent development and neighbour composition within RADIUS_M (lot points, US feet)."""
    xy = np.c_[p.x.to_numpy(), p.y.to_numpy()]
    tree = shapely.STRtree(shapely.points(xy))
    i, k = tree.query(shapely.points(xy), predicate="dwithin", distance=RADIUS_M * FT_PER_M)
    if "recent0" in p and p.recent0.notna().any():  # land-use panels: lots that changed use in the window before t0
        recent = p.recent0.fillna(0).to_numpy()
    else:
        recent = ((p.year0.to_numpy() >= p.t0.iloc[0] - 5) & (p.year0.to_numpy() < p.t0.iloc[0])).astype(float)
    n = len(p)
    cnt = np.bincount(i, minlength=n).astype(float)
    rec = np.bincount(i, weights=recent[k], minlength=n) / np.maximum(cnt, 1)
    g = np.where(p.lu16.to_numpy() >= 0, GROUP_OF[np.clip(p.lu16.to_numpy(), 0, None)], -1)
    comp = np.zeros((n, N_GROUPS), dtype="float32")
    ok = g[k] >= 0
    np.add.at(comp, (i[ok], g[k][ok]), 1.0)
    comp /= np.maximum(comp.sum(1, keepdims=True), 1)
    return np.c_[rec, np.log1p(cnt), comp].astype("float32")


def blocks(p: pd.DataFrame) -> dict[str, np.ndarray]:
    lu, zn, pl = p.lu16.to_numpy(), p.zone16.to_numpy(), p.plan16.to_numpy()
    lot = np.maximum(p.lot_area.to_numpy(), 1.0)
    rank = lambda c: np.where(c >= 0, HOMES_RANK[np.clip(c, 0, None)], np.nan)
    g = lambda c: np.where(c >= 0, GROUP_OF[np.clip(c, 0, None)], -1)
    E = np.c_[_onehot(lu), np.log(lot), np.log1p(p.fa0.to_numpy()), p.fa0.to_numpy() / lot,
              p.t0.to_numpy() - p.year0.to_numpy(), p.units0.to_numpy(), p.land_share0.to_numpy()]
    Z = np.c_[_onehot(zn), np.where(zn >= 0, (g(zn) != g(lu)).astype(float), np.nan), rank(zn) - rank(lu)]
    P = np.c_[_onehot(pl), np.where(pl >= 0, (g(pl) != g(lu)).astype(float), np.nan), rank(pl) - rank(lu)]
    return {"E": E.astype("float32"), "C": _context(p), "Z": Z.astype("float32"), "P": P.astype("float32")}


# ----------------------------------------------------------------------------- metrics
def _auc(s, y):
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _ap(s, y):
    o = np.argsort(-s)
    hits = y[o]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / max(hits.sum(), 1))


def _top(s, y, share):
    k = max(1, int(round(share * len(s))))
    return float(y[np.argsort(-s)[:k]].mean())


def metrics(s, y) -> dict:
    return {"auc": _auc(s, y), "ap": _ap(s, y), "p_top1": _top(s, y, 0.01), "p_top5": _top(s, y, 0.05),
            "base_rate": float(y.mean())}


# ----------------------------------------------------------------------------- within a city
def crossfit(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    import lightgbm as lgb

    s = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def _gain_ci(s_full, s_base, y, tiles, metric, reps=200):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(tiles)
    pos = {t: np.nonzero(tiles == t)[0] for t in uniq}
    out = []
    for _ in range(reps):
        idx = np.concatenate([pos[t] for t in rng.choice(uniq, len(uniq))])
        out.append(metric(s_full[idx], y[idx]) - metric(s_base[idx], y[idx]))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


LOT_BINS = (0, 5_000, 10_000, 25_000, 100_000, np.inf)  # lot area, sq ft


def calibration(prob: np.ndarray, y: np.ndarray, prevalence: float, bins: int = 10) -> dict:
    """How well out-of-time probabilities match outcomes, not only how well they rank.

    reliability: lots in ten equal-count bins of predicted probability, mean predicted vs observed rate;
    ece: bin-size-weighted mean |predicted - observed| over those bins; observed_over_predicted: event totals;
    calibration intercept and slope: logistic regression of the outcome on logit(p) (perfect: 0 and 1);
    brier and brier_skill: against a constant forecast of the training-period prevalence."""
    from sklearn.linear_model import LogisticRegression

    prob = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, bool)
    order = np.argsort(prob)
    rel = []
    for chunk in np.array_split(order, bins):
        rel.append({"mean_predicted": float(prob[chunk].mean()), "observed_rate": float(y[chunk].mean()),
                    "n": int(len(chunk))})
    ece = float(sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in rel) / len(y))
    lg = np.log(prob / (1 - prob)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(lg, y)
    brier = float(np.mean((prob - y) ** 2))
    ref = float(np.mean((prevalence - y) ** 2))
    return {"bins": bins, "reliability": rel, "ece": ece,
            "observed_over_predicted": float(y.sum() / prob.sum()),
            "calibration_intercept": float(fit.intercept_[0]), "calibration_slope": float(fit.coef_[0, 0]),
            "brier": brier, "brier_prevalence": ref, "brier_skill": 1 - brier / ref if ref > 0 else None,
            "prevalence_train": prevalence, "prevalence_test": float(y.mean())}


def recalibrator(p_oof: np.ndarray, y: np.ndarray):
    """Logistic recalibration (intercept and slope on logit p) fitted on out-of-fold training-window probabilities;
    returns a function to apply to test-window probabilities. Uses no test-window information."""
    from sklearn.linear_model import LogisticRegression

    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p_oof), np.asarray(y, bool))
    return lambda q: fit.predict_proba(lg(q))[:, 1]


def yield_model(tr: pd.DataFrame):
    """Expected homes added if a lot develops, learned from the training window's developed lots: the mean in the
    cell of (zoning class, else existing class) x lot-size bin with at least 20 lots, else the class mean, else the
    overall mean. Applied to the test window, it turns a probability of development into expected homes."""
    d = tr[tr.developed.astype(bool) & tr.homes_added.notna()]
    cls = lambda q: np.where(q.zone16.to_numpy() >= 0, q.zone16.to_numpy(), q.lu16.to_numpy())
    size = lambda q: np.digitize(q.lot_area.to_numpy(), LOT_BINS[1:-1])
    t = pd.DataFrame({"c": cls(d), "s": size(d), "h": d.homes_added.to_numpy()})
    cell = t.groupby(["c", "s"]).h.agg(["mean", "size"])
    by_c = t.groupby("c").h.agg(["mean", "size"])
    overall = float(t.h.mean()) if len(t) else 0.0

    def predict(q: pd.DataFrame) -> np.ndarray:
        c, s_ = cls(q), size(q)
        out = np.full(len(q), overall)
        for i, (ci, si) in enumerate(zip(c, s_)):
            if (ci, si) in cell.index and cell.loc[(ci, si), "size"] >= 20:
                out[i] = cell.loc[(ci, si), "mean"]
            elif ci in by_c.index and by_c.loc[ci, "size"] >= 20:
                out[i] = by_c.loc[ci, "mean"]
        return out
    return predict


def scenario(p: pd.DataFrame, likely: np.ndarray, train: pd.DataFrame | None = None) -> dict:
    """Homes added by district: observed vs zoning and plan build-out, likelihood-weighted, the expected yield
    (probability x homes a developing lot of its kind added in the training window), and the trend."""
    acres = np.maximum(p.lot_area.to_numpy(), 1) / 43560
    dens = p.units0.to_numpy() / acres
    lu = p.lu16.to_numpy()
    cap = {}  # homes per acre each class reaches in the city today (90th percentile of lots of that class with homes)
    for c in range(len(CLASSES)):
        m = (lu == c) & (p.units0.to_numpy() > 0)
        cap[c] = float(np.percentile(dens[m], 90)) if m.sum() >= 30 else 0.0
    inc = lambda cls: np.clip(np.array([cap.get(int(c), 0.0) if c >= 0 else 0.0 for c in cls]) * acres - p.units0.to_numpy(), 0, None)
    zone_inc, plan_inc = inc(p.zone16.to_numpy()), inc(p.plan16.to_numpy())
    rec = (p.year0.to_numpy() >= p.t0.iloc[0] - 5) & (p.year0.to_numpy() < p.t0.iloc[0])
    years = p.t1.iloc[0] - p.t0.iloc[0]
    d = pd.DataFrame({"district": p.cd.to_numpy(), "observed": p.homes_added.to_numpy(),
                      "zoning_buildout": zone_inc, "plan_buildout": plan_inc,
                      "likely_zoning": likely * zone_inc, "likely_plan": likely * plan_inc,
                      "trend": np.where(rec, p.units0.to_numpy(), 0.0) * years / 5,
                      "likely_yield": likely * yield_model(train)(p) if train is not None else 0.0}).groupby("district").sum()
    res = {"city_total": {k: float(v) for k, v in d.sum().items()}, "capacity_homes_per_acre": {CLASSES[c].key: round(v, 1) for c, v in cap.items() if v}}
    for k in d.columns.drop("observed"):
        if d[k].sum() <= 0:
            continue
        res[k] = {"ratio_total": float(d[k].sum() / max(d.observed.sum(), 1)),
                  "district_spearman": float(d[k].rank().corr(d.observed.rank())),
                  "district_mae": float((d[k] - d.observed).abs().mean()),
                  "district_bias": float((d[k] - d.observed).mean()),
                  "district_median_ratio": float(np.median(d[k] / d.observed.clip(lower=1)))}
    return res


def destination(p: pd.DataFrame, b: dict, folds: np.ndarray, has: dict) -> dict:
    """Among lots developed in the window: how well does each layer, and each model, name what they became?
    Scored on the 16 classes and on the nine analysis groups."""
    import lightgbm as lgb

    m = p.developed.to_numpy() & (p.dest16.to_numpy() >= 0)
    y = p.dest16.to_numpy()[m]
    gy = GROUP_OF[y]
    acc = lambda pred: {"class": float((pred == y).mean()),
                        "group": float((np.where(pred >= 0, GROUP_OF[np.clip(pred, 0, None)], -1) == gy).mean()),
                        "coverage": float((pred >= 0).mean())}
    out = {"n": int(m.sum()), "layers": {"existing (no change)": acc(p.lu16.to_numpy()[m])}}
    if has["Z"]:
        out["layers"]["zoning class"] = acc(p.zone16.to_numpy()[m])
    if has["P"]:
        out["layers"]["plan class"] = acc(p.plan16.to_numpy()[m])
    out["models"] = {}
    fm = folds[m]
    for st in ("E", "E+C", "E+Z", "E+P", "E+Z+P", "E+C+Z+P"):
        parts = st.split("+")
        if not all(has[x] for x in parts):
            continue
        X = np.concatenate([b[x][m] for x in parts], 1)
        pred = np.full(len(y), -1)
        for f in np.unique(fm):
            tr, te = fm != f, fm == f
            if len(np.unique(y[tr])) < 2:
                continue
            clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                                     verbose=-1, random_state=SEED)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        out["models"][st] = acc(pred)
    return out


def _lgbm():
    import lightgbm as lgb

    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)


def _tiles(p: pd.DataFrame) -> np.ndarray:
    t = TILE_M * FT_PER_M
    return ((p.x // t).astype(int) * 10007 + (p.y // t).astype(int)).to_numpy()


def _available(p: pd.DataFrame, b: dict) -> dict:
    has = {k: bool(np.isfinite(v).any() and (np.nan_to_num(v) != 0).any()) for k, v in b.items()}
    has["Z"] = bool((p.zone16 >= 0).mean() > 0.5)
    has["P"] = bool((p.plan16 >= 0).mean() > 0.5)
    return has


def _ablation(scores: dict, y: np.ndarray, tiles: np.ndarray) -> tuple[dict, dict]:
    sets = {k: metrics(v, y) for k, v in scores.items()}
    gains = {}
    for add, base in (("E+C", "E"), ("E+Z", "E"), ("E+P", "E"), ("E+Z+P", "E+Z"), ("E+Z+P", "E+P"),
                      ("E+C+Z", "E+C"), ("E+C+P", "E+C"), ("E+C+Z+P", "E+C"), ("E+C+Z+P", "E")):
        if add in scores and base in scores:
            gains[f"{add} vs {base}"] = {
                "auc": [sets[add]["auc"] - sets[base]["auc"], _gain_ci(scores[add], scores[base], y, tiles, _auc)],
                "ap": [sets[add]["ap"] - sets[base]["ap"], _gain_ci(scores[add], scores[base], y, tiles, _ap)]}
    return sets, gains


def evaluate_city(geoid: str) -> dict:
    """Spatial cross-fitting on the full window, and the temporal holdout (train on the first window, forecast the
    second) for the ablation and the scenario assessment."""
    t_start = time.time()
    builder, windows = PANELS[geoid]
    res = {"geoid": geoid}
    # (1) full window, 5-fold spatial-block cross-fitting
    p = panel(geoid, *windows[-1])
    y = p.developed.to_numpy().astype(bool)
    b, folds, tiles = blocks(p), spatial_folds(p, tile_m=TILE_M), _tiles(p)
    has = _available(p, b)
    scores = {}
    for st in SETS:
        parts = st.split("+")
        if all(has[x] for x in parts):
            scores[st] = crossfit(np.concatenate([b[x] for x in parts], 1), y, folds)
            print(f"  {geoid} spatial {st:10s} AUC {_auc(scores[st], y):.3f}", flush=True)
    sets, gains = _ablation(scores, y, tiles)
    res["spatial"] = {"window": list(windows[-1]), "n_lots": int(len(p)), "developed": int(y.sum()),
                      "homes_added": float(np.nansum(p.homes_added)), "blocks": has, "sets": sets, "gains": gains,
                      "destination": destination(p, b, folds, has)}
    # (2) temporal holdout: fitted on the first window, scored on the second
    if len(windows) >= 3:
        tr, te = panel(geoid, *windows[0]), panel(geoid, *windows[1])
        btr, bte = blocks(tr), blocks(te)
        ytr, yte = tr.developed.to_numpy().astype(bool), te.developed.to_numpy().astype(bool)
        scores = {}
        for st in SETS:
            parts = st.split("+")
            if all(has[x] for x in parts):
                m = _lgbm().fit(np.concatenate([btr[x] for x in parts], 1), ytr)
                scores[st] = m.predict_proba(np.concatenate([bte[x] for x in parts], 1))[:, 1]
                print(f"  {geoid} temporal {st:10s} AUC {_auc(scores[st], yte):.3f}", flush=True)
        sets, gains = _ablation(scores, yte, _tiles(te))
        full = max(scores, key=len)
        homes = bool(te.homes_added.notna().any())
        res["temporal"] = {"train": list(windows[0]), "test": list(windows[1]), "n_lots": int(len(te)),
                           "developed": int(yte.sum()), "homes_added": float(np.nansum(te.homes_added)),
                           "sets": sets, "gains": gains, "scenario_model": full,
                           "scenario": scenario(te, scores[full], tr) if homes else None,
                           "calibration": {k: calibration(scores[k], yte, float(ytr.mean())) for k in ("E", full)},
                           "scenario_existing_only": scenario(te, scores["E"]) if homes and "E" in scores else None,
                           "destination": destination(te, bte, spatial_folds(te, tile_m=TILE_M), has)}
        # conservative: lots whose zoning or plan changed after t0 (current layers only) are left out of both windows
        if "zone_after_t0" in te and (te.zone_after_t0.any() or te.plan_after_t0.any()):
            ktr = ~(tr.zone_after_t0.to_numpy() | tr.plan_after_t0.to_numpy())
            kte = ~(te.zone_after_t0.to_numpy() | te.plan_after_t0.to_numpy())
            sc = {}
            for st in scores:
                parts = st.split("+")
                m = _lgbm().fit(np.concatenate([btr[x][ktr] for x in parts], 1), ytr[ktr])
                sc[st] = m.predict_proba(np.concatenate([bte[x][kte] for x in parts], 1))[:, 1]
            sets_c, gains_c = _ablation(sc, yte[kte], _tiles(te)[kte])
            res["temporal_conservative"] = {"left_out_test": int((~kte).sum()), "left_out_developed": int(yte[~kte].sum()),
                                            "sets": sets_c, "gains": gains_c}
    res["seconds"] = round(time.time() - t_start, 1)
    return res


BUILDING_OUTCOME = ("0644000", "4819000", "3651000", "5553000", "1714000")  # a new or enlarged building (Chicago: permit)


def transfer(cities=BUILDING_OUTCOME) -> dict:
    """Leave-one-city-out: fitted on the other cities' first windows, scored on the held-out city's second window,
    against the city's own out-of-time model. The plan block is missing (unknown) where a city has no plan."""
    data = {}
    for g in cities:
        w = PANELS[g][1]
        tr, te = panel(g, *w[0]), panel(g, *w[1])
        data[g] = (blocks(tr), tr.developed.to_numpy().astype(bool), blocks(te), te.developed.to_numpy().astype(bool))
    out = {}
    for g in cities:
        out[g] = {}
        for st in ("E", "E+C", "E+C+Z", "E+C+Z+P"):
            parts = st.split("+")
            X = np.concatenate([np.concatenate([data[o][0][x] for x in parts], 1) for o in cities if o != g])
            Y = np.concatenate([data[o][1] for o in cities if o != g])
            m = _lgbm().fit(X, Y)
            s_other = m.predict_proba(np.concatenate([data[g][2][x] for x in parts], 1))[:, 1]
            own = _lgbm().fit(np.concatenate([data[g][0][x] for x in parts], 1), data[g][1])
            s_own = own.predict_proba(np.concatenate([data[g][2][x] for x in parts], 1))[:, 1]
            y = data[g][3]
            out[g][st] = {"other_cities": metrics(s_other, y), "own_city": metrics(s_own, y)}
            print(f"  transfer {g} {st:9s} other {out[g][st]['other_cities']['auc']:.3f} own {out[g][st]['own_city']['auc']:.3f}", flush=True)
    return out


def profile() -> dict:
    """What distinguishes the lots that developed in each test window from all lots: the share that were vacant,
    and the share whose zoning or plan class lies in another analysis group than the existing use."""
    out = {}
    for g, (_, windows) in PANELS.items():
        p = panel(g, *windows[1])
        d = p.developed.to_numpy().astype(bool)
        grp = lambda c: np.where(c >= 0, GROUP_OF[np.clip(c, 0, None)], -1)
        lu = p.lu16.to_numpy()
        row = {"developed": int(d.sum()), "lots": int(len(p)),
               "vacant": {"developed": float((lu[d] == VACANT).mean()), "all": float((lu == VACANT).mean())}}
        for k in ("zone16", "plan16"):
            c = p[k].to_numpy()
            if (c >= 0).mean() > 0.5:
                differs = (c >= 0) & (grp(c) != grp(lu))
                row[k.replace("16", "_differs")] = {"developed": float(differs[d].mean()), "all": float(differs.mean())}
        out[g] = row
    return out


def main(geoids=None) -> dict:
    """Evaluate the cities (one results file each, so cities can run in parallel), then merge them."""
    geoids = geoids or list(PANELS)
    per_city = OUT.parent / "integration"
    per_city.mkdir(parents=True, exist_ok=True)
    for g in [x for x in geoids if x not in ("transfer", "profile")]:
        print(f"== {g}", flush=True)
        (per_city / f"{g}.json").write_text(json.dumps(evaluate_city(g), indent=2))
    if geoids == ["profile"]:
        (OUT.parent / "integration_profile.json").write_text(json.dumps(profile(), indent=2))
        return json.loads(OUT.read_text())
    if geoids == ["transfer"]:
        (per_city / "transfer.json").write_text(json.dumps(transfer(), indent=2))
    merged = {f.stem: json.loads(f.read_text()) for f in sorted(per_city.glob("*.json"))}
    OUT.write_text(json.dumps(merged, indent=2))
    return merged


if __name__ == "__main__":
    r = main(sys.argv[1:] or None)
    for g, x in r.items():
        if g == "transfer":
            continue
        for kind in ("spatial", "temporal"):
            if kind in x:
                print(g, kind, {k: round(v["auc"], 3) for k, v in x[kind]["sets"].items()})
