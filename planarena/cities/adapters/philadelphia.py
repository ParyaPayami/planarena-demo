"""Tier B adapter: Philadelphia, PA, from the Office of Property Assessment (OPA) records and PWD parcels.

Sources (OpenDataPhilly): OPA property assessments (category, building description, livable area, lot area, stories,
year built, zoning, market value; a point per account) and the Water Department parcel polygons. Each OPA point is
placed in the parcel that contains it, so condominium units and several accounts on one parcel are summed. Owner and
mailing fields are never read.

Land use: the City Planning Commission's parcel land-use layer (official), translated to the 16 classes; OPA
categories split single- from multi-family homes where only the two-digit code is given.
Homes: OPA has no dwelling-unit count. Single-family records count as one home each; the rest of each census block's
2020 housing units are allocated to its multi-family and mixed-use parcels by livable area (estimated, flagged).
Zoning: district families (Philadelphia Zoning Code; codes are written without hyphens, e.g. RSA5, CMX3, SPPOA) set
the allowed uses; capacity uses the context FAR.
Adopted plan: Philadelphia2035's land-use recommendations exist only as the district plans' PDF maps; they are
digitised by philadelphia_plan.py (run it after a first build, then rebuild) and joined by parcel.

Run:  python -I -m planarena.cities.adapters.philadelphia
"""
from __future__ import annotations

import json
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.common import (districts, finish_twin, local_crs, plausible_floor_area, reconcile_lot_area,
                                     round_by_block)
from planarena.cities.registry import city_dir, register
from planarena.sources import crosswalk
from planarena.sources.arcgis import fetch_layer
from planarena.taxonomy import INDEX as TAX_INDEX, group_of
from planarena.config import DATA, USE_INDEX as U, USE_KEYS, WEB_CRS
from planarena.national import census, places
from planarena.national.common import download
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal

GEOID = "4260000"
LAND_USE_URL = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Land_Use/FeatureServer/0"
RAW = DATA / "raw" / "philadelphia"
PLAN = city_dir(GEOID) / "plan_digitised.parquet"  # written by philadelphia_plan.py
OPA_URL = "https://opendata-downloads.s3.amazonaws.com/opa_properties_public.csv"
PWD_URL = ("https://hub.arcgis.com/api/v3/datasets/84baed491de44f539889f2af178ad85c_0/downloads/data"
           "?format=shp&spatialRefId=4326&where=1%3D1")
COLS = ["parcel_number", "category_code", "building_code_description_new", "total_livable_area", "total_area",
        "number_stories", "year_built", "market_value", "taxable_land", "exempt_land", "zoning", "location", "shape"]
ALL = (1 << len(USE_KEYS)) - 1
bit = lambda *keys: sum(1 << U[k] for k in keys)
ZONING = [("SPPO", bit("opn", "pub", "vac")), ("RSD", bit("res", "pub", "opn", "vac")),
          ("RSA", bit("res", "pub", "opn", "vac")), ("RTA", bit("res", "pub", "opn", "vac")),
          ("RM", bit("res", "mix", "pub", "opn", "prk", "vac")), ("RMX", bit("res", "mix", "com", "pub", "opn", "prk", "vac")),
          ("CMX", bit("res", "mix", "com", "pub", "opn", "prk", "vac")), ("CA", bit("mix", "com", "pub", "opn", "prk", "vac")),
          ("IRMX", bit("res", "mix", "com", "ind", "pub", "opn", "prk", "vac")),
          ("ICMX", bit("mix", "com", "ind", "tra", "pub", "opn", "prk", "vac")),
          ("I1", bit("com", "ind", "tra", "pub", "opn", "prk", "vac")), ("I2", bit("com", "ind", "tra", "pub", "opn", "prk", "vac")),
          ("I3", bit("ind", "tra", "pub", "opn", "prk", "vac")),
          ("IP", bit("ind", "tra", "pub", "opn", "prk", "vac"))]


def allowed(zone: str) -> int:
    z = zone.upper() if isinstance(zone, str) else ""
    for prefix, mask in ZONING:
        if z.startswith(prefix):
            return mask
    return ALL  # special-purpose and unknown districts: not screened


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    crs = local_crs(pl["lat"], pl["lon"])
    log("downloading OPA assessments and PWD parcels")
    opa = pd.read_csv(download(OPA_URL, RAW / "opa_properties_public.csv"), usecols=COLS, dtype=str)
    z = download(PWD_URL, RAW / "pwd_parcels.zip")
    ext = RAW / "pwd_parcels"
    if not ext.exists():
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if ".." not in n and not n.startswith("/"):
                    zf.extract(n, ext)
    g = gpd.read_file(next(ext.rglob("*.shp")), columns=["parcelid", "address"], engine="pyogrio")
    g = g[g.geometry.notna()].to_crs(crs).reset_index(drop=True)
    g["geometry"] = g.geometry.make_valid().map(_polygonal)
    pts = gpd.GeoSeries.from_wkt(opa["shape"].fillna("POINT EMPTY").str.split(";").str[-1], crs=2272).to_crs(crs)
    ok = pts.notna() & ~pts.is_empty
    tree = shapely.STRtree(g.geometry.values)
    i, k = tree.query(pts[ok].values, predicate="within")
    lot_of = np.full(len(opa), -1)
    lot_of[np.nonzero(ok.to_numpy())[0][i]] = k
    log(f"{len(opa):,} OPA accounts; {int((lot_of >= 0).sum()):,} placed in {len(g):,} parcels")
    for c in ("total_livable_area", "total_area", "number_stories", "year_built", "market_value", "taxable_land", "exempt_land"):
        opa[c] = pd.to_numeric(opa[c], errors="coerce").fillna(0)
    opa["cat"] = pd.to_numeric(opa.category_code, errors="coerce")
    opa["sf"] = (opa.cat == 1).astype(float)
    opa["lot"] = lot_of
    placed = opa[opa.lot >= 0]
    # most common OPA category per parcel (splits single- from multi-family where the land-use code does not)
    m = placed.dropna(subset=["cat"]).groupby(["lot", "cat"]).size().reset_index(name="n")
    cat_mode = m.sort_values(["lot", "n"], ascending=[True, False]).drop_duplicates("lot").set_index("lot").cat
    a = placed.groupby("lot").agg(
        fa=("total_livable_area", "sum"), lot_sqft=("total_area", "max"), floors=("number_stories", "max"),
        year=("year_built", "max"), value=("market_value", "sum"), land=("taxable_land", "sum"), exland=("exempt_land", "sum"),
        zone=("zoning", "first"), opa_address=("location", "first"), single_family=("sf", "sum"), accounts=("cat", "size"))
    a["opa_cat"] = cat_mode.reindex(a.index)
    md = placed.dropna(subset=["building_code_description_new"]).groupby(["lot", "building_code_description_new"]).size()
    md = md.reset_index(name="n").sort_values(["lot", "n"], ascending=[True, False]).drop_duplicates("lot")
    a["opa_desc"] = md.set_index("lot").building_code_description_new.reindex(a.index)
    p = g.join(a, how="left")
    p["address"] = p.address.where(p.address.notna() & (p.address.astype(str).str.len() > 0), p.opa_address)
    # official land use (City Planning Commission): the land-use polygon under each parcel's representative point
    log("downloading the city land-use layer")
    lu = fetch_layer(LAND_USE_URL, RAW / "land_use" / "land_use.parquet",
                     ["objectid", "c_dig1", "c_dig2", "c_dig3", "year"]).to_crs(crs)
    i, k = shapely.STRtree(lu.geometry.values).query(p.geometry.representative_point().values, predicate="within")
    # where polygons overlap (street right-of-way drawn over blocks), the smallest one containing the point is the parcel
    hits = pd.DataFrame({"i": i, "k": k, "area": lu.geometry.area.to_numpy()[k]}).sort_values(["i", "area"])
    under = hits.drop_duplicates("i").set_index("i").k
    p["lu_d2"] = pd.Series(lu.c_dig2.to_numpy()[under.to_numpy()], index=under.index).reindex(p.index)
    p["lu_d3"] = pd.Series(lu.c_dig3.to_numpy()[under.to_numpy()], index=under.index).reindex(p.index)
    p["lu16"] = [TAX_INDEX.get(crosswalk.philadelphia(d2, d3, c, d), -1)
                 for d2, d3, c, d in zip(p.lu_d2, p.lu_d3, p.opa_cat, p.opa_desc)]
    p["use"] = [USE_KEYS[group_of(c)] if c >= 0 else "unknown" for c in p.lu16]
    log(f"{int(p.lu_d2.notna().sum()):,} of {len(p):,} parcels have an official land-use code")
    log(f"{int(p.accounts.notna().sum()):,} parcels with OPA records")

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
    jobs_block = census.lodes_jobs("42", "PA").reindex(blocks.GEOID).fillna(0).to_numpy()

    use_a = p.use.map(U).fillna(-1).astype(int).to_numpy()
    lot, n_lot_fixed = reconcile_lot_area(p.lot_sqft.fillna(0).to_numpy(), p.geometry.area.to_numpy())
    fa, n_fa_dropped = plausible_floor_area(p.fa.fillna(0).to_numpy().astype(float), lot)
    res_area = np.where(use_a == U["res"], fa, np.where(use_a == U["mix"], 0.7 * fa, 0.0))  # mixed: assumption
    # homes: single-family records count one each; the rest of each block's housing units go to its multi-family and
    # mixed parcels by residential floor area
    sf = p.single_family.fillna(0).to_numpy()
    other_res = np.where((use_a == U["res"]) & (sf == 0) | (use_a == U["mix"]), res_area, 0.0)
    left = np.clip(blocks.HU100.to_numpy() - pd.Series(sf).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy(), 0, None)
    tot = pd.Series(other_res).groupby(blk).sum().reindex(range(len(blocks)), fill_value=0).to_numpy()
    units = sf + np.divide(other_res * left[blk], tot[blk], out=np.zeros(len(p)), where=tot[blk] > 0)
    units = round_by_block(units, blk)
    nonres = fa - res_area
    com_like = np.isin(use_a, [U["com"], U["mix"]])
    ll = gpd.GeoSeries(rp, crs=crs).to_crs(WEB_CRS)
    zone = p.zone.fillna("NA").astype(str)
    df = pd.DataFrame({
        "bbl": np.arange(len(p), dtype="int64"), "address": p.address.fillna("").astype(str).to_numpy(), "cd": 0,
        "block": blk.astype("int32"), "zone": zone.to_numpy(), "overlay": "", "special": "", "bldg_class": "",
        "use": use_a.astype("int16"), "lu16": p.lu16.to_numpy().astype("int16"),
        "lu_code": (p.lu_d3.fillna(p.lu_d2).fillna(-1).astype(int).astype(str)).to_numpy(),
        "lot_area": lot, "bldg_area": fa, "res_area": res_area,
        "com_area": np.where(com_like, nonres, 0.0), "office_area": np.where(com_like, 0.5 * nonres, 0.0),
        "retail_area": np.where(com_like, 0.3 * nonres, 0.0), "other_area": np.where(com_like, 0.2 * nonres, 0.0),
        "factory_area": np.where(use_a == U["ind"], nonres, 0.0), "storage_area": 0.0, "garage_area": 0.0,
        "num_bldgs": (fa > 0).astype(float), "floors": p.floors.fillna(0).to_numpy(), "units_res": units,
        "units_total": units, "assess_land": (p.land.fillna(0) + p.exland.fillna(0)).to_numpy(),
        "assess_tot": p.value.fillna(0).to_numpy(), "year_built": p.year.fillna(0).to_numpy(),
        "built_far": np.divide(fa, lot, out=np.zeros(len(p)), where=lot > 0), "historic": False, "landmark": False,
        "x": rp.x.to_numpy(), "y": rp.y.to_numpy(), "lon": ll.x.to_numpy(), "lat": ll.y.to_numpy(),
        "height_m": p.floors.fillna(0).to_numpy() * FLOOR_HEIGHT_M, "census_block": blocks.GEOID.to_numpy()[blk],
    })
    bg = df.census_block.str[:12]
    built = df.bldg_area > 0
    ctx = df[built].groupby(bg[built]).built_far.quantile(0.9)
    df["max_far"] = np.maximum(bg.map(ctx).fillna(df.built_far[built].quantile(0.9)).to_numpy(), df.built_far.to_numpy())
    df["resid_far"] = df["comm_far"] = df["facil_far"] = df["max_far"]
    df["allowed"] = zone.map(allowed).to_numpy().astype("int64")
    df["zone16"] = np.array([TAX_INDEX.get(crosswalk.philadelphia_zone(z), -1) for z in zone], dtype="int16")
    plan_report = {}
    if PLAN.exists():  # the adopted plan, digitised from the district plans' PDF maps (philadelphia_plan.py)
        dig = pd.read_parquet(PLAN)
        lab = df.bbl.map(dict(zip(dig.bbl, dig.plan_label))).fillna("").to_numpy()
        df["plan_designation"] = lab
        df["plan16"] = np.array([TAX_INDEX.get(crosswalk.philadelphia_plan(l), -1) if l else -1 for l in lab],
                                dtype="int16")
        plan_report = json.loads(PLAN.with_suffix(".json").read_text())
    df["cd"] = districts(blocks, blk)
    meta = finish_twin(df, p.geometry.values, crs, out, "PA", float(blocks.POP100.sum() / max(blocks.HU100.sum(), 1)),
                       jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": "Philadelphia OPA property assessments and PWD parcels (OpenDataPhilly)",
        "source_url": "https://opendataphilly.org/datasets/philadelphia-properties-and-assessment-history/",
        "data_quality": {"missing_land_use": int((df.use < 0).sum()), "opa_accounts": int(len(opa)),
                         "accounts_not_in_a_parcel": int((lot_of < 0).sum()),
                         "parcels_without_opa_record": int(p.accounts.isna().sum()),
                         "lot_area_from_polygon": n_lot_fixed, "floor_area_implausible_dropped": n_fa_dropped,
                         "homes_census_2020": int(blocks.HU100.sum()),
                         "missing_plan": int((df.plan16 < 0).sum()) if "plan16" in df else int(len(df)),
                         "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (PWD parcels)", "floor area": "observed (OPA livable area)",
                    "land use": "observed (City Planning Commission land-use layer; OPA splits homes)",
                    "homes": "single-family observed; other homes estimated from Census 2020 block housing units",
                    "jobs": "observed by block (LODES), modelled by floor area", "assessed value": "observed (OPA market value)",
                    "zoning": "simplified (district families set allowed uses; context FAR for capacity)",
                    "adopted plan": "digitised from the Philadelphia2035 district plans' Proposed Land Use maps (PDF); "
                                    "registration error and agreement checks per district in plan_digitised.json"
                                    if plan_report else "missing (published only as PDF maps)",
                    "year built": "observed", "landmarks and historic districts": "missing",
                    "commercial mix": "assumed 50% office, 30% retail, 20% other"},
        "plan_digitised": {k: {x: y for x, y in v.items() if x not in ("legend", "colours")} for k, v in plan_report.items()},
        "missing_indicators": [], "zoning_rules": "Philadelphia Zoning Code district families (simplified)",
        "levels": {str(d): f"Area {d}" for d in sorted(df.cd.unique())},
    }, t0=t0)
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {len(df):,} parcels")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "use_counts", "data_quality", "build_seconds")}, indent=1))
