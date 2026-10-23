"""Steps every city builder shares: graph, files, web layers, coefficients, jobs rates and metadata."""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

from planarena.config import USE_KEYS, WEB_CRS
from planarena.taxonomy import KEYS as TAX_KEYS
from planarena.national import census, energy
from planarena.twin.build import FLOOR_HEIGHT_M, _polygonal, add_grid, build_graph


def reconcile_lot_area(recorded: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, int]:
    """Recorded lot area where it agrees with the parcel polygon within a factor of 3, else the polygon's area
    (assessor files carry some lot areas that are off by orders of magnitude). Returns the areas and how many were
    replaced."""
    ok = (recorded > 0) & (recorded <= 3 * polygon) & (recorded >= polygon / 3)
    return np.where(ok, recorded, polygon).astype(float), int((~ok).sum())


def local_crs(lat: float, lon: float) -> str:
    """Transverse Mercator centred on the city, in US feet (the units the engine uses everywhere)."""
    return f"+proj=tmerc +lat_0={lat:.5f} +lon_0={lon:.5f} +k=1 +x_0=0 +y_0=0 +datum=NAD83 +units=us-ft +no_defs"


def round_by_block(x: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """Integer units whose block sums equal the rounded block totals (largest remainder)."""
    out = np.floor(x)
    rem = x - out
    target = pd.Series(x).groupby(blk).sum().round()
    short = (target - pd.Series(out).groupby(blk).sum()).astype(int)
    order = pd.DataFrame({"b": blk, "r": rem}).sort_values("r", ascending=False)
    order["rank"] = order.groupby("b").cumcount()
    add = order[order["rank"] < order.b.map(short)].index
    out[add] += 1
    return out


def districts(blocks: gpd.GeoDataFrame, blk: np.ndarray) -> np.ndarray:
    """Planning areas where the city publishes none: census tracts grouped by k-means on their centroids, about
    75,000 residents each (1-15)."""
    from scipy.cluster.vq import kmeans2

    tract = blocks.GEOID.str[:11]
    t = pd.DataFrame({"tract": tract, "x": blocks.geometry.centroid.x, "y": blocks.geometry.centroid.y,
                      "pop": blocks.POP100}).groupby("tract").agg(x=("x", "mean"), y=("y", "mean"), pop=("pop", "sum"))
    k = min(int(np.clip(round(t["pop"].sum() / 75000), 1, 15)), len(t))
    lab = np.zeros(len(t), int) if k == 1 else kmeans2(t[["x", "y"]].to_numpy(), k, seed=7, minit="++")[1]
    code = dict(zip(t.index, lab + 1))
    return tract.map(code).to_numpy()[blk].astype("int32")


def overture_buildings(geoid: str, geoms, crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Per lot: floor area (footprint x floors) and the most floors of the Overture buildings whose representative
    point lies in it. Floors are Overture's num_floors, else height / floor height, else 1 (estimated). NaN floors
    where a lot has no building. Used only where the city's records have no building data."""
    from planarena.national import overture

    lots = gpd.GeoSeries(geoms, crs=crs)
    bld = overture.fetch(geoid, tuple(lots.to_crs(4326).total_bounds), "buildings")
    bld = bld[~bld.is_underground.fillna(False).astype(bool)].to_crs(crs)
    fl = pd.to_numeric(bld.num_floors, errors="coerce").astype(float).fillna(
        pd.to_numeric(bld.height, errors="coerce").astype(float) / FLOOR_HEIGHT_M)
    fl = fl.round().clip(lower=1).fillna(1).to_numpy()
    i, k = shapely.STRtree(lots.values).query(bld.geometry.representative_point().values, predicate="within")
    o = pd.DataFrame({"lot": k, "fa": bld.geometry.area.to_numpy()[i] * fl[i], "fl": fl[i]})
    n = len(lots)
    return o.groupby("lot").fa.sum().reindex(range(n)).fillna(0).to_numpy(), o.groupby("lot").fl.max().reindex(range(n)).to_numpy()


MAX_PLAUSIBLE_FAR = 50.0  # above any US building (the tallest Manhattan towers are near FAR 30-40 on their lots)


def plausible_floor_area(floor_area: np.ndarray, lot_area: np.ndarray) -> tuple[np.ndarray, int]:
    """Floor area above MAX_PLAUSIBLE_FAR times the lot is a recording error (one Florida record lists 268 million
    sq ft on a 26,524 sq ft lot); it is treated as unrecorded (0). Returns the areas and how many were dropped."""
    bad = (floor_area > MAX_PLAUSIBLE_FAR * np.maximum(lot_area, 1.0))
    return np.where(bad, 0.0, floor_area), int(bad.sum())


def finish_twin(df: pd.DataFrame, geoms, crs: str, out: Path, state: str, pph: float, jobs_block: np.ndarray,
                blk: np.ndarray, log, meta: dict, t0: float | None = None, minimal: bool = False) -> dict:
    """Write a built city: parcels, geometry, graph, web layers, calibrated.yaml and twin_meta.json.

    df follows the parcels.parquet schema (planarena/twin/build.py tabulate) plus census_block; geoms are lot
    polygons in crs (US feet); jobs_block[k] is the LODES jobs of census block k, and blk[i] lot i's block index.
    """
    t0 = t0 or time.time()
    # the three official layers: existing use (lu16), adopted plan (plan16, with the plan's own designation) and the
    # zoning district's use class (zone16); -1 where the city has no such layer
    for c in ("lu16", "plan16", "zone16"):
        df[c] = df[c].astype("int16") if c in df else np.full(len(df), -1, dtype="int16")
    if "plan_designation" not in df:
        df["plan_designation"] = ""
    grid = add_grid(df)
    gl = gpd.GeoDataFrame({"i": np.arange(len(df))}, geometry=geoms, crs=crs)
    graph = build_graph(gl, df)
    df.to_parquet(out / "parcels.parquet", index=False)
    gl.to_parquet(out / "parcels.geoparquet")
    np.savez_compressed(out / "graph.npz", **graph)
    if minimal:  # historical snapshot: no web layers, coefficients or jobs fit
        meta = meta | {"crs": crs, "n_parcels": int(len(df)), "grid": grid,
                       "graph_edges": {k: int(v.shape[1]) for k, v in graph.items()}}
        (out / "twin_meta.json").write_text(json.dumps(meta, indent=2))
        return meta
    write_web(gl, df, out / "web")
    log("twin written")
    coef = energy.coefficients(state)
    coef["persons_per_household"] = {"value": round(pph, 2), "low": round(pph * 0.95, 2), "high": round(pph * 1.05, 2),
                                     "status": "derived",
                                     "source": "Census 2020 population per housing unit in the city's blocks "
                                               "(includes vacant units, so it understates household size)."}
    coef["calibrated_on"] = time.strftime("%Y-%m-%d")
    bounds = gpd.GeoSeries(gl.geometry, crs=crs).to_crs(WEB_CRS).total_bounds
    meta = meta | {
        "crs": crs, "n_parcels": int(len(df)),
        "use_counts": {USE_KEYS[u] if u >= 0 else "unknown": int(n) for u, n in df.use.value_counts().sort_index().items()},
        "class_counts": _counts(df.lu16), "plan_counts": _counts(df.plan16), "zone_counts": _counts(df.zone16),
        "bbox_wgs84": [float(x) for x in bounds], "grid": grid,
        "graph_edges": {k: int(v.shape[1]) for k, v in graph.items()}, "floor_height_m": FLOOR_HEIGHT_M,
    }
    (out / "twin_meta.json").write_text(json.dumps(meta, indent=2))
    _write_coef(out, coef)  # the engine needs a coefficient file to fit the jobs rates
    coef["jobs_sqft_per_worker"] = jobs_rates(out, jobs_block, blk)
    _write_coef(out, coef)
    meta["build_seconds"] = round(time.time() - t0, 1)
    (out / "twin_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "web" / "twin_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _counts(cls: pd.Series) -> dict:
    """Parcels per 16-class key; empty when the layer is missing."""
    if (cls < 0).all():
        return {}
    return {TAX_KEYS[u] if u >= 0 else "unknown": int(n) for u, n in cls.value_counts().sort_index().items()}


def _write_coef(out: Path, coef: dict) -> None:
    (out / "calibrated.yaml").write_text("# GENERATED by planarena.cities. Do not edit by hand.\n"
                                         + yaml.safe_dump(coef, sort_keys=False))


def jobs_rates(out: Path, jobs_block: np.ndarray, blk: np.ndarray) -> dict:
    """Floor area per worker by type, fitted to the place's LODES jobs by block (as for NYC)."""
    from scipy.optimize import nnls

    from planarena.indicators.engine import load_engine

    e = load_engine(out)
    keys = {"office": "office", "retail": "retail", "other": "other_commercial", "factory": "factory",
            "storage": "storage", "pubfac": "public_facility", "trafac": "transport_utility"}
    X = pd.DataFrame({c: e.obs[c] for c in keys}).groupby(blk).sum()
    y = pd.Series(jobs_block).reindex(X.index).fillna(0).to_numpy()
    cols = [c for c in keys if X[c].sum() > 0]
    rates, _ = nnls(X[cols].to_numpy(), y)
    fallback = {"office": 609, "retail": 1700, "other_commercial": 1378, "factory": 1429, "storage": 2400,
                "public_facility": 1060, "transport_utility": 1429}
    n_jobs = int(round(jobs_block[np.unique(blk)].sum()))  # blocks that hold at least one lot
    res = {"status": "calibrated",
           "source": f"Relative rates fitted to LEHD LODES8 {census.LODES_YEAR} workplace jobs by 2020 census block "
                     "(non-negative least squares on the twin's estimated floor-area components); total scaled to the "
                     f"place's LODES total ({n_jobs:,}). A type with no floor area or a zero fitted rate takes its "
                     "EIA CBECS 2018 value. Range +/-25% (assumption).",
           "calibration_target_jobs": n_jobs}
    fit = dict(zip(cols, rates))
    bounded = []
    for c, k in keys.items():
        r = fit.get(c, 0.0)
        v = float(round(1.0 / r, 1)) if r > 1e-9 else float(fallback[k])
        # a fitted rate more than 4x away from the CBECS value means the block fit cannot separate this type (e.g.
        # little floor area of it, or its jobs counted at other addresses): the CBECS value is used instead
        if not fallback[k] / 4 <= v <= fallback[k] * 4:
            bounded.append(f"{k} (fitted {v:,.0f})")
            v = float(fallback[k])
        res[k] = {"value": v, "low": round(v * 0.8, 1), "high": round(v * 1.25, 1)}
    if bounded:
        res["replaced_by_cbecs"] = ", ".join(bounded)
        res["source"] += " Fitted rates more than 4x from the CBECS value were replaced by it: " + res["replaced_by_cbecs"] + "."
    return res


def write_web(gl: gpd.GeoDataFrame, df: pd.DataFrame, web: Path) -> None:
    import shutil

    web.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(web / "tiles", ignore_errors=True)  # vector tiles are cached on demand; a rebuild invalidates them
    (web / "levels").mkdir(exist_ok=True)
    w = gpd.GeoDataFrame({"i": np.arange(len(df), dtype="int32"), "u": df.use.to_numpy(), "h": df.height_m.round(1).to_numpy(),
                          "cd": df.cd.to_numpy()},
                         geometry=gl.geometry.simplify(1.5, preserve_topology=True).map(_polygonal).values, crs=gl.crs).to_crs(WEB_CRS)
    g = shapely.make_valid(w.geometry.values)
    snapped = shapely.set_precision(g, 1e-6, mode="pointwise")
    ok = shapely.is_valid(snapped) & ~shapely.is_empty(snapped)
    w["geometry"] = np.where(ok, snapped, g)
    opts = {"COORDINATE_PRECISION": 6, "RFC7946": "YES"}
    from planarena.cities.tiles import map_mode

    if map_mode(len(df)) == "geojson":  # large cities are drawn from vector tiles and points instead
        w.to_file(web / "parcels.geojson", driver="GeoJSON", engine="pyogrio", layer_options=opts)
    for cd, part in w.groupby("cd"):
        part.to_file(web / "levels" / f"{cd}.geojson", driver="GeoJSON", engine="pyogrio", layer_options=opts)


