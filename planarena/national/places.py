"""Every US place: the Census gazetteer for search, TIGERweb for boundaries.

Places are incorporated places and census-designated places (2024 gazetteer), plus county subdivisions in the
six New England states, where towns are the unit of local government.
"""
from __future__ import annotations

import io
import re
import zipfile
from functools import lru_cache

import pandas as pd

from planarena.national.common import NATIONAL, arcgis_query, download

GAZ = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/"
TIGERWEB = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
PLACE_LAYERS = {"place": [TIGERWEB + "Places_CouSub_ConCity_SubMCD/MapServer/4",   # incorporated places
                          TIGERWEB + "Places_CouSub_ConCity_SubMCD/MapServer/5"],  # census-designated places
                "cousub": [TIGERWEB + "Places_CouSub_ConCity_SubMCD/MapServer/1"]}
NEW_ENGLAND = {"CT", "ME", "MA", "NH", "RI", "VT"}
LSAD_SUFFIX = re.compile(r"\s+(city|town|village|borough|CDP|municipality|city and borough|urban county|"
                         r"consolidated government.*|metropolitan government.*|unified government.*|plantation|"
                         r"charter township|township|comunidad|zona urbana)$", re.I)


def _gaz(name: str) -> pd.DataFrame:
    z = download(GAZ + name, NATIONAL / "gazetteer" / name)
    with zipfile.ZipFile(z) as zf:
        txt = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
    df = pd.read_csv(io.StringIO(txt), sep="\t", dtype={"GEOID": str})
    df.columns = [c.strip() for c in df.columns]
    return df


@lru_cache(maxsize=1)
def gazetteer() -> pd.DataFrame:
    p = _gaz("2024_Gaz_place_national.zip").assign(kind="place")
    c = _gaz("2024_Gaz_cousubs_national.zip")
    c = c[c.USPS.isin(NEW_ENGLAND) & (c.ALAND > 0)].assign(kind="cousub")
    df = pd.concat([p, c], ignore_index=True)
    df["short"] = df.NAME.str.replace(LSAD_SUFFIX, "", regex=True)
    df["label"] = df.short + ", " + df.USPS
    # In New England the town (county subdivision) governs; drop a place that duplicates its town.
    town = set(zip(df.short[df.kind == "cousub"], df.USPS[df.kind == "cousub"]))
    df = df[~((df.kind == "place") & df.USPS.isin(NEW_ENGLAND) & pd.Series(list(zip(df.short, df.USPS))).isin(town).to_numpy())]
    return df[["GEOID", "kind", "NAME", "short", "label", "USPS", "ALAND", "INTPTLAT", "INTPTLONG"]].rename(
        columns={"GEOID": "geoid", "NAME": "name", "USPS": "state", "ALAND": "aland_m2",
                 "INTPTLAT": "lat", "INTPTLONG": "lon"})


def search(q: str, limit: int = 12) -> list[dict]:
    """Places whose name starts with (then contains) q; larger places first."""
    g = gazetteer()
    q = q.strip().lower()
    if not q:
        return []
    name, _, st = q.partition(",")
    st = st.strip().upper()
    s = g.short.str.lower()
    hit = g[s.str.startswith(name.strip())]
    if len(hit) < limit:
        hit = pd.concat([hit, g[s.str.contains(name.strip(), regex=False) & ~g.index.isin(hit.index)]])
    if st:
        hit = hit[hit.state == st]
    hit = hit.sort_values("aland_m2", ascending=False).head(limit)
    return [{"geoid": r.geoid, "label": r.label, "name": r.name, "state": r.state, "kind": r.kind,
             "lat": float(r.lat), "lon": float(r.lon), "land_km2": round(r.aland_m2 / 1e6, 1)} for r in hit.itertuples()]


def place(geoid: str) -> dict:
    g = gazetteer()
    r = g[g.geoid == geoid]
    if r.empty:
        raise KeyError(f"unknown place GEOID {geoid}")
    r = r.iloc[0]
    return {"geoid": geoid, "label": r.label, "name": r["name"], "state": r.state, "kind": r.kind,
            "lat": float(r.lat), "lon": float(r.lon), "land_km2": round(r.aland_m2 / 1e6, 1)}


def boundary(geoid: str):
    """Place polygon (EPSG:4326) from TIGERweb."""
    import geopandas as gpd

    kind = place(geoid)["kind"]
    for layer in PLACE_LAYERS[kind]:
        f = arcgis_query(layer, {"where": f"GEOID='{geoid}'", "outFields": "GEOID,NAME", "returnGeometry": "true"},
                         NATIONAL / "boundaries" / f"{geoid}_{layer.rsplit('/', 1)[1]}.geojson")
        g = gpd.read_file(f)
        if len(g):
            return g.set_crs(4326, allow_override=True).geometry.union_all()
    raise KeyError(f"no boundary for {geoid}")


@lru_cache(maxsize=1)
def counties() -> pd.DataFrame:
    """US counties (2024 gazetteer): GEOID (5 digits), name, state."""
    c = _gaz("2024_Gaz_counties_national.zip")
    return c[["GEOID", "NAME", "USPS"]].rename(columns={"GEOID": "geoid", "NAME": "name", "USPS": "state"})
