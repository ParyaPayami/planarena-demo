"""2020 Census blocks with population and housing-unit counts (TIGERweb), and LEHD LODES jobs by block."""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from planarena.national.common import NATIONAL, arcgis_query, download

BLOCKS_2020 = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/12"
LODES = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/wac/{st}_wac_S000_JT00_{year}.csv.gz"
LODES_YEAR = 2023


def blocks(geoid: str, bbox: tuple[float, float, float, float]):
    """Census 2020 blocks intersecting bbox (lon/lat), with POP100 and HU100."""
    import geopandas as gpd

    f = arcgis_query(BLOCKS_2020, {"where": "1=1", "geometry": ",".join(f"{v:.6f}" for v in bbox),
                                   "geometryType": "esriGeometryEnvelope", "inSR": "4326",
                                   "spatialRel": "esriSpatialRelIntersects",
                                   "outFields": "GEOID,POP100,HU100,AREALAND", "returnGeometry": "true"},
                     NATIONAL / "blocks" / f"{geoid}.geojson", page=1000)
    g = gpd.read_file(f).set_crs(4326, allow_override=True)
    for c in ("POP100", "HU100", "AREALAND"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)
    return g


def lodes_jobs(state_fips: str, state_abbr: str, year: int = LODES_YEAR) -> pd.Series:
    """Jobs by 2020 block GEOID (all jobs, all workers), from the state's LODES workplace file."""
    st = state_abbr.lower()
    f = download(LODES.format(st=st, year=year), NATIONAL / "lodes" / f"{st}_wac_S000_JT00_{year}.csv.gz")
    with gzip.open(f, "rt") as fh:
        w = pd.read_csv(fh, usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
    return w.set_index("w_geocode").C000.astype(np.float64)
