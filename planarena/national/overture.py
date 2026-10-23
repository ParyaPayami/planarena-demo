"""Overture Maps (open, CDLA-Permissive / ODbL) buildings, places and land use for one city, by bounding box.

DuckDB reads only the row groups whose bounding boxes intersect the city from Overture's public parquet on S3,
so a city costs a few minutes and tens to hundreds of MB, not the whole country.
"""
from __future__ import annotations

import json

from planarena.national.common import NATIONAL, record

RELEASE = "2026-09-23.1"
S3 = "s3://overturemaps-us-west-2/release/{rel}/theme={theme}/type={type}/*"
QUERIES = {
    "buildings": ("buildings", "building",
                  "id, subtype, class, height, num_floors, is_underground"),
    "places": ("places", "place",
               "id, names.primary AS name, basic_category, array_to_string(taxonomy.hierarchy, '>') AS hierarchy, "
               "confidence, operating_status"),
    "land_use": ("base", "land_use", "id, subtype, class"),
}


def _con():
    import duckdb

    c = duckdb.connect()
    c.sql("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; "
          "SET enable_progress_bar=false;")
    return c


def fetch(geoid: str, bbox: tuple[float, float, float, float], what: str, release: str = RELEASE):
    """GeoDataFrame (EPSG:4326) of one Overture layer inside bbox, cached as GeoParquet."""
    import geopandas as gpd
    import shapely

    out = NATIONAL / "overture" / geoid / f"{what}.parquet"
    if not out.exists():
        theme, typ, cols = QUERIES[what]
        x0, y0, x1, y1 = bbox
        sql = (f"SELECT {cols}, ST_AsWKB(geometry) AS wkb FROM read_parquet('{S3.format(rel=release, theme=theme, type=typ)}', "
               f"hive_partitioning=1) WHERE bbox.xmin < {x1} AND bbox.xmax > {x0} AND bbox.ymin < {y1} AND bbox.ymax > {y0}")
        df = _con().sql(sql).df()
        geom = shapely.from_wkb(df.pop("wkb").map(bytes).to_numpy())
        out.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(df, geometry=geom, crs=4326).to_parquet(out)
        record(out, S3.format(rel=release, theme=theme, type=typ), {"bbox": json.dumps(list(bbox))},
               release=release, rows=int(len(df)), licence="Overture Maps: CDLA-Permissive-2.0 / ODbL-1.0 by layer")
    return gpd.read_parquet(out)
