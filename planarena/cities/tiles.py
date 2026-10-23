"""Map data for large cities: Mapbox vector tiles made on demand, and a compact array of parcel points.

A city with more than TILE_MIN_PARCELS parcels is drawn from tiles (zoom 13-15; deck.gl over-zooms beyond 15) and,
when zoomed out, from parcel points. Smaller cities keep one GeoJSON file. Tiles are cut from the twin's own lot
geometry (no external tiler) and cached on disk next to the city's web layers.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

TILE_MIN_PARCELS = 100_000
MIN_ZOOM, MAX_ZOOM = 13, 15
EXTENT = 4096
WORLD = 20037508.342789244  # half the Web Mercator world width, metres
_LOCK = threading.Lock()


def map_mode(n_parcels: int) -> str:
    return "tiles" if n_parcels > TILE_MIN_PARCELS else "geojson"


class TileSource:
    def __init__(self, twin_dir: Path, cache_dir: Path):
        import geopandas as gpd
        import pandas as pd
        import shapely

        g = gpd.read_parquet(twin_dir / "parcels.geoparquet").to_crs(3857)
        p = pd.read_parquet(twin_dir / "parcels.parquet", columns=["use", "cd", "height_m"])
        self.geoms = shapely.make_valid(g.geometry.values)
        self.tree = shapely.STRtree(self.geoms)
        self.use = p.use.to_numpy().astype(int)
        self.cd = p.cd.to_numpy().astype(int)
        self.h = np.round(p.height_m.to_numpy().astype(float), 1)
        self.cache = cache_dir

    def tile(self, z: int, x: int, y: int) -> bytes:
        f = self.cache / str(z) / str(x) / f"{y}.pbf"
        if f.exists():
            return f.read_bytes()
        data = self._encode(z, x, y)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        return data

    def _encode(self, z: int, x: int, y: int) -> bytes:
        import mapbox_vector_tile
        import shapely

        size = 2 * WORLD / (1 << z)
        x0, y1 = -WORLD + x * size, WORLD - y * size
        x1, y0 = x0 + size, y1 - size
        pad = size * 16 / EXTENT  # a few pixels of buffer so polygon edges join across tiles
        idx = self.tree.query(shapely.box(x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        if not len(idx):
            return b""
        clipped = shapely.clip_by_rect(self.geoms[idx], x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        clipped = shapely.simplify(clipped, size / EXTENT / 2, preserve_topology=True)
        feats = [{"geometry": g, "properties": {"i": int(i), "u": int(self.use[i]), "h": float(self.h[i]),
                                                "cd": int(self.cd[i])}}
                 for g, i in zip(clipped, idx) if not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon")]
        return mapbox_vector_tile.encode([{"name": "parcels", "features": feats}],
                                         default_options={"quantize_bounds": (x0, y0, x1, y1), "extents": EXTENT,
                                                          "y_coord_down": False})


@lru_cache(maxsize=3)
def source(city: str) -> TileSource:
    from planarena.cities import registry

    return TileSource(registry.city_dir(city), registry.web_dir(city) / "tiles")


def tile(city: str, z: int, x: int, y: int) -> bytes:
    with _LOCK:
        src = source(city)
    return src.tile(z, x, y)


def points(engine) -> bytes:
    """Little-endian arrays for the map's point layer and selection tools: int16 use, int16 district,
    float32 longitude, float32 latitude, float32 radius (metres, from lot area)."""
    p = engine.p
    # half the side of a square lot of the same area, capped so parks and campuses do not cover the map
    r = np.clip(np.sqrt(np.clip(p.lot_area.to_numpy(), 0, None)) / 3.280839895 / 2, 2.0, 25.0)
    return b"".join([p.use.to_numpy().astype("<i2").tobytes(), p.cd.to_numpy().astype("<i2").tobytes(),
                     p.lon.to_numpy().astype("<f4").tobytes(), p.lat.to_numpy().astype("<f4").tobytes(),
                     r.astype("<f4").tobytes()])
