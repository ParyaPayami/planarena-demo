"""Download a public ArcGIS REST layer or table (FeatureServer or MapServer) to (Geo)Parquet, with provenance.

Pages through the layer by object id in parallel, keeps only the requested fields (owner and mailing fields are
never requested), records URL, fields, feature count, retrieval date and SHA-256 next to the file (SOURCE.json).
Downloaded data is untrusted: read it with python -I.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from planarena.national.common import record

PRIVATE = ("OWNER", "OWN_", "MAIL", "TAXPAYER")  # never requested, whatever a caller asks for


def _get(url: str, params: dict) -> dict:
    cmd = ["curl", "-sfL", "-m", "300", "-A", "Mozilla/5.0", "-G", url]
    for k, v in params.items():
        cmd += ["--data-urlencode", f"{k}={v}"]
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            if "error" not in d:
                return d
    raise RuntimeError(f"{url}: {r.stderr.decode()[:200] or d.get('error')}")


def layer_info(url: str) -> dict:
    return _get(url, {"f": "json"})


def _ids(url: str, where: str, extra: dict | None = None) -> tuple[str, int, list[list[int]]]:
    """Object-id field, page size and the matching object ids in pages (``extra`` may hold a geometry filter)."""
    info = layer_info(url)
    oid = info.get("objectIdField") or next(f["name"] for f in info["fields"] if f["type"] == "esriFieldTypeOID")
    page = int(info.get("maxRecordCount") or 1000)
    flt = {k: v for k, v in (extra or {}).items() if k in ("geometry", "geometryType", "inSR", "spatialRel")}
    ids = sorted(_get(url + "/query", {"where": where, "returnIdsOnly": "true", "f": "json"} | flt)["objectIds"] or [])
    return oid, page, [ids[i:i + page] for i in range(0, len(ids), page)]


def _paged(url: str, where: str, fields: list[str], fmt: str, parse, workers: int, extra: dict | None = None):
    """All matching records, queried by object-id range in parallel. A page that comes back short (a server's
    transfer limit) is split and queried again, so every matching record is fetched once."""
    oid, page, chunks = _ids(url, where, extra)
    params = {"outFields": ",".join(fields), "f": fmt} | ({"outSR": "4326"} if fmt == "geojson" else {}) | (extra or {})

    def one(chunk):
        d = _get(url + "/query", params | {"where": f"{oid} >= {chunk[0]} AND {oid} <= {chunk[-1]} AND ({where})",
                                           "resultRecordCount": str(page)})
        rows = d.get("features", [])
        if len(rows) < len(chunk) and len(chunk) > 1:
            h = len(chunk) // 2
            return one(chunk[:h]) + one(chunk[h:])
        return [parse(rows)]

    with ThreadPoolExecutor(workers) as ex:
        parts = [x for part in ex.map(one, chunks) for x in part]
    return parts, sum(len(c) for c in chunks)


def _keep(fields: list[str]) -> list[str]:
    return [f for f in fields if not f.upper().startswith(PRIVATE)]


def fetch_layer(url: str, out: Path, fields: list[str], where: str = "1=1", workers: int = 8, extra: dict | None = None):
    """GeoDataFrame (EPSG:4326) of a layer, cached at ``out`` (GeoParquet). ``extra`` adds query parameters (e.g.
    geometryPrecision for layers with very large polygons)."""
    import geopandas as gpd
    import pandas as pd

    if out.exists():
        return gpd.read_parquet(out)
    fields = _keep(fields)
    parts, expected = _paged(url, where, fields, "geojson",
                             lambda rows: gpd.GeoDataFrame.from_features(rows, crs=4326), workers, extra)
    g = gpd.GeoDataFrame(pd.concat([p for p in parts if len(p)], ignore_index=True), crs=4326)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out)
    record(out, url, {"where": where, "outFields": ",".join(fields)} | (extra or {}), features=int(len(g)),
           expected=expected, layer=layer_info(url).get("name"))
    return g


def fetch_table(url: str, out: Path, fields: list[str], where: str = "1=1", workers: int = 8):
    """DataFrame of a table (or a layer's attributes only), cached at ``out`` (Parquet)."""
    import pandas as pd

    if out.exists():
        return pd.read_parquet(out)
    fields = _keep(fields)
    parts, expected = _paged(url, where, fields, "json",
                             lambda rows: pd.DataFrame([r["attributes"] for r in rows]), workers)
    t = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out, index=False)
    record(out, url, {"where": where, "outFields": ",".join(fields)}, features=int(len(t)), expected=expected,
           layer=layer_info(url).get("name"))
    return t
