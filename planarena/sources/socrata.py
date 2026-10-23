"""Download a public Socrata (SODA) dataset to Parquet, with provenance.

Only the named columns are requested (owner, taxpayer and mailing columns never are, whatever a caller asks for).
Pages are ordered by the row id so the download is complete and repeatable. Records the URL, query, row count,
retrieval date and SHA-256 next to the file (SOURCE.json). Downloaded data is untrusted: read it with python -I.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from planarena.national.common import record
from planarena.sources.arcgis import PRIVATE

PRIVATE_SODA = PRIVATE  # owner, taxpayer and mailing columns


def _get(url: str, params: dict) -> list:
    cmd = ["curl", "-sfL", "-m", "600", "-A", "Mozilla/5.0", "-G", url]
    for k, v in params.items():
        cmd += ["--data-urlencode", f"{k}={v}"]
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            if isinstance(d, list):
                return d
    raise RuntimeError(f"{url}: {r.stderr.decode()[:200]}")


def fetch(domain: str, dataset: str, out: Path, columns: list[str], where: str | None = None, page: int = 50000):
    """DataFrame of the named columns of a Socrata dataset (all rows matching ``where``), cached at ``out``."""
    import pandas as pd

    if out.exists():
        return pd.read_parquet(out)
    columns = [c for c in columns if not c.upper().startswith(PRIVATE_SODA)]
    url = f"https://{domain}/resource/{dataset}.json"
    q = {"$select": ",".join(columns), "$order": ":id", "$limit": str(page)} | ({"$where": where} if where else {})
    n = int(_get(url, {"$select": "count(*)"} | ({"$where": where} if where else {}))[0]["count"])
    rows, offset = [], 0
    while offset < n:
        got = _get(url, q | {"$offset": str(offset)})
        if not got:
            break
        rows += got
        offset += len(got)
    t = pd.DataFrame(rows, columns=columns)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out, index=False)
    record(out, url, {k: v for k, v in q.items() if k != "$limit"}, rows=int(len(t)), expected=n)
    return t


def fetch_geojson(domain: str, dataset: str, out: Path, where: str | None = None):
    """GeoDataFrame (EPSG:4326) of a Socrata map dataset, cached at ``out`` (GeoParquet)."""
    import geopandas as gpd

    if out.exists():
        return gpd.read_parquet(out)
    url = f"https://{domain}/resource/{dataset}.geojson"
    params = {"$limit": "1000000"} | ({"$where": where} if where else {})
    cmd = ["curl", "-sfL", "-m", "600", "-A", "Mozilla/5.0", "-G", url]
    for k, v in params.items():
        cmd += ["--data-urlencode", f"{k}={v}"]
    raw = out.with_suffix(".geojson")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(subprocess.run(cmd, check=True, capture_output=True).stdout)
    g = gpd.read_file(raw, engine="pyogrio").set_crs(4326, allow_override=True)
    g.to_parquet(out)
    raw.unlink()
    record(out, url, params, rows=int(len(g)))
    return g
