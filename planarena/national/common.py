"""Download helpers shared by the national sources."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from planarena.config import DATA

NATIONAL = DATA / "national"
CITIES = DATA / "cities"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out: Path, query: dict | None = None, timeout: int = 1800) -> Path:
    """Fetch url to out (once) and record its source next to it."""
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    cmd = ["curl", "-sfL", "-m", str(timeout), "-A", "Mozilla/5.0", "-o", str(tmp)]
    if query:
        cmd += ["-G", url] + sum((["--data-urlencode", f"{k}={v}"] for k, v in query.items()), [])
    else:
        cmd.append(url)
    subprocess.run(cmd, check=True)
    tmp.rename(out)
    record(out, url, query)
    return out


def record(path: Path, url: str, query: dict | None = None, **extra) -> None:
    src = path.parent / "SOURCE.json"
    meta = json.loads(src.read_text()) if src.exists() else {}
    meta[path.name] = {"url": url, "query": query, "sha256": sha256(path), "bytes": path.stat().st_size,
                       "retrieved": dt.date.today().isoformat()} | extra
    src.write_text(json.dumps(meta, indent=2))


def arcgis_query(layer_url: str, params: dict, out: Path, page: int = 2000) -> Path:
    """Page through an ArcGIS REST query into one GeoJSON file (cached)."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    feats, offset = [], 0
    while True:
        q = params | {"f": "geojson", "outSR": "4326", "resultOffset": str(offset), "resultRecordCount": str(page)}
        r = subprocess.run(["curl", "-sfL", "-m", "300", "-G", layer_url + "/query"]
                           + sum((["--data-urlencode", f"{k}={v}"] for k, v in q.items()), []),
                           check=True, capture_output=True)
        d = json.loads(r.stdout)
        if "error" in d:
            raise RuntimeError(f"{layer_url}: {d['error']}")
        got = d.get("features", [])
        feats += got
        if len(got) < page and not d.get("exceededTransferLimit") and not d.get("properties", {}).get("exceededTransferLimit"):
            break
        offset += len(got)
        if not got:
            break
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    record(out, layer_url + "/query", {k: v for k, v in params.items() if k != "geometry"}, features=len(feats))
    return out
