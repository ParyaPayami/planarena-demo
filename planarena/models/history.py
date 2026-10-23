"""Annual MapPLUTO snapshots of Manhattan, 2015-2025, for rolling-origin evaluation of the transition model.

Run with:  python -I -m planarena.models.history      (downloads about 1.7 GB on first run; builds one twin per year)
Each release is downloaded from NYC DCP into data/raw/pluto_<release>/, Manhattan is extracted, and a twin
(parcels, geometry, graph) is built into data/history/<release>/. SHA-256 of every archive is recorded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from planarena.config import DATA, PROCESSED

OLD = "https://www.nyc.gov/assets/planning/download/zip/data-maps/open-data/mn_mappluto_{v}.zip"
NEW = "https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/mappluto/nyc_mappluto_{v}_arc_shp.zip"
# year: (release, url template, Manhattan-only archive?)
RELEASES = {
    2015: ("15v1", OLD, True), 2016: ("16v2", OLD, True), 2017: ("17v1", OLD, True), 2018: ("18v1", OLD, True),
    2019: ("19v1", NEW, False), 2020: ("20v8", "https://www.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyc_mappluto_{v}_shp.zip", False),
    2021: ("21v1", NEW, False), 2022: ("22v1", NEW, False), 2023: ("23v1", NEW, False), 2024: ("24v1", NEW, False),
    2025: ("25v3_1", NEW, False),
}
HIST = DATA / "history"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir(year: int) -> Path:
    """Processed twin for a year; 2025 is the live twin."""
    return PROCESSED if year == 2025 else HIST / RELEASES[year][0]


def fetch(year: int) -> Path:
    import pyogrio

    v, tmpl, mn_only = RELEASES[year]
    d = DATA / "raw" / f"pluto_{v}"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"mn_mappluto_{v}.gpkg"
    if out.exists():
        return out
    url = tmpl.format(v=v)
    z = d / url.rsplit("/", 1)[1]
    if not z.exists():
        subprocess.run(["curl", "-sL", "-m", "3600", "-A", "Mozilla/5.0", "-o", str(z), url], check=True)
    ext = d / "extracted"
    if not ext.exists():
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if ".." not in name and not name.startswith("/"):
                    zf.extract(name, ext)
    shp = [p for p in ext.rglob("*.shp") if "unclipped" not in p.name.lower() and "dcp_mappinglot" not in p.name.lower()]
    shp = sorted(shp, key=lambda p: p.stat().st_size, reverse=True)[0]
    g = pyogrio.read_dataframe(shp) if mn_only else pyogrio.read_dataframe(shp, where="Borough = 'MN'")
    pyogrio.write_dataframe(g, out, driver="GPKG")
    (d / "SOURCE.json").write_text(json.dumps({"release": v, "url": url, "archive_sha256": sha256(z),
                                               "extract": out.name, "extract_sha256": sha256(out), "lots": int(len(g))}, indent=2))
    return out


def build_all(years=None) -> dict:
    from planarena.twin.build import build

    out = {}
    for y in sorted(years or RELEASES):
        if y == 2025:
            continue
        sd = snapshot_dir(y)
        if not (sd / "parcels.parquet").exists():
            gpkg = fetch(y)
            m = build(out_dir=sd, pluto_path=gpkg, web=False)
            out[y] = {"release": RELEASES[y][0], "parcels": m["n_parcels"], "missing_use": m["data_quality"]["missing_land_use"]}
            print(y, out[y], flush=True)
    return out


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=1))
