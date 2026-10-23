"""Download the official MapPLUTO release and extract Manhattan for the twin.

Run with:  python -I -m planarena.twin.fetch      (about 300 MB download; skipped if already present)
Writes data/raw/pluto_25v3_1/mn_mappluto_25v3_1.gpkg and SOURCE.json (URL and SHA-256 of the official archive).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile

from planarena.config import DATA

RELEASE = "25v3_1"
URL = f"https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/mappluto/nyc_mappluto_{RELEASE}_arc_shp.zip"
DIR = DATA / "raw" / f"pluto_{RELEASE}"
ZIP = DIR / f"nyc_mappluto_{RELEASE}_arc_shp.zip"
OUT = DIR / f"mn_mappluto_{RELEASE}.gpkg"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> dict:
    import pyogrio

    DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        subprocess.run(["curl", "-sL", "-m", "1800", "-A", "Mozilla/5.0", "-o", str(ZIP), URL], check=True)
    ext = DIR / "extracted"
    if not (ext / "MapPLUTO.shp").exists():
        with zipfile.ZipFile(ZIP) as z:
            for name in z.namelist():  # extract only plain file names into our own folder
                if "/" not in name.strip("/") and ".." not in name:
                    z.extract(name, ext)
    src = DIR / "SOURCE.json"
    if OUT.exists() and src.exists() and json.loads(src.read_text()).get("extract_sha256") == sha256(OUT):
        return json.loads(src.read_text())  # already extracted: re-writing a GeoPackage changes its bytes
    g = pyogrio.read_dataframe(ext / "MapPLUTO.shp", where="Borough = 'MN'")
    pyogrio.write_dataframe(g, OUT, driver="GPKG")
    meta = {"release": RELEASE, "url": URL, "archive_sha256": sha256(ZIP), "extract": OUT.name,
            "extract_sha256": sha256(OUT), "filter": "Borough = 'MN'", "lots": int(len(g))}
    (DIR / "SOURCE.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(fetch(), indent=2))
