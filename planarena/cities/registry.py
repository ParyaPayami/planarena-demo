"""Which cities are built, where their files are, and which tier of data they use.

* Manhattan (Tier B, NYC MapPLUTO) lives in data/processed and web/public/data, as before.
* Every other city lives in data/cities/<GEOID>/ (twin, coefficients, web layers), listed in data/cities/registry.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from planarena.config import CITY_NAME, PROCESSED, WEB_DATA
from planarena.national.common import CITIES

DEFAULT = "manhattan"
REGISTRY = CITIES / "registry.json"


def city_dir(city_id: str) -> Path:
    return PROCESSED if city_id == DEFAULT else CITIES / city_id


def web_dir(city_id: str) -> Path:
    return WEB_DATA if city_id == DEFAULT else CITIES / city_id / "web"


def _read() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def cities() -> dict:
    out = {}
    if (PROCESSED / "parcels.parquet").exists():
        out[DEFAULT] = {"id": DEFAULT, "label": CITY_NAME, "tier": "B", "source": "NYC DCP MapPLUTO"}
    for k, v in _read().items():
        if (city_dir(k) / "parcels.parquet").exists():
            out[k] = v
    return out


def register(city_id: str, label: str, tier: str, meta: dict) -> None:
    r = _read()
    r[city_id] = {"id": city_id, "label": label, "tier": tier, "source": meta.get("source"),
                  "parcels": meta.get("n_parcels"), "built": time.strftime("%Y-%m-%d")}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(r, indent=2))


def exists(city_id: str) -> bool:
    return city_id in cities()
