"""Central configuration: paths, land-use classes, pillar definitions."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
WEB_DATA = ROOT / "web" / "public" / "data"

# Manhattan extract of the official MapPLUTO 25v3.1 release (python -I -m planarena.twin.fetch).
PLUTO_PATH = Path(os.environ.get("PLANARENA_PLUTO", DATA / "raw" / "pluto_25v3_1" / "mn_mappluto_25v3_1.gpkg"))
CITY_NAME = os.environ.get("PLANARENA_CITY", "Manhattan, New York City")

PROJECTED_CRS = "EPSG:2263"  # NY State Plane Long Island, US feet
WEB_CRS = "EPSG:4326"
FT_PER_M = 3.280839895


@dataclass(frozen=True)
class LandUse:
    key: str
    label: str
    color: str
    pluto_codes: tuple[str, ...]
    built: bool  # does this use normally carry building floor area?


# MapPLUTO LandUse codes (01-11) grouped into nine planner-facing classes.
LAND_USES: tuple[LandUse, ...] = (
    LandUse("res", "Residential", "#F2C14E", ("01", "02", "03"), True),
    LandUse("mix", "Mixed residential & commercial", "#F08A4B", ("04",), True),
    LandUse("com", "Commercial & office", "#D7263D", ("05",), True),
    LandUse("ind", "Industrial & manufacturing", "#8E6CC0", ("06",), True),
    LandUse("tra", "Transportation & utility", "#7D8597", ("07",), True),
    LandUse("pub", "Public facilities & institutions", "#3E7CB1", ("08",), True),
    LandUse("opn", "Open space & recreation", "#3FA34D", ("09",), False),
    LandUse("prk", "Parking", "#B8BEC6", ("10",), False),
    LandUse("vac", "Vacant land", "#E3D5B8", ("11",), False),
)
USE_KEYS = tuple(u.key for u in LAND_USES)
USE_INDEX = {k: i for i, k in enumerate(USE_KEYS)}
UNKNOWN_USE = -1
PLUTO_TO_USE = {code: USE_INDEX[u.key] for u in LAND_USES for code in u.pluto_codes}

PILLARS = ("social", "economic", "environmental")
