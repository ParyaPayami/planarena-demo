"""The ten PlanArena cities and how each is built from official parcel-level land use.

Every city's baseline is the land use its own government (or regional planning agency) publishes for each parcel;
the adopted plan (future land use) and zoning are separate layers. Nothing is estimated: a city whose official data
is not yet ingested is listed as not built.

Run:  python -I -m planarena.cities.build "<city>"      (or a Census place GEOID)
"""
from __future__ import annotations

import importlib
import json
import sys

# GEOID -> (label, adapter module under planarena.cities.adapters, or None while its sources are being ingested)
CITIES = {
    "3651000": ("New York, NY", "nyc"),
    "0644000": ("Los Angeles, CA", "losangeles"),
    "1714000": ("Chicago, IL", "chicago"),
    "4835000": ("Houston, TX", "houston"),
    "0455000": ("Phoenix, AZ", "phoenix"),
    "4260000": ("Philadelphia, PA", "philadelphia"),
    "4865000": ("San Antonio, TX", "sanantonio"),
    "0666000": ("San Diego, CA", "sandiego"),
    "4819000": ("Dallas, TX", "dallas"),
    "5553000": ("Milwaukee, WI", "milwaukee"),
}


def resolve(target: str) -> str:
    """GEOID of one of the ten cities, from a GEOID or a name such as 'Chicago' or 'Dallas, TX'."""
    if target in CITIES:
        return target
    key = target.lower().replace(",", " ").split()
    for geoid, (label, _) in CITIES.items():
        if key and label.lower().replace(",", " ").split()[: len(key)] == key:
            return geoid
    raise SystemExit(f"{target!r} is not one of the PlanArena cities: " + ", ".join(v[0] for v in CITIES.values()))


def build(target: str) -> dict:
    geoid = resolve(target)
    label, adapter = CITIES[geoid]
    if adapter is None:
        raise SystemExit(f"{label}: official land-use sources are not ingested yet")
    return importlib.import_module(f"planarena.cities.adapters.{adapter}").build()


if __name__ == "__main__":
    m = build(" ".join(sys.argv[1:]))
    print(json.dumps({k: m.get(k) for k in ("city", "tier", "n_parcels", "use_counts", "build_seconds")}, indent=1))
