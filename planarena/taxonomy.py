"""PlanArena's land-use classification: 16 planning classes, each in one of nine analysis groups.

Every city's official land-use codes (existing use, the adopted plan's designations) are translated into these classes
by the crosswalks in planarena/sources/crosswalk.py. The 16 classes are what a planner sees and edits; the nine
groups (planarena/config.py LAND_USES) are what models trained at group level use (redevelopment, precedent, pattern).
"""
from __future__ import annotations

from dataclasses import dataclass

from planarena.config import USE_INDEX as GROUP


@dataclass(frozen=True)
class Klass:
    key: str
    label: str
    group: str  # one of the nine analysis groups
    color: str
    built: bool


CLASSES: tuple[Klass, ...] = (
    Klass("sf", "Single-family homes", "res", "#F6D55C", True),
    Klass("mf_small", "Small multi-family (2-4 homes)", "res", "#F2B134", True),
    Klass("mf", "Multi-family (5+ homes)", "res", "#E8912D", True),
    Klass("mix", "Mixed use (homes with shops or offices)", "mix", "#E2683C", True),
    Klass("ret", "Retail, food and services", "com", "#D7263D", True),
    Klass("off", "Offices", "com", "#A4243B", True),
    Klass("hot", "Hotels and lodging", "com", "#C94277", True),
    Klass("ind", "Industry and manufacturing", "ind", "#8E6CC0", True),
    Klass("whs", "Warehousing and logistics", "ind", "#6A4C93", True),
    Klass("edu", "Schools and universities", "pub", "#3E7CB1", True),
    Klass("hlt", "Hospitals and health care", "pub", "#2D6A9F", True),
    Klass("civ", "Civic, religious and government", "pub", "#5B8DB8", True),
    Klass("opn", "Parks and open space", "opn", "#3FA34D", False),
    Klass("tra", "Transportation and utilities", "tra", "#7D8597", True),
    Klass("prk", "Parking", "prk", "#B8BEC6", False),
    Klass("vac", "Vacant land", "vac", "#E3D5B8", False),
)
KEYS = tuple(k.key for k in CLASSES)
INDEX = {k: i for i, k in enumerate(KEYS)}
UNKNOWN = -1
TO_GROUP = tuple(GROUP[k.group] for k in CLASSES)  # class index -> analysis-group index


def group_of(cls: int) -> int:
    """Analysis group of a class index (unknown stays unknown)."""
    return TO_GROUP[cls] if cls >= 0 else -1
