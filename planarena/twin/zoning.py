"""Simplified zoning rules-as-code for NYC zoning districts.

Maps a district code (MapPLUTO ``ZoneDist1``, plus any commercial overlay) to the
set of PlanArena land-use classes the district normally permits.

STATUS: provisional simplification. NYC's use regulations are far richer:
Zoning Resolution Articles II-IV, use groups revised by "City of Yes for
Economic Opportunity" (2024). The Studio treats zoning as advisory and shows a
warning. The optimizer treats it as a hard constraint by default.
"""
from __future__ import annotations

import re

from planarena.config import USE_INDEX

ALL = frozenset(USE_INDEX)
_R = frozenset({"res", "pub", "opn", "prk", "vac"})
_C_LOCAL = frozenset({"res", "mix", "com", "pub", "opn", "prk", "vac"})
_C_GENERAL = _C_LOCAL | {"tra"}
_C3 = frozenset({"com", "pub", "opn", "prk", "vac"})
_C8 = frozenset({"com", "ind", "tra", "pub", "opn", "prk", "vac"})
_M1 = frozenset({"ind", "com", "tra", "pub", "opn", "prk", "vac"})
_M_HEAVY = frozenset({"ind", "tra", "com", "opn", "prk", "vac"})
_MX = ALL  # paired M1/R districts, e.g. "M1-5/R10", permit residential uses

RULE_SOURCE = (
    "NYC Zoning Resolution district purposes and use groups, simplified by "
    "PlanArena; provisional, verify against ZR Articles II-IV."
)


def allowed_uses(zone: str | None, overlay: str | None = None) -> frozenset[str]:
    """Return the land-use keys a district permits, using the simplified rules above."""
    if not zone or zone in {"NA", "nan"}:
        return ALL
    z = zone.upper().strip()
    if z == "PARK":
        return frozenset({"opn"})
    if z == "BPC":  # Battery Park City special district
        return _C_LOCAL
    if "/R" in z:  # MX special mixed-use districts
        return _MX
    fam = re.match(r"^([RCM])(\d+)", z)
    if not fam:
        return ALL
    kind, num = fam.group(1), int(fam.group(2))
    if kind == "R":
        base = _R
        if overlay and str(overlay).upper().startswith(("C1", "C2")):
            base = base | {"mix", "com"}
        return base
    if kind == "C":
        if num in (1, 2):
            return _C_LOCAL
        if num == 3:
            return _C3
        if num in (4, 5, 6):
            return _C_GENERAL
        if num == 7:
            return _C3
        return _C8
    if kind == "M":
        return _M1 if num == 1 else _M_HEAVY
    return ALL


def allowed_mask(zone: str | None, overlay: str | None = None) -> int:
    """Bitmask over USE_INDEX order (bit i set means use i is permitted)."""
    mask = 0
    for key in allowed_uses(zone, overlay):
        mask |= 1 << USE_INDEX[key]
    return mask
