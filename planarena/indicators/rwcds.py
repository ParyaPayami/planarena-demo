"""CEQR-style development-site screen for a proposed rezoning (reasonable worst-case development scenario).

Criteria follow DCP practice as written in the East Harlem Rezoning FEIS (CEQR 17DCP048M), Chapter 1,
"General criteria for determining development sites":
  include  lots with a substantial increase in permitted FAR, of 2,000 sq ft or more, that are underutilized
           (vacant, a single floor, or built to half or less of the maximum FAR under the proposed zoning)
  exclude  recent construction; schools, libraries, government offices, large medical centres and houses of
           worship (PlanArena: public-facility lots); buildings with six or more homes built before 1974
           (likely rent-stabilised); designated landmarks
"Substantial increase" is a judgement in the FEIS; PlanArena uses +1.0 FAR (assumption).
Development on a site is built to the proposed maximum FAR times a utilisation factor.
"""
from __future__ import annotations

import numpy as np

from planarena.config import USE_INDEX

U = USE_INDEX
MIN_FAR_INCREASE = 1.0  # assumption


def screen(engine, new_max_far: np.ndarray, area: np.ndarray, year: int, min_increase: float = MIN_FAR_INCREASE):
    """Boolean mask of development sites inside ``area`` under the proposed maximum FAR."""
    p = engine.p
    lot = engine.lot
    built_far = np.divide(engine.bldg, lot, out=np.zeros(engine.n), where=lot > 0)
    old_max = p["max_far"].to_numpy()
    use = engine.base_use
    floors = p["floors"].to_numpy()
    yb = p["year_built"].to_numpy()
    units = p["units_res"].to_numpy()
    underused = (engine.bldg <= 0) | (floors <= 1) | (built_far <= 0.5 * new_max_far) | np.isin(use, [U["vac"], U["prk"]])
    include = area & (new_max_far - old_max >= min_increase) & (lot >= 2000) & underused
    exclude = ((yb >= year - 5) & (engine.bldg > 0)) | (use == U["pub"]) | ((units >= 6) & (yb > 0) & (yb < 1974)) \
        | p["landmark"].to_numpy() | (use == U["opn"])
    return include & ~exclude


def project(engine, sites: np.ndarray, new_max_far: np.ndarray, residential_ok: np.ndarray,
            utilisation: float = 1.0) -> dict:
    """Homes and floor area if every screened site is built to the proposed FAR (x utilisation)."""
    use = engine.base_use.copy()
    far = np.full(engine.n, np.nan)
    idx = np.nonzero(sites)[0]
    use[idx] = np.where(residential_ok[idx], U["mix"], U["com"])
    far[idx] = new_max_far[idx] * utilisation
    st = engine.parcel_state(use, far)
    base = engine.baseline_state
    return {"sites": int(len(idx)), "homes_with_action": float(st["units"][idx].sum()),
            "homes_existing": float(base["units"][idx].sum()),
            "homes_increment": float(st["units"][idx].sum() - base["units"][idx].sum()),
            "floor_area_increment_sqft": float(st["fa_total"][idx].sum() - engine.bldg[idx].sum())}
