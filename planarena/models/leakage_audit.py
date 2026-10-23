"""Leakage audit of the prediction tests: when each feature family is measured, how lots are followed over time, and
how many outcome labels the new-building rule had to correct.

1. Feature timestamps: for each panel (Manhattan annual snapshots and the seven cross-city panels), the source and
   date of each feature family against the decision year t0. Families measured after t0 are marked, with the variant
   that removes them.
2. Lot identity over time (Manhattan): each t0 lot is followed to the t1 lot that contains its representative
   point. Counted per pair: t0 lots with no t1 lot, merges (t1 lots reached from two or more t0 lots; their t0 floor
   area is summed before growth is measured) and splits (t0 lots containing the representative points of two or more
   t1 lots; growth is measured on the piece that contains the t0 point).
3. New-building rule: a lot already present at t0 with a year built of t0 or later is not a new building in
   (t0, t1]. Late releases (MapPLUTO 20v8, December 2020) already list buildings completed in 2020; counted per pair.

Run:  python -I -m planarena.models.leakage_audit     -> results/validation/leakage_audit.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from planarena.config import ROOT

OUT = ROOT / "results" / "validation" / "leakage_audit.json"

# Feature families and their timing relative to the decision year t0, per panel (from the panel builders).
TIMESTAMPS = {
    "Manhattan (rolling origins, 2015-2025)": {
        "outcome": "new building (year built >= t0, not on the t0 snapshot) or floor area +25% and +2,000 sq ft, "
                   "MapPLUTO t0 -> t1",
        "existing": "MapPLUTO release of year t0: use, lot and floor area, floors, year built, units, assessed land "
                    "and total value",
        "context": "same release: neighbours' use composition; share of neighbours built in the five years before t0",
        "zoning": "same release: zoning district, maximum FARs, overlays, landmark and historic status",
        "plan": "none (no citywide adopted land-use plan)",
        "after_t0": "none",
    },
    "Los Angeles": {
        "outcome": "assessor rolls of January t0 and t1",
        "existing": "SCAG 2020 land use; assessor roll of t0",
        "context": "assessor roll of t0 (buildings built in the five years before t0)",
        "zoning": "SCAG 2019 compilation", "plan": "SCAG 2019 General Plan compilation",
        "after_t0": "none",
    },
    "Dallas": {
        "outcome": "DCAD certified files of t0 and t1",
        "existing": "DCAD codes of t0", "context": "DCAD file of t0",
        "zoning": "current layer (undated)", "plan": "forwardDallas! 2.0, adopted 2024 (after t0 = 2022)",
        "after_t0": "zoning and plan may include changes after t0; no dates to remove them",
    },
    "Phoenix": {
        "outcome": "MAG existing land-use vintages of t0 and t1 (change of use group)",
        "existing": "MAG vintage of t0; assessor", "context": "assessor (year built)",
        "zoning": "current layer; zoning cases dated t0 or later flagged",
        "plan": "General Plan 2015 as amended; polygons added after t0 flagged",
        "after_t0": "flagged lots removed in the conservative variant",
    },
    "San Diego": {
        "outcome": "SANDAG land-use vintages of t0 and t1 (change of use group)",
        "existing": "SANDAG vintage of t0", "context": "assessor (year built)",
        "zoning": "current layer; implemented on or after t0 flagged", "plan": "General and community plans (undated)",
        "after_t0": "zoning flags removed in the conservative variant; plan undated",
    },
    "New York City": {
        "outcome": "MapPLUTO releases of t0 and t1", "existing": "MapPLUTO of t0", "context": "MapPLUTO of t0",
        "zoning": "MapPLUTO of t0", "plan": "none", "after_t0": "none",
    },
    "Milwaukee": {
        "outcome": "MPROP year built in (t0, t1]", "existing": "MPROP of t0", "context": "MPROP of t0",
        "zoning": "MPROP of t0", "plan": "none (PDF only)", "after_t0": "none",
    },
    "Chicago": {
        "outcome": "new-construction permits issued in [t0, t1)",
        "existing": "CMAP inventory before t0; Cook County Assessor of t0", "context": "assessor of t0",
        "zoning": "current layer; districts created on or after t0 flagged", "plan": "none",
        "after_t0": "flagged lots removed in the conservative variant",
    },
}


def _identity(y0: int, y1: int) -> dict:
    import geopandas as gpd
    import shapely

    from planarena.models.history import snapshot_dir

    g0 = gpd.read_parquet(snapshot_dir(y0) / "parcels.geoparquet").geometry
    g1 = gpd.read_parquet(snapshot_dir(y1) / "parcels.geoparquet").geometry
    a, b = shapely.STRtree(g1.values).query(g0.representative_point().values, predicate="within")
    j1 = np.full(len(g0), -1)
    j1[a] = b
    hits = np.bincount(j1[j1 >= 0], minlength=len(g1))
    c, d = shapely.STRtree(g1.representative_point().values).query(g0.values, predicate="contains")
    per_t0 = np.bincount(c, minlength=len(g0))
    return {"t0_lots": int(len(g0)), "t1_lots": int(len(g1)), "t0_lots_without_t1_lot": int((j1 < 0).sum()),
            "merges_t1_lots_from_two_or_more": int((hits >= 2).sum()),
            "t0_lots_in_merges": int(hits[hits >= 2].sum()),
            "splits_t0_lots_into_two_or_more": int((per_t0 >= 2).sum())}


def _manhattan_rule(y0: int, y1: int) -> dict:
    from planarena.models import rolling
    from planarena.models.history import snapshot_dir

    lab = rolling._y(y0, y1)
    r = rolling._rows(lab)
    yb0 = pd.read_parquet(snapshot_dir(y0) / "parcels.parquet", columns=["year_built"]).year_built.to_numpy()
    # labels the rule removed: lots already built at t0 with a year built of t0 or later, without floor-area growth
    guarded = r & (yb0 >= y0) & ~lab.floor_area_growth.to_numpy().astype(bool)
    return {"developed": int((lab.developed.to_numpy() & r).sum()), "removed_by_new_building_rule": int(guarded.sum())}


def main() -> dict:
    from planarena.models.integration import PANELS, panel

    res = {"test": __doc__.strip().split("\n\n")[1].replace("\n", " "), "feature_timestamps": TIMESTAMPS,
           "manhattan_pairs": {}, "cross_city_new_building_rule": {}}
    for y0, y1 in ((2015, 2016), (2015, 2020), (2019, 2020), (2020, 2021), (2020, 2025), (2024, 2025)):
        res["manhattan_pairs"][f"{y0}->{y1}"] = _identity(y0, y1) | _manhattan_rule(y0, y1)
        print(y0, y1, res["manhattan_pairs"][f"{y0}->{y1}"], flush=True)
    for g in ("0644000", "4819000"):
        for t0, t1 in PANELS[g][1]:
            p = panel(g, t0, t1)
            already = (p.year0.to_numpy() >= t0)
            res["cross_city_new_building_rule"][f"{g} {t0}->{t1}"] = {
                "lots": int(len(p)), "developed": int(p.developed.sum()),
                "lots_already_built_at_t0_with_year_built_t0_or_later": int(already.sum())}
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
