"""Tier B adapter: all of New York City (five boroughs), from the citywide NYC DCP MapPLUTO 25v3.1 release.

Same records, schema and zoning rules-as-code as the Manhattan demo (planarena/twin/build.py), for every lot in the
city. Like every other Tier B city, the coefficients come from national sources for New York State (RECS, CBECS,
eGRID) and the jobs rates are fitted to the city's LODES blocks; the Manhattan demo keeps its LL84-calibrated
coefficients. Districts are the 59 community districts (and the joint interest areas of parks and airports).
There is no conversion-precedent table outside Manhattan, so the optimizer's precedent filter does not apply here.

Run:  python -I -m planarena.cities.adapters.nyc
"""
from __future__ import annotations

import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from planarena.cities.common import finish_twin
from planarena.sources import crosswalk
from planarena.taxonomy import INDEX as TAX_INDEX
from planarena.cities.registry import city_dir, register
from planarena.config import PROJECTED_CRS
from planarena.national import census, places
from planarena.twin.build import PLUTO_COLS, tabulate
from planarena.twin.fetch import DIR, RELEASE, URL, fetch
from planarena.twin.zoning import RULE_SOURCE

GEOID = "3651000"
BOROUGHS = {"1": ("Manhattan", "061"), "2": ("Bronx", "005"), "3": ("Brooklyn", "047"), "4": ("Queens", "081"),
            "5": ("Staten Island", "085")}


def build() -> dict:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    fetch()  # downloads and unpacks the citywide archive if needed
    pl = places.place(GEOID)
    out = city_dir(GEOID)
    out.mkdir(parents=True, exist_ok=True)
    shp = DIR / "extracted" / "MapPLUTO.shp"
    g = gpd.read_file(shp, columns=PLUTO_COLS + ["BCTCB2020"], engine="pyogrio")
    g = g.to_crs(PROJECTED_CRS) if g.crs is None or g.crs.to_string() != PROJECTED_CRS else g
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    g["geometry"] = g.geometry.make_valid()
    log(f"{len(g):,} lots in five boroughs")
    df = tabulate(g)
    df["zone16"] = np.array([TAX_INDEX.get(crosswalk.nyc_zone(z), -1) for z in df.zone], dtype="int16")
    bct = g["BCTCB2020"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    df["census_block"] = ["36" + BOROUGHS[b[0]][1] + b[1:] if len(b) == 11 and b[0] in BOROUGHS else "" for b in bct]

    bnd = places.boundary(GEOID)
    blocks = census.blocks(GEOID, bnd.bounds)
    blocks = blocks[blocks.GEOID.str[:5].isin([f"36{c}" for _, c in BOROUGHS.values()]) & (blocks.AREALAND > 0)]
    blocks = blocks.to_crs(PROJECTED_CRS).reset_index(drop=True)
    pos = pd.Series(np.arange(len(blocks)), index=blocks.GEOID.to_numpy())
    blk = pos.reindex(df.census_block.to_numpy()).fillna(-1).astype(int).to_numpy().copy()
    miss = blk < 0
    if miss.any():  # lots without a 2020 block code (or a code not in TIGERweb): nearest block
        rp = g.geometry.representative_point().values[miss]
        blk[miss] = shapely.STRtree(blocks.geometry.values).nearest(rp)
        df.loc[miss, "census_block"] = blocks.GEOID.to_numpy()[blk[miss]]
    jobs_block = census.lodes_jobs("36", "NY").reindex(blocks.GEOID).fillna(0).to_numpy()
    names = {int(cd): f"{BOROUGHS[str(cd)[0]][0]} CD {cd % 100}" if cd % 100 < 50 else
             f"{BOROUGHS[str(cd)[0]][0]} joint interest area {cd}" for cd in sorted(df.cd.unique()) if str(cd)[0] in BOROUGHS}
    meta = finish_twin(df, g.geometry.values, PROJECTED_CRS, out, "NY",
                       float(blocks.POP100.sum() / max(blocks.HU100.sum(), 1)), jobs_block, blk, log, {
        "city": pl["label"], "geoid": GEOID, "tier": "B",
        "source": f"NYC DCP MapPLUTO {g['Version'].iloc[0]} (all boroughs)", "source_url": URL,
        "source_archive": json.loads((DIR / "SOURCE.json").read_text()),
        "data_quality": {"missing_land_use": int((df.use < 0).sum()), "lots_without_2020_block": int(miss.sum()),
                         "zero_building_area": int((df.bldg_area <= 0).sum()),
                         "homes_census_2020": int(blocks.HU100.sum()),
                         "jobs_lodes": int(round(jobs_block[np.unique(blk)].sum()))},
        "columns": {"lot boundaries": "observed (tax lots)", "floor area": "observed (MapPLUTO)",
                    "land use": "observed (MapPLUTO)", "homes": "observed (MapPLUTO)",
                    "jobs": "observed by block (LODES), modelled by floor area", "assessed value": "observed (DOF)",
                    "zoning": "simplified rules-as-code of the NYC Zoning Resolution", "year built": "observed",
                    "landmarks and historic districts": "observed",
                    "conversion precedent": "Manhattan only; not applied here"},
        "missing_indicators": [], "zoning_rules": RULE_SOURCE, "release": RELEASE,
        "levels": {str(k): v for k, v in names.items()},
    }, t0=t0)
    meta["coefficients_chosen"] = choose_coefficients(out)["chosen"]
    (out / "twin_meta.json").write_text(json.dumps(meta, indent=2))
    register(GEOID, pl["label"], "B", meta)
    log(f"done: {len(df):,} lots; energy and emission coefficients: {meta['coefficients_chosen']}")
    return meta


def choose_coefficients(out) -> dict:
    """Pick the citywide energy and emission coefficients on buildings neither candidate was fitted on.

    Candidates: national (RECS/CBECS intensities, eGRID New York State and EPA hub factors), or the Manhattan
    calibration (LL84 2024 intensities by use, Local Law 97 factors). Both are scored against LL84 2023 emissions of
    buildings outside Manhattan (LL97 factors on reported fuel use); the one with the smaller median error is kept.
    Jobs rates stay the citywide LODES fit either way.
    """
    import yaml

    from planarena.calibrate import LL97, MAX_SITE_EUI
    from planarena.config import DATA, ROOT
    from planarena.indicators.engine import CALIBRATED_PATH, load_engine

    d = pd.DataFrame(json.loads((DATA / "raw" / "ll84_2023_citywide" / "ll84_nyc_2023_bbl.json").read_text()))
    for c in list(LL97) + ["site_eui_kbtu_ft"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["bbl"] = pd.to_numeric(d.nyc_borough_block_and_lot.astype(str).str.split(";").str[0].str.strip(), errors="coerce")
    d["t"] = sum(d[c].clip(lower=0).fillna(0) * v for c, v in LL97.items())
    implausible = d.site_eui_kbtu_ft > MAX_SITE_EUI  # reporting errors (planarena.calibrate.plausible_ll84)
    obs = d[(d.t > 0) & ~implausible].groupby("bbl").t.sum()
    from planarena.national import energy

    own = out / "calibrated.yaml"
    base = yaml.safe_load(own.read_text())  # jobs rates and household size: kept whichever is chosen
    base.pop("coefficient_choice", None)
    nat = energy.coefficients("NY")
    mn = yaml.safe_load(CALIBRATED_PATH.read_text())
    keys = ("energy_site_eui_kbtu_per_sqft", "emissions_tco2e_per_kbtu_by_use", "emissions_tco2e_per_kbtu")
    candidates = {"national": base | {k: nat[k] for k in keys}, "manhattan_ll84": base | {k: mn[k] for k in keys}}
    res = {"test": choose_coefficients.__doc__.strip().split("\n\n")[0] + " " +
           choose_coefficients.__doc__.strip().split("\n\n")[1].replace("\n", " "), "scores": {}}
    for name, coef in candidates.items():
        own.write_text(yaml.safe_dump(coef, sort_keys=False))
        e = load_engine(out)
        s = pd.Series(e.baseline_state["ghg"], index=e.p.bbl.to_numpy()).groupby(level=0).sum()
        j = obs.index.intersection(s.index)
        outer = j[(j // 1_000_000_000) != 1]
        p, o = s[outer].to_numpy(), obs[outer].to_numpy()
        res["scores"][name] = {"buildings_outside_manhattan": int(len(outer)),
                               "median_abs_pct_error": round(float(np.median(np.abs(p - o) / o)), 3),
                               "total_ratio": round(float(p.sum() / o.sum()), 3),
                               "log_r": round(float(np.corrcoef(np.log1p(p), np.log1p(o))[0, 1]), 3),
                               "city_total_t": round(float(e.baseline_state["ghg"].sum()))}
    res["records_excluded_implausible_eui"] = int(implausible.sum())
    res["chosen"] = min(res["scores"], key=lambda k: res["scores"][k]["median_abs_pct_error"])
    chosen = candidates[res["chosen"]]
    chosen["coefficient_choice"] = {"chosen": res["chosen"], "evidence": "results/validation/nyc_citywide_coefficients.json"}
    own.write_text("# GENERATED by planarena.cities.adapters.nyc. Do not edit by hand.\n" + yaml.safe_dump(chosen, sort_keys=False))
    (ROOT / "results" / "validation" / "nyc_citywide_coefficients.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("city", "n_parcels", "use_counts", "data_quality", "build_seconds")}, indent=1))
