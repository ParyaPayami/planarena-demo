"""Calibrate impact-engine coefficients from primary sources and write planarena/indicators/calibrated.yaml.

Sources (verified on 2026-10-16):
* NYC Local Law 97 of 2019, Admin. Code §28-320.3.1.1 (2024-2029 GHG coefficients), read from the official
  text at https://www.nyc.gov/assets/buildings/local_laws/ll97of2019.pdf
* NYC LL84 Energy and Water Data Disclosure, calendar 2024, Manhattan (NYC Open Data 5zyy-y8am), giving
  median site EUI by building type and the borough's fuel mix
* EIA CBECS 2018 Table B2, median floorspace per worker by building activity
  (https://www.eia.gov/consumption/commercial/data/2018/bc/html/b2.php)
* BLS QCEW, New York County employment, December 2024 = 2,480,984
  (https://www.bls.gov/regions/northeast/news-release/2025/countyemploymentandwages_newyork_20250612.htm)
* U.S. Census QuickFacts, New York County, persons per household 2020-2024 = 2.00 (2019-2023: 2.01).
  The official page is bot-protected; the value is corroborated by two search-index copies of it.

Run with:  python -m planarena.calibrate   (it downloads the LL84 extract if the reference file is missing)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date

import numpy as np
import pandas as pd
import yaml

from planarena.config import DATA, ROOT

REF = DATA / "reference"
LL84_FILE = REF / "ll84_manhattan_2024.json"
LL84_QUERY = ("https://data.cityofnewyork.us/resource/5zyy-y8am.json?"
              "$select=primary_property_type,site_eui_kbtu_ft,property_gfa_calculated,natural_gas_use_kbtu,"
              "electricity_use_grid_purchase,district_steam_use_kbtu,fuel_oil_2_use_kbtu,fuel_oil_4_use_kbtu"
              "&$where=upper(borough)='MANHATTAN' AND report_year='2024'&$limit=50000")
LL84_BBL_FILE = REF / "ll84_manhattan_2024_bbl.json"
LL84_BBL_QUERY = LL84_QUERY.replace("$select=", "$select=nyc_borough_block_and_lot,")
OUT = ROOT / "planarena/indicators/calibrated.yaml"

LL97 = {  # tCO2e per kBtu of site energy, Admin. Code §28-320.3.1.1 (2024-2029)
    "electricity_use_grid_purchase": 0.000288962 / 3.412,  # law states 0.000288962 tCO2e per kWh
    "natural_gas_use_kbtu": 0.00005311,
    "district_steam_use_kbtu": 0.00004493,
    "fuel_oil_2_use_kbtu": 0.00007421,
    "fuel_oil_4_use_kbtu": 0.00007529,
}

GROUPS = {
    "res": ["Multifamily Housing", "Residence Hall/Dormitory", "Senior Living Community", "Other - Lodging/Residential"],
    "com": ["Office", "Financial Office", "Retail Store", "Hotel", "Bank Branch", "Fitness Center/Health Club/Gym",
            "Supermarket/Grocery Store", "Restaurant", "Other - Recreation", "Other - Mall", "Strip Mall",
            "Enclosed Mall", "Other - Services", "Other - Entertainment/Public Assembly"],
    "mix": ["Mixed Use Property"],
    "ind": ["Self-Storage Facility", "Non-Refrigerated Warehouse", "Distribution Center",
            "Manufacturing/Industrial Plant", "Refrigerated Warehouse"],
    "pub": ["K-12 School", "College/University", "Hospital (General Medical & Surgical)", "Medical Office",
            "Worship Facility", "Library", "Museum", "Community Center and Social Meeting Hall", "Police Station",
            "Fire Station", "Other - Public Services", "Performing Arts", "Courthouse", "Laboratory",
            "Other - Education", "Pre-school/Daycare", "Outpatient Rehabilitation/Physical Therapy",
            "Ambulatory Surgical Center", "Urgent Care/Clinic/Other Outpatient", "Other - Specialty Hospital",
            "Prison/Incarceration", "Social/Meeting Hall"],
    "tra": ["Parking", "Transportation Terminal/Station", "Other - Utility", "Energy/Power Station"],
}


MAX_SITE_EUI = 1000.0  # kBtu per sq ft: above any building type (hospitals and data centres stay below)


def plausible_ll84(d: pd.DataFrame) -> pd.DataFrame:
    """LL84 records without reporting errors: site EUI, where reported, at most MAX_SITE_EUI. (Some records report
    tens or hundreds of thousands, e.g. one Bronx senior-living building at 561,197 kBtu/sq ft.)"""
    if "site_eui_kbtu_ft" not in d:
        return d
    return d[~(pd.to_numeric(d["site_eui_kbtu_ft"], errors="coerce") > MAX_SITE_EUI)]


def fetch_ll84(query: str = LL84_QUERY, out=LL84_FILE) -> None:
    REF.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-s", "-m", "180", "-G", query.split("?")[0],
                    *sum((["--data-urlencode", p] for p in query.split("?")[1].split("&")), []),
                    "-o", str(out)], check=True)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate() -> dict:
    if not LL84_FILE.exists():
        fetch_ll84()
    d = pd.DataFrame(json.loads(LL84_FILE.read_text()))
    for c in d.columns[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ok = d[(d.site_eui_kbtu_ft > 5) & (d.site_eui_kbtu_ft < 1000) & (d.property_gfa_calculated > 0)]
    eui, ef_by_use = {}, {}
    for k, types in GROUPS.items():
        g0 = ok[ok.primary_property_type.isin(types)]
        s = g0.site_eui_kbtu_ft
        # Floor-area-weighted mean: the unbiased estimator for totals (medians understate skewed sums).
        wmean = float((s * g0.property_gfa_calculated).sum() / g0.property_gfa_calculated.sum())
        eui[k] = {"value": round(wmean, 1), "low": round(min(float(s.quantile(0.25)), wmean), 1),
                  "high": round(max(float(s.quantile(0.75)), wmean), 1), "median": round(float(s.median()), 1),
                  "n_buildings": int(len(s))}
        g = d[d.primary_property_type.isin(types)]
        tot = {f: float(g[f].clip(lower=0).sum()) for f in LL97}
        t = sum(tot.values())
        ef_by_use[k] = sum(tot[f] / t * LL97[f] for f in LL97) if t else None
    tot = {f: float(d[f].clip(lower=0).sum()) for f in LL97}
    t = sum(tot.values())
    shares = {f: tot[f] / t for f in tot}
    ef = sum(shares[f] * LL97[f] for f in LL97)
    vals = [v for v in ef_by_use.values() if v]
    stamp = date.today().isoformat()
    eui_pluto, ef_pluto = pluto_use_intensities()
    out = {
        "calibrated_on": stamp,
        "reference_sha256": {LL84_FILE.name: sha256(LL84_FILE)},
        "energy_site_eui_kbtu_per_sqft": {
            "status": "derived",
            "source": f"NYC LL84 benchmarking, calendar 2024, Manhattan (NYC Open Data 5zyy-y8am, retrieved {stamp}), "
                      "matched to MapPLUTO lots by BBL. Per PlanArena land use: site energy of the fuels LL97 counts "
                      "(electricity, natural gas, district steam, #2 and #4 oil) divided by MapPLUTO gross floor area "
                      "(ratio estimator, the unbiased estimator of totals); range Q25-Q75 of building-level values, widened "
                      "to include the estimate. Grouping by PlanArena use rather than LL84 property type was chosen on "
                      "held-out buildings (docs/VALIDATION.md, section A). Covers buildings >= 25,000 sq ft only.",
            **eui_pluto,
        },
        "emissions_tco2e_per_kbtu_by_use": {
            "status": "derived",
            "source": "NYC Local Law 97 of 2019 §28-320.3.1.1 coefficients weighted by each PlanArena use's LL84 2024 "
                      "fuel mix (sum of tCO2e / sum of counted site energy).",
            **ef_pluto,
        },
        "emissions_tco2e_per_kbtu": {
            "value": round(ef, 8), "low": round(min(vals), 8), "high": round(max(vals), 8), "status": "derived",
            "source": "NYC Local Law 97 of 2019 §28-320.3.1.1 coefficients (verified from the law text) weighted by "
                      "Manhattan LL84 2024 site-energy fuel shares: "
                      + ", ".join(f"{k.split('_use')[0]} {v:.1%}" for k, v in shares.items())
                      + ". Range = min/max across building-type fuel mixes.",
        },
        "jobs_sqft_per_worker": lodes_jobs_rates(stamp) or {
            "status": "calibrated",
            "source": "EIA CBECS 2018 Table B2 median floorspace per worker by activity (verified), scaled so the "
                      "baseline matches BLS QCEW New York County employment, December 2024 (2,480,984; verified). "
                      "Mapping: office=Office 609; retail=Retail other than mall 1,700; other_commercial=Service 1,378; "
                      "factory=Other 1,429; storage=Warehouse and storage 2,400; public_facility=Education 1,060; "
                      "transport_utility=Other 1,429. Range +/-25% (assumption).",
            "calibration_target_jobs": 2480984,
            "office": {"value": 609, "low": 609 * 0.8, "high": 609 * 1.25},
            "retail": {"value": 1700, "low": 1700 * 0.8, "high": 1700 * 1.25},
            "other_commercial": {"value": 1378, "low": 1378 * 0.8, "high": 1378 * 1.25},
            "factory": {"value": 1429, "low": 1429 * 0.8, "high": 1429 * 1.25},
            "storage": {"value": 2400, "low": 2400 * 0.8, "high": 2400 * 1.25},
            "public_facility": {"value": 1060, "low": 1060 * 0.8, "high": 1060 * 1.25},
            "transport_utility": {"value": 1429, "low": 1429 * 0.8, "high": 1429 * 1.25},
        },
        "persons_per_household": {
            "value": 2.00, "low": 1.9, "high": 2.1, "status": "corroborated",
            "source": "U.S. Census Bureau QuickFacts, New York County: persons per household 2020-2024 = 2.00 "
                      "(2019-2023 = 2.01). The page is bot-protected, so the value was corroborated from search-index "
                      "copies; confirm on census.gov.",
        },
    }
    OUT.write_text("# GENERATED by python -m planarena.calibrate. Do not edit by hand.\n" + yaml.safe_dump(out, sort_keys=False))
    return out


LODES_WAC = DATA / "raw" / "lodes8_ny_2023" / "ny_wac_S000_JT00_2023.csv.gz"
LODES_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/ny/wac/ny_wac_S000_JT00_2023.csv.gz"
QCEW_NY_COUNTY_DEC_2024 = 2480984


def lodes_jobs_rates(stamp: str) -> dict | None:
    """Floor area per worker by floor-area type, fitted to LODES 2023 jobs by census block (non-negative least
    squares on MapPLUTO floor-area components). The engine scales the total to QCEW. Chosen over the CBECS rates by
    a forward test (rates fitted on 2015 lots and LODES 2015, tested on 2025 lots against LODES 2023)."""
    import gzip

    import pyogrio
    from scipy.optimize import nnls

    from planarena.config import PLUTO_PATH
    from planarena.indicators.engine import load_engine

    if not LODES_WAC.exists():
        LODES_WAC.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-m", "300", "-o", str(LODES_WAC), LODES_URL], check=True)
    with gzip.open(LODES_WAC, "rt") as fh:
        w = pd.read_csv(fh, dtype={"w_geocode": str}, usecols=["w_geocode", "C000"])
    w = w[w.w_geocode.str.startswith("36061")].set_index("w_geocode").C000
    src = pyogrio.read_dataframe(PLUTO_PATH, columns=["BBL", "BCTCB2020"], read_geometry=False)
    blk = dict(zip(pd.to_numeric(src.BBL, errors="coerce"), src.BCTCB2020.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)))
    e = load_engine()
    keys = {"office": "office", "retail": "retail", "other": "other_commercial", "factory": "factory",
            "storage": "storage", "pubfac": "public_facility", "trafac": "transport_utility"}
    X = pd.DataFrame({c: e.obs[c] for c in keys})
    X["block"] = ["36061" + str(blk.get(int(b), "00000000000"))[1:] for b in e.p.bbl]
    X = X.groupby("block").sum()
    rates, _ = nnls(X.to_numpy(), w.reindex(X.index).fillna(0).to_numpy())
    out = {"status": "calibrated",
           "source": f"Relative rates fitted to LEHD LODES8 2023 workplace jobs by 2020 census block (non-negative least "
                     f"squares on MapPLUTO floor-area components; retrieved {stamp}); total scaled to BLS QCEW New York "
                     f"County employment, December 2024 ({QCEW_NY_COUNTY_DEC_2024:,}). Chosen over EIA CBECS 2018 rates by a "
                     "forward test (fitted on 2015, tested on 2023; docs/VALIDATION.md). Range +/-25% (assumption). "
                     "A type whose fitted rate is zero falls back to its CBECS value.",
           "calibration_target_jobs": QCEW_NY_COUNTY_DEC_2024, "sha256_lodes": sha256(LODES_WAC)}
    cbecs = {"office": 609, "retail": 1700, "other_commercial": 1378, "factory": 1429, "storage": 2400,
             "public_facility": 1060, "transport_utility": 1429}
    for (c, k), r in zip(keys.items(), rates):
        v = float(round(1.0 / float(r), 1)) if r > 1e-9 else float(cbecs[k])
        out[k] = {"value": v, "low": round(v * 0.8, 1), "high": round(v * 1.25, 1)}
    return out


def pluto_use_intensities() -> tuple[dict, dict]:
    """Counted-fuel site energy per sq ft of MapPLUTO floor area, and tCO2e per kBtu, by PlanArena land use."""
    from planarena.config import PROCESSED, USE_KEYS

    if not LL84_BBL_FILE.exists():
        fetch_ll84(LL84_BBL_QUERY, LL84_BBL_FILE)
    d = plausible_ll84(pd.DataFrame(json.loads(LL84_BBL_FILE.read_text())))
    for c in list(LL97) + ["property_gfa_calculated"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["bbl"] = pd.to_numeric(d.nyc_borough_block_and_lot.astype(str).str.split(";").str[0].str.strip(), errors="coerce")
    d["fuel"] = sum(d[c].clip(lower=0).fillna(0) for c in LL97)
    d["t"] = sum(d[c].clip(lower=0).fillna(0) * v for c, v in LL97.items())
    d = d[d.fuel > 0].drop_duplicates("bbl")
    p = pd.read_parquet(PROCESSED / "parcels.parquet")[["bbl", "use", "bldg_area"]]
    p["bbl"] = p.bbl.astype("int64")
    m = d.merge(p, on="bbl", how="inner")
    m = m[(m.bldg_area > 0) & (m.use >= 0)]
    eui, ef = {}, {}
    for u in ("res", "mix", "com", "ind", "tra", "pub"):
        g = m[m.use == USE_KEYS.index(u)]
        if len(g) < 10:
            continue
        val = float(g.fuel.sum() / g.bldg_area.sum())
        per = g.fuel / g.bldg_area
        eui[u] = {"value": round(val, 1), "low": round(min(float(per.quantile(0.25)), val), 1),
                  "high": round(max(float(per.quantile(0.75)), val), 1), "n_buildings": int(len(g))}
        ef[u] = {"value": round(float(g.t.sum() / g.fuel.sum()), 8), "n_buildings": int(len(g))}
    return eui, ef


def validate_emissions() -> dict:
    """Building-level check of modelled emissions against LL84 metered energy x LL97 factors.

    In-sample: the EUIs are fitted to the same LL84 year, so this measures how well use-type
    averages reproduce individual buildings, not out-of-sample skill.
    """
    from planarena.indicators.engine import load_engine

    if not LL84_BBL_FILE.exists():
        fetch_ll84(LL84_BBL_QUERY, LL84_BBL_FILE)
    e = load_engine()
    d = plausible_ll84(pd.DataFrame(json.loads(LL84_BBL_FILE.read_text())))
    for c in d.columns[2:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["bbl"] = pd.to_numeric(d.nyc_borough_block_and_lot.astype(str).str.split(";").str[0].str.strip(), errors="coerce")
    d["observed_t"] = sum(d[f].clip(lower=0).fillna(0) * v for f, v in LL97.items())
    d = d[d.observed_t > 0].drop_duplicates("bbl")
    idx = {int(b): i for i, b in enumerate(e.p.bbl)}
    d["i"] = d.bbl.map(idx)
    d = d.dropna(subset=["i"])
    d["modelled_t"] = e.baseline_state["ghg"][d.i.astype(int).values]
    m = d[d.modelled_t > 0]
    r = float(np.corrcoef(np.log(m.observed_t), np.log(m.modelled_t))[0, 1])
    out = {"n_buildings": int(len(m)), "log_log_r": round(r, 3), "log_r2": round(r * r, 3),
           "median_abs_pct_error": round(float(np.median(np.abs(m.modelled_t - m.observed_t) / m.observed_t)), 3),
           "aggregate_ratio_modelled_to_observed": round(float(m.modelled_t.sum() / m.observed_t.sum()), 3),
           "observed_mt": round(float(m.observed_t.sum() / 1e6), 3), "modelled_mt": round(float(m.modelled_t.sum() / 1e6), 3),
           "note": "In-sample (EUIs fitted on the same LL84 year); LL84 covers buildings >= 25,000 sq ft.",
           "reference_sha256": sha256(LL84_BBL_FILE)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "validation_emissions_ll84.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=1, default=str)[:1500])
    print(json.dumps(validate_emissions(), indent=1))
