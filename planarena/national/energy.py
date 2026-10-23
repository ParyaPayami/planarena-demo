"""Building energy and emission coefficients for any US place, from national open microdata.

* Residential site EUI: EIA RECS 2020 microdata, by state (weighted kBtu of electricity, natural gas, propane and
  fuel oil per energy-consuming sq ft).
* Non-residential site EUI: EIA CBECS 2018 microdata, by census division and PlanArena use (principal building
  activity grouped into commercial, public, industrial/warehouse, transport/other); national value where a cell has
  fewer than 30 buildings. CBECS excludes manufacturing, so "industrial" is warehouse and storage.
* Emission factors: electricity from EPA eGRID2023 state total output CO2e rates; fuels and district heat from the
  EPA GHG Emission Factors Hub (2025), CH4 and N2O converted with the hub's AR5 100-year GWPs. Each use's factor is
  weighted by that use's fuel mix in the same microdata.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from planarena.national.common import NATIONAL, download, sha256

E = NATIONAL / "energy"
URLS = {
    "cbecs": "https://www.eia.gov/consumption/commercial/data/2018/xls/cbecs2018_final_public.csv",
    "recs": "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv",
    "egrid": "https://www.epa.gov/system/files/documents/2025-06/egrid2023_data_rev2.xlsx",
    "hub": "https://www.epa.gov/system/files/other-files/2025-01/ghg-emission-factors-hub-2025.xlsx",
}
# Census divisions (CBECS CENDIV codes)
DIVISIONS = {
    1: "CT ME MA NH RI VT", 2: "NJ NY PA", 3: "IL IN MI OH WI", 4: "IA KS MN MO NE ND SD",
    5: "DE DC FL GA MD NC SC VA WV", 6: "AL KY MS TN", 7: "AR LA OK TX", 8: "AZ CO ID MT NV NM UT WY",
    9: "AK CA HI OR WA",
}
STATE_DIVISION = {s: d for d, states in DIVISIONS.items() for s in states.split()}
# CBECS 2018 principal building activity -> PlanArena use
PBA_USE = {2: "com", 6: "com", 15: "com", 18: "com", 23: "com", 24: "com", 25: "com", 26: "com",
           4: "pub", 7: "pub", 8: "pub", 12: "pub", 13: "pub", 14: "pub", 16: "pub", 17: "pub",
           5: "ind", 11: "ind", 91: "tra"}
KBTU_PER_MWH = 3412.14
LB_PER_T = 2204.62
MIN_N = 30


def _f(key: str):
    return download(URLS[key], E / URLS[key].rsplit("/", 1)[1])


@lru_cache(maxsize=1)
def fuel_factors() -> dict:
    """t CO2e per kBtu for natural gas, fuel oil No. 2, propane and steam/heat, read from the EPA hub."""
    d = pd.read_excel(_f("hub"), header=None)
    rows = {str(r[1]).strip() if pd.notna(r[1]) else str(r[0]).strip(): r for r in d.itertuples(index=False)}
    lab = lambda r: next((str(v).strip() for v in r if isinstance(v, str)), "")
    gwp, fac = {}, {}
    for r in d.itertuples(index=False):
        vals = [v for v in r if pd.notna(v)]
        name = lab(vals)
        nums = [float(v) for v in vals if isinstance(v, (int, float))]
        if name in ("CH4", "N2O") and nums and name not in gwp:
            gwp[name] = nums[-1]
        for key, label, idx in (("natural_gas", "Natural Gas", 1), ("fuel_oil", "Distillate Fuel Oil No. 2", 1),
                                ("propane", "Propane", 1), ("district_heat", "Steam and Heat", 0)):
            if name == label and key not in fac and len(nums) >= idx + 3:
                fac[key] = nums[idx:idx + 3]  # kg CO2, g CH4, g N2O per mmBtu
    del rows
    if set(fac) != {"natural_gas", "fuel_oil", "propane", "district_heat"} or set(gwp) != {"CH4", "N2O"}:
        raise ValueError(f"EPA hub parse failed: {fac} {gwp}")
    out = {k: (co2 + ch4 / 1000 * gwp["CH4"] + n2o / 1000 * gwp["N2O"]) / 1000 / 1000 for k, (co2, ch4, n2o) in fac.items()}
    out["_detail"] = {"kg_co2_g_ch4_g_n2o_per_mmbtu": fac, "gwp_ar5": gwp}
    return out


@lru_cache(maxsize=1)
def electricity_factors() -> dict:
    """t CO2e per kBtu of electricity by state (eGRID2023 STC2ERTA, lb/MWh)."""
    s = pd.read_excel(_f("egrid"), "ST23", header=1)
    return {r.PSTATABB: float(r.STC2ERTA) / LB_PER_T / KBTU_PER_MWH for r in s.itertuples() if pd.notna(r.STC2ERTA)}


@lru_cache(maxsize=1)
def _cbecs() -> pd.DataFrame:
    c = pd.read_csv(_f("cbecs"), usecols=["PBA", "CENDIV", "SQFT", "FINALWT", "ELBTU", "NGBTU", "FKBTU", "DHBTU"])
    c["use"] = c.PBA.map(PBA_USE)
    c = c[c.use.notna() & (c.SQFT > 0)].fillna(0)
    return c


@lru_cache(maxsize=1)
def _recs() -> pd.DataFrame:
    r = pd.read_csv(_f("recs"), usecols=["state_postal", "NWEIGHT", "TOTSQFT_EN", "BTUEL", "BTUNG", "BTULP", "BTUFO"])
    return r[r.TOTSQFT_EN > 0].fillna(0)


def _summ(df, w, area, fuels: dict) -> dict:
    tot = sum(df[c] for c in fuels.values())
    wa = (df[w] * df[area]).sum()
    eui = float((df[w] * tot).sum() / wa)
    b = (tot / df[area]).to_numpy()
    q25, q75 = np.quantile(b, [0.25, 0.75]) if len(b) > 3 else (eui, eui)
    share = {k: float((df[w] * df[c]).sum() / (df[w] * tot).sum()) for k, c in fuels.items()}
    return {"value": round(eui, 1), "low": round(float(min(q25, eui)), 1), "high": round(float(max(q75, eui)), 1),
            "n_buildings": int(len(df)), "fuel_share": share}


def coefficients(state: str) -> dict:
    """calibrated.yaml blocks for a state: EUI and emission factor by use."""
    div = STATE_DIVISION[state]
    ff, el = fuel_factors(), electricity_factors()
    ef_fuel = {"electricity": el[state], "natural_gas": ff["natural_gas"], "fuel_oil": ff["fuel_oil"],
               "propane": ff["propane"], "district_heat": ff["district_heat"]}
    r = _recs()
    rs = r[r.state_postal == state]
    res = _summ(rs, "NWEIGHT", "TOTSQFT_EN", {"electricity": "BTUEL", "natural_gas": "BTUNG", "propane": "BTULP",
                                                "fuel_oil": "BTUFO"})
    c = _cbecs()
    nonres = {}
    for u in ("com", "pub", "ind", "tra"):
        cell = c[(c.use == u) & (c.CENDIV == div)]
        scope = f"division {div}"
        if len(cell) < MIN_N:
            cell, scope = c[c.use == u], "national"
        nonres[u] = _summ(cell, "FINALWT", "SQFT", {"electricity": "ELBTU", "natural_gas": "NGBTU",
                                                     "fuel_oil": "FKBTU", "district_heat": "DHBTU"}) | {"scope": scope}
    eui = {"res": res} | nonres
    # Mixed residential and commercial: the city's assumed residential share applies in the engine; the EUI here
    # is the floor-area-weighted mean of residential and commercial at a 70/30 split (assumption).
    eui["mix"] = {k: round(0.7 * res[k] + 0.3 * nonres["com"][k], 1) for k in ("value", "low", "high")}
    eui["mix"]["fuel_share"] = {k: 0.7 * res["fuel_share"].get(k, 0) + 0.3 * nonres["com"]["fuel_share"].get(k, 0)
                                for k in set(res["fuel_share"]) | set(nonres["com"]["fuel_share"])}
    efu = {u: {"value": float(sum(s * ef_fuel[f] for f, s in v["fuel_share"].items()))} for u, v in eui.items()}
    vals = [v["value"] for v in efu.values()]
    sources = {k: {"url": URLS[k], "sha256": sha256(_f(k))} for k in URLS}
    return {
        "energy_site_eui_kbtu_per_sqft": {
            "status": "derived",
            "source": f"EIA RECS 2020 microdata ({state}) for residential; EIA CBECS 2018 microdata (census division "
                      f"{div}, national where fewer than {MIN_N} buildings) for non-residential; weighted site energy "
                      "per sq ft; range Q25-Q75 of building values. Mixed use: 70/30 residential/commercial (assumption).",
        } | {u: {k: v for k, v in d.items() if k != "fuel_share"} for u, d in eui.items()},
        "emissions_tco2e_per_kbtu_by_use": {
            "status": "derived",
            "source": f"Electricity: EPA eGRID2023 {state} total output CO2e rate; fuels and district heat: EPA GHG "
                      "Emission Factors Hub 2025 (AR5 GWPs); weighted by each use's fuel mix in RECS/CBECS.",
        } | {u: {"value": round(v["value"], 9)} for u, v in efu.items()},
        "emissions_tco2e_per_kbtu": {"value": round(float(np.mean(vals)), 9), "low": round(min(vals), 9),
                                     "high": round(max(vals), 9), "status": "derived",
                                     "source": "Mean of the use-specific factors (range = min/max across uses)."},
        "national_sources": sources,
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(coefficients(sys.argv[1] if len(sys.argv) > 1 else "WI"), indent=1, default=float)[:3000])
    print(json.dumps(fuel_factors()["_detail"]), re.sub(r"\s+", " ", str({k: round(v * 1e6, 2) for k, v in electricity_factors().items() if k in ("WI", "NY", "CA", "WV")})))
