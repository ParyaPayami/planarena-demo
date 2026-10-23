"""Validation against independent, observed data. Writes results/validation/*.json and docs/VALIDATION.md.

Downloaded inputs live in data/raw/<dataset>/ (untrusted data; never imported as code). Run steps with
    python -I -m planarena.validate <step>        steps: emissions, jobs, backtest, units, east_harlem,
                                                   optimizer, report, all
The land-use backtest trains PyTorch and LightGBM models in separate processes (OpenMP on macOS).

1. emissions   Out-of-sample: coefficients fitted on LL84 2024, tested on LL84 2023 metered buildings.
2. jobs        Modelled jobs by census block and tract against LEHD LODES 2023 workplace jobs.
3. backtest    2015 -> 2025: models trained on MapPLUTO 15v1 predict which parcels changed use and to what.
4. units       Homes model: replay of 2015-2025 development (observed use and floor area in, homes out).
5. east_harlem The East Harlem rezoning (2017): PlanArena on the FEIS projected development sites against
               the FEIS RWCDS and against what was built by 2025.
6. optimizer   Optimality gap of the optimizer against exhaustive search on small study areas.
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from planarena.config import DATA, PLUTO_PATH, PROCESSED, ROOT, USE_INDEX, USE_KEYS

RAW = DATA / "raw"
P15 = DATA / "processed_15v1"
OUT = ROOT / "results" / "validation"
PLUTO_CUR = PLUTO_PATH
U = USE_INDEX
N_USES = len(USE_KEYS)


def _save(name: str, obj: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
    return obj


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC by the rank-sum formula (ties get average ranks)."""
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _logr(a, b):
    m = (a > 0) & (b > 0)
    return float(np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1]), int(m.sum())


# ----------------------------------------------------------------------------- 1. emissions
def emissions() -> dict:
    from planarena.calibrate import LL97, plausible_ll84
    from planarena.indicators.engine import load_engine

    f = RAW / "ll84_2023" / "ll84_manhattan_2023_bbl.json"
    d = plausible_ll84(pd.DataFrame(json.loads(f.read_text())))
    for c in list(LL97) + ["property_gfa_calculated"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["bbl"] = pd.to_numeric(d.nyc_borough_block_and_lot.astype(str).str.split(";").str[0].str.strip(), errors="coerce")
    d["observed_t"] = sum(d[c].clip(lower=0).fillna(0) * v for c, v in LL97.items())
    d = d[d.observed_t > 0].drop_duplicates("bbl")
    e = load_engine()
    idx = {int(b): i for i, b in enumerate(e.p.bbl)}
    d["i"] = d.bbl.map(idx)
    d = d.dropna(subset=["i"])
    d["modelled_t"] = e.baseline_state["ghg"][d.i.astype(int).values]
    m = d[d.modelled_t > 0]
    r, n = _logr(m.observed_t.to_numpy(), m.modelled_t.to_numpy())
    return _save("emissions_ll84_2023", {
        "test": "Out of sample: EUIs and fuel mix fitted on LL84 calendar 2024; tested on calendar 2023 metered energy x LL97 factors",
        "n_buildings": n, "log_log_r": round(r, 3),
        "median_abs_pct_error": round(float(np.median(np.abs(m.modelled_t - m.observed_t) / m.observed_t)), 3),
        "aggregate_ratio_modelled_to_observed": round(float(m.modelled_t.sum() / m.observed_t.sum()), 3),
        "observed_mt": round(float(m.observed_t.sum() / 1e6), 3), "modelled_mt": round(float(m.modelled_t.sum() / 1e6), 3),
        "source": "NYC Open Data 5zyy-y8am, report_year 2023, Manhattan", "source_sha256": _sha(f),
    })


# ----------------------------------------------------------------------------- 2. jobs
def jobs() -> dict:
    import pyogrio

    from planarena.indicators.engine import load_engine

    wac_f = RAW / "lodes8_ny_2023" / "ny_wac_S000_JT00_2023.csv.gz"
    with gzip.open(wac_f, "rt") as fh:
        wac = pd.read_csv(fh, dtype={"w_geocode": str}, usecols=["w_geocode", "C000"])
    wac = wac[wac.w_geocode.str.startswith("36061")]
    src = pyogrio.read_dataframe(PLUTO_CUR, columns=["BBL", "BCTCB2020"], read_geometry=False)
    src["bbl"] = pd.to_numeric(src.BBL, errors="coerce")
    e = load_engine()
    p = pd.DataFrame({"bbl": e.p.bbl.astype("int64"), "jobs": e.baseline_state["jobs"]})
    p = p.merge(src[["bbl", "BCTCB2020"]].dropna(), on="bbl", how="left")
    p = p.dropna(subset=["BCTCB2020"])
    code = p.BCTCB2020.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
    p["block"] = "36061" + code.str[1:]
    p["tract"] = p.block.str[:11]
    mb = p.groupby("block").jobs.sum()
    ob = wac.set_index("w_geocode").C000
    in_sample = "LODES" in e.coef["jobs_sqft_per_worker"].get("source", "")
    res = {"test": "Modelled baseline jobs (floor area per worker by type, scaled to QCEW 2024) against LEHD LODES8 "
                   "workplace jobs, 2023, Manhattan, by 2020 census block and tract",
           "in_sample": in_sample,
           "note": ("In sample: the rates are fitted on LODES 2023. Out-of-sample evidence is the forward test in "
                    "jobs_rates.json.") if in_sample else "Rates from EIA CBECS; not fitted on LODES.",
           "lodes_total_jobs": int(ob.sum()), "modelled_total_jobs": round(float(p.jobs.sum())),
           "parcels_without_block": int(len(e.p) - len(p)),
           "source": "https://lehd.ces.census.gov/data/lodes/LODES8/ny/wac/ny_wac_S000_JT00_2023.csv.gz",
           "source_sha256": _sha(wac_f)}
    for level, mod, obs in (("block", mb, ob), ("tract", p.groupby("tract").jobs.sum(),
                                                ob.groupby(ob.index.str[:11]).sum())):
        j = pd.concat([mod.rename("modelled"), obs.rename("observed")], axis=1).fillna(0)
        r, n = _logr(j.observed.to_numpy(), j.modelled.to_numpy())
        big = j[j.observed >= 100]
        res[level] = {
            "units": int(len(j)), "log_log_r": round(r, 3), "n_both_positive": n,
            "spearman": round(float(j.observed.corr(j.modelled, method="spearman")), 3),
            "median_abs_pct_error_where_observed_ge_100": round(float(np.median(np.abs(big.modelled - big.observed) / big.observed)), 3),
            "share_of_observed_jobs_in_units_with_modelled_jobs": round(float(j.observed[j.modelled > 0].sum() / j.observed.sum()), 3),
        }
    return _save("jobs_lodes_2023", res)


# ----------------------------------------------------------------------------- 3. backtest (2015 -> 2025)
def _ensure_twin15() -> None:
    if not (P15 / "parcels.parquet").exists():
        from planarena.twin.build import build

        build(out_dir=P15, pluto_path=RAW / "pluto_15v1" / "extracted" / "Manhattan" / "MNMapPLUTO.shp", web=False)


def backtest_gnn(epochs: int = 250) -> None:
    """Torch process: final GNN on the 2015 twin; every parcel predicted while hidden (10 random chunks)."""
    import torch

    from planarena.models.features import context_features
    from planarena.models.train import train_final

    _ensure_twin15()
    df = pd.read_parquet(P15 / "parcels.parquet")
    graph = dict(np.load(P15 / "graph.npz"))
    t0 = time.time()
    model, suit, ctx = train_final(df, graph, context_features(df), epochs)
    np.save(P15 / "suit_gnn.npy", suit)
    torch.save({"state_dict": model.state_dict()}, P15 / "gnn.pt")
    print(f"GNN 2015 trained in {time.time() - t0:.0f}s", flush=True)


def backtest_lgbm(chunks: int = 10) -> None:
    """LightGBM process: the same hidden-chunk protocol as the GNN, with neighbour composition features."""
    import lightgbm as lgb

    from planarena.models.common import SEED, neighbour_composition
    from planarena.models.features import context_features, zone_family

    _ensure_twin15()
    df = pd.read_parquet(P15 / "parcels.parquet")
    graph = dict(np.load(P15 / "graph.npz"))
    x = context_features(df)
    use = df.use.to_numpy().astype("int64")
    chunk = np.random.default_rng(SEED).integers(0, chunks, len(df))
    fam = df.zone.map(zone_family).to_numpy()
    suit = np.zeros((len(df), N_USES))
    prior_out = np.zeros((len(df), N_USES))
    for c in range(chunks):
        hid = chunk == c
        tr = ~hid & (use >= 0)
        feats = np.concatenate([x, neighbour_composition(use, hid, graph)], 1)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, verbose=-1, random_state=SEED)
        clf.fit(feats[tr], use[tr])
        suit[np.ix_(hid, clf.classes_)] = clf.predict_proba(feats[hid])
        prior = pd.crosstab(fam[tr], use[tr]).reindex(columns=range(N_USES), fill_value=0) + 1
        prior = prior.div(prior.sum(1), axis=0)
        glob = np.bincount(use[tr], minlength=N_USES) + 1.0
        prior_out[hid] = np.stack([prior.loc[z].to_numpy() if z in prior.index else glob / glob.sum() for z in fam[hid]])
        print(f"lgbm chunk {c}", flush=True)
    np.save(P15 / "suit_lgbm.npy", suit)
    np.save(P15 / "suit_zoning.npy", prior_out)


def _matched() -> pd.DataFrame:
    a = pd.read_parquet(P15 / "parcels.parquet").reset_index(names="i15")
    b = pd.read_parquet(PROCESSED / "parcels.parquet").reset_index(names="i25")
    cols = ["bbl", "use", "units_res", "bldg_area", "res_area", "lot_area", "zone", "max_far", "built_far", "cd",
            "year_built"]
    m = a[["i15"] + cols].merge(b[["i25"] + cols], on="bbl", suffixes=("15", "25"))
    return m


def backtest_eval() -> dict:
    m = _matched()
    ok = (m.use15 >= 0) & (m.use25 >= 0)
    m = m[ok].reset_index(drop=True)
    changed = (m.use15 != m.use25).to_numpy()
    i15 = m.i15.to_numpy()
    cur = m.use15.to_numpy().astype(int)
    new = m.use25.to_numpy().astype(int)
    rows = np.arange(len(m))
    scores, dest = {}, {}
    for name in ("gnn", "lgbm", "zoning"):
        P = np.load(P15 / f"suit_{name}.npy")[i15]
        scores[name] = 1 - P[rows, cur]  # misfit of the current use
        Q = P.copy()
        Q[rows, cur] = 0
        Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
        ch = changed
        rank = (Q[ch] > Q[ch, new[ch]][:, None]).sum(1)
        dest[name] = {"top1": round(float((rank == 0).mean()), 3), "top3": round(float((rank < 3).mean()), 3)}
    underbuilt = 1 - np.clip(np.divide(m.built_far15, m.max_far15, out=np.zeros(len(m)), where=m.max_far15 > 0), 0, 1)
    scores["underbuilt (1 - built/max FAR)"] = underbuilt.to_numpy()
    base_rate = changed.mean()
    k = max(1, int(0.05 * len(m)))
    det = {}
    for name, s in scores.items():
        top = np.argsort(-s)[:k]
        det[name] = {"auc": round(_auc(s, changed), 3), "precision_top5pct": round(float(changed[top].mean()), 3),
                     "lift_top5pct": round(float(changed[top].mean() / base_rate), 2)}
    # Physical change: the use changed and the building was replaced or substantially altered (new year built
    # after 2014, or floor area changed by more than 25%). Filters out reclassifications in PLUTO.
    b15, b25 = m.bldg_area15.to_numpy(), m.bldg_area25.to_numpy()
    rebuilt = (m.year_built25.to_numpy() >= 2015) | (np.abs(b25 - b15) > 0.25 * np.maximum(b15, 1))
    phys = changed & rebuilt
    det_phys = {}
    for name, s in scores.items():
        top = np.argsort(-s)[:k]
        det_phys[name] = {"auc": round(_auc(s, phys), 3), "precision_top5pct": round(float(phys[top].mean()), 3),
                          "lift_top5pct": round(float(phys[top].mean() / phys.mean()), 2)}
    dest_phys = {}
    for name in ("gnn", "lgbm", "zoning"):
        P = np.load(P15 / f"suit_{name}.npy")[i15]
        Q = P.copy()
        Q[rows, cur] = 0
        Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
        rank = (Q[phys] > Q[phys, new[phys]][:, None]).sum(1)
        dest_phys[name] = {"top1": round(float((rank == 0).mean()), 3), "top3": round(float((rank < 3).mean()), 3)}
    trans = pd.crosstab(pd.Series([USE_KEYS[u] for u in cur], name="2015"), pd.Series([USE_KEYS[u] for u in new], name="2025"))
    # Destination baseline without any model: the most common destination of each 2015 use is unknowable
    # in advance (it would use the outcome), so the reference is chance among the other 8 uses.
    return _save("backtest_2015_2025", {
        "test": "Models trained on MapPLUTO 15v1 (2015) only. Each parcel is predicted while hidden. Outcome: land use in "
                "MapPLUTO 25v3.1 for parcels matched by BBL.",
        "matched_parcels": int(len(m)), "changed": int(changed.sum()), "base_rate": round(float(base_rate), 4),
        "change_detection": det, "destination_of_changed_parcels": dest | {"chance": {"top1": round(1 / 8, 3), "top3": round(3 / 8, 3)}},
        "physical_change": {"n": int(phys.sum()), "base_rate": round(float(phys.mean()), 4),
                            "change_detection": det_phys, "destination": dest_phys},
        "transitions": {r: {c: int(v) for c, v in trans.loc[r].items() if v} for r in trans.index},
        "caveat": "Some recorded changes are reclassifications in PLUTO rather than physical change.",
        "source_sha256_15v1": _sha(RAW / "pluto_15v1" / "mn_mappluto_15v1.zip"),
    })


def backtest() -> dict:
    py = [sys.executable, "-I", "-m", "planarena.validate"]
    if not (P15 / "suit_gnn.npy").exists():
        subprocess.run(py + ["backtest_gnn"], check=True, cwd=ROOT)
    if not (P15 / "suit_lgbm.npy").exists():
        subprocess.run(py + ["backtest_lgbm"], check=True, cwd=ROOT)
    return backtest_eval()


# ----------------------------------------------------------------------------- 4. units (homes) replay
def units() -> dict:
    from planarena.indicators.engine import load_engine

    _ensure_twin15()
    m = _matched()
    dev = (m.units_res25 - m.units_res15 >= 5) & (m.bldg_area25 > 1.2 * m.bldg_area15) & m.use25.isin([U["res"], U["mix"]])
    d = m[dev & (m.lot_area15 > 0)]
    e = load_engine(P15)
    use = e.base_use.copy()
    far = np.full(e.n, np.nan)
    idx = d.i15.to_numpy()
    use[idx] = d.use25.to_numpy()
    far[idx] = (d.bldg_area25 / d.lot_area15).to_numpy()
    st = e.parcel_state(use, far)
    pred = st["units"][idx]
    obs = d.units_res25.to_numpy().astype(float)
    eis = d.res_area25.to_numpy() / 900.0  # the CEQR dwelling-unit factor applied to the observed residential floor area
    r, n = _logr(obs, pred)
    return _save("units_replay_2015_2025", {
        "test": "Parcels developed 2015-2025 (>= 5 new homes and > 20% more floor area). Inputs: the 2025 use and the "
                "observed 2025 floor area. Output: homes, compared with homes recorded in MapPLUTO 25v3.1.",
        "parcels": int(len(d)), "observed_homes": int(obs.sum()), "modelled_homes": round(float(pred.sum())),
        "aggregate_ratio": round(float(pred.sum() / obs.sum()), 3), "log_log_r": round(r, 3),
        "median_abs_pct_error": round(float(np.median(np.abs(pred - obs) / obs)), 3),
        "reference_ceqr_900sf_on_observed_residential_area": {
            "aggregate_ratio": round(float(eis.sum() / obs.sum()), 3),
            "median_abs_pct_error": round(float(np.median(np.abs(eis - obs) / obs)), 3)},
        "note": "PlanArena does not know the residential share of a new building; the reference uses the observed one.",
    })


# ----------------------------------------------------------------------------- 5. East Harlem rezoning replay
def _parse_rwcds(txt: str, start: str, stop: str) -> pd.DataFrame:
    a = txt.index(start)
    b = txt.index(stop, a + 1)
    rows = []
    for line in txt[a:b].splitlines():
        f = re.split(r"\s{2,}", line.strip())
        if len(f) < 10 or not re.fullmatch(r"\d+[A-Z]?", f[0]) or not re.fullmatch(r"\d{3,4}", f[1]):
            continue
        num = lambda s: float(s.replace(",", "")) if re.fullmatch(r"-?[\d,]+(\.\d+)?", s) else np.nan
        rows.append({"site": f[0], "block": int(f[1]), "lots": [int(x) for x in re.findall(r"\d+", f[2])],
                     "lot_area": num(f[3]), "zoning": f[4], "max_far": num(f[5]), "built_far": num(f[6]),
                     "bldg_fa": num(f[7]), "res_fa": num(f[8]), "du": num(f[9])})
    return pd.DataFrame(rows)


def east_harlem() -> dict:
    import geopandas as gpd

    from planarena.indicators.engine import load_engine

    pdf = RAW / "eis_east_harlem" / "appenda_feis.pdf"
    txt_f = RAW / "eis_east_harlem" / "appa.txt"
    if not txt_f.exists():
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt_f)], check=True)
    txt = txt_f.read_text()
    marker = "Appendix A-3"
    a3 = txt[txt.index(marker):]
    na = _parse_rwcds(a3, "No Action Total", "With Action Total")  # No Action projected-site table
    wa = _parse_rwcds(a3, "With Action Total", "Increment")  # With Action projected-site table
    # Projected sites only: the first No Action / With Action tables before the potential-site tables.
    na, wa = na.drop_duplicates("site"), wa.drop_duplicates("site")
    sites = wa.merge(na[["site", "du", "bldg_fa"]], on="site", suffixes=("", "_no_action"))
    sites["bbls"] = sites.apply(lambda r: [1_000_000_000 + r.block * 10_000 + lot for lot in r.lots], axis=1)

    e15 = load_engine(P15)
    idx15 = {int(b): i for i, b in enumerate(e15.p.bbl)}
    g15 = gpd.read_parquet(P15 / "parcels.geoparquet")
    g25 = gpd.read_parquet(PROCESSED / "parcels.geoparquet")
    p25 = pd.read_parquet(PROCESSED / "parcels.parquet")
    pts25 = g25.geometry.representative_point()
    tree = __import__("shapely").STRtree(pts25.values)

    use = e15.base_use.copy()
    far_eis = np.full(e15.n, np.nan)
    far_rule = np.full(e15.n, np.nan)
    rows = []
    for _, s in sites.iterrows():
        ii = [idx15[b] for b in s.bbls if b in idx15]
        if not ii:
            continue
        resid = s.res_fa / s.bldg_fa if s.bldg_fa > 0 else 0
        u = U["res"] if resid >= 0.9 else U["mix"] if resid > 0 else U["com"]
        use[ii] = u
        far_eis[ii] = s.built_far
        far_rule[ii] = s.max_far * e15.coef["assumptions"]["buildout_utilization"]["value"]
        site_geom = g15.geometry.iloc[ii].union_all()
        hit = tree.query(site_geom, predicate="contains")
        rows.append({"site": s.site, "i15": ii, "i25": hit.tolist(), "eis_du": s.du, "eis_du_no_action": s.du_no_action,
                     "existing_du_2015": float(e15.p.units_res.iloc[ii].sum()),
                     "du_2025": float(p25.units_res.iloc[hit].sum())})
    base_units = e15.baseline_state["units"]
    st_eis = e15.parcel_state(use, far_eis)
    st_rule = e15.parcel_state(use, far_rule)
    for r in rows:
        r["planarena_du_same_far"] = float(st_eis["units"][r["i15"]].sum())
        r["planarena_du_zoning_rule"] = float(st_rule["units"][r["i15"]].sum())
    t = pd.DataFrame(rows)
    tot = lambda c: float(t[c].sum())
    eis_inc = tot("eis_du") - tot("eis_du_no_action")
    built_inc = tot("du_2025") - tot("existing_du_2015")
    res = {
        "test": "East Harlem Rezoning FEIS (CEQR 17DCP048M, 2017), Appendix A-3 projected development sites. PlanArena "
                "is run on the same sites: (a) with the FEIS site FAR and use, (b) from zoning alone (max FAR x "
                "build-out utilization). Built outcome: homes recorded in MapPLUTO 25v3.1 on the same land.",
        "sites_parsed": int(len(sites)), "sites_matched_2015": int(len(t)),
        "feis_with_action_du": tot("eis_du"), "feis_no_action_du": tot("eis_du_no_action"), "feis_increment_du": eis_inc,
        "planarena_du_same_far": round(tot("planarena_du_same_far")),
        "planarena_du_zoning_rule": round(tot("planarena_du_zoning_rule")),
        "ratio_same_far_to_feis": round(tot("planarena_du_same_far") / tot("eis_du"), 3),
        "ratio_zoning_rule_to_feis": round(tot("planarena_du_zoning_rule") / tot("eis_du"), 3),
        "site_log_r_same_far": round(_logr(t.eis_du.to_numpy(), t.planarena_du_same_far.to_numpy())[0], 3),
        "existing_du_2015": tot("existing_du_2015"), "du_2025": tot("du_2025"), "built_increment_2015_2025": built_inc,
        "share_of_feis_increment_built_by_2025": round(built_inc / eis_inc, 3) if eis_inc else None,
        "feis_analysis_year": 2027,
        "parse_check": {
            "feis_table_totals": {"with_action_du": 5960, "with_action_bldg_sf": 6433375, "no_action_du": 2472,
                                  "no_action_bldg_sf": 2978556},
            "parsed": {"with_action_du": tot("eis_du"), "with_action_bldg_sf": float(sites.bldg_fa.sum()),
                       "no_action_du": tot("eis_du_no_action"), "no_action_bldg_sf": float(sites.bldg_fa_no_action.sum())},
            "note": "Floor areas match the FEIS totals exactly; dwelling units differ by 1-3 because the FEIS rounds each row."},
        "source": "https://www.nyc.gov/assets/planning/download/pdf/applicants/env-review/east-harlem/appenda_feis.pdf",
        "source_sha256": _sha(pdf),
        "planarena_baseline_units_on_sites_2015": round(float(sum(base_units[r["i15"]].sum() for r in rows))),
    }
    return _save("east_harlem_rezoning", res)


# ----------------------------------------------------------------------------- 5b. SoHo/NoHo (2021) site-level
P20 = DATA / "processed_20v8"


def _soho_tables(txt: str) -> list[pd.DataFrame]:
    """Appendix A2 tables in order: No Action (projected, potential), With Action (...), Increment (...)."""
    tables, cur = [], []
    for line in txt.splitlines():
        tok = line.split()
        if tok and tok[0] == "TOTAL":
            tables.append(pd.DataFrame(cur))
            cur = []
            continue
        grp = [k for k, t_ in enumerate(tok) if t_ in ("Projected", "Potential")]
        if len(tok) > 10 and grp:
            g = grp[0]  # existing zoning can span two tokens (split lots), so locate the group column
            num = lambda t: (0.0 if t == "-" else float(t.replace(",", "")) if re.fullmatch(r"-?[\d,]+(\.\d+)?", t)
                             else np.nan)
            cur.append({"dsow": tok[0], "site": tok[1], "lot_area": num(tok[3]), "zone_new": tok[g - 1], "group": tok[g],
                        "res_zfa": num(tok[g + 1]), "res_gfa": num(tok[g + 2]), "du": num(tok[g + 3]),
                        "total_gfa": num(tok[-6]) if len(tok) > 30 else np.nan})
    return tables


def soho_noho() -> dict:
    from planarena.indicators.engine import load_engine

    pdf = RAW / "eis_soho_noho" / "appenda-feis.pdf"
    txt_f = RAW / "eis_soho_noho" / "appa.txt"
    if not txt_f.exists():
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt_f)], check=True)
    txt = txt_f.read_text()
    t = _soho_tables(txt[txt.index("Appendix A2"):txt.index("Appendix A3")])
    no_action, with_action = t[0], t[2]
    # Appendix A3: BBL -> DSOW site id
    a3 = txt[txt.index("Appendix A3"):]
    bbl_site = {}
    for line in a3.splitlines():
        for bbl, dsow in re.findall(r"(1\d{9})\s+\S+\s+(\S+)", line):
            bbl_site[int(bbl)] = dsow
    e20 = load_engine(P20)
    idx20 = {int(b): i for i, b in enumerate(e20.p.bbl)}
    p25 = pd.read_parquet(PROCESSED / "parcels.parquet")
    idx25 = {int(b): i for i, b in enumerate(p25.bbl)}
    use = e20.base_use.copy()
    far = np.full(e20.n, np.nan)
    rows = []
    for _, r in with_action.iterrows():
        bbls = [b for b, d in bbl_site.items() if d == r.dsow]
        ii = [idx20[b] for b in bbls if b in idx20]
        if not ii or not r.total_gfa > 0:
            continue
        resid = r.res_gfa / r.total_gfa
        use[ii] = U["res"] if resid >= 0.9 else U["mix"] if resid > 0 else U["com"]
        far[ii] = r.total_gfa / r.lot_area
        na = no_action[no_action.dsow == r.dsow]
        rows.append({"dsow": r.dsow, "i": ii, "feis_du": r.du, "feis_du_no_action": float(na.du.sum()) if len(na) else 0.0,
                     "du_2020": float(e20.p.units_res.iloc[ii].sum()),
                     "du_2025": float(sum(p25.units_res.iloc[idx25[b]] for b in bbls if b in idx25))})
    st = e20.parcel_state(use, far)
    for r in rows:
        r["planarena_du"] = float(st["units"][r["i"]].sum())
    d = pd.DataFrame(rows)
    proj = with_action[with_action.group == "Projected"]
    res = {
        "test": "SoHo/NoHo Neighborhood Plan FEIS (2021), Appendix A2 projected development sites (BBLs from Appendix A3). "
                "PlanArena is run on the same sites with the FEIS gross floor area and use; built outcome from MapPLUTO "
                "25v3.1 (BBL match).",
        "projected_sites_parsed": int(len(proj)), "sites_matched_2020": int(len(d)),
        "parse_check": {"appendix_total_projected_du": 1861, "parsed": float(proj.du.sum()),
                        "note": "Appendix A2 totals 1,861 projected DUs; the FEIS text reports 1,858."},
        "feis_with_action_du": float(d.feis_du.sum()), "feis_no_action_du": float(d.feis_du_no_action.sum()),
        "feis_text_with_action_du": 1858, "feis_text_increment_du": 1826, "feis_appendix_total_du": float(proj.du.sum()),
        "planarena_du_same_gfa": round(float(d.planarena_du.sum())),
        "ratio_to_feis": round(float(d.planarena_du.sum() / d.feis_du.sum()), 3),
        "site_log_r": round(_logr(d.feis_du.to_numpy(), d.planarena_du.to_numpy())[0], 3),
        "du_2020": float(d.du_2020.sum()), "du_2025": float(d.du_2025.sum()),
        "feis_unit_factor_sf": 850, "feis_analysis_year": 2031, "adopted": "2021-12",
        "source": "https://www.nyc.gov/assets/planning/download/pdf/applicants/env-review/soho-noho/appenda-feis.pdf",
        "source_sha256": _sha(pdf),
    }
    return _save("soho_noho_rezoning", res)


# ----------------------------------------------------------------------------- 5c. area-level replays (all three)
REZONINGS = {
    "East Harlem (2017)": {"cd": 111, "before": "15", "year": 2017, "feis_increment_du": 3488, "horizon": 2027,
                           "feis": "East Harlem Rezoning FEIS, CEQR 17DCP048M, Table 1-1a"},
    "Inwood (2018)": {"cd": 112, "before": "15", "year": 2018, "feis_increment_du": 3908, "horizon": 2033,
                      "feis": "Inwood Rezoning, CEQR 17DME007M, OEC project page (Technical Memorandum 003, 15-year)"},
    "SoHo/NoHo (2021)": {"cd": 102, "before": "20", "year": 2021, "feis_increment_du": 1826, "horizon": 2031,
                         "feis": "SoHo/NoHo Neighborhood Plan FEIS (2021), Chapter 1"},
}


def rezonings() -> dict:
    from planarena.indicators import rwcds
    from planarena.indicators.engine import load_engine
    from planarena.twin.zoning import allowed_uses

    import geopandas as gpd
    import shapely

    p25 = pd.read_parquet(PROCESSED / "parcels.parquet")
    pts25 = gpd.read_parquet(PROCESSED / "parcels.geoparquet").geometry.representative_point().values
    tree25 = shapely.STRtree(pts25)
    out = {}
    for name, cfg in REZONINGS.items():
        d0 = P15 if cfg["before"] == "15" else P20
        e0 = load_engine(d0)
        b = e0.p[["bbl", "zone", "max_far", "units_res", "cd"]].reset_index(names="i0").merge(
            p25[["bbl", "zone", "max_far", "units_res"]], on="bbl", suffixes=("_0", "_1"))
        b = b[b.cd == cfg["cd"]]
        rez = b[b.zone_0 != b.zone_1]
        new_max = e0.p["max_far"].to_numpy().copy()
        new_max[rez.i0.to_numpy()] = rez.max_far_1.to_numpy()
        area = np.zeros(e0.n, bool)
        area[rez.i0.to_numpy()] = True
        res_ok = np.zeros(e0.n, bool)
        res_ok[rez.i0.to_numpy()] = [("res" in allowed_uses(z)) for z in rez.zone_1]
        sites = rwcds.screen(e0, new_max, area, cfg["year"])
        full = rwcds.project(e0, sites, new_max, res_ok, 1.0)
        util = rwcds.project(e0, sites, new_max, res_ok, e0.coef["assumptions"]["buildout_utilization"]["value"])
        # Observed homes by 2025 on the same land: 2025 lots whose representative point falls inside the
        # rezoned (or screened) t0 lots, so merged lots of new buildings are followed.
        g0 = gpd.read_parquet(d0 / "parcels.geoparquet").geometry.values
        def built_on(idx0):
            if not len(idx0):
                return 0.0
            hit = tree25.query(shapely.union_all(g0[idx0]), predicate="contains")
            return float(p25.units_res.iloc[hit].sum() - e0.p.units_res.iloc[idx0].sum())
        observed = built_on(rez.i0.to_numpy())
        observed_sites = built_on(np.nonzero(sites)[0])
        out[name] = {
            "rezoned_lots_matched": int(len(rez)), "screened_sites": full["sites"],
            "feis_increment_du": cfg["feis_increment_du"], "feis_source": cfg["feis"], "feis_horizon": cfg["horizon"],
            "planarena_increment_du_at_max_far": round(full["homes_increment"]),
            "planarena_increment_du_at_utilisation": round(util["homes_increment"]),
            "ratio_max_far_to_feis": round(full["homes_increment"] / cfg["feis_increment_du"], 3),
            "observed_increment_du_rezoned_lots_to_2025": observed,
            "observed_increment_du_on_screened_sites_to_2025": observed_sites,
            "observed_share_of_feis_increment": round(observed / cfg["feis_increment_du"], 3),
        }
    return _save("rezoning_replays", {
        "test": "Area-level replay. Rezoned lots = zoning changed between the release before adoption and MapPLUTO "
                "25v3.1 within the community district (BBL match); homes built are counted on the same land by geometry. PlanArena applies its CEQR-style development-site "
                "screen (planarena/indicators/rwcds.py) under the new maximum FAR and projects homes; compared with the "
                "FEIS increment and with homes recorded by 2025. Mandatory Inclusionary Housing bonus FAR is not modelled.",
        "rezonings": out,
    })


# ----------------------------------------------------------------------------- 0. delta equations (R2)
def delta_forms(sigma: float = 0.2, seed: int = 0) -> dict:
    """Do baseline residuals change the recommendations? Each delta form, each district, with and without a
    synthetic residual (log-normal noise, sd ``sigma``) applied to the observed floor-area components and homes."""
    from planarena.indicators.engine import load_engine
    from planarena.optimize.suggest import SuggestConfig, Suggester

    def engine_for(form, noisy):
        e = load_engine()
        e.delta_form = form
        if noisy:
            rng = np.random.default_rng(seed)
            f = np.exp(rng.normal(0, sigma, e.n))
            for k in e.obs:
                e.obs[k] = e.obs[k].copy() * f
            e.obs_units = e.obs_units.copy() * f
            e.baseline_state = e.parcel_state(e.base_use)
            e._variant_cache = {}
            e._base_acc = e.access(e.baseline_state)
            e._city_base = e.aggregate(e.baseline_state, e._base_acc, np.ones(e.n, bool))
        return e

    cds = [f"cd:{c}" for c in range(101, 113)]
    out, sets = {}, {}
    for form in ("replacement", "difference", "hybrid"):
        for noisy in (False, True):
            e = engine_for(form, noisy)
            sug = Suggester(e, None)
            e.clip_events = 0
            sug.per_use_values()
            for cd in cds:
                r = sug.suggest(SuggestConfig(scope=cd, w_fit=0.0))
                sets[(form, noisy, cd)] = set(r["changed"].tolist())
            out.setdefault(form, {})["clip_events_per_use_table" if not noisy else "clip_events_noisy"] = int(e.clip_events)
    jac = lambda a, b: len(a & b) / max(len(a | b), 1)
    for form in out:
        js = [jac(sets[(form, False, cd)], sets[(form, True, cd)]) for cd in cds]
        out[form]["jaccard_clean_vs_noisy_median"] = round(float(np.median(js)), 3)
        out[form]["jaccard_clean_vs_noisy_min"] = round(float(np.min(js)), 3)
        out[form]["changes_per_district_median"] = int(np.median([len(sets[(form, False, cd)]) for cd in cds]))
    cross = {f"{a} vs {b}": round(float(np.median([jac(sets[(a, False, cd)], sets[(b, False, cd)]) for cd in cds])), 3)
             for a, b in (("replacement", "hybrid"), ("difference", "hybrid"), ("replacement", "difference"))}
    return _save("delta_forms", {
        "test": "Optimizer recommendations (12 districts, default settings, GNN fit term off) under three delta forms, "
                f"with and without a synthetic baseline residual (log-normal, sd {sigma}) on observed floor-area "
                "components and homes. Jaccard = overlap of the recommended parcel sets (1 = identical).",
        "forms": out, "agreement_between_forms_median_jaccard": cross,
        "note": "Clip events: quantities floored at zero, which happen when a modelled old-use composition exceeds "
                "what was observed (the difference form cannot represent complete conversions without flooring).",
    })


# ----------------------------------------------------------------------------- 2c. change prediction (R1)
def _geo_match(d0: Path, d1: Path) -> np.ndarray:
    """For every t0 lot, the index of the t1 lot containing its representative point (-1 if none)."""
    import geopandas as gpd
    import shapely

    g0 = gpd.read_parquet(d0 / "parcels.geoparquet").geometry
    g1 = gpd.read_parquet(d1 / "parcels.geoparquet").geometry
    a, b = shapely.STRtree(g1.values).query(g0.representative_point().values, predicate="within")
    j = np.full(len(g0), -1)
    j[a] = b
    return j


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[ok], obs[ok]
    if len(o) < 5:
        return {"n": int(len(o))}
    rank = lambda v: pd.Series(v).rank().to_numpy()
    return {"n": int(len(o)), "mae": round(float(np.mean(np.abs(p - o))), 2),
            "median_ae": round(float(np.median(np.abs(p - o))), 2),
            "total_ratio": round(float(p.sum() / o.sum()), 3) if abs(o.sum()) > 1e-9 else None,
            "pearson_r": round(float(np.corrcoef(p, o)[0, 1]), 3) if p.std() > 0 and o.std() > 0 else None,
            "spearman": round(float(np.corrcoef(rank(p), rank(o))[0, 1]), 3) if p.std() > 0 and o.std() > 0 else None,
            "sign_agreement": round(float(np.mean(np.sign(np.round(p, 6)) == np.sign(np.round(o, 6)))), 3)}


def _change_events(min_sqft: float = 2000.0) -> dict:
    """Lots that changed use or floor area between MapPLUTO 15v1 and 25v3.1, with their observed changes in homes,
    assessed value (net of the district trend) and emissions (LL84 2015 -> 2024), and the LODES 2015/2023 job counts
    by 2020 census block. Shared by changes() and planarena.models.impact_test."""
    import pyogrio

    from planarena.calibrate import LL97, plausible_ll84
    from planarena.indicators.engine import load_engine
    from planarena.models.common import spatial_folds
    from planarena.models.transition import size_class

    p0 = pd.read_parquet(P15 / "parcels.parquet")
    p1 = pd.read_parquet(PROCESSED / "parcels.parquet")
    j = _geo_match(P15, PROCESSED)
    ok0 = (j >= 0) & (p0.use.to_numpy() >= 0)
    grp = pd.DataFrame({"i0": np.nonzero(ok0)[0], "j1": j[ok0]})
    agg = grp.groupby("j1").i0.apply(list)
    fa0 = agg.map(lambda ii: float(p0.bldg_area.iloc[ii].sum()))
    lot0 = agg.map(lambda ii: float(p0.lot_area.iloc[ii].sum()))
    lots0_a, uses0_a = p0.lot_area.to_numpy(), p0.use.to_numpy()
    use0 = agg.map(lambda ii: int(uses0_a[ii][np.argmax(lots0_a[ii])]))  # use of the largest original lot
    ev = pd.DataFrame({"j1": agg.index, "i0": agg.values, "fa0": fa0.values, "lot0": lot0.values, "use0": use0.values})
    ev["use1"] = p1.use.to_numpy()[ev.j1]
    ev["fa1"] = p1.bldg_area.to_numpy()[ev.j1]
    ev = ev[ev.use1 >= 0]
    conv = ev.use1 != ev.use0
    dens = (np.abs(ev.fa1 - ev.fa0) > 0.25 * np.maximum(ev.fa0, 1)) & (np.abs(ev.fa1 - ev.fa0) > min_sqft)
    ev = ev[conv | dens].copy()
    ev["kind"] = np.where(ev.use1 != ev.use0, "use change", "density only")
    # Density-only events need evidence of construction: a building completed since 2015 on the lot. Others are
    # mostly record changes in PLUTO (e.g. floor area missing in 2015). Inconsistent 2015 records are excluded.
    h0 = ev.i0.map(lambda ii: float(p0.units_res.iloc[ii].sum())).to_numpy()
    per = np.divide(ev.fa0.to_numpy(), h0, out=np.full(len(ev), 1000.0), where=h0 > 0)
    consistent = (ev.fa0.to_numpy() >= 1000) | (h0 <= 0)
    consistent &= (h0 <= 0) | ((per >= 250) & (per <= 5000))
    built = p1.year_built.to_numpy()[ev.j1] >= 2015
    dens_ev = (ev.kind == "density only").to_numpy()
    audit = {"density_events_total": int(dens_ev.sum()),
             "density_with_new_building": int((dens_ev & built).sum()),
             "density_inconsistent_2015_record": int((dens_ev & ~consistent).sum()),
             "density_without_construction_evidence": int((dens_ev & ~built).sum())}
    ev = ev[~dens_ev | (built & consistent)].copy()
    ev.loc[ev.kind == "density only", "kind"] = "density only (new building)"

    # Observed changes
    ev["d_homes"] = p1.units_res.to_numpy()[ev.j1] - ev.i0.map(lambda ii: float(p0.units_res.iloc[ii].sum()))
    single = grp.groupby("j1").i0.transform("count") == 1
    unchanged = grp[single.to_numpy()].copy()
    unchanged = unchanged[~unchanged.j1.isin(ev.j1)]
    v0u = p0.assess_tot.to_numpy()[unchanged.i0]
    v1u = p1.assess_tot.to_numpy()[unchanged.j1]
    cdu = p0.cd.to_numpy()[unchanged.i0]
    okv = (v0u > 0) & (v1u > 0)
    growth = pd.Series(v1u[okv] / v0u[okv]).groupby(cdu[okv]).median()
    g_city = float(np.median(v1u[okv] / v0u[okv]))
    ev["cd0"] = ev.i0.map(lambda ii: int(p0.cd.iloc[ii[0]]))
    gf = ev.cd0.map(lambda c: growth.get(c, g_city))
    ev["d_value"] = p1.assess_tot.to_numpy()[ev.j1] / gf - ev.i0.map(lambda ii: float(p0.assess_tot.iloc[ii].sum()))

    def ll84(f, bbl_field="nyc_borough_block_and_lot"):
        d = plausible_ll84(pd.DataFrame(json.loads(Path(f).read_text())))
        for c in LL97:
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        d["bbl"] = pd.to_numeric(d[bbl_field].astype(str).str.split(";").str[0].str.strip(), errors="coerce")
        d["t"] = sum(d[c].clip(lower=0).fillna(0) * v for c, v in LL97.items())
        return d[d.t > 0].groupby("bbl").t.sum()
    em0 = ll84(RAW / "ll84_2015" / "ll84_manhattan_2015_bbl.json")
    em1 = ll84(DATA / "reference" / "ll84_manhattan_2024_bbl.json")
    bbl0, bbl1 = p0.bbl.to_numpy(), p1.bbl.to_numpy()
    def em_change(r):
        b1 = int(bbl1[r.j1])
        befores = [(int(bbl0[i]), float(p0.bldg_area.iloc[i])) for i in r.i0]
        if b1 not in em1.index or not all(b in em0.index or fa <= 0 for b, fa in befores):
            return np.nan
        return float(em1[b1] - sum(em0.get(b, 0.0) for b, _ in befores))
    ev["d_ghg"] = ev.apply(em_change, axis=1)

    # Census blocks of the 2025 lots, for jobs
    src = pyogrio.read_dataframe(PLUTO_CUR, columns=["BBL", "BCTCB2020"], read_geometry=False)
    blk = dict(zip(pd.to_numeric(src.BBL, errors="coerce"), src.BCTCB2020.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)))
    ev["block"] = ["36061" + str(blk.get(int(bbl1[j]), "00000000000"))[1:] for j in ev.j1]
    def wac(f):
        with gzip.open(f, "rt") as fh:
            w = pd.read_csv(fh, dtype={"w_geocode": str}, usecols=["w_geocode", "C000"])
        return w[w.w_geocode.str.startswith("36061")].set_index("w_geocode").C000
    w15, w23 = wac(RAW / "lodes8_ny_2015" / "ny_wac_S000_JT00_2015.csv.gz"), wac(RAW / "lodes8_ny_2023" / "ny_wac_S000_JT00_2023.csv.gz")
    allb = w15.index.union(w23.index)
    w15, w23 = w15.reindex(allb, fill_value=0), w23.reindex(allb, fill_value=0)
    quiet = ~allb.isin(ev.block.unique())
    jg = float(w23[quiet].sum() / w15[quiet].sum())

    return {"ev": ev, "p0": p0, "p1": p1, "w15": w15, "w23": w23, "jg": jg, "audit": audit}


def changes(min_sqft: float = 2000.0) -> dict:
    """Prediction of change (R1): lots that changed use or floor area between MapPLUTO 15v1 and 25v3.1.

    The 2015 engine gets the 2015 state plus the observed new use and floor area and predicts the change in homes,
    jobs, assessed value and emissions. Observed changes: homes and value from MapPLUTO (value net of the district
    trend of unchanged lots), jobs from LODES 2015 -> 2023 by 2020 census block (net of the trend of blocks without
    changes), emissions from LL84 calendar 2015 -> 2024 metered energy x LL97 factors.
    """
    d = _change_events(min_sqft)
    ev, p0, p1, w15, w23, jg, audit = d["ev"], d["p0"], d["p1"], d["w15"], d["w23"], d["jg"], d["audit"]
    from planarena.indicators.engine import load_engine
    from planarena.models.common import spatial_folds
    from planarena.models.transition import size_class

    # Predictions per delta form
    lodes15_total = int(w15.sum())
    res = {}
    lot0_arr, frac_cache = p0.lot_area.to_numpy(), None
    for form in ("replacement", "difference", "hybrid"):
        e = load_engine(P15)
        e.delta_form = form
        e.coef["jobs_sqft_per_worker"]["calibration_target_jobs"] = lodes15_total  # target known in 2015
        e._calibrate_jobs()
        e.baseline_state = e.parcel_state(e.base_use)
        use = e.base_use.copy()
        far = np.full(e.n, np.nan)
        for r in ev.itertuples():
            ii = np.array(r.i0)
            use[ii] = r.use1
            far[ii] = r.fa1 / max(r.lot0, 1.0)
        st = e.parcel_state(use, far)
        base = e.baseline_state
        pred = {k: ev.i0.map(lambda ii, k=k: float((st[k][ii] - base[k][ii]).sum())).to_numpy()
                for k in ("homes", "jobs", "value", "ghg")}
        out = {}
        for kind in ("all", "use change", "density only (new building)"):
            m = np.ones(len(ev), bool) if kind == "all" else (ev.kind == kind).to_numpy()
            out[kind] = {"homes": _metrics(pred["homes"][m], ev.d_homes.to_numpy()[m]),
                         "assessed_value": _metrics(pred["value"][m], ev.d_value.to_numpy()[m]),
                         "emissions": _metrics(pred["ghg"][m], ev.d_ghg.to_numpy()[m])}
        bj = pd.Series(pred["jobs"]).groupby(ev.block.to_numpy()).sum()
        obs_j = (w23.reindex(bj.index, fill_value=0) - jg * w15.reindex(bj.index, fill_value=0)).to_numpy()
        out["all"]["jobs_by_block"] = _metrics(bj.to_numpy(), obs_j)
        res[form] = out

    # Baselines: no change, and the mean observed change of the same transition type and size class
    # in other spatial folds (cross-fitted, so a lot never informs its own baseline).
    folds = spatial_folds(p0.iloc[ev.i0.map(lambda ii: ii[0])].reset_index(drop=True), k=5)
    key = (ev.use0.astype(str) + ">" + ev.use1.astype(str) + ":" + pd.Series(size_class(ev.fa0.to_numpy()), index=ev.index).astype(str)).to_numpy()
    base_pred = {}
    for col in ("d_homes", "d_value", "d_ghg"):
        v = ev[col].to_numpy()
        bp = np.full(len(ev), np.nan)
        for f in range(5):
            tr, te = folds != f, folds == f
            means = pd.Series(v[tr]).groupby(key[tr]).mean()
            overall = float(np.nanmean(v[tr]))
            bp[te] = [means.get(k, overall) if np.isfinite(means.get(k, np.nan)) else overall for k in key[te]]
        base_pred[col] = bp
    zero = np.zeros(len(ev))
    for name, preds in (("no change", {c: zero for c in base_pred}), ("transition-type mean (cross-fitted)", base_pred)):
        out = {}
        for kind in ("all", "use change", "density only (new building)"):
            m = np.ones(len(ev), bool) if kind == "all" else (ev.kind == kind).to_numpy()
            out[kind] = {"homes": _metrics(preds["d_homes"][m], ev.d_homes.to_numpy()[m]),
                         "assessed_value": _metrics(preds["d_value"][m], ev.d_value.to_numpy()[m]),
                         "emissions": _metrics(preds["d_ghg"][m], ev.d_ghg.to_numpy()[m])}
        res[name] = out
    return _save("change_prediction", {
        "test": changes.__doc__.strip().replace("\n", " "),
        "events": int(len(ev)), "use_change_events": int((ev.kind == "use change").sum()),
        "density_only_events": int((ev.kind == "density only (new building)").sum()),
        "density_event_audit": audit,
        "emissions_events": int(np.isfinite(ev.d_ghg).sum()), "blocks_with_events": int(ev.block.nunique()),
        "jobs_trend_factor_quiet_blocks": round(jg, 4), "lodes_2015_total": lodes15_total,
        "methods": res,
        "unvalidated": ["access to parks, shops and facilities after a change", "runoff after a change",
                        "emissions of buildings under 25,000 sq ft (not in LL84)", "jobs at lot level (LODES is by block)"],
        "sources_sha256": {"ll84_2015": _sha(RAW / "ll84_2015" / "ll84_manhattan_2015_bbl.json"),
                           "lodes_2015": _sha(RAW / "lodes8_ny_2015" / "ny_wac_S000_JT00_2015.csv.gz")},
    })


# ----------------------------------------------------------------------------- 6. optimizer gap
def optimizer(n_areas: int = 20, budget: int = 2, seed: int = 0) -> dict:
    import itertools

    from planarena.indicators.engine import load_engine
    from planarena.optimize.suggest import SuggestConfig, Suggester

    e = load_engine()
    sug = Suggester(e, None)
    pen = SuggestConfig().conversion_penalty
    rng = np.random.default_rng(seed)
    blocks = e.p.groupby(["cd", "block"]).size()
    out = []
    for (cd, blk) in rng.permutation(blocks.index.to_numpy()):
        if len(out) >= n_areas:
            break
        cand = np.nonzero((e.p.block.to_numpy() == blk) & (e.cd == cd))[0]
        cfg = SuggestConfig(scope=f"cd:{int(cd)}", candidate_parcels=cand, w_fit=0.0,
                            max_change_share=1.0, rounds=budget)
        _, eligible, allowed = sug.feasible(cfg, e.base_use)
        el = np.nonzero(eligible)[0]
        moves = [(p, u) for p in el for u in np.nonzero(allowed[p])[0] if u != e.base_use[p]]
        if not 4 <= len(el) <= 12 or not 4 <= len(moves) <= 40:
            continue
        cfg.max_change_share = budget / len(el)
        J = lambda use, k: e.evaluate(use, scope=cfg.scope)["overall"] - pen * k
        best, best_set = 0.0, ()
        for k in range(1, budget + 1):
            for combo in itertools.combinations(moves, k):
                if len({p for p, _ in combo}) < k:
                    continue
                use = e.base_use.copy()
                for p, u in combo:
                    use[p] = u
                v = J(use, k)
                if v > best:
                    best, best_set = v, combo
        r = sug.suggest(cfg)
        got = J(r["use"], len(r["changed"]))
        cfg.polish = 30
        t0 = time.time()
        rp = sug.suggest(cfg)
        polish_s = time.time() - t0
        cfg.polish = 0
        got_p = J(rp["use"], len(rp["changed"]))
        rand = []
        for _ in range(20):
            use = e.base_use.copy()
            pick = rng.choice(len(moves), size=min(budget, len(moves)), replace=False)
            done = set()
            for j in pick:
                p, u = moves[j]
                if p not in done:
                    use[p] = u
                    done.add(p)
            rand.append(J(use, len(done)))
        out.append({"cd": int(cd), "block": int(blk), "eligible": int(len(el)), "moves": len(moves),
                    "exact": best, "optimizer": got, "optimizer_polished": got_p, "polish_seconds": round(polish_s, 2),
                    "random_mean": float(np.mean(rand)),
                    "gap": (best - got) / best if best > 1e-12 else 0.0,
                    "gap_polished": (best - got_p) / best if best > 1e-12 else 0.0})
        print(out[-1], flush=True)
    t = pd.DataFrame(out)
    pos = t[t.exact > 1e-12]
    return _save("optimizer_gap", {
        "test": f"Exhaustive search over every feasible plan with up to {budget} changes on small study areas "
                "(one tax block; 4-12 eligible parcels), objective = overall score minus the conversion penalty, "
                "GNN fit term off. Optimizer = iterated best response with the same constraints.",
        "areas": int(len(t)), "areas_with_positive_optimum": int(len(pos)),
        "median_gap": round(float(pos.gap.median()), 4) if len(pos) else None,
        "mean_gap": round(float(pos.gap.mean()), 4) if len(pos) else None,
        "share_exact_optimum_found": round(float((pos.gap <= 1e-9).mean()), 3) if len(pos) else None,
        "optimizer_over_random_mean": round(float((pos.optimizer - pos.random_mean).mean()), 4) if len(pos) else None,
        "polished": {"median_gap": round(float(pos.gap_polished.median()), 4) if len(pos) else None,
                     "mean_gap": round(float(pos.gap_polished.mean()), 4) if len(pos) else None,
                     "share_exact_optimum_found": round(float((pos.gap_polished <= 1e-9).mean()), 3) if len(pos) else None,
                     "median_seconds": round(float(t.polish_seconds.median()), 2)},
        "material_areas": {"rule": "optimum > 0.01 score points",
                           "n": int((pos.exact > 0.01).sum()),
                           "share_exact_first_order": round(float((pos[pos.exact > 0.01].gap <= 1e-9).mean()), 3) if (pos.exact > 0.01).any() else None,
                           "share_exact_polished": round(float((pos[pos.exact > 0.01].gap_polished <= 1e-9).mean()), 3) if (pos.exact > 0.01).any() else None},
        "areas_detail": out,
    })


# ----------------------------------------------------------------------------- 2a. local error (R8)
def existing_breakdowns(seed: int = 0) -> dict:
    """Existing conditions in detail: where the aggregate agreement hides local error.

    Emissions: errors by land use, building size and district; buildings present in both LL84 years vs one;
    held-out buildings (EUIs refitted on a random half of LL84 2024 buildings, tested on the other half);
    a citywide single-EUI baseline. Jobs: a citywide jobs-per-floor-area baseline, and errors by district.
    """
    import pyogrio

    from planarena.calibrate import GROUPS, LL97, plausible_ll84
    from planarena.indicators.engine import load_engine
    from planarena.models.transition import SIZE_LABELS, size_class

    e = load_engine()
    U_ = {k: v for k, v in USE_INDEX.items()}
    ef = e._cval("emissions_tco2e_per_kbtu")
    def ll84(f):
        d = plausible_ll84(pd.DataFrame(json.loads(Path(f).read_text())))
        for c in list(LL97) + ["property_gfa_calculated"]:
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        d["bbl"] = pd.to_numeric(d.nyc_borough_block_and_lot.astype(str).str.split(";").str[0].str.strip(), errors="coerce")
        d["obs_t"] = sum(d[c].clip(lower=0).fillna(0) * v for c, v in LL97.items())
        d["site_kbtu"] = sum(d[c].clip(lower=0).fillna(0) * (3.412 if c == "electricity_use_grid_purchase" and False else 1.0) for c in LL97)
        return d[d.obs_t > 0].drop_duplicates("bbl")
    d24 = ll84(DATA / "reference" / "ll84_manhattan_2024_bbl.json")
    d23 = ll84(RAW / "ll84_2023" / "ll84_manhattan_2023_bbl.json")
    idx = {int(b): i for i, b in enumerate(e.p.bbl)}
    for d in (d24, d23):
        d["i"] = d.bbl.map(idx)
    d23 = d23.dropna(subset=["i"]).copy()
    d23["i"] = d23.i.astype(int)
    d23["model_t"] = e.baseline_state["ghg"][d23.i]
    d23 = d23[d23.model_t > 0]
    d23["in_both_years"] = d23.bbl.isin(set(d24.bbl))
    d23["use"] = [USE_KEYS[u] if u >= 0 else "unknown" for u in e.base_use[d23.i]]
    d23["size"] = [SIZE_LABELS[c] for c in size_class(e.bldg[d23.i])]
    d23["district"] = e.cd[d23.i]
    def m(g):
        o, mo = g.obs_t.to_numpy(), g.model_t.to_numpy()
        if len(o) < 10:
            return {"n": int(len(o))}
        return {"n": int(len(o)), "median_abs_pct_error": round(float(np.median(np.abs(mo - o) / o)), 3),
                "log_r": round(float(np.corrcoef(np.log(o), np.log(mo))[0, 1]), 3), "total_ratio": round(float(mo.sum() / o.sum()), 3)}
    em = {"all": m(d23),
          "by_use": {k: m(g) for k, g in d23.groupby("use")},
          "by_size": {k: m(g) for k, g in d23.groupby("size")},
          "by_district": {str(k): m(g) for k, g in d23.groupby("district")},
          "by_overlap": {("in 2023 and 2024" if k else "2023 only"): m(g) for k, g in d23.groupby("in_both_years")}}
    # Citywide single-EUI baseline: floor-area-weighted mean EUI of all LL84 2024 buildings
    d24 = d24.dropna(subset=["i"]).copy()
    d24["i"] = d24.i.astype(int)
    gfa = d24.property_gfa_calculated
    site = sum(d24[c].clip(lower=0).fillna(0) for c in LL97)
    eui_city = float(site.sum() / gfa[gfa > 0].sum())
    base = d23.copy()
    base["model_t"] = e.bldg[base.i] * eui_city * ef
    em["baseline_single_citywide_eui"] = m(base[base.model_t > 0])
    # Held-out buildings: refit the use-group EUIs on a random half of LL84 2024 buildings
    rng = np.random.default_rng(seed)
    d24["half"] = rng.random(len(d24)) < 0.5
    d24["group"] = None
    for k, types in GROUPS.items():
        d24.loc[d24.primary_property_type.isin(types), "group"] = k
    d24["site"] = site
    fit = d24[d24.half & (gfa > 0) & d24.group.notna()]
    eui_fit = (fit.groupby("group").site.sum() / fit.groupby("group").property_gfa_calculated.sum()).to_dict()
    test = d23[~d23.bbl.isin(set(fit.bbl))].copy()
    test["model_t"] = [e.bldg[i] * eui_fit.get(USE_KEYS[e.base_use[i]] if e.base_use[i] >= 0 else "", np.nan) * ef for i in test.i]
    test = test.dropna(subset=["model_t"])
    em["held_out_buildings"] = m(test[test.model_t > 0]) | {"note": "EUIs refitted on a random half of LL84 2024 buildings; "
                                                                     "tested on 2023 buildings not in that half"}
    # Jobs: citywide floor-area baseline and errors by district (blocks)
    wac_f = RAW / "lodes8_ny_2023" / "ny_wac_S000_JT00_2023.csv.gz"
    with gzip.open(wac_f, "rt") as fh:
        wac = pd.read_csv(fh, dtype={"w_geocode": str}, usecols=["w_geocode", "C000"])
    ob = wac[wac.w_geocode.str.startswith("36061")].set_index("w_geocode").C000
    src = pyogrio.read_dataframe(PLUTO_CUR, columns=["BBL", "BCTCB2020"], read_geometry=False)
    blk = dict(zip(pd.to_numeric(src.BBL, errors="coerce"), src.BCTCB2020.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)))
    p = pd.DataFrame({"bbl": e.p.bbl.astype("int64"), "jobs": e.baseline_state["jobs"], "cd": e.cd,
                      "nonres": np.clip(e.bldg - e.p.res_area.to_numpy(), 0, None)})
    p["block"] = ["36061" + str(blk.get(int(b), "00000000000"))[1:] for b in p.bbl]
    rate = ob.sum() / p.nonres.sum()
    p["jobs_base"] = p.nonres * rate
    gb = p.groupby("block").agg(model=("jobs", "sum"), base=("jobs_base", "sum"), cd=("cd", "first"))
    gb["obs"] = ob.reindex(gb.index).fillna(0)
    big = gb[gb.obs >= 100]
    jm = lambda col, g: {"n": int(len(g)), "median_abs_pct_error": round(float(np.median(np.abs(g[col] - g.obs) / g.obs)), 3)}
    jobs = {"blocks_with_100_jobs_or_more": {"model": jm("model", big), "baseline_floor_area_only": jm("base", big)},
            "by_district_median_error": {str(k): jm("model", g)["median_abs_pct_error"] for k, g in big.groupby("cd") if len(g) >= 10}}
    return _save("existing_breakdowns", {"test": existing_breakdowns.__doc__.strip(), "emissions_2023": em, "jobs_2023": jobs,
                                         "citywide_eui_kbtu_sqft": round(eui_city, 1)})


# ----------------------------------------------------------------------------- 2b. jobs rates (R8)
def jobs_rates() -> dict:
    """Jobs per sq ft by floor-area type fitted to LODES with non-negative least squares on 2015 lots and LODES 2015
    blocks, then tested forward on 2025 lots against LODES 2023, against the CBECS-based model and a flat rate."""
    import pyogrio
    from scipy.optimize import nnls

    from planarena.indicators.engine import load_engine

    def wac(f):
        with gzip.open(f, "rt") as fh:
            w = pd.read_csv(fh, dtype={"w_geocode": str}, usecols=["w_geocode", "C000"])
        return w[w.w_geocode.str.startswith("36061")].set_index("w_geocode").C000

    def comps(e, blocks):
        o = e.obs
        X = pd.DataFrame({k: o[k] for k in ("office", "retail", "other", "factory", "storage", "pubfac", "trafac")})
        X["block"] = blocks
        return X.groupby("block").sum()

    src = pyogrio.read_dataframe(PLUTO_CUR, columns=["BBL", "BCTCB2020"], read_geometry=False)
    blk = dict(zip(pd.to_numeric(src.BBL, errors="coerce"), src.BCTCB2020.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)))
    e25, e15 = load_engine(), load_engine(P15)
    b25 = ["36061" + str(blk.get(int(b), "00000000000"))[1:] for b in e25.p.bbl]
    # 2015 lots: assign blocks by geometry to the 2025 lots that contain them
    j = _geo_match(P15, PROCESSED)
    b15 = [b25[k] if k >= 0 else "none" for k in j]
    X15, X25 = comps(e15, b15), comps(e25, b25)
    y15, y23 = wac(RAW / "lodes8_ny_2015" / "ny_wac_S000_JT00_2015.csv.gz"), wac(RAW / "lodes8_ny_2023" / "ny_wac_S000_JT00_2023.csv.gz")
    X15 = X15[X15.index != "none"]
    t15 = y15.reindex(X15.index).fillna(0).to_numpy()
    rates, _ = nnls(X15.to_numpy(), t15)
    scale = y23.sum() / y15.sum()
    pred_fit = X25.to_numpy() @ rates * scale
    flat = X25.sum(axis=1).to_numpy() * (y23.sum() / X25.to_numpy().sum())
    jb = pd.Series(e25.baseline_state["jobs"]).groupby(np.array(b25)).sum().reindex(X25.index).to_numpy()
    obs = y23.reindex(X25.index).fillna(0).to_numpy()
    big = obs >= 100
    def mm(pr):
        return {"median_abs_pct_error_blocks_ge_100": round(float(np.median(np.abs(pr[big] - obs[big]) / obs[big])), 3),
                "log_r": round(float(np.corrcoef(np.log1p(obs[big]), np.log1p(pr[big]))[0, 1]), 3),
                "total_ratio": round(float(pr.sum() / obs.sum()), 3)}
    sqft = {k: round(1 / r, 1) if r > 0 else None for k, r in zip(X15.columns, rates)}
    return _save("jobs_rates", {
        "test": jobs_rates.__doc__.strip(), "blocks_tested": int(big.sum()),
        "nnls_sqft_per_worker_fitted_2015": sqft, "growth_scale_2015_to_2023": round(float(scale), 4),
        "methods": {"CBECS composition (current model)": mm(jb), "NNLS rates fitted on 2015": mm(pred_fit),
                    "NNLS rates fitted on 2015, total calibrated": mm(pred_fit * obs.sum() / pred_fit.sum()),
                    "flat rate per sq ft of non-residential floor area": mm(flat)}})


# ----------------------------------------------------------------------------- 6b. district-scale benchmark (R7)
def optimizer_benchmark(budget: int = 10, n_cand: int = 150, sa_evals: int = 400, seeds=(0, 1, 2)) -> dict:
    """Heuristics on the same objective in every district: J = overall score - conversion penalty x changes,
    GNN fit term off, fixed budget. Reports the best plan found by each method, runtime and exact evaluations."""
    from planarena.indicators.engine import load_engine
    from planarena.optimize.suggest import SuggestConfig, Suggester

    e = load_engine()
    sug = Suggester(e, None)
    sug.per_use_values()
    pen = SuggestConfig().conversion_penalty
    rows = []
    for cd in range(101, 113):
        scope = f"cd:{cd}"
        cfg = SuggestConfig(scope=scope, w_fit=0.0, rounds=3)
        mask, eligible, allowed = sug.feasible(cfg, e.base_use)
        cfg.max_change_share = budget / max(eligible.sum(), 1)
        nev = [0]

        def J(use):
            nev[0] += 1
            return e.evaluate(use, scope=scope)["overall"] - pen * int((use != e.base_use).sum())

        # First-order gains, used to shortlist candidate moves for the exact-evaluation heuristics
        ctx = sug._state(mask, e.base_use)
        ag = sug._access_gain(e.base_use, ctx["state"], ctx["acc"], mask, sug.per_use_values())
        g = sug._gain(sug._coefficients(ctx["base"], cfg.pillar_weights, None), sug.per_use_values(), e.base_use, ag)
        g[~allowed] = -np.inf
        g[~eligible] = -np.inf
        g[np.arange(e.n), e.base_use] = -np.inf
        flat = np.argsort(-g, axis=None)[:n_cand]
        moves = [(int(i // g.shape[1]), int(i % g.shape[1])) for i in flat if np.isfinite(g.flat[i])]
        # Surrogate optimum: with one budget and one choice per parcel, the best first-order plan is the top moves
        # of distinct parcels (exact for the linear surrogate, not for the true objective).
        sur, seen = e.base_use.copy(), set()
        for p_, u_ in moves:
            if p_ not in seen and len(seen) < budget:
                sur[p_] = u_
                seen.add(p_)
        res = {"district": cd, "eligible": int(eligible.sum()), "moves_shortlisted": len(moves), "budget": budget}
        t0 = time.time(); nev[0] = 0
        res["surrogate_top_moves"] = {"J": J(sur), "s": round(time.time() - t0, 2), "evals": nev[0]}
        for name, pol in (("first_order", 0), ("first_order_plus_exact_polish", 30)):
            cfg.polish = pol
            t0 = time.time()
            r = sug.suggest(cfg)
            res[name] = {"J": J(r["use"]), "s": round(time.time() - t0, 2), "changes": int(len(r["changed"]))}
        cfg.polish = 0
        # Greedy with exact evaluation over the shortlist
        t0 = time.time(); nev[0] = 0
        cur, curJ, used = e.base_use.copy(), J(e.base_use), set()
        for _ in range(budget):
            best = None
            for p_, u_ in moves:
                if p_ in used:
                    continue
                t = cur.copy(); t[p_] = u_
                v = J(t)
                if best is None or v > best[0]:
                    best = (v, p_, u_)
            if best is None or best[0] <= curJ:
                break
            curJ, cur[best[1]] = best[0], best[2]
            used.add(best[1])
        res["greedy_exact"] = {"J": curJ, "s": round(time.time() - t0, 2), "evals": nev[0]}
        # Simulated annealing over subsets of the shortlist (exact evaluations), several seeds
        sa = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            t0 = time.time(); nev[0] = 0
            plan = {}
            curJ = J(e.base_use)
            bestJ = curJ
            T0 = 0.05
            for k in range(sa_evals):
                prop = dict(plan)
                p_, u_ = moves[rng.integers(len(moves))]
                if p_ in prop and rng.random() < 0.5:
                    prop.pop(p_)
                elif len(prop) < budget or p_ in prop:
                    prop[p_] = u_
                else:
                    prop.pop(list(prop)[rng.integers(len(prop))]); prop[p_] = u_
                t = e.base_use.copy()
                for a, b in prop.items():
                    t[a] = b
                v = J(t)
                T = T0 * (1 - k / sa_evals) + 1e-6
                if v >= curJ or rng.random() < np.exp((v - curJ) / T):
                    plan, curJ = prop, v
                    bestJ = max(bestJ, v)
            sa.append({"J": bestJ, "s": round(time.time() - t0, 2), "evals": nev[0]})
        res["simulated_annealing"] = {"J_mean": float(np.mean([x["J"] for x in sa])), "J_sd": float(np.std([x["J"] for x in sa])),
                                      "s_mean": round(float(np.mean([x["s"] for x in sa])), 2), "evals": sa_evals, "seeds": len(seeds)}
        # Random restarts: best of N random feasible plans from the shortlist
        rng = np.random.default_rng(0)
        t0 = time.time(); nev[0] = 0
        bestJ = -np.inf
        for _ in range(50):
            t = e.base_use.copy()
            pick = rng.permutation(len(moves))
            used = set()
            for idx in pick:
                p_, u_ = moves[idx]
                if p_ not in used and len(used) < budget:
                    t[p_] = u_; used.add(p_)
            bestJ = max(bestJ, J(t))
        res["random_restarts_50"] = {"J": bestJ, "s": round(time.time() - t0, 2), "evals": nev[0]}
        rows.append(res)
        print(cd, {k: (v.get("J", v.get("J_mean")) if isinstance(v, dict) else v) for k, v in res.items() if isinstance(v, dict)}, flush=True)
    methods = ["surrogate_top_moves", "first_order", "first_order_plus_exact_polish", "greedy_exact", "simulated_annealing", "random_restarts_50"]
    val = lambda r, m: r[m].get("J", r[m].get("J_mean"))
    summary = {}
    for m in methods:
        rel = [val(r, m) / max(val(r, mm) for mm in methods) if max(val(r, mm) for mm in methods) > 0 else 1.0 for r in rows]
        summary[m] = {"median_share_of_best_found": round(float(np.median(rel)), 3),
                      "districts_where_best": int(sum(abs(val(r, m) - max(val(r, mm) for mm in methods)) < 1e-9 for r in rows)),
                      "median_seconds": round(float(np.median([r[m].get("s", r[m].get("s_mean")) for r in rows])), 2)}
    return _save("optimizer_benchmark", {
        "test": f"Every district, budget {budget} changes, objective = overall score - conversion penalty x changes (GNN term "
                f"off). Exact-evaluation heuristics search the {n_cand} best first-order moves. No method guarantees a global "
                "optimum; 'share of best found' compares each method with the best plan any method found.",
        "summary": summary, "districts": rows})


# ----------------------------------------------------------------------------- report
def report() -> str:
    """Docs are generated by planarena.report from the result files."""
    from planarena import report as rp

    rp.main()
    return "docs written"


STEPS = {"jobs_rates": jobs_rates, "existing_breakdowns": existing_breakdowns, "delta_forms": delta_forms, "changes": changes, "optimizer_benchmark": optimizer_benchmark, "emissions": emissions, "jobs": jobs, "backtest": backtest, "backtest_gnn": backtest_gnn,
         "soho_noho": soho_noho, "rezonings": rezonings,
         "backtest_lgbm": backtest_lgbm, "backtest_eval": backtest_eval, "units": units,
         "east_harlem": east_harlem, "optimizer": optimizer, "report": report}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = (["emissions", "jobs_rates", "jobs", "existing_breakdowns", "units", "east_harlem", "soho_noho", "rezonings",
             "delta_forms", "changes", "backtest", "optimizer", "optimizer_benchmark", "report"]
            if step == "all" else [step])
    for s in todo:
        t0 = time.time()
        r = STEPS[s]()
        if isinstance(r, dict):
            print(s, json.dumps({k: v for k, v in r.items() if not isinstance(v, (list, dict))}, default=float)[:900])
        print(f"[{s}] {time.time() - t0:.0f}s", flush=True)
