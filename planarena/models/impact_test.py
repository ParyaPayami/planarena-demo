"""Held-out test of the impact calculator on observed change (Manhattan, 2015 -> 2025), with coefficients that were
available before the changes.

The question is conditional: given a lot's 2015 state and the use and floor area it actually had in 2025, how well
does the engine reproduce the change in homes, assessed value, building emissions and jobs? To keep selection,
fitting and evaluation apart:

* Coefficients: energy intensities and emission factors are refitted on LL84 calendar 2015 (not 2024), and the
  relative floor area per worker on LODES 2015 (not 2023), against the 2015 twin. Nothing after 2015 enters them.
* Split: the change events are divided by spatial folds (600 m tiles) into a selection set (two of five folds) and an
  evaluation set (three folds). The delta form, and a scale factor per indicator, are chosen on the selection set
  only; every reported number is on the evaluation set.
* Baselines use the same information (2015 state, new use, new floor area) and are trained on the selection set:
  the mean change of the same transition type and size class, and a gradient-boosted regression.

Run:  python -I -m planarena.models.impact_test     -> results/validation/impact_heldout.json
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from planarena.config import DATA, ROOT, USE_KEYS

OUT = ROOT / "results" / "validation" / "impact_heldout.json"
RAW = DATA / "raw"
SIZE_BINS = (0, 10_000, 50_000, 200_000, np.inf)  # floor area after the change, sq ft
SIZE_LABEL = ("<10k sq ft", "10k-50k", "50k-200k", ">=200k")
SELECTION_FOLDS = (0, 1)


def _ll84_2015() -> pd.DataFrame:
    from planarena.calibrate import LL97, plausible_ll84

    d = plausible_ll84(pd.DataFrame(json.loads((RAW / "ll84_2015" / "ll84_manhattan_2015_bbl.json").read_text())))
    for c in LL97:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["bbl"] = pd.to_numeric(d.nyc_borough_block_and_lot.astype(str).str.split(";").str[0].str.strip(), errors="coerce")
    d["fuel"] = sum(d[c].clip(lower=0).fillna(0) for c in LL97)
    d["t"] = sum(d[c].clip(lower=0).fillna(0) * v for c, v in LL97.items())
    return d[d.fuel > 0].drop_duplicates("bbl")


def pre_period_coefficients(p0: pd.DataFrame, engine) -> dict:
    """Energy intensity and emission factor by use from LL84 2015 x the 2015 twin, and floor area per worker from
    LODES 2015 by 2020 census block (non-negative least squares, as in planarena.calibrate)."""
    import pyogrio
    from scipy.optimize import nnls

    from planarena.config import PLUTO_PATH

    d = _ll84_2015()
    p = p0[["bbl", "use", "bldg_area"]].copy()
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
        eui[u] = {"value": val, "low": min(float(per.quantile(0.25)), val), "high": max(float(per.quantile(0.75)), val),
                  "n_buildings": int(len(g))}
        ef[u] = {"value": float(g.t.sum() / g.fuel.sum()), "n_buildings": int(len(g))}
    with gzip.open(RAW / "lodes8_ny_2015" / "ny_wac_S000_JT00_2015.csv.gz", "rt") as fh:
        w = pd.read_csv(fh, dtype={"w_geocode": str}, usecols=["w_geocode", "C000"])
    w = w[w.w_geocode.str.startswith("36061")].set_index("w_geocode").C000
    src = pyogrio.read_dataframe(PLUTO_PATH, columns=["BBL", "BCTCB2020"], read_geometry=False)
    blk = dict(zip(pd.to_numeric(src.BBL, errors="coerce"),
                   src.BCTCB2020.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)))
    keys = {"office": "office", "retail": "retail", "other": "other_commercial", "factory": "factory",
            "storage": "storage", "pubfac": "public_facility", "trafac": "transport_utility"}
    X = pd.DataFrame({c: engine.obs[c] for c in keys})
    X["block"] = ["36061" + str(blk.get(int(b), "00000000000"))[1:] for b in engine.p.bbl]
    X = X.groupby("block").sum()
    rates, _ = nnls(X.to_numpy(), w.reindex(X.index).fillna(0).to_numpy())
    cbecs = {"office": 609, "retail": 1700, "other_commercial": 1378, "factory": 1429, "storage": 2400,
             "public_facility": 1060, "transport_utility": 1429}
    jobs = {k: (1.0 / float(r) if r > 1e-9 else float(cbecs[k])) for (c, k), r in zip(keys.items(), rates)}
    return {"eui": eui, "ef": ef, "jobs_sqft_per_worker": jobs, "lodes_2015_total": int(w.sum()),
            "ll84_2015_buildings_matched": int(len(m))}


def _apply(engine, coef: dict) -> None:
    for u, v in coef["eui"].items():
        engine.coef["energy_site_eui_kbtu_per_sqft"][u] = {"value": v["value"], "low": v["low"], "high": v["high"]}
    efu = engine.coef.setdefault("emissions_tco2e_per_kbtu_by_use", {})
    for u, v in coef["ef"].items():
        efu[u] = {"value": v["value"]}
    for k, v in coef["jobs_sqft_per_worker"].items():
        engine.coef["jobs_sqft_per_worker"][k] = {"value": v, "low": 0.8 * v, "high": 1.25 * v}
    engine.coef["jobs_sqft_per_worker"]["calibration_target_jobs"] = coef["lodes_2015_total"]
    engine._calibrate_jobs()
    engine.baseline_state = engine.parcel_state(engine.base_use)


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[ok], obs[ok]
    if len(o) < 10:
        return {"n": int(len(o))}
    rank = lambda v: pd.Series(v).rank().to_numpy()
    return {"n": int(len(o)), "pearson_r": float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 and o.std() > 0 else None,
            "spearman": float(np.corrcoef(rank(p), rank(o))[0, 1]) if p.std() > 0 and o.std() > 0 else None,
            "mae": float(np.mean(np.abs(p - o))), "median_ae": float(np.median(np.abs(p - o))),
            "bias": float(np.mean(p - o)), "total_ratio": float(p.sum() / o.sum()) if abs(o.sum()) > 1e-9 else None,
            "mean_abs_observed": float(np.mean(np.abs(o)))}


def _score(pred: dict, obs: dict, m: np.ndarray) -> float:
    """Selection criterion: mean over indicators of MAE / mean |observed| (lower is better)."""
    out = []
    for k in pred:
        ok = m & np.isfinite(obs[k]) & np.isfinite(pred[k])
        if ok.sum() >= 10:
            out.append(np.mean(np.abs(pred[k][ok] - obs[k][ok])) / max(np.mean(np.abs(obs[k][ok])), 1e-9))
    return float(np.mean(out))


def main() -> dict:
    from planarena.indicators.engine import load_engine
    from planarena.models.common import spatial_folds
    from planarena.models.transition import size_class
    from planarena.validate import P15, _change_events

    d = _change_events()
    ev, p0, w15, w23, jg = d["ev"].reset_index(drop=True), d["p0"], d["w15"], d["w23"], d["jg"]
    folds = spatial_folds(p0.iloc[ev.i0.map(lambda ii: ii[0])].reset_index(drop=True), k=5)
    sel = np.isin(folds, SELECTION_FOLDS)
    evl = ~sel
    obs = {"homes": ev.d_homes.to_numpy(float), "value": ev.d_value.to_numpy(float), "ghg": ev.d_ghg.to_numpy(float)}

    # engine predictions with pre-period coefficients, for each delta form
    coef = None
    pred = {}
    e = load_engine(P15)
    coef = pre_period_coefficients(p0, e)
    _apply(e, coef)
    use, far = e.base_use.copy(), np.full(e.n, np.nan)
    for r in ev.itertuples():
        ii = np.array(r.i0)
        use[ii] = r.use1
        far[ii] = r.fa1 / max(r.lot0, 1.0)
    for form in ("replacement", "difference", "hybrid"):
        for rule in ("stock", "recent"):  # unit size of a changed building: the stock's, or recent construction's
            e.delta_form, e.unit_rule = form, rule
            st, base = e.parcel_state(use, far), e.baseline_state
            pred[f"{form}, {rule} unit size"] = {k: ev.i0.map(lambda ii, k=k: float((st[k][ii] - base[k][ii]).sum())).to_numpy()
                                                 for k in ("homes", "value", "ghg", "jobs")}
    sel_scores = {f: _score({k: pred[f][k] for k in obs}, obs, sel) for f in pred}
    sel_homes = {f: _metrics(pred[f]["homes"][sel], obs["homes"][sel]) for f in pred}
    form = min(sel_scores, key=sel_scores.get)
    # one scale factor per indicator, fitted on the selection set (total observed / total predicted)
    scale = {}
    for k in obs:
        ok = sel & np.isfinite(obs[k]) & np.isfinite(pred[form][k])
        scale[k] = float(obs[k][ok].sum() / pred[form][k][ok].sum()) if abs(pred[form][k][ok].sum()) > 1e-9 else 1.0
    scaled = {k: pred[form][k] * scale[k] for k in obs}
    use_scaled = _score(scaled, obs, sel) < _score({k: pred[form][k] for k in obs}, obs, sel)

    # baselines, trained on the selection set only
    import lightgbm as lgb

    key = (ev.use0.astype(str) + ">" + ev.use1.astype(str) + ":"
           + pd.Series(size_class(ev.fa0.to_numpy())).astype(str)).to_numpy()
    units0 = ev.i0.map(lambda ii: float(p0.units_res.iloc[ii].sum())).to_numpy()
    value0 = ev.i0.map(lambda ii: float(p0.assess_tot.iloc[ii].sum())).to_numpy()
    X = np.c_[np.log1p(ev.fa0), np.log1p(ev.fa1), np.log1p(ev.lot0), ev.use0, ev.use1, units0, np.log1p(value0)]
    base = {"transition mean": {}, "gradient boosting": {}}
    for k in obs:
        ok = sel & np.isfinite(obs[k])
        means = pd.Series(obs[k][ok]).groupby(key[ok]).mean()
        overall = float(np.mean(obs[k][ok]))
        base["transition mean"][k] = np.array([means.get(q, overall) for q in key])
        mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                verbose=-1, random_state=0).fit(X[ok], obs[k][ok])
        base["gradient boosting"][k] = mdl.predict(X)

    stock = form.replace("recent", "stock")
    methods = {f"engine ({form}; pre-period coefficients)": {k: pred[form][k] for k in obs},
               f"engine ({stock}; pre-period coefficients)": {k: pred[stock][k] for k in obs},
               f"engine ({form}) x selection-set scale": scaled,
               "engine as deployed (hybrid, stock unit size; pre-period coefficients)":
                   {k: pred["hybrid, stock unit size"][k] for k in obs},
               "mean change of the same transition (selection set)": base["transition mean"],
               "gradient-boosted regression (selection set)": base["gradient boosting"],
               "no change": {k: np.zeros(len(ev)) for k in obs}}
    kind = ev.kind.to_numpy()
    size = np.digitize(ev.fa1.to_numpy(), SIZE_BINS[1:-1])
    res = {"test": __doc__.strip().split("\n\n")[0].replace("\n", " "), "events": int(len(ev)),
           "selection_events": int(sel.sum()), "evaluation_events": int(evl.sum()),
           "pre_period_coefficients": coef, "selection_scores": sel_scores, "selection_homes": sel_homes,
           "chosen_form": form,
           "selection_scale": scale, "scaled_preferred_on_selection": bool(use_scaled), "methods": {}}
    for name, pr in methods.items():
        out = {"all": {k: _metrics(pr[k][evl], obs[k][evl]) for k in obs}}
        for kd in ("use change", "density only (new building)"):
            m = evl & (kind == kd)
            out[kd] = {k: _metrics(pr[k][m], obs[k][m]) for k in obs}
        out["by_size"] = {SIZE_LABEL[b]: {k: _metrics(pr[k][evl & (size == b)], obs[k][evl & (size == b)]) for k in obs}
                          for b in range(len(SIZE_LABEL))}
        res["methods"][name] = out
    # transition types with at least 30 evaluation events (engine, chosen form)
    tt = (ev.use0.map(lambda u: USE_KEYS[u]) + " > " + ev.use1.map(lambda u: USE_KEYS[u])).to_numpy()
    res["by_transition"] = {}
    for t in pd.Series(tt[evl]).value_counts().index:
        m = evl & (tt == t)
        if m.sum() >= 30:
            res["by_transition"][t] = {k: _metrics(pred[form][k][m], obs[k][m]) for k in obs}
    # jobs: by 2020 census block, evaluation blocks only (blocks with no selection-set event)
    blocks_sel = set(ev.block[sel])
    eb = ev.block.to_numpy()
    jm = np.array([b not in blocks_sel for b in eb]) & evl
    bj = pd.Series(pred[form]["jobs"][jm]).groupby(eb[jm]).sum()
    obs_j = (w23.reindex(bj.index, fill_value=0) - jg * w15.reindex(bj.index, fill_value=0)).to_numpy()
    res["jobs_by_block"] = _metrics(bj.to_numpy(), obs_j)
    res["points_homes"] = {"predicted": [round(float(v), 2) for v in pred[form]["homes"][evl]],
                           "observed": [round(float(v), 2) for v in obs["homes"][evl]]}
    for k in ("value", "ghg"):
        ok = evl & np.isfinite(obs[k])
        res[f"points_{k}"] = {"predicted": [round(float(v), 2) for v in pred[form][k][ok]],
                              "observed": [round(float(v), 2) for v in obs[k][ok]]}
    OUT.write_text(json.dumps(res, indent=1))
    for name, r in res["methods"].items():
        a = r["all"]
        print(f"{name:55s} homes r {a['homes'].get('pearson_r') or 0:.3f} total {a['homes'].get('total_ratio') or 0:.2f} "
              f"MAE {a['homes']['mae']:.1f} | value r {a['value'].get('pearson_r') or 0:.3f} | ghg r {a['ghg'].get('pearson_r') or 0:.3f}")
    return res


if __name__ == "__main__":
    main()
