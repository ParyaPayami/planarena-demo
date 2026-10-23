"""Rolling-origin evaluation, calibration and label audit of the redevelopment-transition model (R4, R5).

Uses the annual Manhattan snapshots built by planarena.models.history (MapPLUTO 2015-2025).

* For a horizon h and a decision year t, the model is trained only on pairs (t0 -> t0 + h) whose outcome is
  known by t (t0 + h <= t), and tested on (t -> t + h). Features come from the snapshot of the pair's start year.
* The same test pair is also scored with 5-fold spatial cross-validation inside that period, which is what the
  earlier within-period model selection measured. The gap between the two is the optimism of non-temporal tests.
* Calibration: reliability bins and ECE of the raw probabilities, and after isotonic calibration fitted on the most
  recent training pair predicted out of sample (a model trained on the pairs before it).
* Uncertainty: 600 m tile block bootstrap of AUC and average precision; three LightGBM seeds.
* Label audit: redevelopment labels against DOB certificates of occupancy (new building, alteration type 1).

LightGBM only (no torch in this process).  Run:  python -I -m planarena.models.rolling
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from planarena.config import DATA, ROOT
from planarena.models.common import SEED, spatial_folds
from planarena.models.history import RELEASES, snapshot_dir
from planarena.models.transition import _ap, _auc, _ece, features, labels

CACHE = DATA / "transition" / "rolling"
OUT = ROOT / "results" / "validation" / "rolling_origin.json"
YEARS = sorted(RELEASES)
# Approximate release dates of the snapshots (for matching DOB certificates to a period).
RELEASE_DATE = {y: date(y, 7, 1) for y in YEARS} | {2015: date(2015, 3, 1), 2020: date(2020, 12, 1), 2025: date(2025, 9, 1)}


def _x(year: int) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"x_{year}.npy"
    if not f.exists():
        d = snapshot_dir(year)
        df = pd.read_parquet(d / "parcels.parquet")
        np.save(f, features(df, dict(np.load(d / "graph.npz")), year))
    return np.load(f)


def _y_plain(y0: int, y1: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"y_{y0}_{y1}.parquet"
    if not f.exists():
        labels(snapshot_dir(y0), snapshot_dir(y1), y0).to_parquet(f)
    return pd.read_parquet(f)


_y = _y_plain  # label function used by the evaluation; swapped for the verified variant in verified_labels_check


def _rows(lab: pd.DataFrame) -> np.ndarray:
    return (lab.matched & (lab.use0 >= 0)).to_numpy()


def _fit(X, y, seed):
    import lightgbm as lgb

    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed)
    m.fit(X, y)
    return m


def _pool(pairs):
    xs, ys = [], []
    for y0, y1 in pairs:
        lab = _y(y0, y1)
        r = _rows(lab)
        xs.append(_x(y0)[r])
        ys.append(lab.developed.to_numpy()[r])
    return np.concatenate(xs), np.concatenate(ys)


def _bootstrap(p, y, tiles, n=200, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(tiles)
    idx_by = {t: np.nonzero(tiles == t)[0] for t in uniq}
    aucs, aps = [], []
    for _ in range(n):
        pick = np.concatenate([idx_by[t] for t in rng.choice(uniq, len(uniq), replace=True)])
        if y[pick].sum() == 0:
            continue
        aucs.append(_auc(p[pick], y[pick]))
        aps.append(_ap(p[pick], y[pick]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"auc_95ci": q(aucs), "ap_95ci": q(aps)}


def _reliability(p, y, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [{"mean_predicted": round(float(p[idx == b].mean()), 4), "observed_rate": round(float(y[idx == b].mean()), 4),
             "n": int((idx == b).sum())} for b in range(len(edges) - 1) if (idx == b).any()]


def evaluate_origin(h: int, t: int, seeds=(SEED, SEED + 1, SEED + 2)) -> dict:
    from sklearn.isotonic import IsotonicRegression

    train_pairs = [(a, a + h) for a in YEARS if a + h <= t]
    test_lab = _y(t, t + h)
    r = _rows(test_lab)
    Xte, yte = _x(t)[r], test_lab.developed.to_numpy()[r]
    df = pd.read_parquet(snapshot_dir(t) / "parcels.parquet")
    tiles = spatial_folds(df, k=10_000)[r]  # unique-ish 600 m tile ids for the block bootstrap
    Xtr, ytr = _pool(train_pairs)
    probs = []
    for s in seeds:
        probs.append(_fit(Xtr, ytr, s).predict_proba(Xte)[:, 1])
    p = probs[0]
    per_seed = [{"auc": round(_auc(q, yte), 3), "ap": round(_ap(q, yte), 3)} for q in probs]
    out = {"horizon": h, "decision_year": t, "test": f"{t}->{t + h}", "train_pairs": [f"{a}->{b}" for a, b in train_pairs],
           "n_test": int(len(yte)), "positives": int(yte.sum()), "base_rate": round(float(yte.mean()), 4),
           "lightgbm": {"auc": round(_auc(p, yte), 3), "average_precision": round(_ap(p, yte), 3),
                        "auc_sd_over_seeds": round(float(np.std([x["auc"] for x in per_seed])), 4),
                        "ap_sd_over_seeds": round(float(np.std([x["ap"] for x in per_seed])), 4),
                        **_bootstrap(p, yte, tiles), "ece_raw": round(_ece(p, yte), 4),
                        "mean_predicted": round(float(p.mean()), 4), "reliability_raw": _reliability(p, yte)}}
    # Isotonic calibration fitted on the latest training pair, predicted by a model trained on the pairs before it.
    if len(train_pairs) >= 2:
        early, late = train_pairs[:-1], train_pairs[-1]
        Xe, ye = _pool(early)
        Xl, yl = _pool([late])
        iso = IsotonicRegression(out_of_bounds="clip").fit(_fit(Xe, ye, SEED).predict_proba(Xl)[:, 1], yl)
        pc = iso.predict(p)
        out["lightgbm"] |= {"ece_isotonic": round(_ece(pc, yte), 4), "mean_predicted_isotonic": round(float(pc.mean()), 4),
                            "calibrated_on": f"{late[0]}->{late[1]}", "reliability_isotonic": _reliability(pc, yte)}
    # Baselines (no training needed)
    mx, built = df.max_far.to_numpy()[r], df.built_far.to_numpy()[r]
    under = 1 - np.clip(np.divide(built, mx, out=np.ones(len(mx)), where=mx > 0), 0, 1)
    tot, land = df.assess_tot.to_numpy()[r], df.assess_land.to_numpy()[r]
    lshare = np.clip(np.divide(land, tot, out=np.zeros(len(tot)), where=tot > 0), 0, 1)
    out["baselines"] = {"underbuilt share of zoning capacity": {"auc": round(_auc(under, yte), 3), "average_precision": round(_ap(under, yte), 3)},
                        "land-value share": {"auc": round(_auc(lshare, yte), 3), "average_precision": round(_ap(lshare, yte), 3)}}
    # Optimism of a spatial split inside the test period itself
    folds = spatial_folds(df, k=5)[r]
    ps = np.zeros(len(yte))
    for f in range(5):
        tr, te = folds != f, folds == f
        ps[te] = _fit(Xte[tr], yte[tr], SEED).predict_proba(Xte[te])[:, 1]
    out["spatial_cv_within_test_period"] = {"auc": round(_auc(ps, yte), 3), "average_precision": round(_ap(ps, yte), 3)}
    return out


def _certificates(y0: int, y1: int) -> tuple[set, set]:
    """BBLs with a DOB new-building / alteration-type-1 certificate of occupancy issued in the window (y0, y1]."""
    raw = DATA / "raw" / "dob_co"
    a = pd.DataFrame(json.loads((raw / "dob_co_legacy_mn.json").read_text()))
    a["date"] = pd.to_datetime(a.c_o_issue_date, errors="coerce")
    a["kind"] = a.job_type.map({"NB": "new building", "A1": "alteration"})
    b = pd.DataFrame(json.loads((raw / "dob_co_now_mn.json").read_text()))
    b["date"] = pd.to_datetime(b.c_of_o_issuance_date, format="%m/%d/%y %I:%M:%S %p", errors="coerce")
    b["kind"] = np.where(b.job_type.str.upper().str.contains("NEW BUILDING"), "new building",
                         np.where(b.job_type.str.upper().str.contains("ALTERATION TYPE 1"), "alteration", None))
    co = pd.concat([a[["bbl", "date", "kind"]], b[["bbl", "date", "kind"]]]).dropna()
    co["bbl"] = pd.to_numeric(co.bbl, errors="coerce")
    lo, hi = pd.Timestamp(RELEASE_DATE[y0]), pd.Timestamp(RELEASE_DATE[y1])
    co = co[(co.date > lo) & (co.date <= hi)].dropna(subset=["bbl"])
    return set(co[co.kind == "new building"].bbl.astype("int64")), set(co[co.kind == "alteration"].bbl.astype("int64"))


def _evidence(y0: int, y1: int, lab: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nb, alt = _certificates(y0, y1)
    bbl0 = pd.read_parquet(snapshot_dir(y0) / "parcels.parquet").bbl.to_numpy().astype("int64")
    bbl1 = pd.read_parquet(snapshot_dir(y1) / "parcels.parquet").bbl.to_numpy().astype("int64")
    j1 = lab.j1.to_numpy()
    on = lambda s_: np.array([(bbl0[i] in s_) or (j1[i] >= 0 and bbl1[j1[i]] in s_) for i in range(len(bbl0))])
    ev_nb = on(nb)
    return ev_nb, ev_nb | on(alt)


def _y_verified(y0: int, y1: int) -> pd.DataFrame:
    """Redeveloped = a new building completed in the period (year built), or floor-area growth confirmed by a DOB
    new-building / alteration-type-1 certificate in the window. Removes unconfirmed floor-area changes."""
    f = CACHE / f"yv_{y0}_{y1}.parquet"
    if not f.exists():
        lab = _y_plain(y0, y1).copy()
        _, ev_any = _evidence(y0, y1, lab)
        lab["developed"] = lab.new_building.to_numpy() | (lab.floor_area_growth.to_numpy() & ev_any)
        lab.to_parquet(f)
    return pd.read_parquet(f)


def verified_labels_check(cases=((5, 2020), (3, 2022), (1, 2024))) -> list[dict]:
    """Retrain and test with verified labels for selected horizons (robustness of the label definition)."""
    global _y
    plain = _y
    out = []
    try:
        _y = _y_verified
        for h, t in cases:
            o = evaluate_origin(h, t, seeds=(SEED,))
            out.append({"horizon": h, "test": o["test"], "positives": o["positives"], "base_rate": o["base_rate"],
                        "auc": o["lightgbm"]["auc"], "average_precision": o["lightgbm"]["average_precision"],
                        "best_baseline_auc": max(v["auc"] for v in o["baselines"].values())})
    finally:
        _y = plain
    return out


def label_audit(y0: int, y1: int) -> dict:
    """Agreement of PLUTO-derived redevelopment labels with DOB certificates of occupancy in the same window."""
    raw = DATA / "raw" / "dob_co"
    a = pd.DataFrame(json.loads((raw / "dob_co_legacy_mn.json").read_text()))
    a["date"] = pd.to_datetime(a.c_o_issue_date, errors="coerce")
    a["kind"] = a.job_type.map({"NB": "new building", "A1": "alteration"})
    b = pd.DataFrame(json.loads((raw / "dob_co_now_mn.json").read_text()))
    b["date"] = pd.to_datetime(b.c_of_o_issuance_date, format="%m/%d/%y %I:%M:%S %p", errors="coerce")
    b["kind"] = np.where(b.job_type.str.upper().str.contains("NEW BUILDING"), "new building",
                         np.where(b.job_type.str.upper().str.contains("ALTERATION TYPE 1"), "alteration", None))
    co = pd.concat([a[["bbl", "date", "kind"]], b[["bbl", "date", "kind"]]]).dropna()
    co["bbl"] = pd.to_numeric(co.bbl, errors="coerce")
    lo, hi = pd.Timestamp(RELEASE_DATE[y0]), pd.Timestamp(RELEASE_DATE[y1])
    co = co[(co.date > lo) & (co.date <= hi)]
    nb, alt = set(co[co.kind == "new building"].bbl.dropna().astype("int64")), set(co[co.kind == "alteration"].bbl.dropna().astype("int64"))
    lab = _y(y0, y1)
    p0 = pd.read_parquet(snapshot_dir(y0) / "parcels.parquet")
    p1 = pd.read_parquet(snapshot_dir(y1) / "parcels.parquet")
    import geopandas as gpd
    import shapely

    g0 = gpd.read_parquet(snapshot_dir(y0) / "parcels.geoparquet").geometry
    g1 = gpd.read_parquet(snapshot_dir(y1) / "parcels.geoparquet").geometry
    aa, bb = shapely.STRtree(g1.values).query(g0.representative_point().values, predicate="within")
    j1 = np.full(len(g0), -1)
    j1[aa] = bb
    bbl0, bbl1 = p0.bbl.to_numpy().astype("int64"), p1.bbl.to_numpy().astype("int64")
    r = _rows(lab)
    ev_nb = np.array([(bbl0[i] in nb) or (j1[i] >= 0 and bbl1[j1[i]] in nb) for i in range(len(p0))])
    ev_any = ev_nb | np.array([(bbl0[i] in alt) or (j1[i] >= 0 and bbl1[j1[i]] in alt) for i in range(len(p0))])
    dev = lab.developed.to_numpy()
    m = r
    return {"window": f"{RELEASE_DATE[y0]} to {RELEASE_DATE[y1]}", "labelled_developed": int((dev & m).sum()),
            "lots_with_new_building_certificate": int((ev_nb & m).sum()),
            "precision_vs_new_building_or_alteration_certificate": round(float(ev_any[dev & m].mean()), 3),
            "precision_vs_new_building_certificate": round(float(ev_nb[dev & m].mean()), 3),
            "recall_of_new_building_certificates": round(float(dev[ev_nb & m].mean()), 3),
            "note": "Certificates are matched by BBL (the lot or its 2015/2020 predecessor); condominium unit BBLs may be "
                    "missed, so precision is a lower bound."}


def main() -> dict:
    res = {"protocol": __doc__.strip().split("\n\n")[1], "origins": [], "label_audit": {}}
    for h in (1, 3, 5):
        for t in YEARS:
            if t + h <= max(YEARS) and any(a + h <= t for a in YEARS):
                o = evaluate_origin(h, t)
                res["origins"].append(o)
                print(h, t, o["lightgbm"]["auc"], o["lightgbm"]["average_precision"], o["spatial_cv_within_test_period"], flush=True)
    for y0, y1 in ((2015, 2020), (2020, 2025)):
        res["label_audit"][f"{y0}->{y1}"] = label_audit(y0, y1)
    res["verified_labels"] = verified_labels_check()
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()


def verified_only() -> None:
    """Add the verified-label check to an existing rolling_origin.json without recomputing the rest."""
    res = json.loads(OUT.read_text())
    res["verified_labels"] = verified_labels_check()
    OUT.write_text(json.dumps(res, indent=2))
    print(res["verified_labels"])
