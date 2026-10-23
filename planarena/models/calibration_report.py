"""Calibration of the Manhattan redevelopment model's probabilities, not only its ranking.

For every rolling origin of planarena.models.rolling (horizons of one, three and five years; decision years
2016-2024), the model is refitted exactly as there (training pairs known by the decision year, first seed) and its
test-period probabilities are checked with planarena.models.integration.calibration: reliability in ten equal-count
bins, expected calibration error over those bins, observed over predicted events, calibration intercept and slope,
and the Brier score against a constant forecast of the training-period prevalence. The same is reported after the
temporal isotonic recalibration defined in rolling.py (fitted on the latest training pair, predicted by a model
trained on the pairs before it), and after logistic recalibration fitted on out-of-fold probabilities of the training
pairs (five spatial folds); neither uses test-period information.

The seven cross-city panels are scored the same way: the temporal model of planarena.models.integration (fitted on
the first window, existing conditions only and all available blocks), raw and after logistic recalibration on
out-of-fold training-window probabilities.

Run:  python -I -m planarena.models.calibration_report [manhattan|cities]
  -> results/validation/calibration_manhattan.json, results/validation/calibration_cities.json
"""
from __future__ import annotations

import json

import numpy as np

from planarena.config import ROOT
from planarena.models.common import SEED
from planarena.models.integration import calibration, recalibrator

OUT = ROOT / "results" / "validation" / "calibration_manhattan.json"
OUT_CITIES = ROOT / "results" / "validation" / "calibration_cities.json"


def _oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, fit) -> np.ndarray:
    p = np.zeros(len(y))
    for f in np.unique(folds):
        tr = folds != f
        p[~tr] = fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
    return p


def _folds(pairs) -> np.ndarray:
    import pandas as pd

    from planarena.models import rolling
    from planarena.models.common import spatial_folds
    from planarena.models.history import snapshot_dir

    out = []
    for a, b in pairs:
        r = rolling._rows(rolling._y(a, b))
        out.append(spatial_folds(pd.read_parquet(snapshot_dir(a) / "parcels.parquet", columns=["x", "y"]))[r])
    return np.concatenate(out)


def manhattan() -> dict:
    from sklearn.isotonic import IsotonicRegression

    from planarena.models import rolling

    res = {"test": __doc__.strip().split("\n\n")[1].replace("\n", " "), "origins": []}
    for h in (1, 3, 5):
        for t in rolling.YEARS:
            if not (t + h <= max(rolling.YEARS) and any(a + h <= t for a in rolling.YEARS)):
                continue
            pairs = [(a, a + h) for a in rolling.YEARS if a + h <= t]
            lab = rolling._y(t, t + h)
            r = rolling._rows(lab)
            Xte, yte = rolling._x(t)[r], lab.developed.to_numpy()[r].astype(bool)
            Xtr, ytr = rolling._pool(pairs)
            p = rolling._fit(Xtr, ytr, SEED).predict_proba(Xte)[:, 1]
            row = {"horizon": h, "decision_year": t, "test": f"{t}->{t + h}", "n_test": int(len(yte)),
                   "raw": calibration(p, yte, float(np.mean(ytr)))}
            if len(pairs) >= 2:
                Xe, ye = rolling._pool(pairs[:-1])
                Xl, yl = rolling._pool([pairs[-1]])
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(
                    rolling._fit(Xe, ye, SEED).predict_proba(Xl)[:, 1], yl)
                row["isotonic"] = calibration(iso.predict(p), yte, float(np.mean(ytr)))
            oof = _oof(Xtr, ytr, _folds(pairs), lambda X, y: rolling._fit(X, y, SEED))
            row["recalibrated"] = calibration(recalibrator(oof, ytr)(p), yte, float(np.mean(ytr)))
            res["origins"].append(row)
            c = row["raw"]
            print(f"h={h} t={t}: O/E {c['observed_over_predicted']:.2f} slope {c['calibration_slope']:.2f} "
                  f"BSS {c['brier_skill']:.3f}" + (f" | iso O/E {row['isotonic']['observed_over_predicted']:.2f} "
                                                   f"BSS {row['isotonic']['brier_skill']:.3f}" if "isotonic" in row else ""),
                  flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    return res


def cities() -> dict:
    from planarena.models import integration as I
    from planarena.models.common import spatial_folds

    res = {"test": __doc__.strip().split("\n\n")[2].replace("\n", " "), "cities": {}}
    for g, (_, windows) in I.PANELS.items():
        tr, te = I.panel(g, *windows[0]), I.panel(g, *windows[1])
        btr, bte = I.blocks(tr), I.blocks(te)
        ytr, yte = tr.developed.to_numpy().astype(bool), te.developed.to_numpy().astype(bool)
        has = I._available(I.panel(g, *windows[-1]), I.blocks(I.panel(g, *windows[-1])))
        full = max((st for st in I.SETS if all(has[x] for x in st.split("+"))), key=len)
        folds = spatial_folds(tr, tile_m=I.TILE_M)
        out = {"train": list(windows[0]), "test": list(windows[1])}
        for st in ("E", full):
            parts = st.split("+")
            Xtr, Xte = np.concatenate([btr[x] for x in parts], 1), np.concatenate([bte[x] for x in parts], 1)
            p = I._lgbm().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            oof = _oof(Xtr, ytr, folds, lambda X, y: I._lgbm().fit(X, y))
            out[st] = {"raw": calibration(p, yte, float(ytr.mean())),
                       "recalibrated": calibration(recalibrator(oof, ytr)(p), yte, float(ytr.mean()))}
            print(g, st, {k: round(v["observed_over_predicted"], 2) for k, v in out[st].items()},
                  {k: round(v["brier_skill"], 3) for k, v in out[st].items()}, flush=True)
        res["cities"][g] = out
    OUT_CITIES.write_text(json.dumps(res, indent=1))
    return res


def main(which=("manhattan", "cities")) -> None:
    for w in which:
        {"manhattan": manhattan, "cities": cities}[w]()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or ("manhattan", "cities"))
