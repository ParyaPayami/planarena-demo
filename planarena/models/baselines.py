"""Non-graph baselines on the GNN's spatial folds. Runs in its own process: no torch import.

Run with:  python -m planarena.models.baselines [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from planarena.config import PROCESSED, ROOT
from planarena.models.common import (
    N_FOLDS, N_USES, SEED, cls_metrics, neighbour_composition, reg_metrics, spatial_folds, summarise,
)
from planarena.models.features import context_features, value_target, zone_family


def run(folds: int) -> list[dict]:
    df = pd.read_parquet(PROCESSED / "parcels.parquet")
    graph = dict(np.load(PROCESSED / "graph.npz"))
    x = context_features(df)
    use = df.use.to_numpy().astype("int64")
    yv, vok = value_target(df)
    fold = spatial_folds(df, k=folds)
    fam = df.zone.map(zone_family).to_numpy()
    cd = df.cd.to_numpy()
    out = []
    for f in range(folds):
        t0 = time.time()
        test, tr = fold == f, fold != f
        lab = test & (use >= 0)
        trl = tr & (use >= 0)
        y = use[lab]
        res = {"fold": f, "n_test": int(lab.sum())}
        prior = pd.crosstab(fam[trl], use[trl]).reindex(columns=range(N_USES), fill_value=0) + 1
        prior = prior.div(prior.sum(1), axis=0)
        glob = (np.bincount(use[trl], minlength=N_USES) + 1.0)
        glob /= glob.sum()
        pz = np.stack([prior.loc[z].to_numpy() if z in prior.index else glob for z in fam[lab]])
        res["suit_zoning_majority"] = cls_metrics(y, pz)
        feats = np.concatenate([x, neighbour_composition(use, test, graph)], 1)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
        clf.fit(feats[trl], use[trl])
        pl = np.zeros((lab.sum(), N_USES))
        pl[:, clf.classes_] = clf.predict_proba(feats[lab])
        res["suit_lightgbm"] = cls_metrics(y, pl)
        vl, vt = test & vok, tr & vok
        med = pd.Series(yv[vt]).groupby([use[vt], cd[vt]]).median()
        med_u = pd.Series(yv[vt]).groupby(use[vt]).median()
        g = float(np.median(yv[vt]))
        vm = np.array([med.get((u, c), med_u.get(u, g)) for u, c in zip(use[vl], cd[vl])])
        res["value_median_use_cd"] = reg_metrics(yv[vl], vm)
        vfe = np.concatenate([feats, np.eye(N_USES)[np.clip(use, 0, None)]], 1)
        reg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
        reg.fit(vfe[vt], yv[vt])
        res["value_lightgbm"] = reg_metrics(yv[vl], reg.predict(vfe[vl]))
        res["seconds"] = round(time.time() - t0, 1)
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}
                         | {k: round(v["accuracy"] if "accuracy" in v else v["r2"], 3)
                            for k, v in res.items() if isinstance(v, dict)}), flush=True)
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()
    folds = run(a.folds)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "baselines_cv.json").write_text(
        json.dumps({"folds": folds, "summary": summarise(folds)}, indent=2))


if __name__ == "__main__":
    main()
