"""Ablation of the realism layers: do recommended plans look more like what actually gets built?

The optimizer is run on the 2020 Manhattan snapshot in every community district (balanced weights, 2% change
budget, exact re-scoring of the top 30 moves), under three settings:
* no realism layer: conversions without precedent are allowed;
* precedent filter (the default): conversions of existing buildings never observed before 2020 at that size are
  skipped (precedent table as of 2020);
* precedent filter + likelihood layer: candidates restricted to the fifth of parcels most likely to be redeveloped
  within five years (redevelopment model trained on 2015->2020 only).
Recommended parcels are then compared with what happened by 2025 (redevelopment label; 2025 land use). This tests
plausibility, not desirability: a plan can be good without resembling the market.

The settings recommend different numbers of parcels, so they are also compared at equal counts: in each district
every setting is rerun with a budget equal to the smallest count any setting recommended there, and all plans are
scored with one objective (the balanced overall score of the district minus the conversion penalty per change).
The share of each setting's eligible parcels that were redeveloped is the rate a random choice would reach.

Run:  python -I -m planarena.models.ablate_recs
"""
from __future__ import annotations

import json

import numpy as np

from planarena.config import ROOT
from planarena.models.common import SEED

OUT = ROOT / "results" / "validation" / "realism_ablation.json"
T0, T1 = 2020, 2025


def main() -> dict:
    import pandas as pd

    from planarena.indicators.engine import load_engine
    from planarena.models import rolling
    from planarena.models.history import snapshot_dir
    from planarena.models.transition import precedent
    from planarena.optimize.suggest import SuggestConfig, Suggester

    e = load_engine(snapshot_dir(T0))
    e.set_precedent(precedent(as_of=T0, write=False))
    sug = Suggester(e)
    lab = rolling._y(T0, T1)
    developed = lab.developed.to_numpy()
    p25 = pd.read_parquet(snapshot_dir(T1) / "parcels.parquet", columns=["use"]).use.to_numpy()
    use25 = np.where(lab.j1.to_numpy() >= 0, p25[np.clip(lab.j1.to_numpy(), 0, None)], -1)
    X, y = rolling._pool([(2015, 2020)])
    model = rolling._fit(X, y, SEED)
    p = model.predict_proba(rolling._x(T0))[:, 1]
    likely = np.nonzero(p >= np.quantile(p, 0.8))[0]

    settings = {"no realism layer": {"require_precedent": False},
                "precedent filter (default)": {"require_precedent": True},
                "precedent filter + likelihood layer": {"require_precedent": True, "candidate_parcels": likely}}
    res = {"test": "\n\n".join(__doc__.strip().split("\n\n")[1:3]), "snapshot": T0, "outcome_year": T1,
           "settings": {}, "equal_counts": {}}
    pen = SuggestConfig().conversion_penalty

    def run(cd, kw, budget=None):
        cfg = SuggestConfig(scope=f"cd:{cd}", max_change_share=0.02, polish=30, **kw)
        if budget == 0:
            _, eligible, _ = sug.feasible(cfg, e.base_use)
            return np.zeros(0, int), np.zeros(0, "int16"), 0.0, np.nonzero(eligible)[0]
        if budget is not None:
            _, eligible, _ = sug.feasible(cfg, e.base_use)
            cfg.max_change_share = (budget + 0.5) / max(int(eligible.sum()), 1)
        out = sug.suggest(cfg)
        ch = np.asarray(out["changed"], dtype=int)
        _, eligible, _ = sug.feasible(cfg, e.base_use)
        return ch, out["use"][ch], out["evaluation"]["overall"] - pen * len(ch), np.nonzero(eligible)[0]

    def metrics(rec, new_use, J, elig):
        n = len(rec)
        return {
            "recommended_parcels": int(n),
            "share_without_precedent": round(float(1 - e.has_precedent(rec, new_use).mean()), 3) if n else None,
            "share_redeveloped_by_2025": round(float(developed[rec].mean()), 4) if n else None,
            "share_of_eligible_redeveloped": round(float(developed[elig].mean()), 4) if len(elig) else None,
            "share_with_recommended_use_in_2025": round(float((use25[rec] == new_use).mean()), 4) if n else None,
            "share_redeveloped_into_recommended_use": round(float((developed[rec] & (use25[rec] == new_use)).mean()), 4) if n else None,
            "median_district_objective": round(float(np.median(J)), 3),
            "sum_district_objective": round(float(np.sum(J)), 3),
        }

    first = {}
    for name, kw in settings.items():
        runs = [run(cd, kw) for cd in range(101, 113)]
        first[name] = runs
        res["settings"][name] = metrics(np.concatenate([r[0] for r in runs]), np.concatenate([r[1] for r in runs]),
                                        [r[2] for r in runs], np.unique(np.concatenate([r[3] for r in runs])))
        print(name, res["settings"][name], flush=True)
    k = {cd: min(len(first[n][j][0]) for n in settings) for j, cd in enumerate(range(101, 113))}
    res["equal_counts"]["budget_by_district"] = {str(cd): int(v) for cd, v in k.items()}
    for name, kw in settings.items():
        runs = [run(cd, kw, budget=k[cd]) for cd in range(101, 113)]
        res["equal_counts"][name] = metrics(np.concatenate([r[0] for r in runs]), np.concatenate([r[1] for r in runs]),
                                            [r[2] for r in runs], np.unique(np.concatenate([r[3] for r in runs])))
        print("equal counts", name, res["equal_counts"][name], flush=True)
    rows = (lab.matched & (lab.use0 >= 0)).to_numpy()
    res["reference"] = {"share_of_all_lots_redeveloped": round(float(developed[rows].mean()), 4),
                        "share_of_likely_fifth_redeveloped": round(float(developed[likely].mean()), 4)}
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
