"""Does the redevelopment model transfer between cities? Manhattan (MapPLUTO 2015-2025) and Milwaukee (MPROP 2015-2026).

* Horizon: 3 years. Test pairs: Manhattan 2022->2025, Milwaukee 2023->2026 (the latest in each city).
* Training pairs end no later than the test pair's start year in either city (outcomes known at decision time).
* Training sets: the same city, the other city only, and both pooled.
* Features: a portable set computed the same way in both cities (own use; lot, floor area, floors, age, built FAR,
  homes and assessed value per sq ft as within-city percentile ranks; land-value share; FAR relative to the 90th
  percentile of nearby lots; land-use mix and recent construction of neighbours). Zoning and permitted FAR are left
  out because the two cities regulate density differently. Same-city results with the full NYC feature set are
  given for comparison.
* Label: a new building on the lot within the horizon (year built at or after the start year), in both cities.
  Floor-area growth is not used because Milwaukee's floor area is not recorded consistently across years. Milwaukee
  pairs are matched by tax key on today's parcel polygons, so parcels split or merged since the start year are not
  scored; Manhattan pairs are matched by geometry.
* Uncertainty: 600 m tile block bootstrap of AUC and average precision.

LightGBM only.  Run:  python -I -m planarena.models.crosscity
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from planarena.config import DATA, FT_PER_M, ROOT
from planarena.models.common import N_USES, SEED, neighbour_composition

H = 3
TEST = {"manhattan": 2022, "milwaukee": 2023}
CACHE = DATA / "transition" / "crosscity"
OUT = ROOT / "results" / "validation" / "crosscity_redevelopment.json"


def _dir(city: str, year: int):
    if city == "manhattan":
        from planarena.models.history import snapshot_dir
    else:
        from planarena.cities.mke_history import snapshot_dir
    return snapshot_dir(year)


def _years(city: str) -> list[int]:
    if city == "manhattan":
        from planarena.models.history import RELEASES

        return sorted(RELEASES)
    from planarena.cities.mke_history import YEARS

    return YEARS


def labels(city: str, y0: int, y1: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"y_{city}_{y0}_{y1}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    if city == "manhattan":
        from planarena.models.transition import labels as geo_labels

        lab = geo_labels(_dir(city, y0), _dir(city, y1), y0)
    else:  # tax-key match
        p0 = pd.read_parquet(_dir(city, y0) / "parcels.parquet")
        p1 = pd.read_parquet(_dir(city, y1) / "parcels.parquet")
        pos = pd.Series(np.arange(len(p1)), index=p1.bbl.to_numpy())
        pos = pos[~pos.index.duplicated()]
        j1 = pos.reindex(p0.bbl.to_numpy()).fillna(-1).astype(int).to_numpy()
        ok = j1 >= 0
        t1 = np.where(ok, j1, 0)
        fa0, fa1 = p0.bldg_area.to_numpy(), p1.bldg_area.to_numpy()[t1]
        grew = (fa1 - fa0 > 0.25 * fa0) & (fa1 - fa0 > 2000)
        newb = (p1.year_built.to_numpy()[t1] >= y0) & ~(p0.year_built.to_numpy() >= y0)  # not already on the t0 roll
        lab = pd.DataFrame({"matched": ok, "developed": ok & (grew | newb), "dest": np.where(ok, p1.use.to_numpy()[t1], -1),
                            "new_building": ok & newb, "floor_area_growth": ok & grew, "j1": j1,
                            "use0": p0.use.to_numpy()})
    lab.to_parquet(f)
    return lab


def portable(df: pd.DataFrame, graph: dict, year0: int) -> np.ndarray:
    """Features defined the same way in every city (see module docs)."""
    from planarena.models.transition import _neigh_recent

    n = len(df)
    rank = lambda v: pd.Series(v).rank(pct=True).fillna(0.5).to_numpy()
    use = df.use.to_numpy().astype(int)
    own = np.zeros((n, N_USES + 1))
    own[np.arange(n), np.where(use >= 0, use, N_USES)] = 1
    yb = df.year_built.to_numpy().astype(float)
    known = yb > 1700
    lot = df.lot_area.to_numpy()
    tot, land = df.assess_tot.to_numpy(), df.assess_land.to_numpy()
    far = df.built_far.to_numpy()
    src, dst = graph["near"]
    ctx = pd.Series(far[dst]).groupby(src).quantile(0.9).reindex(np.arange(n)).fillna(0).to_numpy()
    ctx = np.maximum(ctx, far)
    cols = np.stack([
        rank(lot), rank(df.bldg_area.to_numpy()), rank(df.floors.to_numpy()), rank(far),
        rank(np.where(known, year0 - yb, np.nan)), (~known).astype(float),
        rank(df.units_res.to_numpy()), rank(np.divide(tot, lot, out=np.zeros(n), where=lot > 0)),
        np.clip(np.divide(land, tot, out=np.zeros(n), where=tot > 0), 0, 1),
        1 - np.clip(np.divide(far, ctx, out=np.ones(n), where=ctx > 0), 0, 1),
        df.historic.to_numpy().astype(float),
    ], 1)
    return np.concatenate([own, cols, neighbour_composition(use, np.zeros(n, bool), graph),
                           _neigh_recent(df, graph, year0)], 1).astype("float32")


BASELINES = {"Land-value share of assessed value": 18, "Built FAR below nearby lots": 19, "Building age": 14}


def X(city: str, year: int, kind: str = "portable") -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"x_{kind}_{city}_{year}.npy"
    if not f.exists():
        d = _dir(city, year)
        df = pd.read_parquet(d / "parcels.parquet")
        graph = dict(np.load(d / "graph.npz"))
        if kind == "portable":
            x = portable(df, graph, year)
        else:
            from planarena.models.transition import features

            x = features(df, graph, year)
        np.save(f, x)
    return np.load(f)


def _rows(lab):
    return (lab.matched & (lab.use0 >= 0)).to_numpy()


def _pool(city, pairs, kind="portable"):
    xs, ys = [], []
    for y0, y1 in pairs:
        lab = labels(city, y0, y1)
        r = _rows(lab)
        xs.append(X(city, y0, kind)[r])
        ys.append(lab.new_building.to_numpy()[r])
    return np.concatenate(xs), np.concatenate(ys)


def _pairs(city, end_by):
    return [(t, t + H) for t in _years(city) if t + H <= end_by and t + H in _years(city)]


def _metrics(p, y, tiles) -> dict:
    from planarena.models.rolling import _bootstrap
    from planarena.models.transition import _ap, _auc

    k = max(1, int(round(0.01 * len(p))))
    top = np.argsort(-p)[:k]
    return {"auc": round(_auc(p, y), 3), "average_precision": round(_ap(p, y), 3),
            "precision_top1pct": round(float(y[top].mean()), 3)} | _bootstrap(p, y, tiles)


def evaluate() -> dict:
    from planarena.models.rolling import _fit

    res = {"protocol": __doc__.strip().split("\n\n")[1], "horizon_years": H, "tests": {}}
    for city, t in TEST.items():
        other = next(c for c in TEST if c != city)
        lab = labels(city, t, t + H)
        r = _rows(lab)
        yt = lab.new_building.to_numpy()[r]
        d = pd.read_parquet(_dir(city, t) / "parcels.parquet", columns=["x", "y"])
        cell = 600 * FT_PER_M
        tiles = ((d.x.to_numpy() // cell) * 100000 + d.y.to_numpy() // cell)[r]
        xt = X(city, t)[r]
        out = {"test_pair": f"{t}->{t + H}", "parcels": int(r.sum()), "redeveloped": int(yt.sum()),
               "base_rate": round(float(yt.mean()), 4), "models": {}, "baselines": {}}
        for name, col in BASELINES.items():
            out["baselines"][name] = _metrics(xt[:, col], yt, tiles)
        sets = {"same city": [(city, _pairs(city, t))], "other city only": [(other, _pairs(other, t))],
                "both cities": [(city, _pairs(city, t)), (other, _pairs(other, t))]}
        for name, parts in sets.items():
            Xs, Ys = zip(*[_pool(c, pr) for c, pr in parts if pr])
            m = _fit(np.concatenate(Xs), np.concatenate(Ys), SEED)
            out["models"][name] = _metrics(m.predict_proba(xt)[:, 1], yt, tiles) | {
                "training_pairs": {c: [f"{a}->{b}" for a, b in pr] for c, pr in parts}}
        Xf, Yf = _pool(city, _pairs(city, t), "full")
        m = _fit(Xf, Yf, SEED)
        out["models"]["same city, full feature set"] = _metrics(m.predict_proba(X(city, t, "full")[r])[:, 1], yt, tiles)
        best = max(out["baselines"], key=lambda k: out["baselines"][k]["auc"])
        out["best_baseline"] = best
        res["tests"][city] = out
        print(city, {k: (v["auc"], v["average_precision"]) for k, v in out["models"].items()}, "| best baseline",
              best, out["baselines"][best]["auc"], flush=True)
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    evaluate()
