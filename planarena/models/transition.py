"""Redevelopment-transition model: how likely is a parcel to be redeveloped within five years, and into what?

Protocol (temporal holdout, no leakage):
    train  features from MapPLUTO 15v1 (2015), outcomes observed in 20v8 (end of 2020)
    test   features from 20v8 (2020),          outcomes observed in 25v3.1 (2025)
    deploy trained on both periods, applied to 25v3.1 -> data/processed/redevelopment.npz (2025-2030)

Outcome per parcel at t0, matched by geometry so lot mergers are followed:
    developed   the t1 lot containing the parcel has a building completed in the period (year built >= t0)
                or its floor area grew by more than 25% and 2,000 sq ft over the t0 lots inside it
    destination land use of that t1 lot (for developed parcels)

Models (PyTorch and LightGBM run in separate processes): a relational GraphSAGE with a development head
and a destination head; LightGBM; their average; baselines (underbuilt share of zoning capacity,
land-value share, zoning-family prior, same-use).

Run:  python -I -m planarena.models.transition all
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from planarena.config import DATA, PROCESSED, ROOT, USE_KEYS
from planarena.models.common import N_USES, SEED, neighbour_composition
from planarena.models.features import context_features, zone_family

PERIODS = {"train": (DATA / "processed_15v1", DATA / "processed_20v8", 2015),
           "test": (DATA / "processed_20v8", PROCESSED, 2020)}
WORK = DATA / "transition"
OUT = ROOT / "results" / "validation" / "transition_2020_2025.json"


# ----------------------------------------------------------------------------- data
def _neigh_recent(df: pd.DataFrame, graph: dict, year0: int) -> np.ndarray:
    recent = (df.year_built.to_numpy() >= year0 - 5).astype(float)
    out = []
    for r in ("adjacent", "near"):
        src, dst = graph[r]
        s = np.bincount(src, weights=recent[dst], minlength=len(df))
        c = np.bincount(src, minlength=len(df))
        out.append(np.divide(s, c, out=np.zeros(len(df)), where=c > 0))
    return np.stack(out, 1)


def features(df: pd.DataFrame, graph: dict, year0: int) -> np.ndarray:
    use = df.use.to_numpy().astype(int)
    own = np.zeros((len(df), N_USES + 1))
    own[np.arange(len(df)), np.where(use >= 0, use, N_USES)] = 1
    mx = df.max_far.to_numpy()
    built = df.built_far.to_numpy()
    under = 1 - np.clip(np.divide(built, mx, out=np.ones(len(df)), where=mx > 0), 0, 1)
    land = df.assess_land.to_numpy()
    tot = df.assess_tot.to_numpy()
    land_share = np.clip(np.divide(land, tot, out=np.zeros(len(df)), where=tot > 0), 0, 1)
    lv_psf = np.log1p(np.divide(tot, df.lot_area.to_numpy(), out=np.zeros(len(df)), where=df.lot_area.to_numpy() > 0)) / 8
    soft = ((built <= 0.5 * mx) & (df.lot_area.to_numpy() >= 2500) & ~df.landmark.to_numpy() & ~df.historic.to_numpy())
    extra = np.stack([under, mx / 15, land_share, lv_psf, soft.astype(float),
                      np.log1p(df.units_res.to_numpy()) / 6], 1)
    hidden = np.zeros(len(df), bool)
    return np.concatenate([context_features(df), own, extra, neighbour_composition(use, hidden, graph),
                           _neigh_recent(df, graph, year0)], 1).astype("float32")


def labels(d0: Path, d1: Path, year0: int) -> pd.DataFrame:
    """Developed / destination for every t0 parcel, matched to the t1 lot containing its representative point."""
    import geopandas as gpd
    import shapely

    p0 = pd.read_parquet(d0 / "parcels.parquet")
    p1 = pd.read_parquet(d1 / "parcels.parquet")
    g0 = gpd.read_parquet(d0 / "parcels.geoparquet").geometry
    g1 = gpd.read_parquet(d1 / "parcels.geoparquet").geometry
    pts = g0.representative_point().values
    tree = shapely.STRtree(g1.values)
    a, b = tree.query(pts, predicate="within")  # a: t0 index, b: t1 index
    j1 = np.full(len(p0), -1)
    j1[a] = b
    ok = j1 >= 0
    t1 = np.where(ok, j1, 0)
    fa0_in_t1 = pd.Series(p0.bldg_area.to_numpy()[ok]).groupby(j1[ok]).sum()
    fa0 = np.zeros(len(p1))
    fa0[fa0_in_t1.index.to_numpy()] = fa0_in_t1.to_numpy()
    fa1 = p1.bldg_area.to_numpy()
    grew = (fa1 - fa0 > 0.25 * fa0) & (fa1 - fa0 > 2000)
    # A building already in the t0 snapshot with a year built of t0 or later (late releases, e.g. 20v8 in December
    # 2020) is not new in (t0, t1].
    yb0_in_t1 = pd.Series(p0.year_built.to_numpy()[ok]).groupby(j1[ok]).max()
    yb0 = np.zeros(len(p1))
    yb0[yb0_in_t1.index.to_numpy()] = yb0_in_t1.to_numpy()
    newb = (p1.year_built.to_numpy() >= year0) & ~(yb0 >= year0)
    dev1 = (grew | newb)
    lab = pd.DataFrame({"matched": ok, "developed": ok & dev1[t1], "dest": np.where(ok, p1.use.to_numpy()[t1], -1),
                        "new_building": ok & newb[t1], "floor_area_growth": ok & grew[t1], "j1": j1})
    lab["use0"] = p0.use.to_numpy()
    return lab


def period(name: str):
    d0, d1, year0 = PERIODS[name]
    df = pd.read_parquet(d0 / "parcels.parquet")
    graph = dict(np.load(d0 / "graph.npz"))
    return df, graph, features(df, graph, year0), labels(d0, d1, year0)


def prepare() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    for name in PERIODS:
        df, graph, x, lab = period(name)
        np.save(WORK / f"x_{name}.npy", x)
        lab.to_parquet(WORK / f"y_{name}.parquet")
        print(name, x.shape, "matched", int(lab.matched.sum()), "developed", int(lab.developed.sum()), flush=True)
    # deployment features: 2025
    df = pd.read_parquet(PROCESSED / "parcels.parquet")
    np.save(WORK / "x_deploy.npy", features(df, dict(np.load(PROCESSED / "graph.npz")), 2025))


def _rows(lab: pd.DataFrame) -> np.ndarray:
    return (lab.matched & (lab.use0 >= 0)).to_numpy()


# ----------------------------------------------------------------------------- LightGBM (own process)
def fit_lgbm() -> None:
    import lightgbm as lgb

    def fit(xs, ys, devs):
        clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
        clf.fit(xs, ys)
        dst = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                                 verbose=-1, random_state=SEED)
        dst.fit(*devs)
        return clf, dst

    def predict(models, x):
        clf, dst = models
        p = clf.predict_proba(x)[:, 1]
        q = np.zeros((len(x), N_USES))
        q[:, dst.classes_] = dst.predict_proba(x)
        return p, q

    xtr, ytr = np.load(WORK / "x_train.npy"), pd.read_parquet(WORK / "y_train.parquet")
    xte, yte = np.load(WORK / "x_test.npy"), pd.read_parquet(WORK / "y_test.parquet")
    r = _rows(ytr)
    dv = r & ytr.developed.to_numpy() & (ytr.dest.to_numpy() >= 0)
    m = fit(xtr[r], ytr.developed.to_numpy()[r], (xtr[dv], ytr.dest.to_numpy()[dv]))
    p, q = predict(m, xte)
    np.savez(WORK / "pred_lgbm.npz", p=p, q=q)
    # deployment: both periods pooled
    r2 = _rows(yte)
    dv2 = r2 & yte.developed.to_numpy() & (yte.dest.to_numpy() >= 0)
    X = np.concatenate([xtr[r], xte[r2]])
    Y = np.concatenate([ytr.developed.to_numpy()[r], yte.developed.to_numpy()[r2]])
    XD = np.concatenate([xtr[dv], xte[dv2]])
    YD = np.concatenate([ytr.dest.to_numpy()[dv], yte.dest.to_numpy()[dv2]])
    p, q = predict(fit(X, Y, (XD, YD)), np.load(WORK / "x_deploy.npy"))
    np.savez(WORK / "deploy_lgbm.npz", p=p, q=q)
    print("lgbm done", flush=True)


# ----------------------------------------------------------------------------- GNN (own process)
def fit_gnn(epochs: int = 300) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv

    from planarena.models.gnn import RELATIONS, edge_dict

    torch.manual_seed(SEED)

    class TransitionGNN(nn.Module):
        def __init__(self, d, h=96):
            super().__init__()
            self.inp = nn.Linear(d, h)
            self.convs = nn.ModuleList([HeteroConv({("parcel", r, "parcel"): SAGEConv(h, h) for r in RELATIONS},
                                                   aggr="sum") for _ in range(2)])
            self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(2)])
            self.dev = nn.Linear(h, 1)
            self.dst = nn.Linear(h, N_USES)

        def forward(self, x, ed):
            h = F.relu(self.inp(x))
            for conv, norm in zip(self.convs, self.norms):
                h = norm(h + F.relu(conv({"parcel": h}, ed)["parcel"]))
                h = F.dropout(h, 0.2, self.training)
            return self.dev(h).squeeze(1), self.dst(h)

    def train(sets, epochs=epochs, hidden=96, wd=1e-4):
        torch.manual_seed(SEED)
        model = TransitionGNN(sets[0][0].shape[1], hidden)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=wd)
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            loss = 0.0
            for x, ed, y, rows, dest in sets:
                lo, ld = model(x, ed)
                pos = y[rows].float()
                pw = (1 - pos.mean()) / pos.mean().clamp(min=1e-3)
                loss = loss + F.binary_cross_entropy_with_logits(lo[rows], pos, pos_weight=pw.sqrt())
                dv = rows & y & (dest >= 0)
                loss = loss + 0.5 * F.cross_entropy(ld[dv], dest[dv])
            loss.backward()
            opt.step()
        model.eval()
        return model

    def tensors(name, dir0):
        x = torch.tensor(np.load(WORK / f"x_{name}.npy"))
        ed = edge_dict(dict(np.load(dir0 / "graph.npz")))
        lab = pd.read_parquet(WORK / f"y_{name}.parquet")
        return (x, ed, torch.tensor(lab.developed.to_numpy()), torch.tensor(_rows(lab)),
                torch.tensor(lab.dest.to_numpy(), dtype=torch.long))

    t0 = time.time()
    tr = tensors("train", PERIODS["train"][0])
    te = tensors("test", PERIODS["test"][0])
    # Standardize with training statistics (both periods for deployment).
    mu, sd = tr[0].mean(0), tr[0].std(0).clamp(min=1e-6)
    norm = lambda s: (((s[0] - mu) / sd),) + s[1:]
    # Model selection inside the training period only: hold out 20% of 600 m tiles, pick by average precision.
    from planarena.models.common import spatial_folds

    df15 = pd.read_parquet(PERIODS["train"][0] / "parcels.parquet")
    val = torch.tensor(spatial_folds(df15, k=5) == 0)
    x, ed, y, rows, dest = norm(tr)
    grid = [(e, h, w) for e in (60, 150, 300) for h in (32, 96) for w in (1e-4, 1e-2)]
    scores = {}
    for e_, h_, w_ in grid:
        m = train([(x, ed, y, rows & ~val, dest)], epochs=e_, hidden=h_, wd=w_)
        with torch.no_grad():
            pv = torch.sigmoid(m(x, ed)[0]).numpy()
        vr = (rows & val).numpy()
        scores[(e_, h_, w_)] = _ap(pv[vr], y.numpy()[vr])
        print("config", e_, h_, w_, "val AP", round(scores[(e_, h_, w_)], 4), flush=True)
    best_e, best_h, best_w = max(scores, key=scores.get)
    (WORK / "gnn_selection.json").write_text(json.dumps(
        {"grid": [{"epochs": k[0], "hidden": k[1], "weight_decay": k[2], "val_ap": round(v, 4)} for k, v in scores.items()],
         "chosen": {"epochs": best_e, "hidden": best_h, "weight_decay": best_w}}, indent=1))
    train_best = lambda sets: train(sets, epochs=best_e, hidden=best_h, wd=best_w)
    model = train_best([norm(tr)])
    with torch.no_grad():
        lo, ld = model(norm(te)[0], te[1])
    np.savez(WORK / "pred_gnn.npz", p=torch.sigmoid(lo).numpy(), q=F.softmax(ld, 1).numpy())
    model = train_best([norm(tr), norm(te)])
    xd = (torch.tensor(np.load(WORK / "x_deploy.npy")) - mu) / sd
    with torch.no_grad():
        lo, ld = model(xd, edge_dict(dict(np.load(PROCESSED / "graph.npz"))))
    np.savez(WORK / "deploy_gnn.npz", p=torch.sigmoid(lo).numpy(), q=F.softmax(ld, 1).numpy())
    print(f"gnn done in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- evaluation
def _auc(s, y):
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ap(s, y):
    o = np.argsort(-s)
    hits = y[o]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / max(hits.sum(), 1))


def _ece(p, y, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def evaluate() -> dict:
    yte = pd.read_parquet(WORK / "y_test.parquet")
    ytr = pd.read_parquet(WORK / "y_train.parquet")
    xte = np.load(WORK / "x_test.npy")
    rows = _rows(yte)
    y = yte.developed.to_numpy()[rows]
    df = pd.read_parquet(PERIODS["test"][0] / "parcels.parquet")
    use0 = df.use.to_numpy()[rows].astype(int)
    pl, pg = np.load(WORK / "pred_lgbm.npz"), np.load(WORK / "pred_gnn.npz")
    scores = {"GNN": pg["p"][rows], "LightGBM": pl["p"][rows], "GNN + LightGBM (mean)": (pg["p"][rows] + pl["p"][rows]) / 2}
    mx, built = df.max_far.to_numpy()[rows], df.built_far.to_numpy()[rows]
    scores["Underbuilt share of zoning capacity"] = 1 - np.clip(np.divide(built, mx, out=np.ones(len(mx)), where=mx > 0), 0, 1)
    tot, land = df.assess_tot.to_numpy()[rows], df.assess_land.to_numpy()[rows]
    scores["Land-value share of assessed value"] = np.clip(np.divide(land, tot, out=np.zeros(len(tot)), where=tot > 0), 0, 1)
    base = y.mean()
    dev = {}
    for k, s in scores.items():
        o = np.argsort(-s)
        r = {"auc": round(_auc(s, y), 3), "average_precision": round(_ap(s, y), 3)}
        for frac in (0.01, 0.05):
            n = max(1, int(frac * len(s)))
            r[f"precision_top{int(frac * 100)}pct"] = round(float(y[o[:n]].mean()), 3)
            r[f"recall_top{int(frac * 100)}pct"] = round(float(y[o[:n]].sum() / y.sum()), 3)
        if k in ("GNN", "LightGBM", "GNN + LightGBM (mean)"):
            r["brier"] = round(float(np.mean((s - y) ** 2)), 4)
            r["ece"] = round(_ece(s, y), 4)
            r["mean_predicted"] = round(float(s.mean()), 4)
        dev[k] = r
    # destination among developed parcels in the test period
    dmask = y & (yte.dest.to_numpy()[rows] >= 0)
    dest = yte.dest.to_numpy()[rows][dmask]
    fam = df.zone.map(zone_family).to_numpy()[rows][dmask]
    tr_rows = _rows(ytr) & ytr.developed.to_numpy() & (ytr.dest.to_numpy() >= 0)
    d15 = pd.read_parquet(PERIODS["train"][0] / "parcels.parquet")
    prior = pd.crosstab(d15.zone.map(zone_family).to_numpy()[tr_rows], ytr.dest.to_numpy()[tr_rows]).reindex(
        columns=range(N_USES), fill_value=0) + 1
    prior = prior.div(prior.sum(axis=1), axis=0)
    glob = np.bincount(ytr.dest.to_numpy()[tr_rows], minlength=N_USES) + 1.0
    Pz = np.stack([prior.loc[f].to_numpy() if f in prior.index else glob / glob.sum() for f in fam])
    # Persistence baseline as a full ranking, built without test outcomes: the same use first, then the
    # destinations observed in the training period for that use, then a seeded random tie-break.
    T = np.zeros((N_USES, N_USES)) + 1e-3
    np.add.at(T, (ytr.use0.to_numpy()[tr_rows], ytr.dest.to_numpy()[tr_rows]), 1.0)
    T = T / T.sum(axis=1, keepdims=True)
    u0d = np.clip(use0[dmask], 0, None)
    Psame = 10.0 * np.eye(N_USES)[u0d] + T[u0d] + np.random.default_rng(SEED).uniform(0, 1e-9, (len(u0d), N_USES))
    cands = {"GNN": pg["q"][rows][dmask], "LightGBM": pl["q"][rows][dmask],
             "GNN + LightGBM (mean)": (pg["q"][rows][dmask] + pl["q"][rows][dmask]) / 2,
             "Zoning-family prior (training period)": Pz,
             "Same use, then training-period transitions": Psame}
    dst = {}
    chg = dest != use0[dmask]
    for k, Q in cands.items():
        rank = (Q > Q[np.arange(len(dest)), dest][:, None]).sum(1)
        dst[k] = {"top1": round(float((rank == 0).mean()), 3), "top3": round(float((rank < 3).mean()), 3),
                  "top1_when_use_changed": round(float((rank[chg] == 0).mean()), 3) if chg.any() else None}
    res = {
        "test": "Temporal holdout. Trained on 2015 features with outcomes to end-2020 (MapPLUTO 15v1 -> 20v8); "
                "tested on 2020 features with outcomes to 2025 (20v8 -> 25v3.1). Developed = new building completed "
                "in the period, or floor area +25% and +2,000 sq ft, on the land of the parcel (geometry match).",
        "train_parcels": int(_rows(ytr).sum()), "train_developed": int((ytr.developed & _rows(ytr)).sum()),
        "test_parcels": int(rows.sum()), "test_developed": int(y.sum()), "test_base_rate": round(float(base), 4),
        "development": dev, "destination": dst, "destination_n": int(dmask.sum()),
        "destination_n_use_changed": int(chg.sum()),
        "features": int(xte.shape[1]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    return res


def deploy() -> dict:
    """Write the 2025 -> 2030 layer: likelihood from the model with the best test average precision,
    destination from the model with the best test top-1 accuracy."""
    res = json.loads(OUT.read_text())
    g, l = np.load(WORK / "deploy_gnn.npz"), np.load(WORK / "deploy_lgbm.npz")
    pick = {"GNN": (g["p"], g["q"]), "LightGBM": (l["p"], l["q"]),
            "GNN + LightGBM (mean)": ((g["p"] + l["p"]) / 2, (g["q"] + l["q"]) / 2)}
    best_p = max(pick, key=lambda k: res["development"][k]["average_precision"])
    best_q = max(pick, key=lambda k: res["destination"][k]["top1"])
    p, q = pick[best_p][0], pick[best_q][1]
    np.savez_compressed(PROCESSED / "redevelopment.npz", p=p.astype("float32"), q=q.astype("float32"))
    meta = {"likelihood_model": best_p, "destination_model": best_q, "horizon_years": 5,
            "trained_on": "2015->2020 and 2020->2025 (MapPLUTO 15v1, 20v8, 25v3.1)",
            "test_auc": res["development"][best_p]["auc"],
            "test_average_precision": res["development"][best_p]["average_precision"],
            "test_precision_top1pct": res["development"][best_p]["precision_top1pct"],
            "test_base_rate": res["test_base_rate"],
            "test_destination_top1": res["destination"][best_q]["top1"],
            "test_destination_top3": res["destination"][best_q]["top3"],
            "mean_probability_2025": round(float(p.mean()), 4)}
    (PROCESSED / "redevelopment_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


SIZE_BINS = [0, 20_000, 100_000, 500_000]  # existing building floor area class edges (sq ft)
SIZE_LABELS = ["none", "<20k", "20k-100k", "100k-500k", ">500k"]


def size_class(bldg_area: np.ndarray) -> np.ndarray:
    return np.where(bldg_area <= 0, 0, 1 + np.searchsorted(SIZE_BINS[1:], bldg_area, side="right"))


def precedent(as_of: int = 2025, write: bool = True) -> dict:
    """Observed land-use changes in Manhattan between consecutive annual snapshots that ended by ``as_of``, by
    existing building size class. Date-aware, so an evaluation at decision year t can use precedent(as_of=t) and never
    sees later conversions. Counts include some PLUTO reclassifications, so the test is permissive: it flags only
    conversions never seen. Requires the annual snapshots (python -I -m planarena.models.history).
    """
    from planarena.models.history import RELEASES, snapshot_dir
    from planarena.models.rolling import _y

    years = sorted(y for y in RELEASES if y <= as_of)
    pairs = [(a, b) for a, b in zip(years, years[1:])]
    counts = np.zeros((N_USES, N_USES, len(SIZE_LABELS)), int)
    for y0, y1 in pairs:
        y = _y(y0, y1)
        p = pd.read_parquet(snapshot_dir(y0) / "parcels.parquet")
        m = (y.matched & (y.use0 >= 0) & (y.dest >= 0) & (y.use0 != y.dest)).to_numpy()
        np.add.at(counts, (y.use0.to_numpy()[m], y.dest.to_numpy()[m], size_class(p.bldg_area.to_numpy()[m])), 1)
    out = {"source": "Consecutive annual MapPLUTO snapshots, Manhattan, lots matched by geometry; any recorded use change",
           "as_of": as_of, "pairs": [f"{a}->{b}" for a, b in pairs], "uses": list(USE_KEYS), "size_labels": SIZE_LABELS,
           "size_bins_sqft": SIZE_BINS, "counts": counts.tolist(), "total_changes": int(counts.sum())}
    if write:
        (PROCESSED / "conversion_precedent.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    py = [sys.executable, "-I", "-m", "planarena.models.transition"]
    if step == "all":
        for s in ("prepare", "lgbm", "gnn"):
            subprocess.run(py + [s], check=True, cwd=ROOT)
        print(json.dumps(evaluate(), indent=1)[:3000])
        print(deploy())
        print({k: v for k, v in precedent().items() if k != "counts"})
    else:
        {"prepare": prepare, "lgbm": fit_lgbm, "gnn": fit_gnn, "precedent": lambda: print({k: v for k, v in precedent().items() if k != "counts"}),
         "eval": lambda: print(json.dumps(evaluate(), indent=1)), "deploy": lambda: print(deploy())}[step]()
