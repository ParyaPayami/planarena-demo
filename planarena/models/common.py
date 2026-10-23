"""Evaluation utilities shared by the GNN and the baselines. Uses only numpy and pandas.

The GNN (PyTorch) and the baselines (LightGBM) run in separate processes:
loading both OpenMP runtimes in one process deadlocks on macOS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from planarena.config import FT_PER_M, USE_KEYS

N_USES = len(USE_KEYS)
TILE_M, N_FOLDS, SEED = 600, 5, 7


def spatial_folds(df: pd.DataFrame, tile_m=TILE_M, k=N_FOLDS, seed=SEED) -> np.ndarray:
    tf = tile_m * FT_PER_M
    tx = ((df.x - df.x.min()) // tf).astype(int)
    ty = ((df.y - df.y.min()) // tf).astype(int)
    tiles = (tx * 10007 + ty).to_numpy()
    uniq = np.unique(tiles)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {t: i % k for t, i in zip(uniq[perm], range(len(uniq)))}
    return np.array([fold_of[t] for t in tiles])


def neighbour_composition(use: np.ndarray, hidden: np.ndarray, graph: dict) -> np.ndarray:
    """Share of each land use among a parcel's neighbours. Hidden parcels count as unknown."""
    n, feats = len(use), []
    vis = np.where(hidden, -1, use)
    for r in ("adjacent", "near"):
        src, dst = graph[r]
        u = vis[dst]
        ok = u >= 0
        c = np.zeros((n, N_USES))
        np.add.at(c, (src[ok], u[ok]), 1.0)
        t = c.sum(1, keepdims=True)
        feats.append(np.divide(c, t, out=np.zeros_like(c), where=t > 0))
    return np.concatenate(feats, 1)


def cls_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    f1s = []
    for c in range(N_USES):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn > 0:
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    top3 = (np.argsort(-p, 1)[:, :3] == y[:, None]).any(1).mean()
    ll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)))
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(np.mean(f1s)),
            "top3_accuracy": float(top3), "log_loss": float(ll)}


def reg_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {"r2": float(1 - ss_res / ss_tot), "mae_log": float(np.mean(np.abs(y - yhat)))}


def summarise(folds: list[dict]) -> dict:
    keys = [k for k in folds[0] if isinstance(folds[0][k], dict)]
    return {k: {m: {"mean": float(np.mean([f[k][m] for f in folds])),
                    "sd": float(np.std([f[k][m] for f in folds]))} for m in folds[0][k]} for k in keys}
