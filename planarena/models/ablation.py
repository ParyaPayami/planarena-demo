"""Does the graph model add anything? Ablation on the same spatial folds as the main cross-validation (R3, R5).

* GNN transductive (as reported): test parcels stay in the graph with their use and value masked.
* GNN inductive: training uses a graph without any edge touching a test parcel; at test time the full graph is
  used with test parcels masked, so test parcels are never seen during training.
* LightGBM: context features + neighbour land-use composition (test parcels hidden), as in the baselines.
* LightGBM + GNN embedding: the same, plus the transductive fold model's embedding of every parcel computed while
  that parcel is hidden (ten chunks), so no parcel's own use leaks into its features.

PyTorch and LightGBM run in separate processes (OpenMP on macOS).
The winning model is deployed as the pattern-consistency layer (deploy step, data/processed/pattern.npy).

Run:  python -I -m planarena.models.ablation all
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from planarena.config import DATA, PROCESSED, ROOT
from planarena.models.common import N_FOLDS, SEED, cls_metrics, neighbour_composition, spatial_folds, summarise
from planarena.models.features import context_features

WORK = DATA / "ablation"
OUT = ROOT / "results" / "validation" / "ablation_gnn.json"


def _data():
    df = pd.read_parquet(PROCESSED / "parcels.parquet")
    graph = dict(np.load(PROCESSED / "graph.npz"))
    return df, graph, context_features(df), df.use.to_numpy().astype("int64"), spatial_folds(df, k=N_FOLDS)


def gnn(epochs: int = 250, chunks: int = 10) -> None:
    import torch

    from planarena.models.features import value_target
    from planarena.models.gnn import RELATIONS, edge_dict
    from planarena.models.train import gnn_predict, train_gnn

    WORK.mkdir(parents=True, exist_ok=True)
    df, graph, x, use, fold = _data()
    yv, vok = value_target(df)
    ed = edge_dict(graph)
    res = []
    for f in range(N_FOLDS):
        t0 = time.time()
        test = fold == f
        lab = test & (use >= 0)
        m_tr = train_gnn(x, use, yv, vok, ed, ~test, test, epochs)
        pt, _, _ = gnn_predict(m_tr, x, use, test, ed)
        # inductive: no edge touching a test parcel during training
        tidx = np.nonzero(test)[0]
        ed_ind = {}
        for k in RELATIONS:
            v = graph[k]
            keep = ~(np.isin(v[0], tidx) | np.isin(v[1], tidx))
            ed_ind[("parcel", k, "parcel")] = torch.as_tensor(v[:, keep], dtype=torch.long)
        m_in = train_gnn(x, use, yv, vok, ed_ind, ~test, test, epochs)
        pi, _, _ = gnn_predict(m_in, x, use, test, ed)
        # embeddings of every parcel while hidden (from the transductive fold model)
        chunk = np.random.default_rng(SEED).integers(0, chunks, len(df))
        emb = None
        for c in range(chunks):
            hid = test | (chunk == c)
            _, _, h = gnn_predict(m_tr, x, use, hid, ed)
            if emb is None:
                emb = np.zeros((len(df), h.shape[1]), "float32")
            sel = (chunk == c) & ~test
            emb[sel] = h[sel]
        _, _, h = gnn_predict(m_tr, x, use, test, ed)
        emb[test] = h[test]
        np.save(WORK / f"emb_fold{f}.npy", emb.astype("float16"))
        res.append({"fold": f, "n_test": int(lab.sum()), "gnn_transductive": cls_metrics(use[lab], pt[lab]),
                    "gnn_inductive": cls_metrics(use[lab], pi[lab]), "seconds": round(time.time() - t0, 1)})
        print(f, round(res[-1]["gnn_transductive"]["accuracy"], 3), round(res[-1]["gnn_inductive"]["accuracy"], 3), flush=True)
    (WORK / "gnn_folds.json").write_text(json.dumps(res))


def lgbm() -> None:
    import lightgbm as lgb

    df, graph, x, use, fold = _data()
    res = []
    for f in range(N_FOLDS):
        test = fold == f
        lab, tr = test & (use >= 0), ~test & (use >= 0)
        feats = np.concatenate([x, neighbour_composition(use, test, graph)], 1)
        emb = np.load(WORK / f"emb_fold{f}.npy").astype("float32")
        row = {"fold": f}
        for name, F_ in (("lightgbm", feats), ("lightgbm_plus_gnn_embedding", np.concatenate([feats, emb], 1))):
            clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1, random_state=SEED)
            clf.fit(F_[tr], use[tr])
            p = np.zeros((lab.sum(), int(use.max()) + 1))
            p[:, clf.classes_] = clf.predict_proba(F_[lab])
            row[name] = cls_metrics(use[lab], p)
        res.append(row)
        print(f, {k: round(v["accuracy"], 3) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    (WORK / "lgbm_folds.json").write_text(json.dumps(res))


def deploy(chunks: int = 10) -> None:
    """Pattern-consistency layer from the best ablation model: LightGBM on context, neighbour mix and the final GNN's
    embedding (gnn_context.npy, each parcel embedded while hidden). Cross-fitted: each tenth of the parcels is
    predicted by a model trained without it, with its uses hidden from the neighbour features."""
    import lightgbm as lgb

    df, graph, x, use, _ = _data()
    ctx = np.load(PROCESSED / "gnn_context.npy").astype("float32")
    chunk = np.random.default_rng(SEED).integers(0, chunks, len(df))
    out = np.zeros((len(df), int(use.max()) + 1), "float32")
    for c in range(chunks):
        hid = chunk == c
        F_ = np.concatenate([x, neighbour_composition(use, hid, graph), ctx], 1)
        tr = ~hid & (use >= 0)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, verbose=-1, random_state=SEED)
        clf.fit(F_[tr], use[tr])
        out[np.ix_(hid, clf.classes_)] = clf.predict_proba(F_[hid])
    np.save(PROCESSED / "pattern.npy", out)
    (PROCESSED / "pattern_meta.json").write_text(json.dumps({
        "model": "LightGBM + GNN embedding", "chosen_by": "results/validation/ablation_gnn.json",
        "cross_fitted_chunks": chunks, "parcels": int(len(df))}, indent=2))
    print("pattern layer written", out.shape, flush=True)


def merge() -> dict:
    g = json.loads((WORK / "gnn_folds.json").read_text())
    l_ = json.loads((WORK / "lgbm_folds.json").read_text())
    folds = [a | {k: v for k, v in b.items() if k != "fold"} for a, b in zip(g, l_)]
    out = {"test": __doc__.strip().split("\n\n")[1], "folds": folds, "summary": summarise(folds)}
    acc = {k: [f[k]["accuracy"] for f in folds] for k in ("gnn_transductive", "gnn_inductive", "lightgbm", "lightgbm_plus_gnn_embedding")}
    out["embedding_improves_lightgbm_in_folds"] = int(sum(a > b for a, b in zip(acc["lightgbm_plus_gnn_embedding"], acc["lightgbm"])))
    OUT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "all":
        py = [sys.executable, "-I", "-m", "planarena.models.ablation"]
        subprocess.run(py + ["gnn"], check=True, cwd=ROOT)
        subprocess.run(py + ["lgbm"], check=True, cwd=ROOT)
        print(json.dumps(merge()["summary"], indent=1)[:2000])
        subprocess.run(py + ["deploy"], check=True, cwd=ROOT)
    else:
        {"gnn": gnn, "lgbm": lgbm, "deploy": deploy, "merge": lambda: print(merge()["summary"])}[step]()
