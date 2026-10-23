"""Train and evaluate the parcel GNN with spatial-block cross-validation.

Protocol:
* 600 m tiles are assigned to 5 folds. The whole test fold is hidden: its land
  uses and values are masked from the model during training AND testing. This
  matches "re-plan an area": predict the uses of a whole district from its
  surroundings.
* Baselines run in a separate process (planarena.models.baselines) on identical folds.
* Final model: trained on all parcels. Suitability is inferred with chunked
  masking, so each parcel is predicted from 90% visible context.

Run with:  python -m planarena.models.train [--epochs 250] [--folds 5] [--skip-cv]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from planarena.config import PROCESSED, ROOT
from planarena.models.common import N_FOLDS, SEED, cls_metrics, reg_metrics, spatial_folds, summarise
from planarena.models.gnn import N_USES, ParcelGNN, context_features, edge_dict, value_target

RESULTS = ROOT / "results"


def train_gnn(x, use, yv, vok, ed, train_mask, hidden, epochs, lr=3e-3, mask_rate=0.2, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ParcelGNN(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X, Ut = torch.tensor(x), torch.tensor(use, dtype=torch.long)
    YV, VOK, H = torch.tensor(yv), torch.tensor(vok), torch.tensor(hidden)
    pool = np.nonzero(train_mask & (use >= 0) & ~hidden)[0]
    for _ in range(epochs):
        model.train()
        m = np.zeros(len(use), bool)
        m[pool[rng.random(len(pool)) < mask_rate]] = True
        h = model.embed(X, Ut, torch.tensor(m) | H, ed)
        tgt = torch.tensor(np.nonzero(m)[0])
        loss = F.cross_entropy(model.suitability_logits(h[tgt]), Ut[tgt])
        vt = tgt[VOK[tgt]]
        if len(vt):
            loss = loss + 0.5 * F.mse_loss(model.value(h[vt], Ut[vt]), YV[vt])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def gnn_predict(model, x, use, hidden, ed):
    model.eval()
    X, Ut = torch.tensor(x), torch.tensor(use, dtype=torch.long)
    h = model.embed(X, Ut, torch.tensor(hidden), ed)
    return F.softmax(model.suitability_logits(h), 1).numpy(), model.value(h, Ut).numpy(), h.numpy()


def run_cv(df, graph, x, epochs, folds):
    use = df.use.to_numpy().astype("int64")
    yv, vok = value_target(df)
    ed = edge_dict(graph)
    fold = spatial_folds(df, k=folds)
    out = []
    for f in range(folds):
        t0 = time.time()
        test = fold == f
        lab, vl = test & (use >= 0), test & vok
        model = train_gnn(x, use, yv, vok, ed, ~test, test, epochs)
        pg, vg, _ = gnn_predict(model, x, use, test, ed)
        res = {"fold": f, "n_test": int(lab.sum()),
               "suit_gnn": cls_metrics(use[lab], pg[lab]), "value_gnn": reg_metrics(yv[vl], vg[vl]),
               "seconds": round(time.time() - t0, 1)}
        print(json.dumps({"fold": f, "acc": round(res["suit_gnn"]["accuracy"], 3),
                          "f1": round(res["suit_gnn"]["macro_f1"], 3),
                          "value_r2": round(res["value_gnn"]["r2"], 3), "s": res["seconds"]}), flush=True)
        out.append(res)
    return out


def train_final(df, graph, x, epochs, chunks=10):
    use = df.use.to_numpy().astype("int64")
    yv, vok = value_target(df)
    ed = edge_dict(graph)
    model = train_gnn(x, use, yv, vok, ed, np.ones(len(df), bool), np.zeros(len(df), bool), epochs)
    chunk = np.random.default_rng(SEED).integers(0, chunks, len(df))
    suit = np.zeros((len(df), N_USES), "float32")
    ctx = np.zeros((len(df), model.suit_head.in_features), "float32")
    for c in range(chunks):
        hid = chunk == c
        p, _, h = gnn_predict(model, x, use, hid, ed)
        suit[hid], ctx[hid] = p[hid], h[hid]
    return model, suit, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--skip-cv", action="store_true")
    a = ap.parse_args()
    df = pd.read_parquet(PROCESSED / "parcels.parquet")
    graph = dict(np.load(PROCESSED / "graph.npz"))
    meta = json.loads((PROCESSED / "twin_meta.json").read_text())
    x = context_features(df)
    RESULTS.mkdir(exist_ok=True)
    if not a.skip_cv:
        print("baselines (separate process)...", flush=True)
        subprocess.run([sys.executable, "-I", "-m", "planarena.models.baselines", "--folds", str(a.folds)],
                       check=True, cwd=ROOT)
        base = json.loads((RESULTS / "baselines_cv.json").read_text())
        print("GNN folds...", flush=True)
        gnn = run_cv(df, graph, x, a.epochs, a.folds)
        merged = [b | {k: v for k, v in g.items() if k.endswith("_gnn")} for b, g in zip(base["folds"], gnn)]
        report = {"data": meta["source"], "n_parcels": len(df), "epochs": a.epochs, "tile_m": 600,
                  "folds": a.folds, "seed": SEED, "features": int(x.shape[1]), "relations": list(graph),
                  "protocol": "spatial-block CV, 600 m tiles; test fold fully masked (uses and values) "
                              "during training and testing",
                  "cv_folds": merged, "cv_summary": summarise(merged)}
        (RESULTS / "gnn_cv.json").write_text(json.dumps(report, indent=2))
    t0 = time.time()
    model, suit, ctx = train_final(df, graph, x, a.epochs)
    torch.save({"state_dict": model.state_dict(), "in_dim": int(x.shape[1])}, PROCESSED / "gnn.pt")
    np.save(PROCESSED / "suitability.npy", suit)
    np.save(PROCESSED / "gnn_context.npy", ctx)
    print(f"final model trained in {time.time() - t0:.0f}s; artifacts saved", flush=True)


if __name__ == "__main__":
    main()
