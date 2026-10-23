"""Does the NSGA-III explorer find the social / economic / environmental trade-off front?

Study areas are Manhattan tax blocks, chosen at random, with the explorer's own options per parcel (zoning,
protections, precedent). Every plan of a block (every combination of options) is enumerated, so the true front is
known:
* small blocks (64-2,000 plans): every plan is also scored by the full impact engine, so the exact front is known,
  and the explorer's front (re-scored exactly, as in the Studio) is compared with it; this measures search and
  first-order (surrogate) error together;
* medium blocks (20,000-1,000,000 plans): the front of the first-order pillar gains the explorer searches is known;
  this measures the search alone. The exhaustive surrogate front is also re-scored exactly (up to 300 plans) and
  compared with the explorer's re-scored front.
Quality is the hypervolume of a front as a share of the hypervolume of the true front (pillar values normalised to
the range of the plans compared; the worst value of each pillar is the reference point), and the share of
true-front plans the explorer returns. A random search with the same number of evaluations as NSGA-III (population
92 x 120 generations) is the baseline. The explorer keeps every plan it evaluates and returns the non-dominated ones.

Run:  python -I -m planarena.optimize.pareto_test     -> results/validation/pareto_test.json
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import spearmanr

from planarena.config import PILLARS, ROOT, USE_INDEX

OUT = ROOT / "results" / "validation" / "pareto_test.json"
SMALL, MEDIUM = (64, 2_000), (20_000, 1_000_000)
POP, GEN = 92, 120


def nondominated(F: np.ndarray) -> np.ndarray:
    from planarena.optimize.pareto import _nondominated

    return _nondominated(F)


def hv_share(front: np.ndarray, true: np.ndarray, everything: np.ndarray) -> float:
    lo, hi = everything.min(0), everything.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    ind = HV(ref_point=np.ones(3))
    h = lambda V: float(ind(1.0 - (V - lo) / span)) if len(V) else 0.0
    t = h(true)
    return round(h(front) / t, 4) if t > 0 else None


def main(n_small: int = 10, n_medium: int = 10, seed: int = 0) -> dict:
    from planarena.indicators.engine import load_engine
    from planarena.optimize.pareto import options_for, pareto, pillar_gain_matrices
    from planarena.optimize.suggest import SuggestConfig, Suggester

    e = load_engine()
    sug = Suggester(e)
    cfg = SuggestConfig(scope="affected")
    use = e.base_use.astype("int16").copy()
    rng = np.random.default_rng(seed)
    blocks = e.p.groupby(["cd", "block"]).size()
    blk, cdv = e.p.block.to_numpy(), e.cd
    res = {"test": __doc__.strip().split("\n\n")[1].replace("\n", " "), "small": [], "medium": []}
    for cd, b in rng.permutation(blocks.index.to_numpy()):
        if len(res["small"]) >= n_small and len(res["medium"]) >= n_medium:
            break
        idx = np.nonzero((blk == b) & (cdv == cd))[0]
        if not 2 <= len(idx) <= 60:
            continue
        opts = options_for(sug, idx, cfg, use)
        active = [k for k, o in enumerate(opts) if len(o) > 1]
        total = int(np.prod([len(opts[k]) for k in active])) if active else 1
        kind = "small" if SMALL[0] <= total <= SMALL[1] else "medium" if MEDIUM[0] <= total <= MEDIUM[1] else None
        if kind is None or len(res[kind]) >= (n_small if kind == "small" else n_medium):
            continue
        t0 = time.time()
        mask = e.study_area(idx)
        G = pillar_gain_matrices(sug, idx, cfg, use, mask)  # (3, n, N_USES); zero at the current use
        g = [G[:, k, opts[k]] for k in active]  # per active parcel: (3, options), option 0 = keep

        def surrogate(X: np.ndarray) -> np.ndarray:
            F = np.zeros((len(X), 3))
            for j, gj in enumerate(g):
                F += gj[:, X[:, j]].T
            return F

        def plan(x) -> np.ndarray:
            u = use.copy()
            for j, k in enumerate(active):
                u[idx[k]] = opts[k][x[j]]
            return u

        def exact(X: np.ndarray) -> np.ndarray:
            out = np.zeros((len(X), 3))
            for r, x in enumerate(X):
                ev = e.evaluate(plan(x), mask=mask)
                out[r] = [ev["pillars"][p] for p in PILLARS]
            return out

        X = np.array(list(itertools.product(*[range(len(opts[k])) for k in active])), dtype=np.int16)
        S = surrogate(X)
        true_s = nondominated(S)
        # the explorer, exactly as the Studio runs it, with every non-dominated plan kept
        t1 = time.time()
        r = pareto(sug, idx.tolist(), cfg, use, seed=seed, generations=GEN, max_plans=10 ** 6, drop_dominated=False)
        nsga_s = time.time() - t1
        def option(f, k) -> int:
            u = f["changes"].get(str(int(idx[k])))
            return int(np.nonzero(opts[k] == (USE_INDEX[u] if u else use[idx[k]]))[0][0])

        Xn = np.array([[option(f, k) for k in active] for f in r["front"]], dtype=np.int16)
        Sn = surrogate(Xn)
        En = np.array([[f["pillars"][p] for p in PILLARS] for f in r["front"]])
        Xr = np.stack([rng.integers(0, len(opts[k]), POP * GEN) for k in active], 1).astype(np.int16)
        Sr = surrogate(Xr)
        rnd = Xr[nondominated(Sr)]
        true_set = {tuple(x) for x in X[true_s]}
        row = {"cd": int(cd), "block": int(b), "parcels": int(len(idx)), "free_parcels": len(active),
               "plans": total, "nsga_evaluations": POP * GEN, "nsga_seconds": round(nsga_s, 2),
               "surrogate": {"true_front": int(len(true_s)), "nsga_front": int(len(Xn)),
                             "hv_share_nsga": hv_share(Sn, S[true_s], S),
                             "hv_share_random": hv_share(surrogate(rnd), S[true_s], S),
                             "share_of_true_front_found": round(len({tuple(x) for x in Xn} & true_set) / len(true_set), 4)}}
        if kind == "small":
            E = exact(X)
            true_e = nondominated(E)
            row["exact"] = {"true_front": int(len(true_e)),
                            "hv_share_nsga": hv_share(En, E[true_e], E),
                            "hv_share_surrogate_front_rescored": hv_share(E[true_s], E[true_e], E),
                            "share_of_true_front_found": round(len({tuple(x) for x in Xn} & {tuple(x) for x in X[true_e]})
                                                               / len(true_e), 4),
                            "spearman_surrogate_exact": {p: round(float(spearmanr(S[:, i], E[:, i])[0]), 4)
                                                         for i, p in enumerate(PILLARS)}}
        else:
            ts = true_s if len(true_s) <= 300 else true_s[np.linspace(0, len(true_s) - 1, 300).astype(int)]
            Et = exact(X[ts])
            allv = np.vstack([Et, En, np.zeros((1, 3))])
            row["exact"] = {"surrogate_front_rescored": int(len(ts)),
                            "hv_share_nsga_vs_surrogate_front_rescored": hv_share(En, Et, allv)}
        row["seconds"] = round(time.time() - t0, 1)
        res[kind].append(row)
        print(kind, json.dumps(row), flush=True)

    def summ(rows, path):
        v = [r for r in (_get(x, path) for x in rows) if r is not None and np.isfinite(r)]
        return {"median": round(float(np.median(v)), 4), "min": round(float(np.min(v)), 4), "n": len(v)} if v else None

    res["summary"] = {
        "small": {k: summ(res["small"], k) for k in ("surrogate.hv_share_nsga", "surrogate.hv_share_random",
                                                    "surrogate.share_of_true_front_found", "exact.hv_share_nsga",
                                                    "exact.hv_share_surrogate_front_rescored",
                                                    "exact.share_of_true_front_found")},
        "medium": {k: summ(res["medium"], k) for k in ("surrogate.hv_share_nsga", "surrogate.hv_share_random",
                                                     "surrogate.share_of_true_front_found",
                                                     "exact.hv_share_nsga_vs_surrogate_front_rescored", "plans",
                                                     "nsga_seconds")},
        "spearman_surrogate_exact_small": {p: summ(res["small"], f"exact.spearman_surrogate_exact.{p}") for p in PILLARS},
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    return res


def _get(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


if __name__ == "__main__":
    main()
