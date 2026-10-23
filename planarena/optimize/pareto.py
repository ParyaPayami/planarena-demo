"""NSGA-III Pareto explorer for a set of parcels: the social, economic and environmental trade-off.

Search uses fast first-order pillar gains from the Suggester's marginal model.
Every plan on the final front is then re-scored with the full impact engine, so
the reported values are always exact, never surrogate estimates. Seeded and
reproducible. The status quo is always a candidate.
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from planarena.config import PILLARS, USE_INDEX, USE_KEYS
from planarena.indicators.engine import INDICATORS
from planarena.optimize.suggest import SuggestConfig, Suggester

U = USE_INDEX
N_USES = len(USE_KEYS)
MAX_PARCELS = 300


def pillar_gain_matrices(sug: Suggester, idx: np.ndarray, cfg: SuggestConfig, use: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
    """Array of shape (3, len(idx), N_USES): first-order pillar-score change per parcel and use."""
    e = sug.e
    ctx = sug._state(mask, use)
    pu = sug.per_use_values()
    ag = sug._access_gain(use, ctx["state"], ctx["acc"], mask, pu)
    out = []
    for p in PILLARS:
        inds = [i.key for i in INDICATORS if i.pillar == p]
        w = {k: (1.0 / len(inds) if k in inds else 0.0) for k in (i.key for i in INDICATORS)}
        coef = sug._coefficients(ctx["base"], cfg.pillar_weights, w)
        out.append(sug._gain(coef, pu, use, ag)[idx])
    return np.stack(out)


def _nondominated(F: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Indices of the plans no other plan beats on every pillar (maximisation); identical plans count once."""
    keep = np.zeros(0, int)
    for s in range(0, len(F), chunk):
        c = np.arange(s, min(s + chunk, len(F)))
        if len(keep):
            A, B = F[c][:, None, :], F[keep][None, :, :]
            c = c[~((B >= A).all(-1) & (B > A).any(-1)).any(1)]
        pool = np.concatenate([keep, c])
        keep = pool[NonDominatedSorting().do(-F[pool], only_non_dominated_front=True)]
    _, first = np.unique(F[keep].round(12), axis=0, return_index=True)
    return np.sort(keep[first])


def options_for(sug: Suggester, idx: np.ndarray, cfg: SuggestConfig, use: np.ndarray) -> list[np.ndarray]:
    """Land-use options per parcel under the same protections the optimizer uses."""
    e = sug.e
    opts = []
    for i in idx:
        cur = int(use[i])
        if cur < 0 or (cfg.protect_landmarks and e.protected[i]) or \
                (cfg.protect_parks and cur == U["opn"]) or (cfg.protect_public and cur == U["pub"]) or \
                (cfg.protect_infrastructure and cur == U["tra"]):
            opts.append(np.array([cur]))
            continue
        allowed = np.ones(N_USES, bool)
        if cfg.respect_zoning:
            allowed = ((int(e.allowed[i]) >> np.arange(N_USES)) & 1).astype(bool)
        if cfg.protect_homes and e.obs_units[i] > 0:
            allowed &= np.isin(np.arange(N_USES), [U["res"], U["mix"]])
        if not cfg.allow_demolition and e.bldg[i] > 0:
            allowed &= ~np.isin(np.arange(N_USES), [U["opn"], U["prk"], U["vac"]])
        if cfg.require_precedent and e.precedent is not None and e.bldg[i] > 0:
            allowed &= e.has_precedent(np.full(N_USES, i), np.arange(N_USES))
        allowed[cur] = True
        o = np.nonzero(allowed)[0]
        opts.append(np.concatenate([[cur], o[o != cur]]))  # option 0 = keep the current use
    return opts


def pareto(sug: Suggester, parcels: list[int], cfg: SuggestConfig | None = None, use: np.ndarray | None = None,
           seed: int = 42, generations: int = 120, max_plans: int = 24, drop_dominated: bool = True) -> dict:
    e = sug.e
    cfg = cfg or SuggestConfig(scope="affected")
    use = (e.base_use if use is None else np.asarray(use)).astype("int16").copy()
    idx = np.array(sorted(set(int(i) for i in parcels)))[:MAX_PARCELS]
    if not len(idx):
        raise ValueError("Select at least one parcel")
    opts = options_for(sug, idx, cfg, use)
    active = np.array([k for k, o in enumerate(opts) if len(o) > 1])
    if not len(active):
        return {"algorithm": "NSGA-III", "front": [], "presets": [], "note": "No selected parcel can change under the current protections."}
    mask = e.study_area(idx) if cfg.scope == "affected" else e.scope_mask(use, cfg.scope)
    gain = pillar_gain_matrices(sug, idx, cfg, use, mask)  # (3, n_sel, N_USES)
    opt_gain = [gain[:, k, opts[k]] for k in active]  # each (3, n_opts)

    archive = []  # every plan evaluated; the front is taken from all of them, not only the final population

    class LandUseProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(active), n_obj=3, xl=0, xu=np.array([len(opts[k]) - 1 for k in active]), vtype=int)

        def _evaluate(self, X, out, *args, **kwargs):
            X = np.clip(np.rint(X), 0, self.xu).astype(int)
            F = np.zeros((len(X), 3))
            for j, g in enumerate(opt_gain):
                F += g[:, X[:, j]].T
            archive.append(X.astype("int16"))
            out["F"] = -F

    ref = get_reference_directions("das-dennis", 3, n_partitions=8)
    alg = NSGA3(ref_dirs=ref, pop_size=max(len(ref), 92), sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                mutation=PM(eta=20, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
    res = minimize(LandUseProblem(), alg, ("n_gen", generations), seed=seed, verbose=False)
    X = np.vstack([np.zeros((1, len(active)), "int16"), np.clip(np.rint(res.pop.get("X")), 0, None).astype("int16")]
                  + archive)  # the status quo is always a candidate
    X = np.unique(X, axis=0)
    F = np.zeros((len(X), 3))
    for j, g in enumerate(opt_gain):
        F += g[:, np.minimum(X[:, j], g.shape[1] - 1)].T
    nd = _nondominated(F)
    nd = nd[np.argsort(-F[nd].sum(axis=1))]
    if len(nd) > max_plans:  # keep a spread: best per pillar, then evenly spaced by total gain
        keep = set(int(nd[np.argmax(F[nd, p])]) for p in range(3))
        keep |= set(int(i) for i in nd[np.linspace(0, len(nd) - 1, max_plans - len(keep)).astype(int)])
        nd = np.array(sorted(keep, key=lambda i: -F[i].sum()))
    front = []
    for i in nd:
        u = use.copy()
        for j, k in enumerate(active):
            u[idx[k]] = opts[k][min(X[i, j], len(opts[k]) - 1)]
        ev = e.evaluate(u, mask=mask)
        changes = {str(int(p)): USE_KEYS[int(u[p])] for p in np.nonzero(u != use)[0]}
        front.append({"changes": changes, "n_changed": len(changes), "pillars": ev["pillars"],
                      "overall": ev["overall"], "estimated": F[i].round(3).tolist()})
    vals = np.array([[f["pillars"][p] for p in PILLARS] for f in front])
    # after exact re-scoring, drop plans that another plan on the front now beats on every pillar
    beaten = ((vals[None, :, :] >= vals[:, None, :]).all(-1) & (vals[None, :, :] > vals[:, None, :]).any(-1)).any(1)
    beaten &= drop_dominated
    dropped = int(beaten.sum())
    front, vals = [f for f, b in zip(front, beaten) if not b], vals[~beaten]
    lo, span = vals.min(0), np.ptp(vals, 0)
    span[span == 0] = 1
    presets = []
    for name, w in (("Balanced", [1, 1, 1]), ("Social first", [1, 0, 0]), ("Economy first", [0, 1, 0]),
                    ("Environment first", [0, 0, 1])):
        k = int(np.argmax(((vals - lo) / span) @ np.array(w, float)))
        presets.append({"name": name, "weights": dict(zip(PILLARS, w)), "index": k})
    return {"algorithm": "NSGA-III (pymoo), first-order search with an archive of every plan evaluated; exact re-scoring",
            "seed": seed,
            "generations": generations, "n_parcels": int(len(idx)), "n_free": int(len(active)),
            "front": front, "presets": presets, "objectives": list(PILLARS), "dropped_after_rescoring": dropped,
            "study_area_parcels": int(mask.sum())}
