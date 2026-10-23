"""Comprehensive stakeholder vote over land-use plans (social choice).

No language model, no messages: each voter's ballot is derived from its utility for each plan,
so every vote is reproducible and auditable. Utilities are the engine's signed indicator scores
weighted by the voter's priorities, so the status quo is worth 0 to everyone.

Electorates
* Stakeholder council: the five stakeholder agents, each with a number of seats (weights).
* Residents' referendum: today's residents vote by tax block (precinct), weighted by population,
  on the change in their own block's access to parks, shops and services, community facilities,
  land-use mix and green cover, plus homes in the study area. Future residents do not vote.
  Differences under 0.01 score points count as indifference (JND, an assumption).

Voting rules (weighted voters; ties in a ballot are allowed)
  plurality, Borda, instant-runoff (IRV), Copeland, Schulze, Kemeny-Young (exact ILP),
  approval (approve plans better than the status quo), score (0-10 per voter) and majority judgment.
The headline rule is Schulze: it is Condorcet-consistent, monotone and independent of clones,
which matters here because compromise plans are often near-copies of each other.

Analysis: pairwise majority matrix, Condorcet winner and loser, Smith set, agreement across rules,
a strategic-voting check (can one voter change the result by misreporting?), robustness to
coefficient uncertainty and seat weights (Monte Carlo), and a double-majority test against the
status quo (council seats and residents).

Parcel-by-parcel ballot: on every eligible parcel the council ranks the allowed uses by each
member's marginal utility (weighted Borda). A change passes only with an absolute majority of seats
over the current use. Passed changes, strongest first, within the change budget, form one more
candidate plan. Gains are first order, so that plan is re-scored by the full engine.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from planarena.agents.gametheory import solution_summary
from planarena.agents.negotiation import Plan, generate_candidates
from planarena.agents.stakeholders import utility_matrix
from planarena.config import USE_KEYS
from planarena.indicators.engine import INDICATORS
from planarena.optimize.suggest import N_USES, SuggestConfig, Suggester

TOL = 1e-6
JND = 0.01  # residents ignore utility differences below 0.01 score points (just-noticeable difference; assumption)
ORDINAL_RULES = ("plurality", "borda", "irv", "copeland", "schulze", "kemeny", "approval", "score",
                 "majority_judgment")
RULE_LABEL = {
    "plurality": "Plurality", "borda": "Borda count", "irv": "Instant runoff", "copeland": "Copeland",
    "schulze": "Schulze", "kemeny": "Kemeny–Young", "approval": "Approval", "score": "Score (0–10)",
    "majority_judgment": "Majority judgment",
}
GRADES = ("strongly oppose", "oppose", "slightly oppose", "neutral", "slightly support", "support",
          "strongly support")
LOCAL_KEYS = ("park_access", "services_access", "facility_access", "mix", "green_cover")


# ----------------------------------------------------------------------------- ballots
def _beats(U: np.ndarray) -> np.ndarray:
    """B[a, b, v]: voter v strictly prefers plan a to plan b."""
    return U[:, None, :] > U[None, :, :] + TOL


def pairwise(U: np.ndarray, w: np.ndarray) -> np.ndarray:
    """P[a, b] = weight of voters who strictly prefer a to b (indifferent voters abstain)."""
    return (_beats(U) * w).sum(axis=2)


def _tops(U: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Each voter's first choices among ``rows`` as equal shares; voters indifferent among all abstain."""
    sub = U if rows is None else U[rows]
    top = sub >= sub.max(axis=0, keepdims=True) - TOL
    k = top.sum(axis=0)
    share = top / np.maximum(k, 1)
    share[:, k == len(sub)] = 0.0
    return share


def plurality(U, w):
    return _tops(U) @ w


def borda(U, w):
    B = _beats(U)
    m = len(U)
    tie = ~B & ~B.transpose(1, 0, 2)
    tie[np.arange(m), np.arange(m)] = False
    return (B.sum(axis=1) + 0.5 * tie.sum(axis=1)) @ w


def irv(U, w) -> tuple[int, list[dict]]:
    m = len(U)
    alive = np.ones(m, bool)
    tiebreak = borda(U, w)
    rounds = []
    while True:
        rows = np.nonzero(alive)[0]
        if len(rows) == 1:
            return int(rows[0]), rounds
        tally = _tops(U, rows) @ w
        active = tally.sum()
        lead = int(np.argmax(tally))
        rec = {"tally": {int(r): float(t) for r, t in zip(rows, tally)}}
        if active <= TOL:  # everyone indifferent among the remaining plans
            rounds.append(rec)
            return int(rows[np.argmax(tiebreak[rows])]), rounds
        if tally[lead] > 0.5 * active:
            rounds.append(rec)
            return int(rows[lead]), rounds
        low = min(range(len(rows)), key=lambda j: (tally[j], tiebreak[rows[j]], -rows[j]))
        rec["eliminated"] = int(rows[low])
        rounds.append(rec)
        alive[rows[low]] = False


def copeland(P):
    return (P > P.T + TOL).sum(axis=1) + 0.5 * ((np.abs(P - P.T) <= TOL).sum(axis=1) - 1)


def schulze(P, tiebreak: np.ndarray | None = None) -> list[int]:
    """Schulze ranking (strongest beatpaths, winning votes)."""
    m = len(P)
    p = np.where(P > P.T + TOL, P, 0.0)
    np.fill_diagonal(p, 0.0)
    for i in range(m):
        p = np.maximum(p, np.minimum(p[:, i:i + 1], p[i:i + 1, :]))
        np.fill_diagonal(p, 0.0)
    wins = (p > p.T + TOL).sum(axis=1)
    tb = np.zeros(m) if tiebreak is None else tiebreak
    return sorted(range(m), key=lambda a: (-wins[a], -tb[a], a))


def kemeny(P, time_limit: float = 10.0) -> tuple[list[int], bool]:
    """Kemeny-Young ranking: the order that agrees with the most pairwise preferences (exact ILP)."""
    m = len(P)
    if m == 1:
        return [0], True
    pairs = [(a, b) for a in range(m) for b in range(m) if a != b]
    col = {pr: i for i, pr in enumerate(pairs)}
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for a, b in itertools.combinations(range(m), 2):  # x_ab + x_ba = 1
        rows += [r, r]; cols += [col[a, b], col[b, a]]; vals += [1, 1]; lo.append(1); hi.append(1); r += 1
    for a, b, c in itertools.combinations(range(m), 3):  # no 3-cycles, either orientation
        for x, y, z in ((a, b, c), (a, c, b)):
            rows += [r, r, r]; cols += [col[x, y], col[y, z], col[z, x]]; vals += [1, 1, 1]
            lo.append(-np.inf); hi.append(2); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, len(pairs))).tocsr()
    c = -np.array([P[a, b] for a, b in pairs])
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(len(pairs)),
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if res.x is None:
        return schulze(P), False
    x = np.rint(res.x)
    wins = np.zeros(m)
    for (a, b), v in zip(pairs, x):
        wins[a] += v
    return sorted(range(m), key=lambda a: (-wins[a], a)), bool(res.status == 0)


def approval(U, w):
    return (U > TOL) @ w


def score_vote(U, w):
    lo, hi = U.min(axis=0), U.max(axis=0)
    rng = hi - lo
    act = rng > TOL
    if not act.any():
        return np.zeros(len(U))
    S = (U[:, act] - lo[act]) / rng[act] * 10.0
    return S @ w[act] / w[act].sum()


def _grades(U):
    M = np.abs(U).max(axis=0)
    g = np.where(M > TOL, 3 + 3 * U / np.where(M > TOL, M, 1.0), 3.0)
    return np.clip(np.rint(g), 0, 6).astype(int)


def majority_judgment(U, w) -> tuple[list[int], list[dict]]:
    G = _grades(U)
    tot = w.sum()
    info = []
    for a in range(len(U)):
        dist = np.bincount(G[a], weights=w, minlength=7) / tot
        med = int(np.searchsorted(np.cumsum(dist), 0.5 - 1e-12))
        p, q = float(dist[med + 1:].sum()), float(dist[:med].sum())
        info.append({"median": med, "grade": GRADES[med], "above": p, "below": q,
                     "key": (med, p if p > q else -q), "distribution": dist.round(4).tolist()})
    order = sorted(range(len(U)), key=lambda a: (info[a]["key"][0], info[a]["key"][1]), reverse=True)
    return order, info


def condorcet(P):
    m = len(P)
    off = ~np.eye(m, dtype=bool)
    win = [a for a in range(m) if np.all((P[a] > P[:, a] + TOL)[off[a]])]
    lose = [a for a in range(m) if np.all((P[a] < P[:, a] - TOL)[off[a]])]
    return (win[0] if win else None), (lose[0] if lose else None)


def smith_set(P) -> list[int]:
    R = P >= P.T - TOL
    for i in range(len(P)):
        R = R | (R[:, i:i + 1] & R[i:i + 1, :])
    return [a for a in range(len(P)) if R[a].all()]


def run_rules(U: np.ndarray, w: np.ndarray, exact_kemeny: bool = True) -> dict:
    """All rules on one electorate. Returns winners (indices), rankings and scores."""
    P = pairwise(U, w)
    bd = borda(U, w)
    out = {"pairwise": P}
    sc = {"plurality": plurality(U, w), "borda": bd, "approval": approval(U, w), "score": score_vote(U, w),
          "copeland": copeland(P)}
    for k, v in sc.items():
        out[k] = {"winner": int(max(range(len(U)), key=lambda a: (v[a], bd[a], -a))), "scores": v}
    w_irv, rounds = irv(U, w)
    out["irv"] = {"winner": w_irv, "rounds": rounds}
    rk = schulze(P, bd)
    out["schulze"] = {"winner": rk[0], "ranking": rk}
    if exact_kemeny:
        rk_k, optimal = kemeny(P)
        out["kemeny"] = {"winner": rk_k[0], "ranking": rk_k, "optimal": optimal}
    order, info = majority_judgment(U, w)
    out["majority_judgment"] = {"winner": order[0], "ranking": order, "grades": info}
    cw, cl = condorcet(P)
    out["condorcet_winner"], out["condorcet_loser"], out["smith_set"] = cw, cl, smith_set(P)
    rules = [r for r in ORDINAL_RULES if r in out]
    counts = np.bincount([out[r]["winner"] for r in rules], minlength=len(U))
    out["consensus"] = {"winner": int(max(range(len(U)), key=lambda a: (counts[a], a == out["schulze"]["winner"]))),
                        "rules_won": counts.tolist(), "n_rules": len(rules)}
    return out


def _winner(rule: str, U, w) -> int:
    if rule == "irv":
        return irv(U, w)[0]
    if rule == "schulze":
        return schulze(pairwise(U, w), borda(U, w))[0]
    if rule == "copeland":
        c, bd = copeland(pairwise(U, w)), borda(U, w)
        return int(max(range(len(U)), key=lambda a: (c[a], bd[a], -a)))
    f = {"plurality": plurality, "borda": borda, "approval": approval, "score": score_vote}[rule]
    v, bd = f(U, w), borda(U, w)
    return int(max(range(len(U)), key=lambda a: (v[a], bd[a], -a)))


def manipulation_check(U, w, rules=("schulze", "irv", "borda", "plurality", "copeland")) -> list[dict]:
    """Can one voter get a plan it prefers by misreporting (push it first, bury the winner last)?

    A standard heuristic search, not an exhaustive one: finding no case is evidence, not proof.
    """
    found = []
    for rule in rules:
        W = _winner(rule, U, w)
        for v in range(U.shape[1]):
            if w[v] <= 0:
                continue
            prefs = U[:, v]
            for t in np.argsort(-prefs):
                if prefs[t] <= prefs[W] + TOL:
                    break
                U2 = U.copy()
                U2[:, v] = prefs
                U2[t, v] = prefs.max() + 1.0
                U2[W, v] = prefs.min() - 1.0
                W2 = _winner(rule, U2, w)
                if prefs[W2] > prefs[W] + TOL:
                    found.append({"rule": rule, "voter": v, "sincere_winner": int(W), "result": int(W2)})
                    break
    return found


# ----------------------------------------------------------------------------- uncertainty
def _score_table(evaluations: list[dict]):
    """Per plan and indicator: (delta, low, high, baseline, direction, kind) for re-scoring samples."""
    keys = [i.key for i in INDICATORS]
    m, K = len(evaluations), len(keys)
    d, lo, hi, b = (np.zeros((m, K)) for _ in range(4))
    for a, ev in enumerate(evaluations):
        rows = {r["key"]: r for r in ev["indicators"]}
        for k, key in enumerate(keys):
            r = rows[key]
            d[a, k] = r["delta"]
            lo[a, k], hi[a, k] = r.get("range", [r["delta"], r["delta"]])
            b[a, k] = r["baseline"]
    direction = np.array([i.direction for i in INDICATORS], float)
    share = np.array([i.kind == "share" for i in INDICATORS])
    return keys, d, lo, hi, b, direction, share


def _scores(delta, b, direction, share):
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(np.abs(b) > 1e-9, 100 * delta / np.abs(b), 0.0)
    raw = np.where(share[None, :], 100 * delta, pct)
    return np.clip(direction[None, :] * raw, -100, 100)


def _triangular(t, lo, mode, hi):
    """Inverse CDF of the triangular distribution (lo, mode, hi) at quantile t (broadcasts)."""
    span = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(span > 0, (mode - lo) / span, 0.5)
        left = lo + np.sqrt(np.clip(t * span * (mode - lo), 0, None))
        right = hi - np.sqrt(np.clip((1 - t) * span * (hi - mode), 0, None))
    return np.where(span > 0, np.where(t < f, left, right), mode)


def robustness(evaluations, stakeholders, seats: np.ndarray, samples: int = 200, seed: int = 0,
               concentration: float = 50.0) -> dict:
    """Schulze win probability per plan under (a) coefficient uncertainty, (b) seat perturbation, (c) both.

    Each indicator's delta is drawn from a triangular distribution over its sensitivity range with the
    central estimate as the mode. One quantile per indicator is shared by all plans, because a
    coefficient that is higher than assumed is higher for every plan (comonotone uncertainty).
    Seats are drawn from a Dirichlet distribution centred on the given seats.
    """
    keys, d, lo, hi, b, direction, share = _score_table(evaluations)
    Wst = np.array([[s.normalized()[k] for s in stakeholders] for k in keys])
    rng = np.random.default_rng(seed)
    act = seats > 0
    m = len(evaluations)
    out = {}
    for mode in ("coefficients", "seats", "both"):
        wins = np.zeros(m)
        for _ in range(samples):
            if mode == "seats":
                U = _scores(d, b, direction, share) @ Wst
            else:
                t = rng.random(len(keys))
                U = _scores(_triangular(t[None, :], lo, d, hi), b, direction, share) @ Wst
            w = seats.copy()
            if mode != "coefficients":
                w = np.zeros_like(seats)
                w[act] = rng.dirichlet(seats[act] / seats[act].sum() * concentration) * seats[act].sum()
            wins[_winner("schulze", U, w)] += 1
        out[mode] = wins / max(samples, 1)
    return out


# ----------------------------------------------------------------------------- residents' referendum
def resident_components(engine, uses: list[np.ndarray], mask: np.ndarray):
    """Per plan, precinct (tax block) and criterion: the score each precinct's residents see (the change in their
    own block's access, mix and green cover, in clipped points, then homes in the study area). Voters are today's
    residents; precinct weight = residents."""
    e = engine
    base_st = e.baseline_state
    base_acc = e.access(base_st)
    R = base_st["residents"]
    sel = np.nonzero(mask & (R > 0))[0]
    blocks = e.p["block"].to_numpy()[sel]
    ids, inv = np.unique(blocks, return_inverse=True)
    Wp = np.bincount(inv, weights=R[sel])
    homes_base = max(float(base_st["homes"][mask].sum()), 1.0)
    C = np.zeros((len(uses), len(ids), len(LOCAL_KEYS) + 1))
    for a, use in enumerate(uses):
        st = e.parcel_state(use)
        acc = e.access(st)
        for j, k in enumerate(LOCAL_KEYS):
            dk = acc[k][sel].astype(float) - base_acc[k][sel].astype(float)
            C[a, :, j] = np.clip(100 * np.bincount(inv, weights=R[sel] * dk) / Wp, -100, 100)
        C[a, :, -1] = np.clip(100 * (float(st["homes"][mask].sum()) - homes_base) / homes_base, -100, 100)
    return C, Wp, ids


def resident_electorate(engine, uses: list[np.ndarray], mask: np.ndarray, weights: dict, jnd: float = None):
    """Utility of each precinct (tax block) for each plan. Voters are today's residents."""
    C, Wp, ids = resident_components(engine, uses, mask)
    w = np.array([weights.get(k, 0.0) for k in LOCAL_KEYS] + [weights.get("homes", 0.0)])
    U, L = C @ w, C[:, :, :-1] @ w[:-1]  # L: local part only, which blocks' own access changes
    j = JND if jnd is None else jnd
    if j <= 0:
        return U, Wp, ids, L
    return np.round(U / j) * j, Wp, ids, np.round(L / j) * j


# ----------------------------------------------------------------------------- parcel-by-parcel ballot
def parcel_ballot(sug: Suggester, stakeholders, seats: np.ndarray, cfg: SuggestConfig,
                  use: np.ndarray | None = None) -> dict:
    e = sug.e
    use = (e.base_use if use is None else np.asarray(use)).astype("int16").copy()
    pu = sug.per_use_values()
    mask, eligible, allowed = sug.feasible(cfg, use)
    ctx = sug._state(mask, use)
    ag = sug._access_gain(use, ctx["state"], ctx["acc"], mask, pu)
    idx = np.nonzero(eligible)[0]
    al = allowed[idx]
    cur = use[idx].astype(int)
    rows = np.arange(len(idx))
    G = []
    for s in stakeholders:
        g = sug._gain(sug._coefficients(ctx["base"], cfg.pillar_weights, s.normalized()), pu, use, ag)[idx]
        g[rows, cur] = 0.0
        g[~al] = -np.inf
        G.append(g)
    G = np.stack(G)  # (voters, parcels, uses)
    eye = np.eye(N_USES, dtype=bool)
    points = np.zeros((len(idx), N_USES))
    for g, sv in zip(G, seats):
        if sv <= 0:
            continue
        a, b = g[:, :, None], g[:, None, :]
        valid = al[:, None, :]
        with np.errstate(invalid="ignore"):
            better = (a > b + TOL) & valid
            tie = (np.abs(a - b) <= TOL) & valid & ~eye[None]
        points += sv * (better.sum(2) + 0.5 * tie.sum(2))
    points[~al] = -np.inf
    best = points.max(axis=1)
    winner = np.where(points[rows, cur] >= best - TOL, cur, points.argmax(axis=1))
    gw = G[:, rows, winner]
    total = seats.sum()
    support = (seats[:, None] * (gw > TOL)).sum(0) / total
    against = (seats[:, None] * (gw < -TOL)).sum(0) / total
    passed = (winner != cur) & (support > 0.5)
    fin = np.where(np.isfinite(G), np.abs(G), 0.0)
    scale = np.maximum(fin.reshape(len(seats), -1).max(axis=1), TOL)
    strength = (seats[:, None] * gw / scale[:, None]).sum(0)
    budget = max(1, int(cfg.max_change_share * len(idx)))
    order = np.nonzero(passed)[0]
    order = order[np.lexsort((-strength[order], -support[order]))][:budget]
    new = use.copy()
    new[idx[order]] = winner[order]
    return {"use": new, "parcels": idx, "winner": winner, "support": support, "against": against,
            "passed": passed, "chosen": idx[order], "budget": budget}


# ----------------------------------------------------------------------------- the vote
def _named(rules_out: dict, ids: list[str]) -> dict:
    res = {}
    for r in ORDINAL_RULES:
        if r not in rules_out:
            continue
        o = rules_out[r]
        item = {"winner": ids[o["winner"]], "label": RULE_LABEL[r]}
        if "scores" in o:
            item["scores"] = {ids[a]: round(float(v), 4) for a, v in enumerate(o["scores"])}
        if "ranking" in o:
            item["ranking"] = [ids[a] for a in o["ranking"]]
        if r == "irv":
            item["rounds"] = [{"tally": {ids[k]: round(v, 4) for k, v in rd["tally"].items()},
                               **({"eliminated": ids[rd["eliminated"]]} if "eliminated" in rd else {})}
                              for rd in o["rounds"]]
        if r == "majority_judgment":
            item["grades"] = {ids[a]: {k: g[k] for k in ("grade", "median", "above", "below")}
                              for a, g in enumerate(o["grades"])}
        if r == "kemeny":
            item["optimal"] = o["optimal"]
        res[r] = item
    return res


def _vs(U, w, a, b) -> dict:
    """Weight preferring plan a to plan b, and b to a."""
    return {"for": float(w[U[a] > U[b] + TOL].sum()), "against": float(w[U[a] < U[b] - TOL].sum()),
            "total": float(w.sum())}


def hold_vote(sug: Suggester, stakeholders, seats: dict | None = None, scope: str = "cd:103",
              max_change_share: float = 0.05, current_use: np.ndarray | None = None,
              candidate_parcels=None, residents: bool = True, samples: int = 200,
              plans: list[Plan] | None = None, with_parcel_ballot: bool = True, seed: int = 0) -> dict:
    e = sug.e
    sv = np.array([float((seats or {}).get(s.key, 1.0)) for s in stakeholders])
    if sv.sum() <= 0:
        raise ValueError("at least one stakeholder needs a seat")
    status_quo = Plan("status-quo", "No change", "status quo", e.base_use.copy(), {})
    ballot = None
    if plans is None:
        plans = generate_candidates(sug, stakeholders, scope=scope, max_change_share=max_change_share,
                                    current_use=current_use, candidate_parcels=candidate_parcels)
        if with_parcel_ballot:
            cfg = SuggestConfig(scope=scope, max_change_share=max_change_share, candidate_parcels=candidate_parcels)
            ballot = parcel_ballot(sug, stakeholders, sv, cfg)
            plans.append(Plan("parcel-ballot", "Parcel-by-parcel ballot", "parcel vote", ballot["use"], {}))
    plans = [status_quo] + list(plans)
    uniq, seen, alias = [], {}, {}
    for p in plans:
        h = p.use.tobytes()
        if h in seen:
            alias[p.id] = seen[h]
            continue
        seen[h] = p.id
        uniq.append(p)
    plans = uniq
    for p in plans:  # exact re-score with the sensitivity envelope
        p.evaluation = e.evaluate(p.use, scope=scope, ranges=True)
    ids = [p.id for p in plans]
    evals = [p.evaluation for p in plans]

    U = utility_matrix(stakeholders, evals)
    council = run_rules(U, sv)
    W = council["schulze"]["winner"]
    prob = robustness(evals, stakeholders, sv, samples=samples, seed=seed) if samples else None
    manip = manipulation_check(U, sv)
    out_council = {
        "voters": [{"key": s.key, "name": s.name, "seats": float(x)} for s, x in zip(stakeholders, sv)],
        "utilities": U.round(4).tolist(),
        "ballots": {s.key: [ids[a] for a in np.argsort(-U[:, v], kind="stable")] for v, s in enumerate(stakeholders)},
        "rules": _named(council, ids),
        "pairwise": council["pairwise"].round(4).tolist(),
        "condorcet_winner": None if council["condorcet_winner"] is None else ids[council["condorcet_winner"]],
        "condorcet_loser": None if council["condorcet_loser"] is None else ids[council["condorcet_loser"]],
        "smith_set": [ids[a] for a in council["smith_set"]],
        "consensus": {"winner": ids[council["consensus"]["winner"]],
                      "rules_won": dict(zip(ids, council["consensus"]["rules_won"])),
                      "n_rules": council["consensus"]["n_rules"]},
        "manipulation": [{"rule": RULE_LABEL[m["rule"]], "voter": stakeholders[m["voter"]].key,
                          "sincere_winner": ids[m["sincere_winner"]], "result": ids[m["result"]]} for m in manip],
        "robustness": None if prob is None else {
            "samples": samples,
            **{k: dict(zip(ids, v.round(4).tolist())) for k, v in prob.items()}},
        "versus_status_quo": _vs(U, sv, W, 0),
        "gains": {s.key: bool(U[W, v] >= -TOL) for v, s in enumerate(stakeholders)},
    }

    out_res = None
    if residents:
        mask = e.scope_mask(e.base_use, scope)
        weights = next((s.normalized() for s in stakeholders if s.key == "residents"), None)
        if weights is not None:
            UR, WR, blocks, LR = resident_electorate(e, [p.use for p in plans], mask, weights)
            if len(WR):
                rr = run_rules(UR, WR, exact_kemeny=len(plans) <= 24)
                WRw = rr["schulze"]["winner"]
                affected = np.abs(LR[W] - LR[0]) >= JND / 2
                out_res = {
                    "precincts": int(len(WR)), "residents": float(WR.sum()),
                    "affected_precincts": int(affected.sum()), "affected_residents": float(WR[affected].sum()),
                    "rules": _named(rr, ids),
                    "condorcet_winner": None if rr["condorcet_winner"] is None else ids[rr["condorcet_winner"]],
                    "smith_set": [ids[a] for a in rr["smith_set"]],
                    "winner": ids[WRw],
                    "council_winner_vs_status_quo": _vs(UR, WR, W, 0),
                    "agrees_with_council": bool(WRw == W),
                    "pairwise": rr["pairwise"].round(2).tolist(),
                }

    vsq = out_council["versus_status_quo"]
    council_majority = vsq["for"] > 0.5 * vsq["total"]
    double = None
    if out_res:
        r = out_res["council_winner_vs_status_quo"]
        voting = r["for"] + r["against"]
        double = bool(council_majority and (voting <= TOL or r["for"] > r["against"]))
    decision = _explain(ids, plans, W, council, out_council, out_res, double, prob)

    res = {
        "candidates": [{"id": p.id, "title": p.title, "origin": p.origin, "n_changed": p.evaluation["n_changed"],
                        "pillars": p.evaluation["pillars"], "overall": p.evaluation["overall"]} for p in plans],
        "aliases": alias,
        "headline_rule": "schulze",
        "council": out_council,
        "residents": out_res,
        "bargaining": solution_summary(U, ids),
        "decision": decision | {"double_majority": double, "council_majority": bool(council_majority)},
        "plans": plans,
    }
    if ballot is not None:
        res["parcel_ballot"] = {
            "eligible": int(len(ballot["parcels"])), "passed": int(ballot["passed"].sum()),
            "changed": int(len(ballot["chosen"])), "budget": ballot["budget"],
            "parcels": ballot["parcels"].tolist(), "winner": [USE_KEYS[u] for u in ballot["winner"]],
            "support": ballot["support"].round(3).tolist(), "chosen": ballot["chosen"].tolist(),
        }
    return res


def _explain(ids, plans, W, council, oc, ores, double, prob) -> dict:
    title = {p.id: p.title for p in plans}
    lines = []
    cw = council["condorcet_winner"]
    lines.append(f"“{title[ids[W]]}” wins the stakeholder council under the Schulze rule"
                 + (", and it is the Condorcet winner: it beats every other plan head to head." if cw == W
                    else "." if cw is None else f"; the Condorcet winner is “{title[ids[cw]]}”."))
    n_won = council["consensus"]["rules_won"][W]
    lines.append(f"It wins under {n_won} of {council['consensus']['n_rules']} voting rules.")
    v = oc["versus_status_quo"]
    lines.append(f"{v['for']:g} of {v['total']:g} seats prefer it to no change; {v['against']:g} prefer no change.")
    if ores:
        r = ores["council_winner_vs_status_quo"]
        voting = r["for"] + r["against"]
        if voting > TOL:
            lines.append(f"Residents' referendum: {100 * r['for'] / voting:.0f}% of residents with a preference "
                         f"choose it over no change ({ores['affected_precincts']} of {ores['precincts']} blocks see "
                         "a local access change).")
        else:
            lines.append("Residents' referendum: no block's access changes, so residents are indifferent.")
        lines.append("The residents' own Schulze winner is the same plan." if ores["agrees_with_council"]
                     else f"The residents' own Schulze winner is “{title[ores['winner']]}”.")
    if prob is not None:
        n = oc["robustness"]["samples"]
        lines.append(f"Robustness over {n} simulated votes: it wins {100 * prob['coefficients'][W]:.0f}% when "
                     f"coefficients vary within their ranges, {100 * prob['seats'][W]:.0f}% when seat weights vary, "
                     f"and {100 * prob['both'][W]:.0f}% when both vary.")
    sch = [m for m in oc["manipulation"] if m["rule"] == RULE_LABEL["schulze"]]
    lines.append("No single stakeholder could change the Schulze result by voting insincerely (heuristic check)."
                 if not sch else
                 "A single insincere ballot could change the Schulze result ("
                 + ", ".join(sorted({m["voter"] for m in sch})) + "). Agent ballots are sincere by construction; "
                 "with human voters this would matter.")
    if double is not None:
        lines.append("It carries a double majority (council seats and affected residents)." if double
                     else "It does not carry a double majority; treat it as contested.")
    return {"plan": ids[W], "title": title[ids[W]], "rule": "Schulze", "explanation": lines}


def vote_all_districts(samples: int = 200, out=None) -> dict:
    """Hold the vote in every community district and save a summary (results/vote_summary.json)."""
    import json

    from planarena.agents.stakeholders import DEFAULT_STAKEHOLDERS
    from planarena.config import PROCESSED, ROOT
    from planarena.indicators.engine import load_engine

    from planarena.models.gnn import attach_value_model

    e = load_engine()
    suit = np.load(PROCESSED / "suitability.npy") if (PROCESSED / "suitability.npy").exists() else None
    attach_value_model(e)  # same models as the app
    sug = Suggester(e, suit)
    rows = []
    for cd in sorted({int(c) for c in e.cd if 101 <= c <= 112}):
        r = hold_vote(sug, DEFAULT_STAKEHOLDERS, scope=f"cd:{cd}", samples=samples)
        c, res = r["council"], r["residents"]
        W = r["decision"]["plan"]
        rows.append({
            "cd": cd, "candidates": len(r["candidates"]), "winner": W,
            "condorcet_winner": c["condorcet_winner"], "smith_set_size": len(c["smith_set"]),
            "rules_won": c["consensus"]["rules_won"][W], "n_rules": c["consensus"]["n_rules"],
            "rules_agreeing_all": len({v["winner"] for v in c["rules"].values()}) == 1,
            "seats_for_vs_status_quo": c["versus_status_quo"]["for"],
            "residents_winner": res["winner"] if res else None,
            "residents_agree": res["agrees_with_council"] if res else None,
            "residents_yes_share": (res["council_winner_vs_status_quo"]["for"]
                                    / max(res["council_winner_vs_status_quo"]["for"]
                                          + res["council_winner_vs_status_quo"]["against"], 1e-9)) if res else None,
            "double_majority": r["decision"]["double_majority"],
            "robust_coefficients": c["robustness"]["coefficients"][W], "robust_seats": c["robustness"]["seats"][W],
            "robust_both": c["robustness"]["both"][W],
            "schulze_manipulable": any(m["rule"] == RULE_LABEL["schulze"] for m in c["manipulation"]),
            "nash_bargaining": (r["bargaining"]["nash_bargaining"] or {}).get("plan"),
            "parcel_ballot_passed": r["parcel_ballot"]["passed"], "parcel_ballot_changed": r["parcel_ballot"]["changed"],
            "parcel_ballot_rank": c["rules"]["schulze"]["ranking"].index(
                r["aliases"].get("parcel-ballot", "parcel-ballot")) + 1,
        })
        print(cd, W, rows[-1]["condorcet_winner"], f"robust {rows[-1]['robust_both']:.2f}")
    n = len(rows)
    summary = {
        "districts": n, "samples": samples,
        "condorcet_winner_exists": sum(r["condorcet_winner"] is not None for r in rows),
        "winner_is_condorcet": sum(r["condorcet_winner"] == r["winner"] for r in rows),
        "all_rules_agree": sum(r["rules_agreeing_all"] for r in rows),
        "residents_agree_with_council": sum(bool(r["residents_agree"]) for r in rows),
        "double_majority": sum(bool(r["double_majority"]) for r in rows),
        "winner_is_nash_bargaining": sum(r["nash_bargaining"] == r["winner"] for r in rows),
        "schulze_manipulable": sum(r["schulze_manipulable"] for r in rows),
        "median_robust_both": float(np.median([r["robust_both"] for r in rows])),
        "median_rules_won": float(np.median([r["rules_won"] for r in rows])),
        "note": "Five stakeholder agents with one seat each; plans may change up to 5% of eligible parcels; "
                "GNN value head attached, as in the app.",
        "by_district": rows,
    }
    path = out or ROOT / "results" / "vote_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = vote_all_districts()
    print({k: v for k, v in s.items() if k != "by_district"})


# ----------------------------------------------------------------------------- sensitivity of the simulated scenario (R6)
def sensitivity(sug: Suggester, stakeholders, scope: str, draws: int = 200, n_random: int = 16, seed: int = 0) -> dict:
    """How much the simulated outcome depends on its assumptions, in one district.

    1. utility weights: each stakeholder's indicator weights drawn from a Dirichlet centred on the assumed weights
    2. candidate generation: random feasible plans (not built from the stakeholder utilities) added to the ballot
    3. residents' indifference threshold (JND) swept
    4. seat allocations: equal seats, then each stakeholder with three seats
    5. rule disagreement by candidate set: how many of the nine rules pick a different winner, and whether a
       Condorcet winner exists, on the generated ballot, on the stakeholders' ideal plans only, on the generated
       ballot plus random plans, and on random plans only (each set with the status quo)
    6. heterogeneous residents: each precinct's indicator weights drawn from a Dirichlet centred on the residents'
       profile, independently per precinct; share of draws in which the residents' winner is unchanged
    """
    rng = np.random.default_rng(seed)
    base = hold_vote(sug, stakeholders, scope=scope, samples=0, residents=True)
    plans, ids = base["plans"], [p.id for p in base["plans"]]
    W0 = base["decision"]["plan"]
    evals = [p.evaluation for p in plans]
    keys, d, lo, hi, b, direction, share = _score_table(evals)
    S = _scores(d, b, direction, share)
    sv = np.ones(len(stakeholders))
    out = {"scope": scope, "winner": W0, "candidates": len(ids)}
    # 1. utility weights
    Wst = np.array([[s.normalized()[k] for s in stakeholders] for k in keys])
    same = 0
    for _ in range(draws):
        Wp = np.stack([rng.dirichlet(Wst[:, j] * 20 + 0.05) for j in range(len(stakeholders))], axis=1)
        same += ids[_winner("schulze", S @ Wp, sv)] == W0
    out["winner_unchanged_under_weight_draws"] = round(same / draws, 3)
    # 2. independent candidates: random feasible plans with the same change budget
    e = sug.e
    cfg = SuggestConfig(scope=scope)
    _, eligible, allowed = sug.feasible(cfg, e.base_use)
    el = np.nonzero(eligible)[0]
    budget = max(1, int(cfg.max_change_share * len(el)))
    rand_plans = []
    for k in range(n_random):
        use = e.base_use.copy()
        for p_ in rng.choice(el, size=min(budget, len(el)), replace=False):
            opts = np.nonzero(allowed[p_])[0]
            opts = opts[opts != use[p_]]
            if len(opts):
                use[p_] = rng.choice(opts)
        rand_plans.append(Plan(f"random-{k}", f"Random plan {k + 1}", "random", use, e.evaluate(use, scope=scope)))
    allp = plans + rand_plans
    Ua = utility_matrix(stakeholders, [p.evaluation for p in allp])
    win_all = [p.id for p in allp][_winner("schulze", Ua, sv)]
    P = pairwise(Ua, sv)
    wi = [p.id for p in allp].index(W0)
    out["random_candidates"] = n_random
    out["winner_with_random_candidates"] = win_all
    out["winner_unchanged_with_random_candidates"] = bool(win_all == W0)
    out["winner_beats_every_random_plan"] = bool(all(P[wi, len(plans) + k] > P[len(plans) + k, wi] for k in range(n_random)))
    # 5. rule disagreement by candidate set
    sets = {"generated": list(range(len(plans))),
            "ideal plans only": [0] + [a for a, p in enumerate(plans) if p.id.startswith("ideal-")],
            "generated + random": list(range(len(allp))),
            "random only": [0] + list(range(len(plans), len(allp)))}
    dis = {}
    for name, rows_ in sets.items():
        rr = run_rules(Ua[rows_], sv)
        winners = [rr[r]["winner"] for r in ORDINAL_RULES if r in rr]
        dis[name] = {"candidates": len(rows_), "distinct_winners": len(set(winners)),
                     "rules_with_modal_winner": int(np.bincount(winners).max()),
                     "condorcet_winner_exists": rr["condorcet_winner"] is not None,
                     "smith_set_size": len(rr["smith_set"]),
                     "schulze_winner": [allp[i].id for i in rows_][rr["schulze"]["winner"]]}
    out["rule_disagreement_by_candidate_set"] = dis
    # 3. residents' indifference threshold
    weights = next((s_.normalized() for s_ in stakeholders if s_.key == "residents"), None)
    mask = e.scope_mask(e.base_use, scope)
    jres = {}
    for j in (0.0, 0.005, 0.01, 0.05, 0.1):
        UR, WR, _, _ = resident_electorate(e, [p.use for p in plans], mask, weights, jnd=j)
        jres[str(j)] = ids[_winner("schulze", UR, WR)] if len(WR) else None
    out["residents_winner_by_jnd"] = jres
    out["residents_winner_stable_across_jnd"] = len(set(jres.values())) == 1
    # 6. heterogeneous residents: one weight draw per precinct
    C, WR, _ = resident_components(e, [p.use for p in plans], mask)
    if len(WR):
        crit = list(LOCAL_KEYS) + ["homes"]
        w0 = np.array([weights.get(k, 0.0) for k in crit])
        w0 = w0 / w0.sum()
        same = 0
        for _ in range(draws // 2):
            Wb = rng.dirichlet(w0 * 20 + 0.05, size=len(WR))  # (precincts, criteria)
            UR = np.round(np.einsum("abk,bk->ab", C, Wb) / JND) * JND
            same += ids[_winner("schulze", UR, WR)] == jres[str(JND)]
        out["residents_winner_unchanged_with_heterogeneous_weights"] = round(same / (draws // 2), 3)
    # 4. seat allocations
    seat_res = {"equal": W0}
    for j, s_ in enumerate(stakeholders):
        w = np.ones(len(stakeholders))
        w[j] = 3
        seat_res[f"{s_.key} x3"] = ids[_winner("schulze", utility_matrix(stakeholders, evals), w)]
    out["winner_by_seats"] = seat_res
    out["share_of_seat_allocations_with_same_winner"] = round(sum(v == W0 for v in seat_res.values()) / len(seat_res), 3)
    return out


def sensitivity_all_districts(out=None) -> dict:
    import json

    from planarena.agents.stakeholders import DEFAULT_STAKEHOLDERS
    from planarena.config import PROCESSED, ROOT
    from planarena.indicators.engine import load_engine
    from planarena.models.gnn import attach_value_model

    e = load_engine()
    attach_value_model(e)
    suit = np.load(PROCESSED / "suitability.npy") if (PROCESSED / "suitability.npy").exists() else None
    sug = Suggester(e, suit)
    rows = []
    for cd in range(101, 113):
        r = sensitivity(sug, DEFAULT_STAKEHOLDERS, f"cd:{cd}")
        rows.append(r)
        print(cd, {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
    summary = {
        "median_winner_unchanged_under_weight_draws": float(np.median([r["winner_unchanged_under_weight_draws"] for r in rows])),
        "districts_winner_unchanged_with_random_candidates": int(sum(r["winner_unchanged_with_random_candidates"] for r in rows)),
        "districts_winner_beats_every_random_plan": int(sum(r["winner_beats_every_random_plan"] for r in rows)),
        "districts_residents_winner_stable_across_jnd": int(sum(r["residents_winner_stable_across_jnd"] for r in rows)),
        "median_share_of_seat_allocations_with_same_winner": float(np.median([r["share_of_seat_allocations_with_same_winner"] for r in rows])),
        "median_residents_winner_unchanged_with_heterogeneous_weights": float(np.median(
            [r.get("residents_winner_unchanged_with_heterogeneous_weights", np.nan) for r in rows])),
        "rule_disagreement_by_candidate_set": {
            name: {"median_candidates": float(np.median([r["rule_disagreement_by_candidate_set"][name]["candidates"] for r in rows])),
                   "median_distinct_winners": float(np.median([r["rule_disagreement_by_candidate_set"][name]["distinct_winners"] for r in rows])),
                   "max_distinct_winners": int(max(r["rule_disagreement_by_candidate_set"][name]["distinct_winners"] for r in rows)),
                   "districts_all_rules_agree": int(sum(r["rule_disagreement_by_candidate_set"][name]["distinct_winners"] == 1 for r in rows)),
                   "districts_with_condorcet_winner": int(sum(r["rule_disagreement_by_candidate_set"][name]["condorcet_winner_exists"] for r in rows))}
            for name in rows[0]["rule_disagreement_by_candidate_set"]},
        "districts": len(rows)}
    res = {"test": sensitivity.__doc__.strip(), "summary": summary, "by_district": rows}
    path = out or ROOT / "results" / "validation" / "vote_sensitivity.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    return res
