"""Multi-agent negotiation over candidate land-use plans.

1. Candidate generation (the agents' option space), using the optimizer:
   - each agent's ideal plan (its own weights)
   - pairwise compromise plans (averages of two agents' weights)
   - a consensus plan (only moves that improve every agent's utility)
   - an equal-weights plan, and the user's current plan, if one is given
2. Negotiation: multilateral monotonic concession, following Zeuthen's risk
   criterion as formalized by Harsanyi (1956). Each round, every agent holds an
   offer. If one offer gives every agent at least the utility of its own offer,
   it is agreed. Otherwise the agent least willing to risk conflict concedes to
   its next-best plan, choosing the one that most raises the Nash product.
3. Solution concepts are computed over the same candidate set, for comparison.
4. Price of anarchy: parcels are split among the agents, each agent
   best-responds on its own parcels, and the equilibrium welfare is compared
   with the cooperative optimum.

Agent decisions are computed here, deterministically and auditably. An LLM may
phrase the messages (llm_voice) but never decides.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from planarena.agents.gametheory import nash_bargaining, solution_summary
from planarena.agents.stakeholders import DEFAULT_STAKEHOLDERS, Stakeholder, utility_matrix
from planarena.indicators.engine import IND_KEYS, INDICATORS
from planarena.optimize.suggest import SuggestConfig, Suggester

LABEL = {i.key: i.label for i in INDICATORS}


@dataclass
class Plan:
    id: str
    title: str
    origin: str
    use: np.ndarray
    evaluation: dict


def _blend(a: dict, b: dict) -> dict:
    return {k: 0.5 * a.get(k, 0) + 0.5 * b.get(k, 0) for k in IND_KEYS}


def generate_candidates(sug: Suggester, stakeholders, scope: str, max_change_share: float = 0.05,
                        current_use: np.ndarray | None = None, candidate_parcels=None) -> list[Plan]:
    base_cfg = dict(scope=scope, max_change_share=max_change_share, candidate_parcels=candidate_parcels)
    plans: list[Plan] = []

    def add(pid, title, origin, cfg):
        r = sug.suggest(cfg)
        plans.append(Plan(pid, title, origin, r["use"], r["evaluation"]))

    for s in stakeholders:
        add(f"ideal-{s.key}", f"{s.name}'s plan", "agent ideal",
            SuggestConfig(indicator_weights=s.normalized(), **base_cfg))
    for a, b in itertools.combinations(stakeholders, 2):
        add(f"blend-{a.key}-{b.key}", f"{a.name} + {b.name}", "pairwise compromise",
            SuggestConfig(indicator_weights=_blend(a.normalized(), b.normalized()), **base_cfg))
    add("balanced", "Balanced (equal pillars)", "equal weights", SuggestConfig(**base_cfg))
    add("consensus", "Consensus (no agent loses)", "unanimity",
        SuggestConfig(unanimity_weights=[s.normalized() for s in stakeholders], **base_cfg))
    if current_use is not None:
        plans.append(Plan("user", "Your plan", "user", np.asarray(current_use, dtype="int16"),
                          sug.e.evaluate(current_use, scope=scope)))
    # Plans that are identical in effect (same use array) are merged.
    uniq, seen = [], set()
    for p in plans:
        h = hash(p.use.tobytes())
        if h not in seen:
            seen.add(h)
            uniq.append(p)
    return uniq


def _top_effects(ev: dict, weights: dict, n: int = 2, positive: bool = True) -> list[str]:
    scored = [(weights.get(r["key"], 0) * r["score"], r) for r in ev["indicators"] if weights.get(r["key"], 0) > 0]
    scored.sort(key=lambda t: -t[0] if positive else t[0])
    out = []
    for v, r in scored[:n]:
        if (v > 0) == positive and abs(v) > 1e-6:
            word = "better" if r["score"] >= 0 else "worse"
            out.append(f"{LABEL[r['key']].lower()} {abs(r['score']):.1f} pts {word}")
    return out


def negotiate(plans: list[Plan], stakeholders=DEFAULT_STAKEHOLDERS, max_rounds: int | None = None) -> dict:
    ids = [p.id for p in plans]
    U = utility_matrix(stakeholders, [p.evaluation for p in plans])
    n_agents = len(stakeholders)
    order = [list(np.argsort(-U[:, i])) for i in range(n_agents)]
    pos = [0] * n_agents
    offer = [order[i][0] for i in range(n_agents)]
    transcript = []

    def say(agent, kind, plan_idx, text):
        transcript.append({"round": rnd, "agent": stakeholders[agent].key if agent is not None else "moderator",
                           "kind": kind, "plan": ids[plan_idx] if plan_idx is not None else None, "text": text})

    rnd = 0
    for i, s in enumerate(stakeholders):
        good = _top_effects(plans[offer[i]].evaluation, s.normalized())
        say(i, "propose", offer[i], f"I propose “{plans[offer[i]].title}”"
            + (f": {', '.join(good)}." if good else "."))
    agreed = None
    max_rounds = max_rounds or len(plans) * n_agents
    for rnd in range(1, max_rounds + 1):
        own = np.array([U[offer[i], i] for i in range(n_agents)])
        for j in range(n_agents):
            if all(U[offer[j], i] >= own[i] - 1e-9 for i in range(n_agents)):
                agreed = offer[j]
                break
        if agreed is not None:
            break
        # Zeuthen risk: the largest relative loss the agent would accept rather than give in.
        risk = np.zeros(n_agents)
        for i in range(n_agents):
            denom = max(own[i], 1e-9)
            others = [U[offer[j], i] for j in range(n_agents) if j != i]
            risk[i] = min((own[i] - o) / denom for o in others) if own[i] > 0 else 0.0
        # The agent least willing to risk conflict concedes one step down its own ranking
        # (minimal concession, the monotonic concession protocol). Agents at the bottom of
        # their ranking cannot concede further.
        # Agents never concede below their disagreement utility (the status quo, 0).
        can = [i for i in range(n_agents)
               if pos[i] + 1 < len(order[i]) and U[order[i][pos[i] + 1], i] >= 0]
        if not can:
            break
        conceder = min(can, key=lambda i: risk[i])
        ranked = order[conceder]
        pos[conceder] += 1
        best = ranked[pos[conceder]]
        old = offer[conceder]
        offer[conceder] = best
        s = stakeholders[conceder]
        lost = _top_effects(plans[old].evaluation, s.normalized(), n=1)
        keep = _top_effects(plans[best].evaluation, s.normalized(), n=1)
        say(conceder, "concede", best,
            f"I can move from “{plans[old].title}” to “{plans[best].title}”"
            + (f", as long as {keep[0]} holds." if keep else "."))
        # The agent most hurt by the new offer voices its objection.
        hurt = int(np.argmin([U[best, i] - U[offer[i], i] for i in range(n_agents)]))
        if hurt != conceder:
            bad = _top_effects(plans[best].evaluation, stakeholders[hurt].normalized(), n=1, positive=False)
            if bad:
                say(hurt, "object", best, f"“{plans[best].title}” still costs us {bad[0]}.")
    fallback = None
    if agreed is None:
        fallback = nash_bargaining(U)
        if fallback is None:
            fallback = int(np.argmax(U.sum(axis=1)))
        rnd += 1
        say(None, "mediate", fallback, f"No unanimous offer; the mediator proposes “{plans[fallback].title}”, "
                                      "the Nash bargaining solution among the plans no agent loses on.")
    final = agreed if agreed is not None else fallback
    rnd += 1
    say(None, "agree", final, f"Agreement on “{plans[final].title}”.")
    return {
        "agreed_plan": ids[final], "agreed_by_protocol": agreed is not None, "rounds": rnd,
        "utilities": {s.key: float(U[final, i]) for i, s in enumerate(stakeholders)},
        "nash_product": float(np.prod(np.clip(U[final], 0, None))),
        "transcript": transcript,
        "solutions": solution_summary(U, ids),
        "utility_matrix": U.round(3).tolist(), "plan_ids": ids,
        "plans": [{"id": p.id, "title": p.title, "origin": p.origin, "n_changed": p.evaluation["n_changed"],
                   "pillars": p.evaluation["pillars"], "overall": p.evaluation["overall"]} for p in plans],
    }


def price_of_anarchy(sug: Suggester, stakeholders, scope: str, max_change_share: float = 0.05,
                     rounds: int = 3, seed: int = 0) -> dict:
    """Fragmented control: each agent decides a random 1/n share of the eligible parcels."""
    e = sug.e
    mask = e.scope_mask(e.base_use, scope)
    idx = np.nonzero(mask & (e.base_use >= 0))[0]
    rng = np.random.default_rng(seed)
    owner = rng.integers(0, len(stakeholders), len(idx))
    use = e.base_use.copy()
    for _ in range(rounds):
        for i, s in enumerate(stakeholders):
            r = sug.suggest(SuggestConfig(indicator_weights=s.normalized(), scope=scope,
                                          max_change_share=max_change_share,
                                          candidate_parcels=idx[owner == i], rounds=1), use=use)
            use = r["use"]
    ev_ne = e.evaluate(use, scope=scope)
    u_ne = np.array([s.utility(ev_ne) for s in stakeholders])
    coop = sug.suggest(SuggestConfig(scope=scope, max_change_share=max_change_share,
                                     indicator_weights={k: sum(s.normalized()[k] for s in stakeholders)
                                                        / len(stakeholders) for k in IND_KEYS}))
    u_coop = np.array([s.utility(coop["evaluation"]) for s in stakeholders])
    w_ne, w_coop = float(u_ne.sum()), float(u_coop.sum())
    return {"welfare_fragmented": w_ne, "welfare_cooperative": w_coop,
            "price_of_anarchy": (w_coop / w_ne) if w_ne > 1e-9 else None,
            "utilities_fragmented": u_ne.round(3).tolist(), "utilities_cooperative": u_coop.round(3).tolist(),
            "note": "Welfare is the sum of agent utilities. PoA > 1 means fragmented control loses welfare."}
