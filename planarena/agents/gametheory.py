"""Cooperative game-theory solution concepts over a finite set of candidate plans.

utilities: array (n_plans, n_agents). The disagreement point d is the status quo,
where every utility is 0.
"""
from __future__ import annotations

import numpy as np


def individually_rational(u: np.ndarray, d: np.ndarray | None = None, tol: float = 1e-9) -> np.ndarray:
    d = np.zeros(u.shape[1]) if d is None else d
    return np.all(u >= d - tol, axis=1)


def nash_bargaining(u: np.ndarray, d: np.ndarray | None = None) -> int | None:
    """argmax over plans of prod_i (u_i - d_i), restricted to individually rational plans (Nash 1950)."""
    d = np.zeros(u.shape[1]) if d is None else d
    ir = individually_rational(u, d)
    if not ir.any():
        return None
    gains = np.clip(u - d, 1e-12, None)
    logp = np.where(ir, np.log(gains).sum(axis=1), -np.inf)
    return int(np.argmax(logp))


def kalai_smorodinsky(u: np.ndarray, d: np.ndarray | None = None) -> int | None:
    """Maximize the minimum gain as a share of each agent's ideal gain (Kalai & Smorodinsky 1975)."""
    d = np.zeros(u.shape[1]) if d is None else d
    ir = individually_rational(u, d)
    if not ir.any():
        return None
    ideal = np.max(u[ir], axis=0) - d
    rel = (u - d) / np.where(ideal > 1e-12, ideal, 1.0)
    return int(np.argmax(np.where(ir, rel.min(axis=1), -np.inf)))


def utilitarian(u: np.ndarray) -> int:
    return int(np.argmax(u.sum(axis=1)))


def egalitarian(u: np.ndarray) -> int:
    return int(np.argmax(u.min(axis=1)))


def pareto_front(u: np.ndarray) -> np.ndarray:
    """Boolean mask of the plans that no other plan Pareto-dominates."""
    n = len(u)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(u >= u[i], axis=1) & np.any(u > u[i], axis=1)
        keep[i] = not dom.any()
    return keep


def solution_summary(u: np.ndarray, plan_ids: list[str]) -> dict:
    out = {}
    for name, idx in (("nash_bargaining", nash_bargaining(u)), ("kalai_smorodinsky", kalai_smorodinsky(u)),
                      ("utilitarian", utilitarian(u)), ("egalitarian", egalitarian(u))):
        out[name] = None if idx is None else {"plan": plan_ids[idx], "index": idx,
                                              "utilities": u[idx].round(3).tolist(),
                                              "welfare": float(u[idx].sum())}
    out["pareto_plans"] = [plan_ids[i] for i in np.nonzero(pareto_front(u))[0]]
    out["individually_rational_plans"] = [plan_ids[i] for i in np.nonzero(individually_rational(u))[0]]
    return out
