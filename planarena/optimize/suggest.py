"""Optimal suggested land use for each parcel.

The objective is a first-order approximation of the impact engine's overall
score: weighted pillar means of signed % changes against the scope baseline.
Each parcel/use pair receives a marginal gain:

    gain[p, u] = sum_k c_k * delta_k[p, u]          additive indicators (exact per parcel)
               + access terms (need-weighted)        park / services / facility effects
               + w_fit * (log P(u|ctx) - log P(cur|ctx))   GNN neighbourhood fit

Parcels are switched in iterated best-response rounds. The global objective is
the potential function of a potential game, so each round can only improve it,
and the dynamics stop at a pure Nash equilibrium of the parcel game, which is a
local optimum. Each round is limited by a change budget, then the access
deficits are recomputed. The final plan is always re-scored by the full engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from planarena.config import PILLARS, USE_INDEX, USE_KEYS
from planarena.indicators.engine import ADDITIVE, INDICATORS, ImpactEngine

U = USE_INDEX
N_USES = len(USE_KEYS)


@dataclass
class SuggestConfig:
    pillar_weights: dict = field(default_factory=lambda: {p: 1.0 for p in PILLARS})
    indicator_weights: dict | None = None  # optional per-indicator weights (stakeholder view)
    scope: str = "city"
    respect_zoning: bool = True
    protect_landmarks: bool = True
    protect_homes: bool = True  # never remove existing homes
    allow_demolition: bool = False  # built lots may not become open space, parking or vacant
    require_precedent: bool = True  # skip conversions of existing buildings never observed in 2015-2025
    protect_parks: bool = True  # existing open space stays (parkland alienation needs state approval in NY)
    protect_public: bool = True  # existing schools, hospitals and civic facilities stay
    protect_infrastructure: bool = True  # transportation and utility lots stay
    conversion_penalty: float = 0.0005  # overall-score cost per converted parcel (realism prior)
    max_change_share: float = 0.05  # at most this share of scope parcels per plan
    rounds: int = 3
    w_fit: float = 0.002  # weight on GNN log-odds fit, in overall-score units
    min_gain: float = 1e-6
    candidate_parcels: np.ndarray | None = None  # restrict to these parcel indices
    unanimity_weights: list | None = None  # consensus mode: list of agent indicator-weight dicts
    polish: int = 0  # exact re-scoring of the top-K first-order moves after the rounds (0 = off)
    polish_passes: int = 2


class Suggester:
    def __init__(self, engine: ImpactEngine, suitability: np.ndarray | None = None):
        self.e = engine
        self.suit = suitability
        self._per_use = None

    # Per-parcel additive indicator values under every land use. Computed once:
    # each parcel's value under use u does not depend on the other parcels.
    def per_use_values(self) -> dict[str, np.ndarray]:
        if self._per_use is None:
            vals = {k: np.zeros((self.e.n, N_USES)) for k in ADDITIVE + ("services_fa", "pubfac_fa", "residents")}
            for u in range(N_USES):
                st = self.e.parcel_state(np.full(self.e.n, u, dtype="int16"))
                for k in vals:
                    vals[k][:, u] = st[k]
            self._per_use = vals
        return self._per_use

    def _state(self, mask: np.ndarray, use: np.ndarray) -> dict:
        st = self.e.parcel_state(use)
        acc = self.e.access(st)
        return {"state": st, "acc": acc, "base": self.e.aggregate(st, acc, mask)}

    @staticmethod
    def _coefficients(base: dict, pillar_weights: dict, indicator_weights: dict | None) -> dict:
        """Score-units per unit change of each indicator. Matches the engine's scoring."""
        wsum = sum(pillar_weights.values()) or 1.0
        n_in = {p: sum(1 for i in INDICATORS if i.pillar == p) for p in PILLARS}
        coef = {}
        for ind in INDICATORS:
            if indicator_weights is not None:
                w = indicator_weights.get(ind.key, 0.0)
            else:
                w = pillar_weights.get(ind.pillar, 0) / wsum / n_in[ind.pillar]
            b = base[ind.key]
            if ind.kind == "share":
                coef[ind.key] = w * ind.direction * 100.0  # percentage points, as in the engine
            else:
                coef[ind.key] = (w * ind.direction * 100.0 / abs(b)) if b and np.isfinite(b) and abs(b) > 1e-9 else 0.0
        return coef

    def _gain(self, coef: dict, pu: dict, use: np.ndarray, ag: dict) -> np.ndarray:
        rows = np.arange(self.e.n)
        gain = np.zeros((self.e.n, N_USES))
        for k in ADDITIVE:
            if coef.get(k):
                gain += coef[k] * (pu[k] - pu[k][rows, use][:, None])
        for k, g in ag.items():
            if coef.get(k):
                gain += coef[k] * g
        return gain

    def _access_gain(self, use, st, acc, mask, pu) -> dict[str, np.ndarray]:
        """First-order change of the five resident-weighted indicators (park, services and facility access, land-use
        mix, green cover) for each (parcel, use): the change in what residents near the parcel can reach, plus the
        change in who is counted when the parcel's own residents change."""
        e = self.e
        n = e.n
        rows = np.arange(n)
        R = st["residents"] * mask
        Rtot = max(R.sum(), 1.0)
        cell = e.cell
        ny, nx = e.ny, e.nx

        def conv(weights, k):
            r = np.bincount(cell, weights=weights, minlength=nx * ny).reshape(ny, nx)
            return np.clip(fftconvolve(r, k, mode="same").ravel(), 0, None)

        def spread(per_cell, k):
            """Kernel sum of a per-cell quantity, read at each parcel's cell (signed)."""
            return fftconvolve(per_cell.reshape(ny, nx), k, mode="same").ravel()[cell]

        gains = {}
        # Park access (threshold): residents near p who lack access and would cross the threshold.
        open_area = np.where(use == U["opn"], e.lot, 0.0)
        A = conv(open_area, e.k_walk)  # open area within walk, per cell
        Rc = np.bincount(cell, weights=R, minlength=nx * ny)
        lacking = np.where(A < e.park_min, Rc, 0.0).reshape(ny, nx)
        near_lacking = np.clip(fftconvolve(lacking, e.k_walk, mode="same").ravel(), 0, None)[cell]
        has = np.where(A >= e.park_min, Rc, 0.0).reshape(ny, nx)
        near_has = np.clip(fftconvolve(has, e.k_walk, mode="same").ravel(), 0, None)[cell]
        frac_new = np.clip(e.lot / e.park_min, 0, 1)
        g_park = np.zeros((n, N_USES))
        g_park[:, U["opn"]] = np.where(use != U["opn"], near_lacking * frac_new, 0.0)
        losing = (use == U["opn"])
        for u in range(N_USES):
            if u != U["opn"]:
                g_park[:, u] = np.where(losing, -near_has * frac_new * 0.5, 0.0)
        gains["park_access"] = g_park / Rtot

        # Services / facilities adequacy = mean over residents of min(ratio / target, 1), where the ratio is floor
        # area per resident within the radius (all residents, as in the engine). Added floor area helps only
        # residents in under-served cells; added residents dilute the ratio there.
        d_res = pu["residents"] - pu["residents"][rows, use][:, None]

        def adequacy_gain(fa_key, kernel, target):
            RW = conv(st["residents"] * 1.0, kernel)
            FA = conv(st[fa_key] * 1.0, kernel)
            ratio = np.divide(FA, RW, out=np.zeros_like(FA), where=RW > 1)
            under = (ratio < target) & (RW > 1)
            weight = np.divide(Rc, RW, out=np.zeros_like(Rc), where=RW > 1) * under / target
            G = spread(weight, kernel)
            dil = spread(-weight * ratio, kernel)  # d(adequacy)/d(residents nearby)
            d_fa = pu[fa_key] - pu[fa_key][rows, use][:, None]
            return (d_fa * G[:, None] + d_res * dil[:, None]) / Rtot

        gains["services_access"] = adequacy_gain("services_fa", e.k_walk, e.t_serv)
        gains["facility_access"] = adequacy_gain("pubfac_fa", e.k_fac, e.t_fac)

        # Land-use mix (normalised entropy of five land groups within the walk radius) and green cover (open-space
        # share of that land). Moving a lot of area a from group g0 to g1 changes each nearby cell's land by
        # a * (e_g1 - e_g0); the gradient of entropy is -(log s_g + H log G) / (T log G). Parking and vacant land
        # belong to no group. Shares are floored at one median lot so that a first lot of a missing group counts
        # a finite amount.
        groups = [("res",), ("mix", "com"), ("ind", "tra"), ("pub",), ("opn",)]
        gid = np.full(N_USES, -1)
        for j, grp in enumerate(groups):
            for k in grp:
                gid[U[k]] = j
        ng = len(groups)
        g_now = np.where(use >= 0, gid[np.clip(use, 0, None)], -1)
        land = np.stack([conv(np.where(g_now == j, e.lot, 0.0), e.k_walk) for j in range(ng)])
        lot_med = float(np.median(e.lot[mask])) if mask.any() else float(np.median(e.lot))
        T = np.maximum(land.sum(0), lot_med)
        s = land / T
        sf = np.maximum(s, lot_med / T)
        H = -(s * np.log(np.where(s > 0, s, 1.0))).sum(0) / np.log(ng)
        d_mix = np.zeros((ng + 1, n))  # last row: no group
        d_green = np.zeros((ng + 1, n))
        for j in range(ng):
            d_mix[j] = spread(Rc * -(np.log(sf[j]) + H * np.log(ng)) / (T * np.log(ng)), e.k_walk)
            d_green[j] = spread(Rc * ((j == ng - 1) - s[-1]) / T, e.k_walk)
        g_use = np.where(gid >= 0, gid, ng)  # group row per use
        g_cur = np.where(g_now >= 0, g_now, ng)
        a = e.lot[:, None]
        gains["mix"] = a * (d_mix[g_use].T - d_mix[g_cur, rows][:, None]) / Rtot
        gains["green_cover"] = a * (d_green[g_use].T - d_green[g_cur, rows][:, None]) / Rtot

        # Who is counted: a parcel's own residents enter every resident-weighted mean with its own value.
        own = {"park_access": acc["park_access"].astype(float), "services_access": acc["services_access"],
               "facility_access": acc["facility_access"], "mix": acc["mix"], "green_cover": acc["green_cover"]}
        dR = d_res * mask[:, None]
        for k, v in own.items():
            mean = float((v * R).sum() / Rtot)
            gains[k] = gains[k] + dR * (v - mean)[:, None] / Rtot
        return gains

    def feasible(self, cfg: SuggestConfig, use: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(scope mask, eligible parcels, allowed uses per parcel) under the plan's protections."""
        e = self.e
        mask = e.scope_mask(use, cfg.scope)
        eligible = mask.copy()
        if cfg.candidate_parcels is not None:
            sel = np.zeros(e.n, bool)
            sel[cfg.candidate_parcels] = True
            eligible &= sel
        if cfg.protect_landmarks:
            eligible &= ~e.protected
        if cfg.protect_parks:
            eligible &= use != U["opn"]
        if cfg.protect_public:
            eligible &= use != U["pub"]
        if cfg.protect_infrastructure:
            eligible &= use != U["tra"]
        eligible &= use >= 0
        allowed = np.ones((e.n, N_USES), bool)
        if cfg.respect_zoning:
            allowed = ((e.allowed[:, None] >> np.arange(N_USES)[None, :]) & 1).astype(bool)
        if cfg.protect_homes:
            homes = e.obs_units > 0
            allowed[homes] &= np.isin(np.arange(N_USES), [U["res"], U["mix"]])[None, :].repeat(homes.sum(), 0)
        if not cfg.allow_demolition:
            built = e.bldg > 0
            unbuilt_uses = np.isin(np.arange(N_USES), [U["opn"], U["prk"], U["vac"]])
            allowed[built[:, None] & unbuilt_uses[None, :]] = False
        if cfg.require_precedent and e.precedent is not None:
            built = np.nonzero(eligible & (e.bldg > 0))[0]
            for u in range(N_USES):
                allowed[built, u] &= e.has_precedent(built, np.full(len(built), u))
        return mask, eligible, allowed

    def suggest(self, cfg: SuggestConfig | None = None, use: np.ndarray | None = None) -> dict:
        cfg = cfg or SuggestConfig()
        e = self.e
        use = (e.base_use if use is None else np.asarray(use)).astype("int16").copy()
        start = use.copy()
        pu = self.per_use_values()
        mask, eligible, allowed = self.feasible(cfg, use)
        budget = max(1, int(cfg.max_change_share * eligible.sum()))
        fit = None
        if self.suit is not None and cfg.w_fit > 0:
            lp = np.log(np.clip(self.suit, 1e-4, 1))
            fit = lp
        rows = np.arange(e.n)
        history = []
        for rnd in range(cfg.rounds):
            ctx = self._state(mask, use)
            ag = self._access_gain(use, ctx["state"], ctx["acc"], mask, pu)
            if cfg.unanimity_weights:
                # Consensus mode: only moves that improve every agent's utility (first-order).
                per_agent = [self._gain(self._coefficients(ctx["base"], cfg.pillar_weights, w), pu, use, ag)
                             for w in cfg.unanimity_weights]
                stack = np.stack(per_agent)
                gain = np.where(np.all(stack > 0, axis=0), stack.sum(axis=0), -np.inf)
            else:
                coef = self._coefficients(ctx["base"], cfg.pillar_weights, cfg.indicator_weights)
                gain = self._gain(coef, pu, use, ag)
            if fit is not None:
                gain += cfg.w_fit * (fit - fit[rows, use][:, None])
            gain -= cfg.conversion_penalty
            gain[~allowed] = -np.inf
            gain[~eligible] = -np.inf
            gain[rows, use] = 0.0
            best = gain.argmax(1)
            bgain = gain[rows, best]
            remaining = budget - int((use != start).sum())
            cand = np.nonzero((bgain > cfg.min_gain) & (best != use))[0]
            if remaining <= 0 or not len(cand):
                break
            take = cand[np.argsort(-bgain[cand])][: max(1, remaining // max(1, cfg.rounds - rnd))]
            use[take] = best[take]
            history.append({"round": rnd + 1, "changed": int(len(take)), "est_gain": float(bgain[take].sum())})
        if cfg.polish and not cfg.unanimity_weights and "gain" in locals():
            use = self._polish(cfg, use, start, gain, budget)
        ev = e.evaluate(use, scope=cfg.scope, pillar_weights=cfg.pillar_weights)
        changed = np.nonzero(use != start)[0]
        return {"use": use, "changed": changed, "evaluation": ev, "history": history,
                "budget": budget, "eligible": int(eligible.sum())}

    def _polish(self, cfg: SuggestConfig, use: np.ndarray, start: np.ndarray, gain: np.ndarray,
                budget: int) -> np.ndarray:
        """Exact local search: try the top-K first-order moves (add, or swap out the weakest change) and
        dropping changes, re-scored by the full engine; keep the best improvement, up to polish_passes times.

        The first-order gains miss saturating access effects on very large parcels; this step corrects
        them at the cost of a few hundred exact evaluations.
        """
        e = self.e
        g = gain.copy()
        g[np.arange(e.n), start] = -np.inf
        flat = np.argsort(-g, axis=None)[: cfg.polish]
        moves = [(int(i // N_USES), int(i % N_USES)) for i in flat if np.isfinite(g.flat[i])]

        def J(u: np.ndarray) -> float:
            ev = e.evaluate(u, scope=cfg.scope, pillar_weights=cfg.pillar_weights)
            if cfg.indicator_weights is not None:
                val = sum(cfg.indicator_weights.get(r["key"], 0.0) * r["score"] for r in ev["indicators"])
            else:
                val = ev["overall"]
            return val - cfg.conversion_penalty * int((u != start).sum())

        cur = J(use)
        for _ in range(cfg.polish_passes):
            changed = np.nonzero(use != start)[0]
            best_val, best_use = cur, None
            for q in sorted(changed, key=lambda q: gain[q, use[q]])[:15]:  # the 15 weakest changes
                t = use.copy()
                t[q] = start[q]
                v = J(t)
                if v > best_val + 1e-9:
                    best_val, best_use = v, t
            for p, u in moves:
                if use[p] == u:
                    continue
                t = use.copy()
                t[p] = u
                if int((t != start).sum()) > budget:
                    rest = [q for q in changed if q != p]
                    if not rest:
                        continue
                    weakest = min(rest, key=lambda q: gain[q, use[q]])
                    t[weakest] = start[weakest]
                v = J(t)
                if v > best_val + 1e-9:
                    best_val, best_use = v, t
            if best_use is None:
                break
            use, cur = best_use, best_val
        return use
