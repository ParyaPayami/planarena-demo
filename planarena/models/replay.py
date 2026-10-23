"""Integration value: are likelihood-weighted projections closer to what was built than capacity-based ones?

For each Manhattan rezoning (East Harlem 2017, Inwood 2018, SoHo/NoHo 2021):
* Sites: PlanArena's CEQR-style development-site screen under the adopted maximum FAR, on the snapshot of the
  adoption year (before adoption).
* Capacity projection: each site built to the new FAR at the build-out utilisation (homes added per site); also
  pro-rated linearly from the review's build year to 2025.
* Likelihood-weighted projection: the same homes per site times the site's probability of redevelopment by 2025,
  from the redevelopment model trained only on snapshot pairs whose outcome was known by the adoption year, scored on
  the adoption-year snapshot with the adopted zoning (FARs and district) in place. The longest horizon with training
  pairs (1, 3 or 5 years, at most the years to 2025) is used and extended to 2025 with a constant hazard.
* Likelihood x empirical yield: a redeveloped site seldom reaches capacity. In the same training pairs, soft sites
  under their existing zoning (the same screen without the FAR increase) that were redeveloped added a share of their
  capacity homes; the projection multiplies each site's probability, its capacity and the mean of that share. Its 95%
  range draws redevelopment from the probabilities and, for each redeveloped site, a share from the training sites.
* Observed: homes added by 2025 on each site's land (2025 lots whose representative point lies inside the site) and
  whether the site was redeveloped (the model's label definition).
* Also reported: how well each ranking of sites (probability, or homes added) finds the sites that were redeveloped.

No human participants.  Run:  python -I -m planarena.models.replay
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from planarena.config import PROCESSED, ROOT, USE_INDEX as U
from planarena.models.common import SEED

OUT = ROOT / "results" / "validation" / "integration_replay.json"
REZONINGS = {"East Harlem (2017)": {"cd": 111, "year": 2017, "review_year": 2027},
             "Inwood (2018)": {"cd": 112, "year": 2018, "review_year": 2033},
             "SoHo/NoHo (2021)": {"cd": 102, "year": 2021, "review_year": 2031}}
END = 2025


def _model(year: int):
    """LightGBM trained on pairs with outcomes known by ``year``; returns (model, horizon)."""
    from planarena.models import rolling

    T = END - year
    for h in (5, 3, 1):
        if h > T:
            continue
        pairs = [(a, a + h) for a in rolling.YEARS if a + h <= year]
        if pairs:
            X, y = rolling._pool(pairs)
            return rolling._fit(X, y, SEED), h, pairs
    raise ValueError(f"no training pairs for {year}")


_YIELD: dict = {}


def _site_homes(e0, idx, year1: int) -> np.ndarray:
    """Homes on each site's land at ``year1`` (lots whose representative point lies inside the site) minus homes at t0."""
    import geopandas as gpd
    import shapely

    from planarena.models.history import snapshot_dir

    d1, g0 = snapshot_dir(year1), e0.geometry
    p1 = pd.read_parquet(d1 / "parcels.parquet", columns=["units_res"])
    pts1 = gpd.read_parquet(d1 / "parcels.geoparquet").geometry.representative_point().values
    tree = shapely.STRtree(pts1)
    return np.array([p1.units_res.iloc[tree.query(g0[i], predicate="contains")].sum() for i in idx], float) \
        - e0.p.units_res.to_numpy()[idx]


def yield_shares(pairs) -> np.ndarray:
    """Homes added as a share of capacity homes, for soft sites redeveloped in the training pairs (existing zoning)."""
    import geopandas as gpd

    from planarena.indicators import rwcds
    from planarena.indicators.engine import load_engine
    from planarena.models.history import snapshot_dir
    from planarena.models.rolling import _y
    from planarena.twin.zoning import allowed_uses

    out = []
    for a, b in pairs:
        if (a, b) not in _YIELD:
            e = load_engine(snapshot_dir(a))
            e.geometry = gpd.read_parquet(snapshot_dir(a) / "parcels.geoparquet").geometry.values
            res_ok = np.array([("res" in allowed_uses(z)) for z in e.p.zone], bool)
            mx = e.p["max_far"].to_numpy()
            sites = rwcds.screen(e, mx, res_ok & (mx > 0), a, min_increase=0.0)
            lab = _y(a, b)
            red = sites & lab.developed.to_numpy()
            idx = np.nonzero(red)[0]
            util = e.coef["assumptions"]["buildout_utilization"]["value"]
            use = e.base_use.copy()
            far = np.full(e.n, np.nan)
            use[idx], far[idx] = U["mix"], mx[idx] * util
            cap = e.parcel_state(use, far)["units"][idx] - e.baseline_state["units"][idx]
            got = _site_homes(e, idx, b)
            ok = cap > 0
            _YIELD[(a, b)] = np.clip(got[ok], 0, None) / cap[ok]
        out.append(_YIELD[(a, b)])
    return np.concatenate(out)


def replay(name: str, cfg: dict) -> dict:
    import geopandas as gpd

    from planarena.indicators import rwcds
    from planarena.indicators.engine import load_engine
    from planarena.models.history import snapshot_dir
    from planarena.models.rolling import _y
    from planarena.models.transition import _auc, features
    from planarena.twin.zoning import allowed_mask, allowed_uses

    R = cfg["year"]
    d0 = snapshot_dir(R)
    e0 = load_engine(d0)
    p25 = pd.read_parquet(PROCESSED / "parcels.parquet")
    b = e0.p[["bbl", "zone", "overlay", "cd"]].reset_index(names="i0").merge(
        p25[["bbl", "zone", "overlay", "max_far", "resid_far", "comm_far", "facil_far"]], on="bbl", suffixes=("_0", "_1"))
    b = b[b.cd == cfg["cd"]]
    rez = b[b.zone_0 != b.zone_1]
    i0 = rez.i0.to_numpy()
    new_max = e0.p["max_far"].to_numpy().copy()
    new_max[i0] = rez.max_far.to_numpy()
    area = np.zeros(e0.n, bool)
    area[i0] = True
    res_ok = np.zeros(e0.n, bool)
    res_ok[i0] = [("res" in allowed_uses(z)) for z in rez.zone_1]
    sites = rwcds.screen(e0, new_max, area, R)
    idx = np.nonzero(sites)[0]
    util = e0.coef["assumptions"]["buildout_utilization"]["value"]
    use = e0.base_use.copy()
    far = np.full(e0.n, np.nan)
    use[idx] = np.where(res_ok[idx], U["mix"], U["com"])
    far[idx] = new_max[idx] * util
    st = e0.parcel_state(use, far)
    added = st["units"][idx] - e0.baseline_state["units"][idx]

    # probability of redevelopment under the adopted zoning
    plan = e0.p.copy()
    plan.loc[i0, "zone"] = rez.zone_1.to_numpy()
    plan.loc[i0, "overlay"] = rez.overlay_1.fillna("").to_numpy()
    for c in ("max_far", "resid_far", "comm_far", "facil_far"):
        plan.loc[i0, c] = rez[c].to_numpy()
    plan["allowed"] = [allowed_mask(z, o) for z, o in zip(plan.zone, plan.overlay)]
    graph = dict(np.load(d0 / "graph.npz"))
    m, h, pairs = _model(R)
    T = END - R
    p_h = m.predict_proba(features(plan, graph, R)[idx])[:, 1]
    p_T = 1 - (1 - p_h) ** (T / h)
    p_now = m.predict_proba(features(e0.p, graph, R)[idx])[:, 1]  # same sites, zoning before adoption

    # observed by 2025
    lab = _y(R, END)
    redeveloped = lab.developed.to_numpy()[idx]
    e0.geometry = gpd.read_parquet(d0 / "parcels.geoparquet").geometry.values
    obs = _site_homes(e0, idx, END)
    cap = float(added.sum())
    share = yield_shares(pairs)
    rng = np.random.default_rng(SEED)
    draws = 20000
    built = rng.random((draws, len(idx))) < p_T
    sh = share[rng.integers(0, len(share), (draws, len(idx)))] if len(share) else np.zeros((draws, len(idx)))
    yld = (built * sh * added).sum(1)
    prorate = (END - R) / (cfg["review_year"] - R)
    out = {
        "adopted": R, "years_to_2025": T, "sites": int(len(idx)), "rezoned_lots": int(len(rez)),
        "model": {"horizon_years": h, "training_pairs": [f"{a}->{c}" for a, c in pairs], "extended_to_2025": T != h},
        "homes_added_by_2025": {
            "observed": float(obs.sum()),
            "capacity": round(cap), "capacity_prorated_to_2025": round(cap * prorate),
            "likelihood_weighted": round(float((p_T * added).sum())),
            # each site redevelops independently with its probability: 2.5th-97.5th percentile of 20,000 draws
            "likelihood_weighted_95pct": [round(float(v)) for v in np.percentile(
                (np.random.default_rng(SEED).random((20000, len(idx))) < p_T) @ added, [2.5, 97.5])],
            "likelihood_x_yield": round(float((p_T * added).sum() * (share.mean() if len(share) else 0.0))),
            "likelihood_x_yield_95pct": [round(float(v)) for v in np.percentile(yld, [2.5, 97.5])],
        },
        "yield_share_of_capacity": {"training_sites_redeveloped": int(len(share)),
                                    "mean": round(float(share.mean()), 3) if len(share) else None,
                                    "median": round(float(np.median(share)), 3) if len(share) else None,
                                    "share_with_no_homes": round(float((share <= 0).mean()), 3) if len(share) else None},
        "sites_redeveloped_by_2025": {"observed": int(redeveloped.sum()), "expected_from_probabilities": round(float(p_T.sum()), 1)},
        "mean_probability_adopted_vs_prior_zoning": [round(float(p_h.mean()), 4), round(float(p_now.mean()), 4)],
    }
    if 0 < redeveloped.sum() < len(idx):
        y = redeveloped.astype(bool)
        out["ranking_auc"] = {"probability": round(_auc(p_T, y), 3), "homes_added": round(_auc(added, y), 3)}
    a = out["homes_added_by_2025"]
    out["error_vs_observed_homes"] = {k: round(a[k] - a["observed"]) for k in
                                      ("capacity", "capacity_prorated_to_2025", "likelihood_weighted", "likelihood_x_yield")}
    out["ratio_to_observed"] = {k: (round(a[k] / a["observed"], 2) if a["observed"] > 0 else None) for k in
                                ("capacity", "capacity_prorated_to_2025", "likelihood_weighted", "likelihood_x_yield")}
    return out


def main() -> dict:
    res = {"test": __doc__.strip().split("\n\n")[1], "rezonings": {}}
    for name, cfg in REZONINGS.items():
        res["rezonings"][name] = r = replay(name, cfg)
        print(name, r["homes_added_by_2025"], r["sites_redeveloped_by_2025"], r.get("ranking_auc"), flush=True)
    tot = {k: sum(r["homes_added_by_2025"][k] for r in res["rezonings"].values())
           for k in ("observed", "capacity", "capacity_prorated_to_2025", "likelihood_weighted", "likelihood_x_yield")}
    res["sum_of_absolute_errors"] = {k: sum(abs(r["error_vs_observed_homes"][k]) for r in res["rezonings"].values())
                                     for k in ("capacity", "capacity_prorated_to_2025", "likelihood_weighted",
                                               "likelihood_x_yield")}
    res["observed_within_model_interval"] = {n: r["homes_added_by_2025"]["likelihood_weighted_95pct"][0]
                                             <= r["homes_added_by_2025"]["observed"]
                                             <= r["homes_added_by_2025"]["likelihood_weighted_95pct"][1]
                                             for n, r in res["rezonings"].items()}
    res["observed_within_yield_interval"] = {n: r["homes_added_by_2025"]["likelihood_x_yield_95pct"][0]
                                             <= r["homes_added_by_2025"]["observed"]
                                             <= r["homes_added_by_2025"]["likelihood_x_yield_95pct"][1]
                                             for n, r in res["rezonings"].items()}
    res["caveats"] = ["Three rezonings; few sites were redeveloped by 2025, so site-level rankings are not informative "
                      "where fewer than five were.", "Capacity projections describe full build-out by the review's build "
                      "year; the comparison is with what was built by 2025, which is the near-term question.",
                      "Homes are counted from MapPLUTO units on each site's land; Mandatory Inclusionary Housing bonus floor "
                      "area is not modelled."]
    res["total_three_rezonings"] = tot
    res["closest_projection_by_rezoning"] = {n: min(("capacity", "capacity_prorated_to_2025", "likelihood_weighted",
                                                     "likelihood_x_yield"),
                                                    key=lambda k: abs(r["error_vs_observed_homes"][k]))
                                             for n, r in res["rezonings"].items()}
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(main()["total_three_rezonings"]))
