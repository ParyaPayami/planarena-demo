"""The paper, with every number and table taken from the result files (like planarena/report.py for the docs).

paper/paper.tex.in holds the text; each @@name@@ in it is filled from values() or tables() below, which read only
results/validation/*.json. A test fails if paper/paper.tex is out of date, if a placeholder is left unfilled, or if a
citation is not in docs/refs.bib.

Run:  python -I -m planarena.paper            -> paper/paper.tex (and paper/paper.pdf when tectonic is installed)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import numpy as np

from planarena.config import ROOT

RES = ROOT / "results" / "validation"
PAPER = ROOT / "paper"
TEMPLATE = PAPER / "paper.tex.in"
OUT = PAPER / "paper.tex"

# the test cities, in the order of the tables
CITY = {"0644000": "Los Angeles", "4819000": "Dallas", "3651000": "New York", "5553000": "Milwaukee",
        "0455000": "Phoenix", "0666000": "San Diego", "1714000": "Chicago"}
OUTCOME = {"0644000": "new building or floor area +25\\%", "4819000": "new building or floor area +25\\%",
           "3651000": "new building or floor area +25\\%", "5553000": "new building",
           "0455000": "change of use group", "0666000": "change of use group", "1714000": "new-construction permit"}
OUTCOME_SOURCE = {"0644000": "county assessor rolls", "4819000": "appraisal district files", "3651000": "MapPLUTO",
                  "5553000": "MPROP", "0455000": "MAG land-use inventory", "0666000": "SANDAG land-use inventory",
                  "1714000": "building permits"}
# the ten cities: short source descriptions (README, "Cities")
TEN = {"3651000": ("New York", "MapPLUTO (City Planning)", "none citywide"),
       "0644000": ("Los Angeles", "SCAG 2020; county assessor", "General Plan (SCAG 2019)"),
       "1714000": ("Chicago", "CMAP 2023; county assessor", "none citywide"),
       "4835000": ("Houston", "City (appraisal codes)", "none"),
       "0455000": ("Phoenix", "County assessor; MAG", "General Plan"),
       "4260000": ("Philadelphia", "City Planning Commission; OPA", "district plans (digitised PDF)"),
       "4865000": ("San Antonio", "Bexar Appraisal District", "adopted area plans"),
       "0666000": ("San Diego", "SANDAG 2025; county assessor", "General and community plans"),
       "4819000": ("Dallas", "City 2026; DCAD", "forwardDallas! 2.0"),
       "5553000": ("Milwaukee", "MPROP", "PDF only (not included)")}


def _load(name: str):
    for f in (RES / f"{name}.json", RES.parent / f"{name}.json"):
        if f.exists():
            return json.loads(f.read_text())
    raise FileNotFoundError(name)


def _f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def _sg(x: float, d: int = 3) -> str:
    if round(abs(x), d) == 0:
        return f"{0:.{d}f}"
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.{d}f}"


def _num(x: float, d: int = 2) -> str:
    """A signed number with a typographic minus, and no negative zero."""
    if round(abs(x), d) == 0:
        return f"{0:.{d}f}"
    return ("$-$" if x < 0 else "") + f"{abs(x):.{d}f}"


WORDS = "zero one two three four five six seven eight nine ten".split()


def _w(n: int) -> str:
    return WORDS[n] if 0 <= n <= 10 else str(n)


def _of(k: int, n: int) -> str:
    """'four of seven', or 'all five' when every case qualifies."""
    return f"all {_w(n)}" if k == n else f"{_w(k)} of {_w(n)}"


def _ratio(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def _pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}\\%"


def _times(x: float) -> str:
    return f"{x:.2f}$\\times$" if x < 10 else f"{x:.0f}$\\times$"


def _list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _gain(g: dict | None, key: str):
    """(value, low, high, significant) of a gain in AUC, or None."""
    if not g or key not in g:
        return None
    v, (lo, hi) = g[key]["auc"]
    return v, lo, hi, lo > 0 or hi < 0


def _gtex(t) -> str:
    if t is None:
        return "--"
    v, lo, hi, sig = t
    s = f"{_sg(v)} [{_f(lo) if lo >= 0 else _sg(lo)}, {_f(hi) if hi >= 0 else _sg(hi)}]"
    return f"\\textbf{{{s}}}" if sig else s


def _data():
    integ = _load("integration_cities")
    cities = [g for g in CITY if g in integ]
    return integ, cities


# ----------------------------------------------------------------------------- values
def values() -> dict[str, str]:
    integ, cities = _data()
    T = {g: integ[g]["temporal"] for g in cities}
    layers = _load("cities_layers")
    V: dict[str, str] = {}

    # ten cities
    parcels = sum(v["parcels"] for v in layers.values())
    V["n_cities"] = str(len(layers))
    V["parcels_total"] = _n(parcels)
    V["parcels_total_m"] = f"{parcels / 1e6:.1f}"
    ratios = [v["homes_twin"] / v["homes_census_2020"] for v in layers.values() if v.get("homes_census_2020")]
    V["homes_ratio_lo"], V["homes_ratio_hi"] = _f(min(ratios), 2), _f(max(ratios), 2)
    with_plan = [g for g, v in layers.items() if v["plan"] > 0]
    with_zone = [g for g, v in layers.items() if v["zoning"] > 0]
    V["n_with_plan"], V["n_with_zoning"] = str(len(with_plan)), str(len(with_zone))
    match = [v["plan_matches_existing_group"] for g, v in layers.items() if v.get("plan_matches_existing_group")]
    V["plan_match_lo"], V["plan_match_hi"] = _pct(min(match)), _pct(max(match))

    # Philadelphia's digitised plan
    phl = layers["4260000"]
    dig = phl["plan_digitised"]
    used = {k: v for k, v in dig.items() if v.get("used")}
    n_plans = sum(len(k.split(" + ")) for k in dig)  # districts
    V["phl_districts"] = str(n_plans)
    V["phl_districts_used"] = str(sum(len(k.split(" + ")) for k in used))
    V["phl_maps"], V["phl_maps_used"] = str(len(dig)), str(len(used))
    V["phl_not_used"] = _list(sorted(k for k, v in dig.items() if not v.get("used")))
    V["phl_share"] = _pct(phl["plan"])
    err = [v["median_error_ft"] for v in used.values()]
    V["phl_err_lo"], V["phl_err_hi"] = _f(min(err), 1), _f(max(err), 1)
    V["phl_err_median"] = _f(float(np.median(err)), 1)
    cov = [v["coverage"] for v in used.values()]
    V["phl_cov_lo"], V["phl_cov_hi"] = _pct(min(cov)), _pct(max(cov))
    ag = [v["agreement_with_existing"] for v in used.values()]
    sh = [v["agreement_if_shifted_300ft"] for v in used.values()]
    V["phl_agree_lo"], V["phl_agree_hi"] = _pct(min(ag)), _pct(max(ag))
    V["phl_shift_lo"], V["phl_shift_hi"] = _pct(min(sh)), _pct(max(sh))
    V["phl_margin_min"] = _f(100 * min(a - b for a, b in zip(ag, sh)), 0)
    chk = _load("philadelphia_plan_check")["overall"]
    V["phl_pix_n"], V["phl_pix_agree"] = _n(chk["parcels_read_both_ways"]), _pct(chk["agreement"], 1)
    V["phl_pix_lo"], V["phl_pix_hi"] = _pct(chk["agreement_range"][0], 1), _pct(chk["agreement_range"][1], 1)
    V["phl_manual_n"] = _n(chk["manual_sample"])

    # test panels
    V["n_test"] = str(len(cities))
    V["test_lots"] = _n(sum(T[g]["n_lots"] for g in cities))
    V["test_developed"] = _n(sum(T[g]["developed"] for g in cities))
    br = [T[g]["sets"]["E"]["base_rate"] for g in cities]
    V["base_lo"], V["base_hi"] = _pct(min(br), 2), _pct(max(br), 1)
    aucE = [T[g]["sets"]["E"]["auc"] for g in cities]
    V["auc_E_lo"], V["auc_E_hi"] = _f(min(aucE)), _f(max(aucE))
    V["auc_E_lo_city"] = CITY[cities[int(np.argmin(aucE))]]
    rest = [a for a in aucE if a != min(aucE)]
    V["auc_E_rest_lo"] = _f(min(rest))
    full = {g: T[g]["sets"].get("E+C+Z+P", T[g]["sets"].get("E+C+Z"))["auc"] for g in cities}
    V["auc_full_lo"], V["auc_full_hi"] = _f(min(full.values())), _f(max(full.values()))

    # gains over the out-of-time test
    ctx = {g: _gain(T[g]["gains"], "E+C vs E") for g in cities}
    sig_pos = lambda d: [g for g, t in d.items() if t and t[3] and t[0] > 0]
    sig_neg = lambda d: [g for g, t in d.items() if t and t[3] and t[0] < 0]
    c_sig = sig_pos(ctx)
    V["n_ctx_sig"] = str(len(c_sig))
    V["ctx_sig_cities"] = _list([CITY[g] for g in c_sig])
    V["ctx_gain_lo"], V["ctx_gain_hi"] = _f(min(ctx[g][0] for g in c_sig)), _f(max(ctx[g][0] for g in c_sig))
    c_ns = [g for g in cities if g not in c_sig]
    V["ctx_ns_cities"] = _list([CITY[g] for g in c_ns])
    phx = ctx.get("0455000")
    V["phx_ctx"], V["phx_ctx_lo"], V["phx_ctx_hi"] = _sg(phx[0]), _sg(phx[1]), _sg(phx[2])
    zg = {g: _gain(T[g]["gains"], "E+C+Z vs E+C") for g in cities}
    pg = {g: _gain(T[g]["gains"], "E+C+P vs E+C") for g in cities}
    zg = {g: t for g, t in zg.items() if t}
    pg = {g: t for g, t in pg.items() if t}
    z_sig, p_sig = sig_pos(zg), sig_pos(pg)
    V["n_z"], V["n_p"] = str(len(zg)), str(len(pg))
    V["n_z_sig"], V["n_p_sig"] = str(len(z_sig)), str(len(p_sig))
    V["z_sig_cities"], V["p_sig_cities"] = _list([CITY[g] for g in z_sig]), _list([CITY[g] for g in p_sig])
    V["z_sig_lo"], V["z_sig_hi"] = _f(min(zg[g][0] for g in z_sig)), _f(max(zg[g][0] for g in z_sig))
    V["p_sig_lo"], V["p_sig_hi"] = _f(min(pg[g][0] for g in p_sig)), _f(max(pg[g][0] for g in p_sig))
    small = [zg[g][0] for g in z_sig if g != "0455000"] + [pg[g][0] for g in p_sig if g != "0455000"]
    V["zp_small_max"] = _f(max(small))
    V["phx_z"], V["phx_p"] = _f(zg["0455000"][0]), _f(pg["0455000"][0])
    V["z_ns_cities"] = _list([CITY[g] for g in zg if g not in z_sig])
    V["p_ns_cities"] = _list([CITY[g] for g in pg if g not in p_sig])
    V["n_neg_sig"] = str(len(sig_neg(zg)) + len(sig_neg(pg)))
    la = T["0644000"]["gains"]
    V["la_zp"] = _sg(la["E+C+Z+P vs E+C"]["auc"][0])
    # best block set per city (out of time)
    best = {g: max(T[g]["sets"], key=lambda k: T[g]["sets"][k]["auc"]) for g in cities}
    V["n_best_has_C"] = str(sum("C" in best[g] for g in cities))
    phx_sets = T["0455000"]["sets"]
    V["phx_best_set"] = best["0455000"].replace("+", "{+}")
    V["phx_best_auc"] = _f(phx_sets[best["0455000"]]["auc"])

    # zoning and plan without context, and what distinguishes developed lots
    zE = {g: _gain(T[g]["gains"], "E+Z vs E") for g in cities}
    zE = {g: t for g, t in zE.items() if t}
    zE_sig = sig_pos(zE)
    V["n_zE_sig"], V["n_zE"] = str(len(zE_sig)), str(len(zE))
    V["zE_lo"], V["zE_hi"] = _f(min(zE[g][0] for g in zE_sig)), _f(max(zE[g][0] for g in zE_sig))
    prof = _load("integration_profile")
    zr = [prof[g]["zone_differs"]["developed"] / prof[g]["zone_differs"]["all"] for g in cities if "zone_differs" in prof[g]]
    vr = [prof[g]["vacant"]["developed"] / prof[g]["vacant"]["all"] for g in cities]
    V["zdiff_ratio_lo"], V["zdiff_ratio_hi"] = _ratio(min(zr)), _ratio(max(zr))
    V["vac_ratio_lo"], V["vac_ratio_hi"] = _ratio(min(vr)), _ratio(max(vr))
    ph = prof["0455000"]
    V["phx_vac_dev"], V["phx_vac_all"] = _pct(ph["vacant"]["developed"]), _pct(ph["vacant"]["all"])
    V["phx_zd_dev"], V["phx_zd_all"] = _pct(ph["zone_differs"]["developed"]), _pct(ph["zone_differs"]["all"])
    V["phx_pd_dev"], V["phx_pd_all"] = _pct(ph["plan_differs"]["developed"]), _pct(ph["plan_differs"]["all"])
    V["mke_dev"] = _n(T["5553000"]["developed"])
    V["existing_min"] = _pct(min(v["existing"] for v in layers.values()))
    V["neg_sentence"] = ("No layer significantly reduced performance once context was included."
                         if not sig_neg(zg) and not sig_neg(pg) else "")

    # conservative variant
    cons = {g: integ[g]["temporal_conservative"] for g in cities if integ[g].get("temporal_conservative")}
    V["n_cons"] = str(len(cons))
    V["cons_cities"] = _list([CITY[g] for g in cons])
    held, lost = {}, {}
    for g, c in cons.items():
        for key, main, name in (("E+C+Z vs E+C", zg, "zoning"), ("E+C+P vs E+C", pg, "plan")):
            if g in main and main[g][3] and main[g][0] > 0:
                t = _gain(c["gains"], key)
                (held if t and t[3] and t[0] > 0 else lost).setdefault(g, []).append(name)
    phrase = lambda d: _list([f"the {' and '.join(v)} gain{'s' if len(v) > 1 else ''} in {CITY[g]}" for g, v in d.items()])
    cap = lambda x: x[:1].upper() + x[1:]
    V["cons_held"], V["cons_lost"] = cap(phrase(held)), cap(phrase(lost))
    V["n_cons_lost"] = str(sum(len(v) for v in lost.values()))

    # spatial cross-fitting
    sp = {g: integ[g]["spatial"]["sets"] for g in cities}
    V["n_ctx_spatial_pos"] = str(sum(sp[g]["E+C"]["auc"] > sp[g]["E"]["auc"] for g in cities))
    zsp = [g for g in cities if "E+C+Z" in sp[g]]
    V["n_z_spatial_pos"] = str(sum(sp[g]["E+C+Z"]["auc"] > sp[g]["E+C"]["auc"] for g in zsp))
    V["n_z_spatial"] = str(len(zsp))

    # destination
    D = {g: T[g].get("destination") for g in cities if T[g].get("destination") and T[g]["destination"]["n"]}
    V["n_dest"] = str(len(D))
    better_z = [g for g, d in D.items() if d["layers"].get("zoning class", {}).get("coverage")
                and d["layers"]["zoning class"]["class"] > d["layers"]["existing (no change)"]["class"]]
    V["n_dest_z_better"] = str(len(better_z))
    V["n_dest_z"] = str(sum(1 for d in D.values() if d["layers"].get("zoning class", {}).get("coverage")))
    dz = [D[g]["layers"]["zoning class"]["class"] - D[g]["layers"]["existing (no change)"]["class"] for g in better_z]
    V["dest_z_lo"], V["dest_z_hi"] = _f(100 * min(dz), 0), _f(100 * max(dz), 0)
    dal = D["4819000"]["layers"]
    V["dal_plan_dest"], V["dal_zone_dest"] = _pct(dal["plan class"]["class"]), _pct(dal["zoning class"]["class"])
    V["dal_none_dest"] = _pct(dal["existing (no change)"]["class"])
    mod = {g: d["models"].get("E+C+Z+P") or d["models"].get("E+Z") for g, d in D.items()}
    best_layer = {g: max(v["class"] for k, v in d["layers"].items() if v.get("coverage")) for g, d in D.items()}
    V["n_model_beats_layers"] = str(sum(mod[g]["class"] >= best_layer[g] for g in D))

    # homes
    S = {g: T[g]["scenario"] for g in cities if T[g].get("scenario")}
    V["n_homes"] = str(len(S))
    V["homes_cities"] = _list([CITY[g] for g in S])
    bo = [S[g][k]["ratio_total"] for g in S for k in ("zoning_buildout", "plan_buildout") if k in S[g]]
    lk = [S[g][k]["ratio_total"] for g in S for k in ("likely_zoning", "likely_plan") if k in S[g]]
    V["bo_lo"], V["bo_hi"] = f"{min(bo):.0f}", f"{max(bo):.0f}"
    V["lk_lo"], V["lk_hi"] = _f(min(lk), 2), _f(max(lk), 2)
    tr_rho = [S[g]["trend"]["district_spearman"] for g in S]
    lk_rho = [max(S[g][k]["district_spearman"] for k in ("likely_zoning", "likely_plan") if k in S[g]) for g in S]
    V["trend_rho_lo"], V["trend_rho_hi"] = _num(min(tr_rho)), _num(max(tr_rho))
    V["lk_rho_lo"], V["lk_rho_hi"] = _num(min(lk_rho)), _num(max(lk_rho))
    V["n_trend_better"] = str(sum(t > l for t, l in zip(tr_rho, lk_rho)))
    ly_rho = [S[g]["likely_yield"]["district_spearman"] for g in S if "likely_yield" in S[g]]
    V["ly_rho_lo"], V["ly_rho_hi"] = _num(min(ly_rho)), _num(max(ly_rho))
    V["n_yield_vs_trend"] = str(sum(S[g]["likely_yield"]["district_spearman"] >= S[g]["trend"]["district_spearman"]
                                    for g in S if "likely_yield" in S[g]))
    tr_tot = [S[g]["trend"]["ratio_total"] for g in S]
    V["trend_tot_lo"], V["trend_tot_hi"] = _f(min(tr_tot), 2), _f(max(tr_tot), 2)

    # transfer
    tr = integ["transfer"]
    tc = [g for g in CITY if g in tr]
    worse = [g for g in tc if tr[g]["E+C+Z+P"]["other_cities"]["auc"] < tr[g]["E+C+Z+P"]["own_city"]["auc"]]
    V["n_transfer"], V["n_transfer_worse"] = str(len(tc)), str(len(worse))
    V["transfer_better_city"] = _list([CITY[g] for g in tc if g not in worse]) if len(worse) < len(tc) else "none"
    ctx_t = [tr[g]["E+C"]["other_cities"]["auc"] - tr[g]["E"]["other_cities"]["auc"] for g in tc]
    V["n_transfer_ctx_pos"] = str(sum(x > 0 for x in ctx_t))
    gap = [tr[g]["E+C+Z+P"]["own_city"]["auc"] - tr[g]["E+C+Z+P"]["other_cities"]["auc"] for g in worse]
    V["transfer_gap_lo"], V["transfer_gap_hi"] = _f(min(gap)), _f(max(gap))

    # Manhattan rezonings and the redevelopment model
    G = _load("integration_replay")["rezonings"]
    h = lambda k: sum(v["homes_added_by_2025"][k] for v in G.values())
    V["g_n"] = str(len(G))
    V["g_observed"], V["g_capacity"] = _n(h("observed")), _n(h("capacity"))
    V["g_prorated"], V["g_likely"] = _n(h("capacity_prorated_to_2025")), _n(h("likelihood_weighted"))
    V["g_cap_ratio"] = f"{h('capacity') / h('observed'):.0f}"
    V["g_inside"] = str(sum(v["homes_added_by_2025"]["likelihood_weighted_95pct"][0] <= v["homes_added_by_2025"]["observed"]
                            <= v["homes_added_by_2025"]["likelihood_weighted_95pct"][1] for v in G.values()))
    V["g_soho_built"] = _n(G["SoHo/NoHo (2021)"]["homes_added_by_2025"]["observed"])
    V["feis_eh"] = _f(_load("east_harlem_rezoning")["ratio_same_far_to_feis"], 2)
    V["feis_sn"] = _f(_load("soho_noho_rezoning")["ratio_to_feis"], 2)
    ro = _load("rolling_origin")["origins"]
    auc = [o["lightgbm"]["auc"] for o in ro]
    base = [max(b["auc"] for b in o["baselines"].values()) for o in ro]
    V["ro_n"], V["ro_auc_lo"], V["ro_auc_hi"] = str(len(ro)), _f(min(auc)), _f(max(auc))
    V["ro_beats"] = str(sum(a > b for a, b in zip(auc, base)))
    tm = _load("transition_2020_2025")["development"]
    V["tm_auc"] = _f(tm["LightGBM"]["auc"])
    # the system's components (Manhattan validation, optimizer, decisions, speed)
    lat = _load("latency")["twins"]
    V["lat_manh_ms"] = f"{lat['Manhattan']['fifty_changes']['median_ms']:.0f}"
    V["lat_nyc_s"] = f"{lat['New York City']['fifty_changes']['median_ms'] / 1000:.2f}"
    V["manh_parcels"], V["nyc_parcels"] = _n(lat["Manhattan"]["parcels"]), _n(lat["New York City"]["parcels"])
    from planarena.indicators.engine import INDICATORS
    V["n_indicators"] = _w(len(INDICATORS))
    em = _load("emissions_ll84_2023")
    V["em_n"], V["em_r"] = _n(em["n_buildings"]), _f(em["log_log_r"])
    V["em_mae"], V["em_ratio"] = _pct(em["median_abs_pct_error"]), _f(em["aggregate_ratio_modelled_to_observed"], 2)
    jb = _load("jobs_lodes_2023")
    V["jobs_tract_r"], V["jobs_block_r"] = _f(jb["tract"]["log_log_r"]), _f(jb["block"]["log_log_r"])
    un = _load("units_replay_2015_2025")
    V["units_n"], V["units_ratio"], V["units_r"] = _n(un["parcels"]), _f(un["aggregate_ratio"], 2), _f(un["log_log_r"])
    cp = _load("change_prediction")
    hy, base = cp["methods"]["hybrid"]["all"], cp["methods"]["transition-type mean (cross-fitted)"]["all"]
    V["cp_events"] = _n(cp["events"])
    V["cp_homes_r"], V["cp_homes_ratio"] = _f(hy["homes"]["pearson_r"]), _f(hy["homes"]["total_ratio"], 2)
    V["cp_value_r"], V["cp_em_r"], V["cp_em_n"] = _f(hy["assessed_value"]["pearson_r"]), _f(hy["emissions"]["pearson_r"]), _n(hy["emissions"]["n"])
    V["cp_base_homes_r"], V["cp_base_value_r"] = _f(base["homes"]["pearson_r"]), _f(base["assessed_value"]["pearson_r"])
    V["cp_base_em_r"] = _f(base["emissions"]["pearson_r"])
    V["cp_jobs_r"] = _f(hy["jobs_by_block"]["pearson_r"])
    tm = _load("transition_2020_2025")
    dev = tm["development"]
    V["tm_auc"], V["tm_gnn_auc"] = _f(dev["LightGBM"]["auc"]), _f(dev["GNN"]["auc"])
    V["tm_base_auc"] = _f(max(v["auc"] for k, v in dev.items() if k not in ("LightGBM", "GNN", "GNN + LightGBM (mean)")))
    V["tm_top1"], V["tm_base_rate"] = _pct(dev["LightGBM"]["precision_top1pct"], 0), _pct(tm["test_base_rate"], 2)
    V["tm_ece"] = _f(dev["LightGBM"]["ece"])
    og = _load("optimizer_gap")
    V["og_areas"], V["og_first"] = str(og["areas"]), _pct(og["share_exact_optimum_found"])
    V["og_polished"], V["og_sec"] = _pct(og["polished"]["share_exact_optimum_found"]), str(og["polished"]["median_seconds"])
    ob = _load("optimizer_benchmark")["summary"]
    V["ob_polish_share"], V["ob_polish_sec"] = _pct(ob["first_order_plus_exact_polish"]["median_share_of_best_found"], 1), str(ob["first_order_plus_exact_polish"]["median_seconds"])
    V["ob_greedy_sec"] = str(ob["greedy_exact"]["median_seconds"])
    V["ob_greedy_best"] = _w(ob["greedy_exact"]["districts_where_best"])
    ra = _load("realism_ablation")
    st = ra["settings"]
    V["ra_noprec"] = _pct(st["no realism layer"]["share_without_precedent"])
    V["ra_red_none"] = _pct(st["no realism layer"]["share_redeveloped_by_2025"])
    V["ra_red_lik"] = _pct(st["precedent filter + likelihood layer"]["share_redeveloped_by_2025"])
    V["ra_all"], V["ra_fifth"] = _pct(ra["reference"]["share_of_all_lots_redeveloped"], 1), _pct(ra["reference"]["share_of_likely_fifth_redeveloped"], 1)
    V["ra_score_prec"] = _f(st["precedent filter (default)"]["median_district_objective"], 2)
    V["ra_score_lik"] = _f(st["precedent filter + likelihood layer"]["median_district_objective"], 2)
    vs = _load("vote_summary")
    V["vs_d"] = _w(vs["districts"])
    for k in ("condorcet_winner_exists", "all_rules_agree", "residents_agree_with_council", "winner_is_nash_bargaining",
              "schulze_manipulable", "double_majority"):
        V["vs_" + k] = _w(vs[k])
    V["vs_robust"] = _pct(vs["median_robust_both"])
    d = vs["districts"]
    V["vs_rules_phrase"] = ("the nine rules never all agree" if not vs["all_rules_agree"]
                            else f"all nine rules agree in {_w(vs['all_rules_agree'])} districts")
    V["vs_nash_phrase"] = ("is never the Nash bargaining solution" if not vs["winner_is_nash_bargaining"]
                           else f"is the Nash bargaining solution in {_w(vs['winner_is_nash_bargaining'])} districts")
    V["vs_resid_phrase"] = (f"in {_w(vs['residents_agree_with_council'])} district"
                            + ("" if vs["residents_agree_with_council"] == 1 else "s"))
    V["dec_guidance"] = ("No rule agreed with all the others in any district of the simulated scenario."
                         if not vs["all_rules_agree"] else
                         f"The nine rules agreed in only {_w(vs['all_rules_agree'])} of {_w(d)} districts.")
    V["of_greedy"] = _of(ob["greedy_exact"]["districts_where_best"], d)
    sv = _load("vote_sensitivity")["summary"]
    V["vs_weights"] = _pct(sv["median_winner_unchanged_under_weight_draws"])

    # held-out impact test
    ih = _load("impact_heldout")
    M = ih["methods"]
    dep = M["engine as deployed (hybrid, stock unit size; pre-period coefficients)"]["all"]
    gb = M["gradient-boosted regression (selection set)"]["all"]
    tmn = M["mean change of the same transition (selection set)"]["all"]
    scl = next(v for k, v in M.items() if "selection-set scale" in k)["all"]
    V["ih_events"], V["ih_sel"], V["ih_eval"] = _n(ih["events"]), _n(ih["selection_events"]), _n(ih["evaluation_events"])
    V["ih_homes_r"], V["ih_homes_mae"] = _f(dep["homes"]["pearson_r"], 2), _f(dep["homes"]["mae"], 1)
    V["ih_none_mae"] = _f(M["no change"]["all"]["homes"]["mae"], 1)
    V["ih_homes_total"], V["ih_gb_homes_total"] = _f(dep["homes"]["total_ratio"], 2), _f(gb["homes"]["total_ratio"], 2)
    V["ih_homes_total_scaled"] = _f(scl["homes"]["total_ratio"], 2)
    V["ih_gb_homes_r"], V["ih_tm_homes_r"] = _f(gb["homes"]["pearson_r"], 2), _f(tmn["homes"]["pearson_r"], 2)
    V["ih_value_r"], V["ih_gb_value_r"] = _f(dep["value"]["pearson_r"], 2), _f(gb["value"]["pearson_r"], 2)
    V["ih_em_r"], V["ih_gb_em_r"], V["ih_em_n"] = _f(dep["ghg"]["pearson_r"], 2), _f(gb["ghg"]["pearson_r"], 2), _n(dep["ghg"]["n"])
    chosen = next(k for k in M if k.startswith("engine (") and "scale" not in k)
    excess = M[chosen]["all"]["homes"]["bias"] * M[chosen]["all"]["homes"]["n"]  # predicted minus observed, total
    into_mix = sum(v["homes"]["bias"] * v["homes"]["n"] for t, v in ih["by_transition"].items() if t.endswith("> mix"))
    V["ih_mix_share"] = _pct(into_mix / excess)
    V["ih_bias_com_mix"] = _f(ih["by_transition"]["com > mix"]["homes"]["bias"], 0)
    V["ih_jobs_r"] = _f(ih["jobs_by_block"]["pearson_r"], 2)
    for k in ("homes", "value", "ghg"):  # direction right, among lots where both changes are non-zero
        pr, ob = np.array(ih[f"points_{k}"]["predicted"]), np.array(ih[f"points_{k}"]["observed"])
        m = (pr != 0) & (ob != 0)
        V[f"ih_sign_{k}"] = _pct(float((np.sign(pr[m]) == np.sign(ob[m])).mean()))

    # calibration
    cm = _load("calibration_manhattan")["origins"]
    oe = [o["raw"]["observed_over_predicted"] for o in cm]
    sl = [o["raw"]["calibration_slope"] for o in cm]
    bs = [o["raw"]["brier_skill"] for o in cm]
    V["cal_oe_lo"], V["cal_oe_hi"] = _f(min(oe), 2), _f(max(oe), 2)
    V["cal_slope_lo"], V["cal_slope_hi"] = _f(min(sl), 2), _f(max(sl), 2)
    V["cal_bss_lo"], V["cal_bss_hi"] = _num(min(bs), 2), _num(max(bs), 2)
    V["cal_bss_neg"] = _of(sum(b < 0 for b in bs), len(bs))
    rc = [o["recalibrated"]["observed_over_predicted"] for o in cm]
    V["cal_rec_oe_lo"], V["cal_rec_oe_hi"] = _f(min(rc), 2), _f(max(rc), 2)
    rs = [o["recalibrated"]["calibration_slope"] for o in cm]
    V["cal_rec_slope_lo"], V["cal_rec_slope_hi"] = _f(min(rs), 2), _f(max(rs), 2)
    V["cal_rec_bss_neg"] = _of(sum(o["recalibrated"]["brier_skill"] < 0 for o in cm), len(cm))
    cc = _load("calibration_cities")["cities"]
    full = lambda v: next(k for k in v if k not in ("E", "train", "test"))
    coe = [v[full(v)]["raw"]["observed_over_predicted"] for v in cc.values()]
    cre = [v[full(v)]["recalibrated"]["observed_over_predicted"] for v in cc.values()]
    V["calc_oe_lo"], V["calc_oe_hi"] = _f(min(coe), 2), _f(max(coe), 2)
    V["calc_rec_oe_lo"], V["calc_rec_oe_hi"] = _f(min(cre), 2), _f(max(cre), 2)
    V["ro_beats_phrase"] = ("every origin" if V["ro_beats"] == V["ro_n"]
                            else f"{_of(int(V['ro_beats']), int(V['ro_n']))} origins")

    # expected yield, and the rezonings
    ly = [S[g]["likely_yield"]["ratio_total"] for g in S if "likely_yield" in S[g]]
    V["ly_lo"], V["ly_hi"] = _f(min(ly), 2), _f(max(ly), 2)
    rz = _load("integration_replay")
    G = rz["rezonings"]
    rat = lambda n: _times(G[n]["ratio_to_observed"]["likelihood_weighted"])
    V["g_eh_ratio"], V["g_in_ratio"], V["g_sn_ratio"] = rat("East Harlem (2017)"), rat("Inwood (2018)"), rat("SoHo/NoHo (2021)")
    for k in ("g_eh_ratio", "g_in_ratio", "g_sn_ratio"):
        V[k] = V[k].replace("$\\times$", "")
    sh = [v["yield_share_of_capacity"]["mean"] for v in G.values()]
    V["g_share"] = f"{_pct(min(sh))}--{_pct(max(sh))}" if round(min(sh), 2) != round(max(sh), 2) else _pct(sh[0])
    V["g_yield"] = _n(sum(v["homes_added_by_2025"]["likelihood_x_yield"] for v in G.values()))
    V["of_g_yield"] = _of(sum(rz["observed_within_yield_interval"].values()), len(G))
    V["g_sae_yield"] = _n(rz["sum_of_absolute_errors"]["likelihood_x_yield"])
    V["g_sae_lik"] = _n(rz["sum_of_absolute_errors"]["likelihood_weighted"])

    # optimizer denominators, NSGA-III test, plausibility at equal counts
    det = og["areas_detail"]
    pos = [a for a in det if a["exact"] > 1e-12]
    V["og_pos"] = _w(len(pos))
    V["og_first_frac"] = _of(sum(a["gap"] <= 1e-9 for a in pos), len(pos))
    V["og_polished_frac"] = _of(sum(a["gap_polished"] <= 1e-9 for a in pos), len(pos))
    pt = _load("pareto_test")
    md = pt["summary"]["medium"]
    V["pt_med_n"], V["pt_small_n"] = _w(len(pt["medium"])), _w(len(pt["small"]))
    V["pt_med_plans"] = f"{_n(min(r['plans'] for r in pt['medium']))}--{_n(max(r['plans'] for r in pt['medium']))}"
    V["pt_med_hv"], V["pt_med_hv_min"] = _pct(md["surrogate.hv_share_nsga"]["median"], 1), _pct(md["surrogate.hv_share_nsga"]["min"], 1)
    V["pt_rand_hv"] = _pct(md["surrogate.hv_share_random"]["median"], 1)
    eq = ra["equal_counts"]
    nm = ("no realism layer", "precedent filter (default)", "precedent filter + likelihood layer")
    V["ra_eq_none"], V["ra_eq_prec"], V["ra_eq_lik"] = (_pct(eq[k]["share_redeveloped_by_2025"], 0) for k in nm)
    into = sum(round(eq[k]["share_redeveloped_into_recommended_use"] * eq[k]["recommended_parcels"]) for k in nm)
    V["ra_into_phrase"] = ("None of the" if into == 0 else
                           f"Only {_w(into)} of the {_n(sum(eq[k]['recommended_parcels'] for k in nm))}")
    J = {k: eq[k]["sum_district_objective"] for k in nm}
    V["ra_cost_prec"] = _pct(1 - J[nm[1]] / J[nm[0]])
    V["ra_cost_lik"] = _pct(1 - J[nm[2]] / J[nm[0]])
    rd = sv["rule_disagreement_by_candidate_set"]
    V["vs_gen_winners"] = _w(int(rd["generated"]["median_distinct_winners"]))
    V["vs_ideal_winners"] = _w(int(rd["ideal plans only"]["median_distinct_winners"]))
    V["vs_rand_winners"] = _w(int(rd["random only"]["median_distinct_winners"]))
    V["vs_het"] = _pct(sv["median_residents_winner_unchanged_with_heterogeneous_weights"])
    V["release"] = RELEASE

    # counts in the prose are words ("seven cities", "all five", "four of seven")
    pairs = {"zE": ("n_zE_sig", "n_zE"), "z": ("n_z_sig", "n_z"), "p": ("n_p_sig", "n_p"),
             "ctx_spatial": ("n_ctx_spatial_pos", "n_test"), "z_spatial": ("n_z_spatial_pos", "n_z_spatial"),
             "dest_z": ("n_dest_z_better", "n_dest_z"), "model_layers": ("n_model_beats_layers", "n_dest"),
             "trend": ("n_trend_better", "n_homes"), "yield_trend": ("n_yield_vs_trend", "n_homes"),
             "transfer": ("n_transfer_worse", "n_transfer"),
             "transfer_ctx": ("n_transfer_ctx_pos", "n_transfer"), "g": ("g_inside", "g_n")}
    for name, (k, n) in pairs.items():
        V[f"of_{name}"] = _of(int(V[k]), int(V[n]))
    V["trend_guidance"] = (f"A trend ranked districts better than build-out-based projections in {V['of_trend']} "
                           f"cities, and likelihood $\\times$ yield as well as the trend in {V['of_yield_trend']}")
    for k in list(V):
        if (k.startswith("n_") or k in ("g_n", "g_inside")) and V[k].isdigit() and int(V[k]) <= 10:
            V[k] = _w(int(V[k]))
    return V


# ----------------------------------------------------------------------------- tables
def _tab(caption: str, label: str, spec: str, head: list[str], rows: list[list[str]], note: str = "",
         size: str = "\\small", sep: str = "4pt", pos: str = "t") -> str:
    L = [f"\\begin{{table}}[{pos}]", "\\centering", size, f"\\setlength{{\\tabcolsep}}{{{sep}}}", f"\\caption{{{caption}}}",
         f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    L += [" & ".join(r) + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}"]
    if note:
        L += ["\\par\\smallskip", f"\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}\\end{{minipage}}"]
    L += ["\\end{table}"]
    return "\n".join(L)


def tables() -> dict[str, str]:
    integ, cities = _data()
    T = {g: integ[g]["temporal"] for g in cities}
    layers = _load("cities_layers")
    out = {}

    rows = []
    for g, (name, src, plan) in TEN.items():
        v = layers[g]
        cov = lambda k: _pct(v[k]) if v[k] > 0 else "--"
        rows.append([name, _n(v["parcels"]), src, cov("existing"), plan, cov("plan"), cov("zoning"),
                     _f(v["homes_twin"] / v["homes_census_2020"], 2)])
    total = sum(v["parcels"] for v in layers.values())
    rows.append(["\\textit{All}", _n(total), "", "", "", "", "", ""])
    out["table_cities"] = _tab(
        "The ten cities: official parcel layers on one schema. Shares are of the city's parcels.", "tab:cities",
        "lrllllll", ["City", "Parcels", "Existing use (source)", "Share", "Adopted plan", "Share", "Zoning",
                     "Homes / Census"], rows,
        "Existing use is the city's or regional agency's parcel land use, refined by assessor records where the layer "
        "groups uses. Homes / Census: homes in the twin over Census 2020 housing units. Houston has no zoning. "
        "Milwaukee's area plans are published as PDFs that could not be downloaded by script.", size="\\scriptsize")

    rows = []
    for g in cities:
        t = T[g]
        rows.append([CITY[g], OUTCOME[g], OUTCOME_SOURCE[g], f"{t['train'][0]}--{t['train'][1]}",
                     f"{t['test'][0]}--{t['test'][1]}", _n(t["n_lots"]), _n(t["developed"]),
                     _pct(t["sets"]["E"]["base_rate"], 2)])
    out["table_panels"] = _tab(
        "Test panels. Models are fitted on the training window and scored on the test window.", "tab:panels",
        "lllllrrr", ["City", "Development", "Source", "Train", "Test", "Lots", "Developed", "Rate"], rows,
        "Lots already under construction at the start of a window are left out. Floor area +25\\% also requires "
        "+2,000 sq ft. Change of use group: the lot changes to a built use of another of the nine groups, or gains "
        "homes.", size="\\scriptsize", pos="h!")

    rows = []
    for g in cities:
        s = T[g]["sets"]
        c = lambda k: _f(s[k]["auc"]) if k in s else "--"
        best = max(s, key=lambda k: s[k]["auc"])
        cells = [c(k) for k in ("E", "E+C", "E+C+Z", "E+C+P", "E+C+Z+P")]
        rows.append([CITY[g]] + cells + [best])
    out["table_auc"] = _tab(
        "Where development happens: out-of-time AUC by block set. E existing conditions, C neighbourhood context, "
        "Z zoning, P adopted plan; -- where the city has no such layer.", "tab:auc", "lrrrrrl",
        ["City", "E", "E+C", "E+C+Z", "E+C+P", "E+C+Z+P", "Best set (any)"], rows,
        "Best set: the highest AUC among all fitted combinations, including those without context.", pos="h!")

    rows = []
    for g in cities:
        gg = T[g]["gains"]
        c = integ[g].get("temporal_conservative")
        rows.append([CITY[g], _gtex(_gain(gg, "E+C vs E")), _gtex(_gain(gg, "E+C+Z vs E+C")),
                     _gtex(_gain(gg, "E+C+P vs E+C")),
                     _gtex(_gain(c["gains"], "E+C+Z vs E+C")) if c else "",
                     _gtex(_gain(c["gains"], "E+C+P vs E+C")) if c else ""])
    out["table_gains"] = _tab(
        "Gain in out-of-time AUC from adding a block, with 95\\% bootstrap intervals over 2 km tiles. Bold: the "
        "interval excludes zero.", "tab:gains", "llllll",
        ["City", "C over E", "Z over E+C", "P over E+C", "Z, conservative", "P, conservative"], rows,
        "Conservative: in each window, lots whose zoning or plan dates from the window's start or later are left "
        "out; possible where a current layer records those dates (Phoenix, San Diego, Chicago). Dallas's current "
        "layers carry no dates, so its zoning and plan gains may include post-decision information.", size="\\scriptsize", sep="2.5pt")

    rows = []
    for g in cities:
        d = T[g].get("destination")
        if not d or not d["n"]:
            continue
        f = lambda x: f"{_pct(x['class'])} / {_pct(x['group'])}" if x and x.get("coverage") else "--"
        st = next(k for k in ("E+C+Z+P", "E+Z+P", "E+Z", "E+C", "E") if k in d["models"])
        rows.append([CITY[g], _n(d["n"]), f(d["layers"].get("existing (no change)")), f(d["layers"].get("zoning class")),
                     f(d["layers"].get("plan class")), f(d["models"][st]) + f" ({st})"])
    out["table_dest"] = _tab(
        "What developed lots became: share named correctly, as one of 16 classes / one of nine groups.", "tab:dest",
        "lrllll", ["City", "Developed", "No change", "Zoning class", "Plan class", "Model (blocks)"], rows,
        "In Phoenix and San Diego development is a change of class (to another group, or to more homes), so ``no "
        "change'' is never right at class level there. Models are cross-fitted over 2 km tiles among the developed "
        "lots. Chicago's permits do not record the new use.", size="\\scriptsize")

    rows = []
    for g in cities:
        sc = T[g].get("scenario")
        if not sc:
            continue
        f = lambda k: f"{_times(sc[k]['ratio_total'])} ({_num(sc[k]['district_spearman'])})" if k in sc else "--"
        rows.append([CITY[g], _n(sc["city_total"]["observed"]), f("zoning_buildout"), f("plan_buildout"),
                     f("likely_zoning"), f("likely_plan"), f("likely_yield"), f("trend")])
    out["table_homes"] = _tab(
        "Homes added in the test window: projection over observed total (Spearman correlation across districts).",
        "tab:homes", "lrllllll", ["City", "Observed", "Zoning build-out", "Plan build-out", "Likely $\\times$ zoning",
                                   "Likely $\\times$ plan", "Likely $\\times$ yield", "Five-year trend"], rows,
        "Observed: gross homes added in the window. Build-out (a proxy, not legal capacity): every lot at the homes "
        "per acre that the 90th percentile of lots of its zoned (or planned) class reach in the city today. Likely: "
        "the build-out increment times the out-of-time probability of development (blocks E+C+Z+P where available). "
        "Yield: the probability times the mean homes added by developing lots of the same class and lot size in the "
        "training window. Trend: homes completed in the five years before the window, scaled to its length.",
        size="\\scriptsize", sep="3pt")

    tr = integ["transfer"]
    rows = []
    for g in [c for c in CITY if c in tr]:
        f = lambda k: f"{_f(tr[g][k]['other_cities']['auc'])} / {_f(tr[g][k]['own_city']['auc'])}" if k in tr[g] else "--"
        rows.append([CITY[g], f("E"), f("E+C"), f("E+C+Z"), f("E+C+Z+P")])
    out["table_transfer"] = _tab(
        "Transfer: AUC of a model fitted on the other cities' training windows / the city's own model, on the city's "
        "test window.", "tab:transfer", "lllll", ["City", "E", "E+C", "E+C+Z", "E+C+Z+P"], rows,
        "Cities whose outcome is a building or a permit. A missing plan layer is coded as unknown.", pos="h!")

    G = _load("integration_replay")["rezonings"]
    rows = []
    for name, v in G.items():
        h = v["homes_added_by_2025"]
        lo, hi = h["likelihood_weighted_95pct"]
        ylo, yhi = h["likelihood_x_yield_95pct"]
        rows.append([name, str(v["sites"]), _n(h["observed"]), _n(h["capacity"]), _n(h["capacity_prorated_to_2025"]),
                     f"{_n(h['likelihood_weighted'])} ({_n(lo)}--{_n(hi)})",
                     f"{_n(h['likelihood_x_yield'])} ({_n(ylo)}--{_n(yhi)})"])
    k = lambda key: _n(sum(v["homes_added_by_2025"][key] for v in G.values()))
    rows.append(["\\textit{All three}", "", k("observed"), k("capacity"), k("capacity_prorated_to_2025"),
                 k("likelihood_weighted"), k("likelihood_x_yield")])
    sae = _load("integration_replay")["sum_of_absolute_errors"]
    rows.append(["\\textit{Sum of absolute errors}", "", "", _n(sae["capacity"]), _n(sae["capacity_prorated_to_2025"]),
                 _n(sae["likelihood_weighted"]), _n(sae["likelihood_x_yield"])])
    out["table_replay"] = _tab(
        "Three Manhattan rezonings: homes on development sites by 2025, observed and projected.", "tab:replay",
        "lrrrrll", ["Rezoning (adopted)", "Sites", "Built", "Capacity", "Pro-rated",
                    "Prob.\\ $\\times$ capacity (95\\%)", "Prob.\\ $\\times$ yield (95\\%)"], rows,
        "Capacity is full build-out by the review's build year; pro-rated scales it linearly to 2025. The probability "
        "comes from a redevelopment model trained only on outcomes known by the adoption year; yield is the mean share "
        "of capacity homes that soft sites redeveloped in those training years added. Ranges are conditional on the "
        "probabilities (Supplement S5).", size="\\footnotesize", sep="3pt")

    from planarena.indicators.engine import INDICATORS
    status = {"homes": "held-out change", "value": "held-out change", "ghg": "existing; held-out change",
              "jobs": "existing; change: no skill", "green_space": "direct accounting"}
    rows = [[i.label.replace("&", "\\&"), i.pillar, i.unit.replace("%", "\\%"), "higher" if i.direction > 0 else "lower",
             status.get(i.key, "not validated")] for i in INDICATORS]
    out["table_indicators"] = _tab(
        "Indicators computed for every edit, and how each is tested (Manhattan).", "tab:indicators", "lllll",
        ["Indicator", "Pillar", "Unit", "Better", "Test"], rows,
        "Unchanged parcels keep their observed values; changed parcels follow the delta equations. Each indicator "
        "carries a sensitivity range from low and high coefficient variants. Access, mix, green cover and runoff are "
        "recomputed from the resulting state but have no observed after-change data to test against.",
        size="\\scriptsize")

    rows = [["Existing conditions", "Does the twin reproduce what is there?", "LL84 energy, LODES jobs, recorded homes",
             "accounting", "\\ref{sec:r-existing}"],
            ["Planning calculations", "Does it reproduce a review's projection from the review's own inputs?",
             "FEIS site tables (two rezonings)", "accounting", "\\ref{sec:r-existing}"],
            ["Effects of change", "Given an observed change, does it predict the effect?",
             "MapPLUTO 2015 vs 2025, held-out folds", "empirical", "\\ref{sec:r-change}"],
            ["Likelihood", "Does it rank lots that redevelop, and how many, forward in time?",
             "Rolling origins 2016--2025", "empirical", "\\ref{sec:r-likely}"],
            ["Official layers", "What do zoning and plans add, in other cities?", "Seven city panels", "empirical",
             "\\ref{sec:r-layers}"],
            ["Housing projections", "Are projected homes close to what was built?", "Four cities; three rezonings",
             "empirical", "\\ref{sec:r-homes}"],
            ["Optimizer", "How close is the best plan found to the best plan?", "Exhaustive search; 12 districts",
             "algorithmic", "\\ref{sec:r-opt}"],
            ["Pareto explorer", "Does NSGA-III find the trade-off front?", "Exhaustive enumeration of blocks",
             "algorithmic", "\\ref{sec:r-opt}"],
            ["Realism", "Are recommended changes plausible?", "2020 plans vs 2025 outcomes", "empirical",
             "\\ref{sec:r-opt}"],
            ["Decision module", "How does the choice depend on rule, seats and candidates?", "Assumed utilities",
             "illustrative", "\\ref{sec:r-dec}"]]
    out["table_design"] = _tab("Evaluation design: each component, its question, data and type of evidence.",
                               "tab:design", "lp{5.0cm}p{4.0cm}ll", ["Component", "Question", "Data", "Evidence",
                                                                     "Section"], rows, size="\\scriptsize")

    ih = _load("impact_heldout")
    M = ih["methods"]
    chosen = next(k for k in M if k.startswith("engine (") and "scale" not in k)
    lab = {"engine as deployed (hybrid, stock unit size; pre-period coefficients)": "PlanArena (deployed form)",
           chosen: "PlanArena (form chosen on selection lots)",
           next(k for k in M if "selection-set scale" in k): "\\quad $\\times$ selection-lot scale",
           "gradient-boosted regression (selection set)": "Gradient-boosted regression",
           "mean change of the same transition (selection set)": "Mean change of the same transition"}
    rows = []
    for k, name in lab.items():
        a = M[k]["all"]
        rows.append([name, _f(a["homes"]["pearson_r"], 2), _f(a["homes"]["mae"], 1), _num(a["homes"]["bias"], 1),
                     _f(a["homes"]["total_ratio"], 2), _f(a["value"]["pearson_r"], 2), _f(a["value"]["total_ratio"], 2),
                     _f(a["ghg"]["pearson_r"], 2), _f(a["ghg"]["total_ratio"], 2)])
    nc = M["no change"]["all"]["homes"]
    out["table_change"] = _tab(
        f"Effects of observed change on {_n(ih['evaluation_events'])} held-out Manhattan lots, 2015--2025: given the "
        "2015 state and the observed new use and floor area, predicted against recorded change (Pearson $r$; mean "
        "absolute error and mean bias in homes per lot; predicted over observed total).",
        "tab:change", "lrrrrrrrr", ["Method", "Homes $r$", "MAE", "Bias", "Total", "Value $r$", "Total",
                                    "Emis.\\ $r$", "Total"], rows,
        f"Selection lots ({_n(ih['selection_events'])}, two of five spatial folds) chose the form and fitted the "
        f"baselines and the scale. Predicting no change gives a homes MAE of {_f(nc['mae'], 1)}. Emissions: "
        f"{_n(M[chosen]['all']['ghg']['n'])} benchmarked buildings.", size="\\scriptsize", sep="3pt")

    ob = _load("optimizer_benchmark")["summary"]
    rows = [[k.replace("_", " "), _pct(x["median_share_of_best_found"], 1), str(x["districts_where_best"]), str(x["median_seconds"])]
            for k, x in ob.items()]
    out["table_opt"] = _tab(
        "Optimizer at district scale (12 Manhattan districts, up to ten changes): share of the best plan any tested "
        "method found (not a proven optimum).",
        "tab:opt", "lrrr", ["Search", "Median share of best", "Districts best", "Median seconds"], rows, size="\\small")

    ra = _load("realism_ablation")
    eq = ra["equal_counts"]
    rows = [[k, _n(v["recommended_parcels"]), _pct(v["share_without_precedent"], 1), _pct(v["share_redeveloped_by_2025"], 1),
             _pct(v["share_of_eligible_redeveloped"], 1), _pct(v["share_redeveloped_into_recommended_use"], 1),
             _f(v["sum_district_objective"], 1)] for k, v in eq.items() if isinstance(v, dict) and "recommended_parcels" in v]
    full = ra["settings"]
    out["table_realism"] = _tab(
        "Realism layers at equal counts: plans recommended on the 2020 snapshot of every Manhattan district, and what "
        "happened by 2025.", "tab:realism", "lrrrrrr",
        ["Optimizer setting", "Parcels", "No precedent", "Redeveloped", "Eligible redev.", "Into rec.\\ use",
         "Objective"], rows,
        "In each district every setting recommends the smallest number any setting recommended there. Eligible redev.: "
        "share of the setting's eligible parcels redeveloped (the rate of a random choice). Objective: the sum over "
        "districts of the overall score minus the conversion penalty. At the full budget the settings recommended "
        + ", ".join(_n(v["recommended_parcels"]) for v in full.values()) + " parcels.", size="\\scriptsize")

    rows = []
    for g in cities:
        s = T[g]["sets"]
        k = "E+C+Z+P" if "E+C+Z+P" in s else "E+C+Z"
        rows.append([CITY[g], _pct(s["E"]["base_rate"], 2), _f(s["E"]["ap"]), _f(s[k]["ap"]), k,
                     _pct(s["E"]["p_top1"], 1), _pct(s[k]["p_top1"], 1)])
    out["table_ap"] = _tab(
        "Out-of-time average precision and precision among the top 1\\% of lots.", "tab:ap", "lrrrlrr",
        ["City", "Base rate", "AP, E", "AP, all", "All blocks", "Top 1\\%, E", "Top 1\\%, all"], rows, pos="h!")
    return out


# ----------------------------------------------------------------------------- supplement
SUPP_TEMPLATE = PAPER / "supplement.tex.in"
SUPP_OUT = PAPER / "supplement.tex"
RELEASE = "v1.1-review"
EVIDENCE = [  # component, claim, type of evidence, test
    ("Existing conditions", "records reproduced", "accounting check", "twin vs assessor, LL84, LEHD"),
    ("Planning calculations", "reviews' projections reproduced", "accounting check", "two environmental reviews"),
    ("Impact engine", "effects of observed change", "empirical, held out", "selection / evaluation folds"),
    ("Job change", "effect on jobs by block", "empirical, held out (no skill)", "LEHD 2015--2023"),
    ("Access, mix, green cover", "effects of change", "not validated", "no after-change data"),
    ("Redevelopment model", "where change happens", "empirical, out of time", "rolling origins; seven cities"),
    ("Calibration", "how often change happens", "empirical, out of time", "reliability, O/E, Brier skill"),
    ("Zoning and plan layers", "added predictive value", "empirical, out of time", "block ablation, conservative variant"),
    ("Housing projections", "homes built", "empirical, out of time", "districts; three rezonings"),
    ("Optimizer", "best plan found", "algorithmic verification", "exhaustive search; benchmark"),
    ("NSGA-III explorer", "trade-off front", "algorithmic verification", "exhaustive enumeration of blocks"),
    ("Realism layers", "plausible recommendations", "empirical, out of time", "2020 plans vs 2025 outcomes"),
    ("Decision module", "how a choice depends on its rule", "illustrative simulation", "assumed utilities and seats"),
]


def _code_settings() -> dict:
    import yaml

    from planarena.agents import voting
    from planarena.optimize.suggest import SuggestConfig
    from planarena.twin.build import CELL_M

    coef = yaml.safe_load((ROOT / "planarena" / "indicators" / "coefficients.yaml").read_text())
    a, acc = coef["assumptions"], coef["access"]
    cfg = SuggestConfig()
    import inspect

    rob = inspect.signature(voting.robustness).parameters
    return {"util": a["buildout_utilization"], "pph": a["persons_per_household"], "walk_m": acc["walk_radius_m"],
            "fac_m": acc["facility_radius_m"], "park_sqft": acc["park_min_sqft"], "pen": cfg.conversion_penalty,
            "share": cfg.max_change_share, "rounds": cfg.rounds, "wfit": cfg.w_fit, "jnd": voting.JND,
            "rob_samples": rob["samples"].default, "rob_conc": rob["concentration"].default, "cell_m": CELL_M}


def supplement_values() -> dict[str, str]:
    c = _code_settings()
    lat = _load("latency")["twins"]["Manhattan"]["derived"]
    V = {"util": _f(c["util"]["value"], 1), "util_lo": _f(c["util"]["low"], 1), "util_hi": _f(c["util"]["high"], 1),
         "pph": _f(c["pph"]["value"], 2), "cell_m": f"{c['cell_m']:.0f}",
         "walk_m": str(c["walk_m"]), "fac_m": str(c["fac_m"]), "park_acres": _f(c["park_sqft"] / 43560, 0),
         "sqft_unit": _n(lat["sqft_per_unit"]), "pen": f"{c['pen']:g}", "share": f"{c['share']:g}",
         "rounds": str(c["rounds"]), "wfit": f"{c['wfit']:g}", "polish": "30", "jnd": f"{c['jnd']:g}",
         "rob_samples": str(c["rob_samples"]), "rob_conc": f"{c['rob_conc']:g}", "release": RELEASE}
    ih = _load("impact_heldout")
    V["ih_events"], V["ih_sel"], V["ih_eval"] = _n(ih["events"]), _n(ih["selection_events"]), _n(ih["evaluation_events"])
    V["ih_jobs_r"], V["ih_jobs_n"] = _f(ih["jobs_by_block"]["pearson_r"], 2), _n(ih["jobs_by_block"]["n"])
    for k in ("homes", "value", "ghg"):
        pr, ob = np.array(ih[f"points_{k}"]["predicted"]), np.array(ih[f"points_{k}"]["observed"])
        m = (pr != 0) & (ob != 0)
        V[f"ih_sign_{k}"] = _pct(float((np.sign(pr[m]) == np.sign(ob[m])).mean()))
    V["n_features"] = str(_load("transition_2020_2025")["features"])
    la = _load("leakage_audit")
    mp = la["manhattan_pairs"]
    V["leak_2020_lots"] = str(mp["2020->2025"]["removed_by_new_building_rule"])
    r1, r5 = mp["2020->2021"], mp["2020->2025"]
    V["leak_2020_1_share"] = _pct(r1["removed_by_new_building_rule"] / (r1["developed"] + r1["removed_by_new_building_rule"]))
    V["leak_2020_5_share"] = _pct(r5["removed_by_new_building_rule"] / (r5["developed"] + r5["removed_by_new_building_rule"]))
    cc = la["cross_city_new_building_rule"].values()
    V["leak_cross_max"] = _pct(max(v["lots_already_built_at_t0_with_year_built_t0_or_later"] / max(v["developed"], 1)
                                   for v in cc), 1)
    pt = _load("pareto_test")
    sm, md = pt["summary"]["small"], pt["summary"]["medium"]
    plans = lambda rows: f"{_n(min(r['plans'] for r in rows))}--{_n(max(r['plans'] for r in rows))}"
    V["pt_small_n"], V["pt_med_n"] = str(len(pt["small"])), str(len(pt["medium"]))
    V["pt_small_plans"], V["pt_med_plans"] = plans(pt["small"]), plans(pt["medium"])
    V["pt_new_rho_min"] = _f(min(v["min"] for v in pt["summary"]["spearman_surrogate_exact_small"].values()), 2)
    old = _load("pareto_test_first_order_v0")
    V["pt_old_env_min"] = _num(old["summary"]["spearman_surrogate_exact_small"]["environmental"]["min"], 2)
    V["pt_old_hv_min"] = _pct(old["summary"]["small"]["exact.hv_share_nsga"]["min"], 1)
    V["nsga_refdirs"], V["nsga_pop"], V["nsga_gen"], V["nsga_show"] = "45", "92", "120", "24"
    return V


def supplement_tables() -> dict[str, str]:
    out = {}
    out["table_evidence"] = _tab("Claims, the evidence behind them, and its type.", "tab:s-evidence", "llll",
                                 ["Component", "Claim", "Evidence", "Test"], [list(r) for r in EVIDENCE])
    la = _load("leakage_audit")
    rows = [[k, v["existing"], v["zoning"], v["plan"], v["after_t0"]] for k, v in la["feature_timestamps"].items()]
    out["table_timing"] = _tab(
        "Source and date of each feature family against the decision year $t_0$. Context features (neighbour "
        "composition, recent construction within 400 m) come from the same record as existing conditions.",
        "tab:s-timing", "p{0.17\\linewidth}p{0.22\\linewidth}p{0.17\\linewidth}p{0.17\\linewidth}p{0.17\\linewidth}",
        ["Panel", "Existing conditions", "Zoning", "Plan", "Measured after $t_0$"], rows, size="\\footnotesize")
    rows = [[k.replace("->", "--"), _n(v["t0_lots"]), _n(v["t0_lots_without_t1_lot"]),
             _n(v["merges_t1_lots_from_two_or_more"]), _n(v["t0_lots_in_merges"]),
             _n(v["splits_t0_lots_into_two_or_more"]), _n(v["developed"]), _n(v["removed_by_new_building_rule"])]
            for k, v in la["manhattan_pairs"].items()]
    out["table_identity"] = _tab("Following Manhattan lots over time, and labels removed by the new-building rule.",
                                 "tab:s-identity", "lrrrrrrr",
                                 ["Pair", "$t_0$ lots", "No $t_1$ lot", "Mergers", "Lots merged", "Splits", "Developed",
                                  "Removed"], rows, size="\\footnotesize")
    ih = _load("impact_heldout")
    M = ih["methods"]
    chosen = next(k for k in M if k.startswith("engine (") and "scale" not in k)
    name = {k: {"engine as deployed (hybrid, stock unit size; pre-period coefficients)": "PlanArena (deployed form)",
                chosen: "PlanArena (form chosen on selection lots)",
                "gradient-boosted regression (selection set)": "Gradient-boosted regression",
                "mean change of the same transition (selection set)": "Mean change of the same transition"}.get(
                    k, "\\quad $\\times$ selection-lot scale") for k in M}
    short = lambda k: k
    rows = []
    for k, v in M.items():
        if k == "no change":
            continue
        a = v["all"]
        rows.append([short(name[k]), _f(a["homes"]["pearson_r"] or 0, 2), _f(a["homes"]["mae"], 1),
                     _num(a["homes"]["bias"], 1), _f(a["homes"]["total_ratio"], 2),
                     _f(a["value"]["pearson_r"] or 0, 2), _f(a["value"]["total_ratio"], 2),
                     _f(a["ghg"]["pearson_r"] or 0, 2), _f(a["ghg"]["total_ratio"], 2), _n(a["ghg"]["n"])])
    nc = M["no change"]["all"]["homes"]
    out["table_ih_main"] = _tab(
        f"Held-out impact test on {_n(ih['evaluation_events'])} evaluation lots: Pearson $r$, mean absolute error "
        "(homes), mean bias (predicted minus observed) and total predicted over total observed.", "tab:s-ih",
        "lrrrrrrrrr", ["Method", "Homes $r$", "MAE", "Bias", "Total", "Value $r$", "Total", "Emis.\\ $r$", "Total",
                       "$n$ emis."], rows,
        note=f"Predicting no change gives a homes MAE of {_f(nc['mae'], 1)}. Bias and MAE in homes per lot.",
        size="\\footnotesize", sep="3pt")
    dep = next(k for k in M if k.startswith("engine as deployed"))
    rows = []
    for b, v in M[dep]["by_size"].items():
        h = v["homes"]
        if not h.get("n"):
            continue
        rows.append([b.replace(">=", "$\\geq$").replace("<", "$<$"), _n(h["n"]), _num(h["pearson_r"] or 0, 2),
                     _f(h["mae"], 1), _num(h["bias"], 1),
                     _num(h["total_ratio"], 2) if h.get("total_ratio") is not None else "--",
                     _num(v["value"]["total_ratio"], 2) if v["value"].get("total_ratio") is not None else "--"])
    for kd in ("use change", "density only (new building)"):
        h, v = M[dep][kd]["homes"], M[dep][kd]["value"]
        rows.append([kd, _n(h["n"]), _num(h["pearson_r"] or 0, 2), _f(h["mae"], 1), _num(h["bias"], 1),
                     _num(h["total_ratio"], 2), _num(v["total_ratio"], 2)])
    out["table_ih_size"] = _tab("Deployed engine by size of the new building (sq ft) and by kind of change "
                                "(evaluation lots).", "tab:s-ih-size", "lrrrrrr",
                                ["Group", "$n$", "Homes $r$", "MAE", "Bias", "Total", "Value total"], rows,
                                note="Total: predicted over observed total; negative where the two totals have opposite "
                                     "signs (small buildings lost homes in total).", size="\\footnotesize")
    rows = []
    for t, v in ih["by_transition"].items():
        h = v["homes"]
        rows.append([t.replace(">", "$\\to$"), _n(h["n"]), _f(h["mae"], 1), _num(h["bias"], 1),
                     _num(h["total_ratio"], 2) if h.get("total_ratio") is not None else "--",
                     _num(v["value"]["total_ratio"], 2) if v["value"].get("total_ratio") is not None else "--"])
    out["table_ih_type"] = _tab("Chosen engine form by transition (groups with at least 30 evaluation lots).",
                                "tab:s-ih-type", "lrrrrr", ["Transition", "$n$", "Homes MAE", "Bias", "Total",
                                                             "Value total"], rows, size="\\footnotesize")
    cm = _load("calibration_manhattan")
    rows = []
    for o in cm["origins"]:
        r, iso, rc = o["raw"], o.get("isotonic"), o["recalibrated"]
        rows.append([str(o["horizon"]), o["test"].replace("->", "--"), _n(o["n_test"]),
                     _n(round(r["prevalence_test"] * o["n_test"])), _f(r["observed_over_predicted"], 2),
                     _num(r["calibration_intercept"], 2), _f(r["calibration_slope"], 2), _num(r["brier_skill"], 3),
                     _f(iso["observed_over_predicted"], 2) if iso else "--", _num(iso["brier_skill"], 3) if iso else "--",
                     _f(rc["observed_over_predicted"], 2), _f(rc["calibration_slope"], 2), _num(rc["brier_skill"], 3)])
    out["table_cal_mn"] = _tab(
        "Calibration of the Manhattan redevelopment model by horizon $h$ (years) and test window: observed over "
        "predicted events (O/E), calibration intercept and slope, and Brier skill against the training-period rate; "
        "then after temporal isotonic recalibration, and after logistic recalibration on out-of-fold training "
        "probabilities.", "tab:s-cal-mn", "llrrrrrrrrrrr",
        ["$h$", "Test", "Lots", "Events", "O/E", "Int.", "Slope", "Skill", "O/E iso", "Skill iso", "O/E rec.",
         "Slope rec.", "Skill rec."], rows, size="\\scriptsize", sep="2.5pt")
    cc = _load("calibration_cities")["cities"]
    rows = []
    for g in [c for c in CITY if c in cc]:
        for k, v in cc[g].items():
            if k in ("train", "test"):
                continue
            r, rc = v["raw"], v["recalibrated"]
            rows.append([CITY[g] if k == "E" else "", k, _pct(r["prevalence_test"], 2), _f(r["observed_over_predicted"], 2),
                         _f(r["calibration_slope"], 2), _num(r["brier_skill"], 3), _f(rc["observed_over_predicted"], 2),
                         _f(rc["calibration_slope"], 2), _num(rc["brier_skill"], 3)])
    out["table_cal_cities"] = _tab(
        "Calibration in each city's test window (existing conditions only, and all blocks): raw, and after logistic "
        "recalibration on out-of-fold training-window probabilities.", "tab:s-cal-cities", "llrrrrrrr",
        ["City", "Blocks", "Rate", "O/E", "Slope", "Skill", "O/E rec.", "Slope rec.", "Skill rec."], rows,
        size="\\footnotesize")
    pt = _load("pareto_test")
    S = pt["summary"]
    md = lambda d: f"{_f(d['median'], 3)} ({_f(d['min'], 3)})" if d else "--"
    rows = [["Small (exact)", str(len(pt["small"])), md(S["small"]["exact.hv_share_nsga"]),
             md(S["small"]["exact.share_of_true_front_found"]), md(S["small"]["surrogate.hv_share_random"])],
            ["Medium (first order)", str(len(pt["medium"])), md(S["medium"]["surrogate.hv_share_nsga"]),
             md(S["medium"]["surrogate.share_of_true_front_found"]), md(S["medium"]["surrogate.hv_share_random"])],
            ["Medium, re-scored", str(len(pt["medium"])), md(S["medium"]["exact.hv_share_nsga_vs_surrogate_front_rescored"]),
             "--", "--"]]
    out["table_nsga"] = _tab("NSGA-III against the true front: median (minimum) over blocks of the hypervolume share "
                             "and of the share of true-front plans returned; random search with the same number of "
                             "evaluations for comparison.", "tab:s-nsga", "lrrrr",
                             ["Blocks", "$n$", "Hypervolume share", "Front plans found", "Random search"], rows,
                             size="\\footnotesize")
    from planarena.agents.stakeholders import DEFAULT_STAKEHOLDERS
    from planarena.indicators.engine import INDICATORS

    keys = [i.key for i in INDICATORS]
    lab = {"homes": "Homes", "jobs": "Jobs", "value": "Value", "park_access": "Park", "services_access": "Services",
           "facility_access": "Facilities", "mix": "Mix", "green_space": "Green sp.", "runoff": "Runoff",
           "ghg": "Emissions", "green_cover": "Green cov."}
    rows = [[s.name] + [(_f(s.normalized()[k], 2) if s.normalized()[k] else "") for k in keys]
            for s in DEFAULT_STAKEHOLDERS]
    out["table_weights"] = _tab("Indicator weights of the five stakeholder agents (assumptions; each sums to one).",
                                "tab:s-weights", "l" + "r" * len(keys), ["Agent"] + [lab[k] for k in keys], rows,
                                size="\\footnotesize", sep="3pt")
    vs = _load("vote_sensitivity")["summary"]
    rd = vs["rule_disagreement_by_candidate_set"]
    rows = [[k, _f(v["median_candidates"], 0), _f(v["median_distinct_winners"], 0), str(v["max_distinct_winners"]),
             str(v["districts_all_rules_agree"]), str(v["districts_with_condorcet_winner"])] for k, v in rd.items()]
    out["table_vote_sets"] = _tab(
        f"Disagreement among the nine rules by candidate set ({vs['districts']} districts): median candidates, median "
        "and maximum number of distinct winners, districts where all rules agree and with a Condorcet winner.",
        "tab:s-vote-sets", "lrrrrr", ["Candidate set", "Candidates", "Winners", "Max", "All agree", "Condorcet"],
        rows, note=f"With each block's resident weights drawn independently, the residents' winner is unchanged in a "
                   f"median {_pct(vs['median_residents_winner_unchanged_with_heterogeneous_weights'])} of draws.",
        size="\\footnotesize")
    return out


# ----------------------------------------------------------------------------- render
def _fill(template, fill: dict) -> str:
    def sub(m):
        k = m.group(1)
        if k not in fill:
            raise KeyError(f"no value for @@{k}@@")
        return fill[k]
    return re.sub(r"@@(\w+)@@", sub, template.read_text())


def render() -> str:
    return _fill(TEMPLATE, {**values(), **tables()})


def render_supplement() -> str:
    return _fill(SUPP_TEMPLATE, {**supplement_values(), **supplement_tables()})


def outputs() -> dict:
    return {OUT: render(), SUPP_OUT: render_supplement()}


def citations(text: str) -> set[str]:
    keys = set()
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", text):
        keys |= {k.strip() for k in m.group(1).split(",")}
    return keys


def main() -> None:
    for p, text in outputs().items():
        p.write_text(text)
        print("written", p.relative_to(ROOT))
    if shutil.which("tectonic"):
        for f in (OUT, SUPP_OUT):
            r = subprocess.run(["tectonic", "--keep-logs", f.name], cwd=PAPER, capture_output=True, text=True)
            print(f.with_suffix(".pdf").name if r.returncode == 0 else r.stderr[-3000:])


if __name__ == "__main__":
    main()
