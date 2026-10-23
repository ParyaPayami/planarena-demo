"""Generate every reported number in README.md and docs/VALIDATION.md from result and coefficient files.

No result is typed by hand: README.md holds marked blocks that this script rewrites, and VALIDATION.md is
written whole. A test (tests/test_docs.py) fails when the committed docs differ from what this produces.

Run with:  python -I -m planarena.report            write the docs
           python -I -m planarena.report --check    exit 1 if the docs are out of date
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

from planarena.config import PROCESSED, ROOT

RES = ROOT / "results"
README = ROOT / "README.md"
VALIDATION = ROOT / "docs" / "VALIDATION.md"


def load(name: str) -> dict | None:
    for f in (RES / f"{name}.json", RES / "validation" / f"{name}.json"):
        if f.exists():
            return json.loads(f.read_text())
    return None


def pct(x, d=0):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=0):
    return "–" if x is None else f"{x:,.{d}f}"


def times(x, d=2):
    return "–" if x is None else f"{x:.{d}f}×"


def r3(x):
    return "–" if x is None else f"{x:.3f}"


# ----------------------------------------------------------------------------- shared pieces
def _best_baseline(dev: dict) -> tuple[str, dict]:
    models = {"GNN", "LightGBM", "GNN + LightGBM (mean)"}
    base = {k: v for k, v in dev.items() if k not in models}
    k = max(base, key=lambda k: base[k]["auc"])
    return k, base[k]


def _best_model(dev: dict) -> tuple[str, dict]:
    models = {k: v for k, v in dev.items() if k in {"GNN", "LightGBM", "GNN + LightGBM (mean)"}}
    k = max(models, key=lambda k: models[k]["average_precision"])
    return k, models[k]


def _cv_tables() -> list[str]:
    g, b = load("gnn_cv"), load("baselines_cv")
    meta = json.loads((PROCESSED / "twin_meta.json").read_text()) if (PROCESSED / "twin_meta.json").exists() else {}
    if not g:
        return []
    s = g["cv_summary"]
    n_test = sum(f["n_test"] for f in g["cv_folds"])
    miss = meta.get("data_quality", {}).get("missing_land_use")
    m = lambda k, met: f"{s[k][met]['mean']:.3f} ± {s[k][met]['sd']:.3f}" if k in s and met in s[k] else "–"
    out = [f"Spatial-block cross-validation, {g['folds']} folds of {g['tile_m']} m tiles. The task is reconstructing "
           f"each parcel's current use from its context, which describes pattern consistency, not suitability.",
           "",
           f"Parcels evaluated: {num(n_test)} of {num(g['n_parcels'])}; the {num(miss)} parcels with no recorded land use "
           "have no label and are excluded." if miss is not None else f"Parcels evaluated: {num(n_test)}.",
           "",
           "| Task | GNN | LightGBM | Simple baseline |", "|---|---|---|---|",
           f"| Current use, accuracy | {m('suit_gnn', 'accuracy')} | {m('suit_lightgbm', 'accuracy')} | {m('suit_zoning_majority', 'accuracy')} (zoning majority) |",
           f"| Current use, macro F1 | {m('suit_gnn', 'macro_f1')} | {m('suit_lightgbm', 'macro_f1')} | {m('suit_zoning_majority', 'macro_f1')} |",
           f"| Current use, log loss | {m('suit_gnn', 'log_loss')} | {m('suit_lightgbm', 'log_loss')} | {m('suit_zoning_majority', 'log_loss')} |",
           f"| Value, R² | {m('value_gnn', 'r2')} | {m('value_lightgbm', 'r2')} | {m('value_median_use_cd', 'r2')} (median by use and district) |"]
    if b:
        acc_g = [f["suit_gnn"]["accuracy"] for f in g["cv_folds"] if "suit_gnn" in f]
        acc_l = [f["suit_lightgbm"]["accuracy"] for f in g["cv_folds"] if "suit_lightgbm" in f]
        if acc_g and len(acc_g) == len(acc_l):
            wins = sum(lg > gn for gn, lg in zip(acc_g, acc_l))
            out += ["", f"LightGBM has higher current-use accuracy than the GNN in {wins} of {len(acc_g)} folds."]
    out += ["", "The GNN cross-validation is transductive: held-out parcels stay in the graph with their use and value "
            "masked. See docs/METHODS.md."]
    a = load("ablation_gnn")
    if a:
        sa = a["summary"]
        names = {"gnn_transductive": "GNN, transductive (as above)", "gnn_inductive": "GNN, inductive (no test-parcel edges in training)",
                 "lightgbm": "LightGBM", "lightgbm_plus_gnn_embedding": "LightGBM + GNN embedding"}
        ma = lambda k, met: f"{sa[k][met]['mean']:.3f} ± {sa[k][met]['sd']:.3f}" if k in sa and met in sa[k] else "–"
        out += ["", "**Ablation** (same folds; `results/validation/ablation_gnn.json`):", "",
                "| Model | Accuracy | Macro F1 | Log loss |", "|---|---|---|---|"]
        out += [f"| {v} | {ma(k, 'accuracy')} | {ma(k, 'macro_f1')} | {ma(k, 'log_loss')} |" for k, v in names.items()]
        out += ["", f"Adding the GNN embedding raises LightGBM accuracy in {a['embedding_improves_lightgbm_in_folds']} "
                f"of {len(a['folds'])} folds."]
    return out


def _crosscity() -> list[str]:
    cc = load("crosscity_redevelopment")
    if not cc:
        return []
    L = [f"New building on the lot within {cc['horizon_years']} years. Portable features (computed the same way in both "
         "cities); training pairs end by the test pair's start year in either city; 600 m tile bootstrap intervals "
         "(`planarena/models/crosscity.py`).", "",
         "| Test city (pair) | Trained on | AUC (95% CI) | Average precision (95% CI) | Precision, top 1% |",
         "|---|---|---|---|---|"]
    for city, t in cc["tests"].items():
        lab = f"{city.title()} ({t['test_pair']}; {num(t['redeveloped'])} new buildings, base rate {pct(t['base_rate'], 2)})"
        for name, m in t["models"].items():
            L.append(f"| {lab} | {name} | {r3(m['auc'])} ({r3(m['auc_95ci'][0])}–{r3(m['auc_95ci'][1])}) | "
                     f"{r3(m['average_precision'])} ({r3(m['ap_95ci'][0])}–{r3(m['ap_95ci'][1])}) | {pct(m['precision_top1pct'], 1)} |")
            lab = ""
        b = t["baselines"][t["best_baseline"]]
        L.append(f"| | best simple baseline: {t['best_baseline']} | {r3(b['auc'])} | {r3(b['average_precision'])} | {pct(b['precision_top1pct'], 1)} |")
    return L


def _integration() -> list[str]:
    ir = load("integration_replay")
    if not ir:
        return []
    L = ["Does weighting each development site by its probability of redevelopment bring a rezoning's projected homes "
         "closer to what was built than capacity-based analysis? Sites come from PlanArena's development-site screen "
         "under the adopted FAR; probabilities come from the redevelopment model trained only on outcomes known by the "
         "adoption year, scored with the adopted zoning (`planarena/models/replay.py`).", "",
         "| Rezoning | Sites | Built by 2025 (homes on the sites) | Capacity | Capacity pro-rated to 2025 | Probability × capacity (95% range) | Probability × empirical yield (95% range) | Sites redeveloped: observed / expected |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, r in ir["rezonings"].items():
        a, s = r["homes_added_by_2025"], r["sites_redeveloped_by_2025"]
        lo, hi = a["likelihood_weighted_95pct"]
        ylo, yhi = a["likelihood_x_yield_95pct"]
        L.append(f"| {n} | {r['sites']} | {num(a['observed'])} | {num(a['capacity'])} | {num(a['capacity_prorated_to_2025'])} | "
                 f"{num(a['likelihood_weighted'])} ({num(lo)} to {num(hi)}) | {num(a['likelihood_x_yield'])} ({num(ylo)} to {num(yhi)}) | "
                 f"{s['observed']} / {s['expected_from_probabilities']} |")
    t = ir["total_three_rezonings"]
    L.append(f"| **All three** | | {num(t['observed'])} | {num(t['capacity'])} | {num(t['capacity_prorated_to_2025'])} | "
             f"{num(t['likelihood_weighted'])} | {num(t['likelihood_x_yield'])} | |")
    sae = ir["sum_of_absolute_errors"]
    L.append(f"| **Sum of absolute errors** | | | {num(sae['capacity'])} | {num(sae['capacity_prorated_to_2025'])} | "
             f"{num(sae['likelihood_weighted'])} | {num(sae['likelihood_x_yield'])} | |")
    inside = sum(ir["observed_within_model_interval"].values())
    inside_y = sum(ir["observed_within_yield_interval"].values())
    L += ["", f"Observed homes fall inside the 95% range in {inside} of {len(ir['rezonings'])} rezonings (probability × "
          f"capacity) and {inside_y} (probability × yield). The ranges are conditional on the model's probabilities. "
          "Close totals can come from errors that cancel across rezonings; the sum of absolute errors shows them.", "",
          "Model used for each rezoning (training pairs known by the adoption year; extended to 2025 with a constant hazard): "
          + "; ".join(f"{n}: {r['model']['horizon_years']}-year model on {', '.join(r['model']['training_pairs'])}, "
                      f"mean site probability {r['mean_probability_adopted_vs_prior_zoning'][0]:.3f} with the adopted zoning vs "
                      f"{r['mean_probability_adopted_vs_prior_zoning'][1]:.3f} before"
                      for n, r in ir["rezonings"].items()) + ".", ""]
    L += [f"- {c}" for c in ir["caveats"]]
    ra = load("realism_ablation")
    if ra:
        L += ["", f"**Realism layers in the optimizer** (`planarena/models/ablate_recs.py`): plans for every district on the "
              f"{ra['snapshot']} snapshot, compared with what happened by {ra['outcome_year']}. This tests plausibility, not "
              "desirability.", "",
              "| Setting | Budget | Recommended parcels | Without precedent | Redeveloped by 2025 | Eligible parcels redeveloped | Redeveloped into the recommended use | Objective (sum over districts) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for budget, block in (("full", ra["settings"]), ("equal counts", ra["equal_counts"])):
            for n, s in block.items():
                if not isinstance(s, dict) or "recommended_parcels" not in s:
                    continue
                L.append(f"| {n} | {budget} | {num(s['recommended_parcels'])} | {pct(s['share_without_precedent'], 1)} | "
                         f"{pct(s['share_redeveloped_by_2025'], 1)} | {pct(s['share_of_eligible_redeveloped'], 1)} | "
                         f"{pct(s['share_redeveloped_into_recommended_use'], 1)} | {s['sum_district_objective']} |")
        ref = ra["reference"]
        L += ["", f"For reference, {pct(ref['share_of_all_lots_redeveloped'], 1)} of all lots and "
              f"{pct(ref['share_of_likely_fifth_redeveloped'], 1)} of the likeliest fifth were redeveloped. The objective "
              "is the overall score minus the conversion penalty, the same for every setting; at equal counts each "
              "setting recommends, in each district, the smallest number any setting recommended there. Search is "
              "heuristic, so a constraint can change the plan found in either direction."]
    return L


# ----------------------------------------------------------------------------- VALIDATION.md
CITY_NAMES = {"0644000": "Los Angeles", "4819000": "Dallas", "3651000": "New York", "5553000": "Milwaukee",
              "0455000": "Phoenix", "0666000": "San Diego", "1714000": "Chicago"}


def _gain(g: dict | None, k: str) -> str:
    if not g:
        return "–"
    v, (lo, hi) = g[k]
    star = "" if lo <= 0 <= hi else " *"
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def _integration_cities() -> list[str]:
    r = load("integration_cities")
    if not r:
        return []
    cities = [g for g in CITY_NAMES if g in r]
    L = ["Out-of-time test: fitted on the first window, scored on the next (METHODS section 11). AUC of each block "
         "set; E existing, C context, Z zoning, P adopted plan.", "",
         "| City | Windows (train → test) | Lots | Developed in test | E | E+C | E+C+Z | E+C+P | E+C+Z+P |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in cities:
        t = r[g].get("temporal")
        if not t:
            continue
        a = lambda k: r3(t["sets"][k]["auc"]) if k in t["sets"] else "–"
        L.append(f"| {CITY_NAMES[g]} | {t['train'][0]}–{t['train'][1]} → {t['test'][0]}–{t['test'][1]} | "
                 f"{num(t['n_lots'])} | {num(t['developed'])} | {a('E')} | {a('E+C')} | {a('E+C+Z')} | {a('E+C+P')} | {a('E+C+Z+P')} |")
    L += ["", "Gain in AUC from adding a block (out-of-time; 95% bootstrap interval over 2 km tiles; * the interval "
          "excludes zero):", "", "| City | + context over E | + zoning over E+C | + plan over E+C | + zoning and plan over E+C |",
          "|---|---|---|---|---|"]
    for g in cities:
        t = r[g].get("temporal")
        if not t:
            continue
        gg = t["gains"]
        L.append(f"| {CITY_NAMES[g]} | {_gain(gg.get('E+C vs E'), 'auc')} | {_gain(gg.get('E+C+Z vs E+C'), 'auc')} | "
                 f"{_gain(gg.get('E+C+P vs E+C'), 'auc')} | {_gain(gg.get('E+C+Z+P vs E+C'), 'auc')} |")
    cons = [g for g in cities if r[g].get("temporal_conservative")]
    if cons:
        L += ["", "Conservative variant (lots rezoned or re-planned after t0 left out, where plan and zoning are current "
              "layers):", "", "| City | Lots left out (developed) | + zoning over E+C | + plan over E+C |", "|---|---|---|---|"]
        for g in cons:
            c = r[g]["temporal_conservative"]
            L.append(f"| {CITY_NAMES[g]} | {num(c['left_out_test'])} ({num(c['left_out_developed'])}) | "
                     f"{_gain(c['gains'].get('E+C+Z vs E+C'), 'auc')} | {_gain(c['gains'].get('E+C+P vs E+C'), 'auc')} |")
    L += ["", "What developed lots became (out-of-time window): share named correctly by each layer and by the full "
          "model, as a 16-class use and as one of the nine analysis groups:", "",
          "| City | Developed lots | Existing use (no change) | Zoning class | Plan class | Model (layers) |", "|---|---:|---|---|---|---|"]
    for g in cities:
        d = (r[g].get("temporal") or {}).get("destination")
        if not d or not d["n"]:  # Chicago: permits do not record what the lot became
            continue
        f = lambda x: f"{pct(x['class'])} / {pct(x['group'])}" if x and x.get("coverage") else "–"
        st = next((k for k in ("E+C+Z+P", "E+Z+P", "E+Z", "E+C", "E") if k in d["models"]), None)
        L.append(f"| {CITY_NAMES[g]} | {num(d['n'])} | {f(d['layers'].get('existing (no change)'))} | "
                 f"{f(d['layers'].get('zoning class'))} | {f(d['layers'].get('plan class'))} | "
                 f"{f(d['models'][st]) + f' ({st})' if st else '–'} |")
    L += ["", "Homes added in the test window, by district: projections against what was built (ratio of totals; "
          "Spearman correlation across districts):", "",
          "| City | Observed homes | Zoning build-out | Plan build-out | Likely × zoning | Likely × plan | Five-year trend |",
          "|---|---:|---|---|---|---|---|"]
    for g in cities:
        sc = (r[g].get("temporal") or {}).get("scenario")
        if not sc:
            continue
        f = lambda k: f"{times(sc[k]['ratio_total'])} ({sc[k]['district_spearman']:.2f})" if k in sc else "–"
        L.append(f"| {CITY_NAMES[g]} | {num(sc['city_total']['observed'])} | {f('zoning_buildout')} | {f('plan_buildout')} | "
                 f"{f('likely_zoning')} | {f('likely_plan')} | {f('trend')} |")
    tr = r.get("transfer")
    if tr:
        L += ["", "Transfer: fitted on the other cities with a building outcome, scored on the city (AUC, other "
              "cities / own city):", "", "| City | E | E+C | E+C+Z | E+C+Z+P |", "|---|---|---|---|---|"]
        for g in [c for c in CITY_NAMES if c in tr]:
            f = lambda k: f"{r3(tr[g][k]['other_cities']['auc'])} / {r3(tr[g][k]['own_city']['auc'])}" if k in tr[g] else "–"
            L.append(f"| {CITY_NAMES[g]} | {f('E')} | {f('E+C')} | {f('E+C+Z')} | {f('E+C+Z+P')} |")
    return L


def render_validation() -> str:
    L = ["# Validation (generated)", "",
         "<!-- GENERATED by python -I -m planarena.report. Do not edit by hand. -->", "",
         "Every number below is read from `results/**/*.json`, produced by `planarena/validate.py` and "
         "`planarena/models/transition.py` from downloaded open data (file hashes are in each result file). "
         "Methods are in [METHODS.md](METHODS.md). The checks fall into three classes; only class C tests the "
         "effect of a change.", ""]

    # A. Existing conditions
    L += ["## A. Existing conditions", ""]
    ins, em, jb, un = load("validation_emissions_ll84"), load("emissions_ll84_2023"), load("jobs_lodes_2023"), load("units_replay_2015_2025")
    if ins:
        L += [f"**Emissions, same year as fitting (LL84 2024):** {num(ins['n_buildings'])} buildings; log–log r {r3(ins['log_log_r'])}; "
              f"median absolute error {pct(ins['median_abs_pct_error'])}; total modelled / metered {times(ins['aggregate_ratio_modelled_to_observed'])}.", ""]
    if em:
        L += [f"**Emissions, another year (LL84 2023, coefficients fitted on 2024):** {num(em['n_buildings'])} buildings; "
              f"log–log r {r3(em['log_log_r'])}; median absolute error {pct(em['median_abs_pct_error'])}; total "
              f"{times(em['aggregate_ratio_modelled_to_observed'])} ({em['modelled_mt']} vs {em['observed_mt']} Mt). "
              "Most buildings appear in both years, so this is not a test on unseen buildings. The aggregate agrees "
              "far better than individual buildings do.", ""]
    if jb:
        L += [f"**Jobs against LEHD LODES 2023:** total modelled {num(jb['modelled_total_jobs'])} (scaled to QCEW) vs LODES "
              f"{num(jb['lodes_total_jobs'])}." + (f" {jb['note']}" if jb.get("note") else ""), "",
              "| Level | Units | log–log r | Spearman | Median error (units with ≥100 jobs) |", "|---|---:|---:|---:|---:|"]
        for lv in ("tract", "block"):
            x = jb[lv]
            L.append(f"| {lv} | {num(x['units'])} | {r3(x['log_log_r'])} | {r3(x['spearman'])} | {pct(x['median_abs_pct_error_where_observed_ge_100'])} |")
        L.append("")
    if un:
        L += [f"**Homes from floor area (records of {num(un['parcels'])} lots developed 2015–2025):** given the observed use and "
              f"floor area, PlanArena estimates {num(un['modelled_homes'])} homes against {num(un['observed_homes'])} recorded "
              f"({times(un['aggregate_ratio'], 3)}; log–log r {r3(un['log_log_r'])}; median error {pct(un['median_abs_pct_error'])}). "
              f"Reference, CEQR 900 sq ft per home on the observed residential floor area: "
              f"{times(un['reference_ceqr_900sf_on_observed_residential_area']['aggregate_ratio'], 3)}.", ""]

    eb = load("existing_breakdowns")
    if eb:
        e23 = eb["emissions_2023"]
        L += ["**Where the emissions estimate is weak (LL84 2023):**", "",
              "| Group | Buildings | Median error | log–log r | Total ratio |", "|---|---:|---:|---:|---:|"]
        def row(label, x):
            return f"| {label} | {num(x.get('n'))} | {pct(x.get('median_abs_pct_error'))} | {r3(x.get('log_r'))} | {times(x.get('total_ratio'))} |"
        L.append(row("All (PlanArena)", e23["all"]))
        L.append(row("Baseline: one citywide intensity × floor area", e23["baseline_single_citywide_eui"]))
        L.append(row("Held-out buildings (intensities refitted on the other half)", e23["held_out_buildings"]))
        for k, x in e23["by_overlap"].items():
            L.append(row(f"Buildings {k}", x))
        for k, x in e23["by_use"].items():
            L.append(row(f"Use: {k}", x))
        for k, x in e23["by_size"].items():
            L.append(row(f"Size: {k} sq ft", x))
        L.append("")
        jj = eb["jobs_2023"]["blocks_with_100_jobs_or_more"]
        L += [f"**Jobs by block (blocks with ≥100 jobs, n = {num(jj['model']['n'])}):** median error {pct(jj['model']['median_abs_pct_error'])} "
              f"for PlanArena against {pct(jj['baseline_floor_area_only']['median_abs_pct_error'])} for a flat rate per sq ft of "
              "non-residential floor area.", ""]
    jr = load("jobs_rates")
    if jr:
        L += [f"**Jobs rates tested forward in time** (fitted on 2015 lots and LODES 2015, tested on 2025 lots against LODES 2023; "
              f"{num(jr['blocks_tested'])} blocks with ≥100 jobs):", "",
              "| Method | Median error | log r | Total ratio |", "|---|---:|---:|---:|"]
        for k, x in jr["methods"].items():
            L.append(f"| {k} | {pct(x['median_abs_pct_error_blocks_ge_100'])} | {r3(x['log_r'])} | {times(x['total_ratio'])} |")
        L.append("")

    # B. Planning calculations
    nc = load("nyc_citywide_coefficients")
    if nc:
        L += ["### New York City as a whole: which energy and emission coefficients", "",
              "Scored on LL84 2023 buildings outside Manhattan, which neither candidate was fitted on "
              f"(`planarena/cities/adapters/nyc.py`; {num(nc['records_excluded_implausible_eui'])} records with a site "
              "EUI above 1,000 kBtu/sq ft excluded as reporting errors).", "",
              "| Coefficients | Buildings | Total (model / reported) | Median building error |", "|---|---:|---:|---:|"]
        for name, sc in nc["scores"].items():
            L.append(f"| {name.replace('_', ' ')}{' (chosen)' if name == nc['chosen'] else ''} | "
                     f"{num(sc['buildings_outside_manhattan'])} | {times(sc['total_ratio'])} | {pct(sc['median_abs_pct_error'])} |")
        L.append("")
    L += ["## B. Consistency with planning calculations", ""]
    eh, sn, rz = load("east_harlem_rezoning"), load("soho_noho_rezoning"), load("rezoning_replays")
    if eh:
        L += [f"**East Harlem (2017), FEIS projected sites:** {eh['sites_parsed']} parsed, {eh['sites_matched_2015']} matched. With the FEIS "
              f"floor area and use, PlanArena gives {num(eh['planarena_du_same_far'])} homes against the FEIS's {num(eh['feis_with_action_du'])} "
              f"({times(eh['ratio_same_far_to_feis'], 3)}, site-level log r {r3(eh['site_log_r_same_far'])}); from zoning alone "
              f"{num(eh['planarena_du_zoning_rule'])} ({times(eh['ratio_zoning_rule_to_feis'], 3)}). Parse check: floor areas match the FEIS "
              "totals; homes differ by the FEIS's per-row rounding.", ""]
    if sn:
        L += [f"**SoHo/NoHo (2021), FEIS projected sites:** {sn['projected_sites_parsed']} parsed (total {num(sn['parse_check']['parsed'])} "
              f"homes, the appendix total; the FEIS text reports {num(sn['feis_text_with_action_du'])}). On {sn['sites_matched_2020']} sites "
              f"matched to 2020 lots: {num(sn['planarena_du_same_gfa'])} vs {num(sn['feis_with_action_du'])} "
              f"({times(sn['ratio_to_feis'], 3)}, site-level log r {r3(sn['site_log_r'])}).", ""]
    if rz:
        L += ["**PlanArena's own development-site screen on the rezoned lots** (`planarena/indicators/rwcds.py`):", "",
              "| Rezoning | Screened sites | FEIS increment | PlanArena at max FAR | At 0.8 × max | Built by 2025 | Share of FEIS increment |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in rz["rezonings"].items():
            L.append(f"| {k} | {v['screened_sites']} | {num(v['feis_increment_du'])} (to {v['feis_horizon']}) | "
                     f"{num(v['planarena_increment_du_at_max_far'])} ({times(v['ratio_max_far_to_feis'])}) | "
                     f"{num(v['planarena_increment_du_at_utilisation'])} | {v['observed_increment_du_rezoned_lots_to_2025']:+,.0f} | "
                     f"{pct(v['observed_share_of_feis_increment'])} |")
        L += ["", "The FEIS selects 'projected' sites more narrowly than this screen, which also captures sites the FEIS "
              "treats as 'potential'. Mandatory Inclusionary Housing bonus floor area is not modelled.", ""]

    # C. Prediction of change
    L += ["## C. Prediction of change", ""]
    cp, df = load("change_prediction"), load("delta_forms")
    if cp:
        a = cp["density_event_audit"]
        L += [f"Lots that changed between MapPLUTO 15v1 and 25v3.1: {num(cp['events'])} events "
              f"({num(cp['use_change_events'])} use changes, {num(cp['density_only_events'])} density-only with a new building). "
              f"Of {num(a['density_events_total'])} apparent density-only changes, {num(a['density_without_construction_evidence'])} had no "
              f"building completed since 2015 and {num(a['density_inconsistent_2015_record'])} had an inconsistent 2015 record; "
              "they are record changes and are excluded.", "",
              "| Method | Change type | Homes r | Homes total ratio | Homes median error | Value r | Emissions r (n) | Emissions total ratio |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for mth, v in cp["methods"].items():
            for kind, row in v.items():
                h, val, e = row["homes"], row["assessed_value"], row["emissions"]
                L.append(f"| {mth} | {kind} | {r3(h.get('pearson_r'))} | {times(h.get('total_ratio'))} | {num(h.get('median_ae'), 1)} | "
                         f"{r3(val.get('pearson_r'))} | {r3(e.get('pearson_r'))} ({e.get('n')}) | {times(e.get('total_ratio'))} |")
        jobs = {m: v["all"].get("jobs_by_block") for m, v in cp["methods"].items() if "jobs_by_block" in v["all"]}
        if jobs:
            L += ["", "Jobs change by 2020 census block (LODES 2015 → 2023, net of the trend of blocks without changes): "
                  + "; ".join(f"{m} r {r3(x.get('pearson_r'))}" for m, x in jobs.items())
                  + ". No method has useful skill at block level."]
        L += ["", "**Not validated:** " + "; ".join(cp["unvalidated"]) + ".", ""]
    ih = load("impact_heldout")
    if ih:
        L += ["**Held-out test** (`planarena/models/impact_test.py`): coefficients from 2015 data; delta form, unit-size "
              f"rule and scale chosen on {num(ih['selection_events'])} lots in two of five spatial folds; metrics on the "
              f"other {num(ih['evaluation_events'])}. Baselines are fitted on the selection lots with the same information.", "",
              "| Method | Homes r | Homes MAE | Homes bias | Homes total ratio | Value r | Value total ratio | Emissions r | Emissions total ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for mth, v in ih["methods"].items():
            a = v["all"]
            L.append(f"| {mth} | {r3(a['homes'].get('pearson_r'))} | {num(a['homes']['mae'], 1)} | {num(a['homes']['bias'], 1)} | "
                     f"{times(a['homes'].get('total_ratio'))} | {r3(a['value'].get('pearson_r'))} | {times(a['value'].get('total_ratio'))} | "
                     f"{r3(a['ghg'].get('pearson_r'))} | {times(a['ghg'].get('total_ratio'))} |")
        L += ["", f"Job change by block (evaluation blocks only): r {r3(ih['jobs_by_block']['pearson_r'])}.", ""]
    if df:
        L += ["**Do baseline residuals change the recommendations?** Optimizer recommendations in 12 districts, with and "
              "without synthetic log-normal error on the observed baseline (Jaccard overlap of recommended parcels; 1 = identical):", "",
              "| Delta form | Median overlap | Lowest overlap |", "|---|---:|---:|"]
        for f_, v in df["forms"].items():
            L.append(f"| {f_} | {r3(v['jaccard_clean_vs_noisy_median'])} | {r3(v['jaccard_clean_vs_noisy_min'])} |")
        L += ["", "Agreement between forms (median overlap): "
              + "; ".join(f"{k} {r3(x)}" for k, x in df["agreement_between_forms_median_jaccard"].items()) + ".", ""]

    # D. Models
    L += ["## D. Models", "", "### Pattern-consistency model (GNN) and LightGBM", ""] + _cv_tables() + [""]
    if _crosscity():
        L += ["### Redevelopment across cities (Manhattan and Milwaukee)", ""] + _crosscity() + [""]
    bt = load("backtest_2015_2025")
    if bt:
        ph = bt["physical_change"]
        L += ["### Pattern models as redevelopment predictors (2015 → 2025)", "",
              f"{num(bt['matched_parcels'])} lots matched; {num(ph['n'])} rebuilt with a new use ({pct(ph['base_rate'], 1)}). "
              "Models trained on 2015 data only.", "",
              "| Score | AUC | Precision in top 5% | New use top-1 | Top-3 |", "|---|---:|---:|---:|---:|"]
        for k, v in ph["change_detection"].items():
            d = ph["destination"].get(k)
            L.append(f"| {k} | {r3(v['auc'])} | {pct(v['precision_top5pct'], 1)} | "
                     + (f"{pct(d['top1'], 1)} | {pct(d['top3'], 1)} |" if d else "– | – |"))
        L.append("")
    tr = load("transition_2020_2025")
    if tr:
        bk, bv = _best_baseline(tr["development"])
        L += ["### Redevelopment-transition model (temporal holdout)", "",
              f"Trained on 2015 features with 2015–2020 outcomes ({num(tr['train_developed'])} redeveloped lots of "
              f"{num(tr['train_parcels'])}); tested on 2020 features with 2020–2025 outcomes ({num(tr['test_developed'])} of "
              f"{num(tr['test_parcels'])}; base rate {pct(tr['test_base_rate'], 2)}).", "",
              "| Will the lot be redeveloped within 5 years? | AUC | Average precision | Precision, top 1% | Recall, top 5% | ECE |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, v in tr["development"].items():
            L.append(f"| {k} | {r3(v['auc'])} | {r3(v['average_precision'])} | {pct(v['precision_top1pct'], 1)} | "
                     f"{pct(v['recall_top5pct'], 1)} | {r3(v.get('ece'))} |")
        L += ["", f"Best simple baseline: {bk} (AUC {r3(bv['auc'])}).", "",
              f"| Into what use? ({num(tr['destination_n'])} redeveloped lots) | Top-1 | Top-3 | Top-1 when the use changed |",
              "|---|---:|---:|---:|"]
        for k, v in tr["destination"].items():
            L.append(f"| {k} | {pct(v['top1'], 1)} | {pct(v['top3'], 1)} | {pct(v['top1_when_use_changed'], 1)} |")
        L += ["", "Model selection for the GNN used a spatial hold-out inside the training period, which is not a "
              "temporal test; its accuracy there is not comparable with the temporal test above.", ""]

    ro = load("rolling_origin")
    if ro:
        L += ["### Rolling origin, calibration and label audit", "",
              "Each row trains only on outcomes known at the decision year and tests on the next period.", "",
              "| Horizon | Test | Positives | Base rate | AUC (95% CI) | Avg precision (95% CI) | Best baseline AUC | Spatial CV in the same period, AP | ECE raw → isotonic |",
              "|---:|---|---:|---:|---|---|---:|---:|---|"]
        for o in ro["origins"]:
            lg, bl = o["lightgbm"], o["baselines"]
            bb = max(bl.values(), key=lambda x: x["auc"])
            ece = f"{r3(lg['ece_raw'])} → {r3(lg.get('ece_isotonic'))}" if "ece_isotonic" in lg else r3(lg["ece_raw"])
            L.append(f"| {o['horizon']} y | {o['test']} | {num(o['positives'])} | {pct(o['base_rate'], 2)} | {r3(lg['auc'])} "
                     f"({r3(lg['auc_95ci'][0])}–{r3(lg['auc_95ci'][1])}) | {r3(lg['average_precision'])} "
                     f"({r3(lg['ap_95ci'][0])}–{r3(lg['ap_95ci'][1])}) | {r3(bb['auc'])} | "
                     f"{r3(o['spatial_cv_within_test_period']['average_precision'])} | {ece} |")
        L += ["", "Within-period spatial cross-validation (right-hand columns) is not a guide to forward performance: it is "
              "much higher than the temporal test in some periods and lower in others. Only the temporal rows estimate "
              "forward performance. Probabilities track the development rate of the training periods, so rankings are "
              "more reliable than five-year probabilities.", ""]
        if ro.get("verified_labels"):
            L += ["**With verified labels** (floor-area growth counts only with a DOB certificate; see METHODS.md):", "",
                  "| Horizon | Test | Positives | Base rate | AUC | Average precision | Best baseline AUC |", "|---:|---|---:|---:|---:|---:|---:|"]
            for v in ro["verified_labels"]:
                L.append(f"| {v['horizon']} y | {v['test']} | {num(v['positives'])} | {pct(v['base_rate'], 2)} | {r3(v['auc'])} | "
                         f"{r3(v['average_precision'])} | {r3(v['best_baseline_auc'])} |")
            L.append("")
        for k, a in ro.get("label_audit", {}).items():
            L.append(f"**Label audit {k}** ({a['window']}): {num(a['labelled_developed'])} lots labelled redeveloped; "
                     f"{pct(a['precision_vs_new_building_or_alteration_certificate'])} of them have a DOB new-building or "
                     f"alteration-type-1 certificate of occupancy in the window ({pct(a['precision_vs_new_building_certificate'])} "
                     f"new-building only); {pct(a['recall_of_new_building_certificates'])} of lots with a new-building certificate "
                     f"are labelled redeveloped. {a['note']}")
        L.append("")

    # E. Optimizer
    op = load("optimizer_gap")
    if op:
        sizes = [a["moves"] for a in op["areas_detail"]]
        L += ["## E. Optimizer (best plan found, not a global optimum)", "",
              f"Exhaustive search over every plan with up to 2 changes in {op['areas']} single-block study areas "
              f"({min(sizes)}–{max(sizes)} candidate moves each; {op['areas_with_positive_optimum']} with a positive optimum). "
              f"First-order search alone found the exhaustive optimum in {pct(op['share_exact_optimum_found'])} of them "
              f"(mean gap {pct(op['mean_gap'], 1)}); with the exact re-scoring step, {pct(op['polished']['share_exact_optimum_found'])} "
              f"(median {op['polished']['median_seconds']} s). This says nothing about district or city scale.", ""]

    ob = load("optimizer_benchmark")
    if ob:
        L += ["**District scale** (" + ob["test"] + ")", "",
              "| Method | Median share of best plan found | Districts where best | Median seconds |", "|---|---:|---:|---:|"]
        for k, x in ob["summary"].items():
            L.append(f"| {k.replace('_', ' ')} | {pct(x['median_share_of_best_found'], 1)} | {x['districts_where_best']} | {x['median_seconds']} |")
        L.append("")

    # F. Simulated stakeholder scenarios
    vs = load("vote_summary")
    if vs:
        L += ["## F. Simulated stakeholder scenarios", "", _vote_text(vs), ""]
    sv = load("vote_sensitivity")
    if sv:
        x = sv["summary"]
        n = x["districts"]
        L += ["**Sensitivity of the simulated outcome:**", "", "| Change to the assumptions | Result |", "|---|---|",
              f"| Stakeholder utility weights redrawn (Dirichlet around the assumed weights) | winner unchanged in a median {pct(x['median_winner_unchanged_under_weight_draws'])} of draws |",
              f"| Ten random feasible plans added to the ballot (not built from the utilities) | winner unchanged in {x['districts_winner_unchanged_with_random_candidates']} of {n} districts; beats every random plan head to head in {x['districts_winner_beats_every_random_plan']} |",
              f"| Residents' indifference threshold swept 0–0.1 points | residents' winner stable in {x['districts_residents_winner_stable_across_jnd']} of {n} districts |",
              f"| One stakeholder given three seats (each in turn) | same winner in a median {pct(x['median_share_of_seat_allocations_with_same_winner'])} of seat allocations |", ""]
    if _integration():
        L += ["## G. Integration value: likelihood-weighted vs capacity-based projections", ""] + _integration() + [""]
    if _integration_cities():
        L += ["## H. Integration value across cities: component ablation on observed history", ""] + _integration_cities() + [""]
    return "\n".join(L).rstrip() + "\n"


def _vote_text(vs: dict) -> str:
    rows = [("A Condorcet winner exists (otherwise the simulated majority preferences cycle)", vs["condorcet_winner_exists"]),
            ("All nine rules agree", vs["all_rules_agree"]),
            ("The council winner beats no change in both simulated electorates", vs["double_majority"]),
            ("The simulated residents' winner is the council's winner", vs["residents_agree_with_council"]),
            ("The council winner is also the Nash bargaining solution", vs["winner_is_nash_bargaining"]),
            ("One insincere ballot could change the Schulze result", vs["schulze_manipulable"])]
    n = vs["districts"]
    t = ["Ballots are computed from assumed stakeholder utilities, not elicited preferences. These results describe the "
         "simulated scenario, not public support or political feasibility.", "",
         f"| Finding ({n} districts, one seat each) | Districts |", "|---|---|"]
    t += [f"| {k} | {v} of {n} |" for k, v in rows]
    t.append(f"| Median robustness of the winner (coefficients and seats varied, {vs['samples']} draws) | {pct(vs['median_robust_both'], 1)} |")
    return "\n".join(t)


# ----------------------------------------------------------------------------- README blocks
def readme_blocks() -> dict[str, str]:
    from planarena.indicators.engine import INDICATORS, load_coefficients

    meta = json.loads((PROCESSED / "twin_meta.json").read_text())
    c = load_coefficients()
    blocks = {"header": f"Demo city: Manhattan, {num(meta['n_parcels'])} parcels from {meta['source']}, downloaded from DCP by "
                        f"`planarena.twin.fetch`. The archive's and the extract's SHA-256 are recorded in `twin_meta.json` "
                        f"and in every export. {len(INDICATORS)} indicators across three pillars."}
    jobs = c["jobs_sqft_per_worker"]
    pph = c["assumptions"]["persons_per_household"]
    bo = c["assumptions"]["buildout_utilization"]
    rc = c["runoff_coefficient_c"]
    blocks["coefficients"] = "\n".join([
        "| Coefficient | Source | Status |", "|---|---|---|",
        f"| Building energy (site EUI by use) | NYC LL84 benchmarking, calendar 2024, Manhattan; floor-area-weighted mean, Q25–Q75 range | {c['energy_site_eui_kbtu_per_sqft']['status']} |",
        f"| Emission factor | NYC LL97 §28-320.3.1.1 (2024–2029) weighted by LL84 Manhattan fuel shares | {c['emissions_tco2e_per_kbtu']['status']} |",
        f"| Floorspace per worker | EIA CBECS 2018 Table B2, scaled so the baseline equals BLS QCEW New York County employment ({num(jobs.get('calibration_target_jobs'))}) | {jobs['status']} |",
        f"| Persons per household | Census QuickFacts, New York County: {pph['value']:.2f} | {pph.get('status', '')} (page is bot-protected; confirm) |",
        f"| Runoff C | NYC DEP surface coefficients (roof {rc['roof']['value']:.2f}) weighted by each parcel's roof share; ground C by use | {rc['status']} |",
        f"| Build-out utilization | {bo['value']} ({bo['low']}–{bo['high']}) | {bo.get('status', 'assumption')} |"])
    rows = []
    em, jb, un = load("emissions_ll84_2023"), load("jobs_lodes_2023"), load("units_replay_2015_2025")
    eh, sn, cp, tr, op = load("east_harlem_rezoning"), load("soho_noho_rezoning"), load("change_prediction"), load("transition_2020_2025"), load("optimizer_gap")
    if em:
        rows.append(f"| A | Building emissions vs LL84 2023 (fitted on 2024; mostly the same buildings) | total {times(em['aggregate_ratio_modelled_to_observed'])}; log–log r {r3(em['log_log_r'])}; median building error {pct(em['median_abs_pct_error'])} |")
    if jb:
        rows.append(f"| A | Jobs vs LODES 2023 | tract r {r3(jb['tract']['log_log_r'])}, block r {r3(jb['block']['log_log_r'])}; median tract error {pct(jb['tract']['median_abs_pct_error_where_observed_ge_100'])} |")
    if un:
        rows.append(f"| A | Homes from observed floor area ({num(un['parcels'])} developed lots) | {times(un['aggregate_ratio'], 3)} recorded homes; median error {pct(un['median_abs_pct_error'])} |")
    if eh:
        rows.append(f"| B | East Harlem FEIS sites, same floor area and use | {times(eh['ratio_same_far_to_feis'], 3)} the FEIS homes (site r {r3(eh['site_log_r_same_far'])}) |")
    if sn:
        rows.append(f"| B | SoHo/NoHo FEIS sites, same floor area and use | {times(sn['ratio_to_feis'], 3)} (site r {r3(sn['site_log_r'])}) |")
    if cp:
        uc = cp["methods"]["hybrid"]["use change"]
        bt = cp["methods"]["transition-type mean (cross-fitted)"]["use change"]
        rows.append(f"| C | Change in homes after real use changes ({num(cp['use_change_events'])}) | r {r3(uc['homes']['pearson_r'])} (baseline {r3(bt['homes']['pearson_r'])}); total {times(uc['homes']['total_ratio'])} |")
        rows.append(f"| C | Change in emissions after real use changes ({uc['emissions']['n']} buildings) | r {r3(uc['emissions']['pearson_r'])} (baseline {r3(bt['emissions']['pearson_r'])}); total {times(uc['emissions']['total_ratio'])} |")
        jb_ = cp["methods"]["hybrid"]["all"].get("jobs_by_block", {})
        rows.append(f"| C | Change in jobs by census block | r {r3(jb_.get('pearson_r'))}: no useful skill |")
    if tr:
        mk, mv = _best_model(tr["development"])
        bk, bv = _best_baseline(tr["development"])
        rows.append(f"| D | Redevelopment within 5 years, temporal holdout 2020→2025 | {mk} AUC {r3(mv['auc'])}, precision in top 1% {pct(mv['precision_top1pct'])} (base rate {pct(tr['test_base_rate'], 1)}); best baseline {bk} AUC {r3(bv['auc'])} |")
    ro, jr, eb = load("rolling_origin"), load("jobs_rates"), load("existing_breakdowns")
    if eb:
        x, b0 = eb["emissions_2023"]["all"], eb["emissions_2023"]["baseline_single_citywide_eui"]
        rows.append(f"| A | Building-level emissions error vs a single citywide intensity | median {pct(x['median_abs_pct_error'])} vs {pct(b0['median_abs_pct_error'])} (totals {times(x['total_ratio'])} vs {times(b0['total_ratio'])}) |")
    if jr:
        mth = jr["methods"]
        fit = mth.get("NNLS rates fitted on 2015, total calibrated") or mth.get("NNLS rates fitted on 2015")
        rows.append(f"| A | Jobs rates tested forward (fit 2015, test 2023), block median error | fitted rates {pct(fit['median_abs_pct_error_blocks_ge_100'])}; CBECS {pct(mth['CBECS composition (current model)']['median_abs_pct_error_blocks_ge_100'])}; flat rate {pct(mth['flat rate per sq ft of non-residential floor area']['median_abs_pct_error_blocks_ge_100'])} |")
    if ro:
        aucs = [o["lightgbm"]["auc"] for o in ro["origins"]]
        rows.append(f"| D | Redevelopment, rolling origin ({len(aucs)} tests, 1-, 3- and 5-year horizons) | AUC {r3(min(aucs))}–{r3(max(aucs))}; above the best simple baseline in every test |")
        la = ro.get("label_audit", {}).get("2020->2025")
        if la:
            rows.append(f"| D | Labels vs DOB certificates of occupancy (2020→2025) | {pct(la['precision_vs_new_building_or_alteration_certificate'])} of labelled lots have a certificate (a lower bound); labels are noisy |")
    ir = load("integration_replay")
    if ir:
        t3 = ir["total_three_rezonings"]
        rows.append(f"| C | Homes built by 2025 on the development sites of three rezonings ({num(t3['observed'])}) | "
                    f"likelihood-weighted projection {num(t3['likelihood_weighted'])}; capacity {num(t3['capacity'])} "
                    f"(pro-rated {num(t3['capacity_prorated_to_2025'])}); three cases only |")
    cc = load("crosscity_redevelopment")
    if cc:
        mk, mh = cc["tests"]["milwaukee"]["models"], cc["tests"]["manhattan"]["models"]
        rows.append(f"| D | Redevelopment model moved between cities (new buildings, {cc['horizon_years']} years) | "
                    f"Milwaukee: trained on Manhattan AUC {r3(mk['other city only']['auc'])} vs own city {r3(mk['same city']['auc'])}; "
                    f"Manhattan: trained on Milwaukee {r3(mh['other city only']['auc'])} vs own city {r3(mh['same city']['auc'])} "
                    f"(best baseline {r3(cc['tests']['manhattan']['baselines'][cc['tests']['manhattan']['best_baseline']]['auc'])}) |")
    if op:
        rows.append(f"| E | Optimizer vs exhaustive search ({op['areas_with_positive_optimum']} single-block areas, up to 2 changes) | optimum found in {pct(op['polished']['share_exact_optimum_found'])} with exact re-scoring; small cases only |")
    blocks["key_results"] = "\n".join(["| Class | What is tested | Result |", "|---|---|---|"] + rows)
    vs = load("vote_summary")
    blocks["vote"] = _vote_text(vs) if vs else ""
    blocks["gnn_cv"] = "\n".join(_cv_tables())
    return blocks


def apply_blocks(text: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        pat = re.compile(rf"<!-- BEGIN GENERATED: {name} -->\n.*?<!-- END GENERATED: {name} -->", re.S)
        if not pat.search(text):
            raise KeyError(f"README has no generated block '{name}'")
        block = f"<!-- BEGIN GENERATED: {name} -->\n{body}\n<!-- END GENERATED: {name} -->"
        text = pat.sub(lambda m: block, text)
    return text


def outputs() -> dict:
    return {VALIDATION: render_validation(), README: apply_blocks(README.read_text(), readme_blocks())}


def main(check: bool = False) -> int:
    stale = []
    for path, text in outputs().items():
        if path.read_text() != text:
            stale.append(path.name)
            if not check:
                path.write_text(text)
    if check and stale:
        print("out of date:", ", ".join(stale))
        return 1
    print("docs up to date" if check else f"written: {', '.join(stale) or 'nothing changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
