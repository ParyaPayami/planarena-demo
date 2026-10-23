"""Figures for the paper, drawn only from result files (like planarena/report.py for numbers).

Run:  python -I -m planarena.figures        -> docs/figures/*.png and *.svg
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from planarena.config import ROOT  # noqa: E402

RES = ROOT / "results" / "validation"
OUT = ROOT / "docs" / "figures"
INK, MUTED, A, B, C, D = "#1f2933", "#7b8794", "#2f6fb0", "#d9822b", "#3f8f5a", "#a33b3b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150})


def _load(name):
    f = RES / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return name


def integration():
    r = _load("integration_replay")
    if not r:
        return None
    names = list(r["rezonings"])
    keys = [("capacity", "Capacity", MUTED), ("capacity_prorated_to_2025", "Capacity pro-rated to 2025", B),
            ("likelihood_weighted", "Probability × capacity", A), ("observed", "Built by 2025", INK)]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(names))
    w = 0.2
    for k, (key, label, col) in enumerate(keys):
        v = [max(r["rezonings"][n]["homes_added_by_2025"][key], 1) for n in names]
        bars = ax.bar(x + (k - 1.5) * w, v, w, label=label, color=col)
        if key == "likelihood_weighted":
            lo = [r["rezonings"][n]["homes_added_by_2025"]["likelihood_weighted_95pct"][0] for n in names]
            hi = [r["rezonings"][n]["homes_added_by_2025"]["likelihood_weighted_95pct"][1] for n in names]
            ax.errorbar(x + (k - 1.5) * w, v, yerr=[np.maximum(np.array(v) - np.maximum(lo, 1), 0), np.array(hi) - np.array(v)],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        for b_, val in zip(bars, v):
            ax.annotate(f"{val:,.0f}", (b_.get_x() + b_.get_width() / 2, val), ha="center", va="bottom", fontsize=6.5,
                        xytext=(0, 1), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("Homes added on development sites")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Projected and built homes after three Manhattan rezonings (bars: model 95% range)", fontsize=9, loc="left")
    return _save(fig, "integration_replay")


def rolling():
    r = _load("rolling_origin")
    if not r:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    cols = {1: A, 3: B, 5: C}
    for h in (1, 3, 5):
        o = sorted([x for x in r["origins"] if x["horizon"] == h], key=lambda x: x["decision_year"])
        if not o:
            continue
        t = [x["decision_year"] for x in o]
        for ax, met, ci in ((axes[0], "auc", "auc_95ci"), (axes[1], "average_precision", "ap_95ci")):
            y = [x["lightgbm"][met] for x in o]
            lo = [x["lightgbm"][ci][0] for x in o]
            hi = [x["lightgbm"][ci][1] for x in o]
            ax.plot(t, y, marker="o", ms=3, color=cols[h], label=f"{h}-year")
            ax.fill_between(t, lo, hi, color=cols[h], alpha=0.15, lw=0)
        b = [max(v["auc"] for v in x["baselines"].values()) for x in o]
        axes[0].plot(t, b, ls="--", lw=0.8, color=cols[h])
    axes[0].set_ylabel("AUC (dashed: best simple baseline)")
    axes[1].set_ylabel("Average precision")
    for ax in axes:
        ax.set_xlabel("Decision year")
    axes[0].legend(frameon=False, fontsize=7, title="Horizon", title_fontsize=7)
    fig.suptitle("Redevelopment model tested forward in time (Manhattan, 2015-2025)", fontsize=9, x=0.01, ha="left")
    return _save(fig, "rolling_origin")


def crosscity():
    r = _load("crosscity_redevelopment")
    if not r:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    rows, y = [], 0
    for city, t in r["tests"].items():
        for name, m in t["models"].items():
            rows.append((f"{city.title()}: {name}", m["auc"], m["auc_95ci"], y))
            y += 1
        b = t["baselines"][t["best_baseline"]]
        rows.append((f"{city.title()}: best baseline", b["auc"], [b["auc"], b["auc"]], y))
        y += 1.6
    for label, v, ci, yy in rows:
        col = MUTED if "baseline" in label else (B if "other city" in label else A)
        ax.errorbar(v, yy, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", ms=4, color=col, ecolor=col, capsize=2)
    ax.set_yticks([r_[3] for r_ in rows], [r_[0] for r_ in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("AUC, new building within 3 years (95% tile bootstrap)")
    ax.set_title("Redevelopment model moved between cities", fontsize=9, loc="left")
    return _save(fig, "crosscity")


def layers():
    r = _load("integration_cities")
    if not r:
        return None
    from planarena.report import CITY_NAMES

    cities = [g for g in CITY_NAMES if g in r and (r[g].get("temporal") or {}).get("gains")]
    keys = [("E+C vs E", "+ context over E", A), ("E+C+Z vs E+C", "+ zoning over E+C", B),
            ("E+C+P vs E+C", "+ plan over E+C", C)]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(cities) + 0.9))
    for i, g in enumerate(cities):
        gains = r[g]["temporal"]["gains"]
        for k, (key, label, col) in enumerate(keys):
            if key not in gains:
                continue
            v, (lo, hi) = gains[key]["auc"]
            y = i + (k - 1) * 0.22
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", ms=3.5, color=col, ecolor=col, capsize=2,
                        label=label if i == 0 or not ax.get_legend_handles_labels()[1].count(label) else None)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(cities)), [CITY_NAMES[g] for g in cities])
    ax.invert_yaxis()
    ax.set_xlabel("Gain in AUC, out-of-time test window (95% bootstrap over 2 km tiles)")
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("What each layer adds to predicting where development happens", fontsize=9, loc="left")
    return _save(fig, "integration_layers")


def impact_scatter():
    """Observed against predicted change on the held-out evaluation lots (Manhattan 2015-2025)."""
    r = _load("impact_heldout")
    if not r or "points_value" not in r:
        return None
    # (result key, title, unit, linear range around zero, decades labelled)
    panels = [("points_homes", "Homes", "homes", 1.0, [1, 10, 100]),
              ("points_value", "Assessed value", "USD", 1e4, [1e5, 1e7]),
              ("points_ghg", "Building emissions", "tCO$_2$e/yr", 10.0, [10, 1000])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (k, title, unit, th, dec) in zip(axes, panels):
        pr, ob = np.array(r[k]["predicted"]), np.array(r[k]["observed"])
        lim = float(np.percentile(np.abs(np.r_[pr, ob]), 99.5)) * 1.1 or 1.0
        ax.scatter(pr, ob, s=4, alpha=0.35, color=A, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
        ticks = [-d for d in dec[::-1]] + [0] + dec
        lab = [("$-$" if t < 0 else "") + (f"{abs(t):g}" if abs(t) < 1e4 else f"$10^{{{int(np.log10(abs(t)))}}}$")
               if t else "0" for t in ticks]
        for axis in ("x", "y"):
            getattr(ax, f"set_{axis}scale")("symlog", linthresh=th)
            getattr(ax, f"set_{axis}lim")(-lim, lim)
            getattr(ax, f"set_{axis}ticks")(ticks, lab)
        ax.set_title(f"{title} (n = {len(pr):,})", fontsize=8, loc="left")
        ax.set_xlabel(f"predicted change ({unit})", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("observed change", fontsize=7)
    return _save(fig, "impact_scatter")


def reliability():
    """Reliability of redevelopment probabilities: Manhattan by horizon (latest origin), and each city's test window."""
    m, c = _load("calibration_manhattan"), (_load("calibration_cities") or {}).get("cities")
    if not m:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for h, col in ((1, A), (3, B), (5, C)):
        rows = [o for o in m["origins"] if o["horizon"] == h]
        if not rows:
            continue
        o = rows[-1]
        rel = o["raw"]["reliability"]
        ax.plot([b["mean_predicted"] for b in rel], [max(b["observed_rate"], 1e-4) for b in rel], marker="o", ms=3,
                color=col, lw=1, label=f"{h}-year, {o['test'].replace('->', '–')}")
    ax.set_title("Manhattan (latest origin per horizon)", fontsize=8, loc="left")
    ax = axes[1]
    cols = [A, B, C, D, MUTED, INK, "#8e6bbf"]
    names = {"0644000": "Los Angeles", "4819000": "Dallas", "3651000": "New York", "5553000": "Milwaukee",
             "0455000": "Phoenix", "0666000": "San Diego", "1714000": "Chicago"}
    if c:
        for (g, name), col in zip(names.items(), cols):
            cal = c.get(g)
            if not cal:
                continue
            k = next(x for x in cal if x not in ("E", "train", "test"))
            rel = cal[k]["raw"]["reliability"]
            ax.plot([b["mean_predicted"] for b in rel], [max(b["observed_rate"], 1e-4) for b in rel], marker="o",
                    ms=2.5, color=col, lw=0.9, label=name)
    ax.set_title("Seven cities, all blocks (test window)", fontsize=8, loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = 1e-4, 1.0
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("mean predicted probability (decile; observed 0 drawn at 0.0001)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    axes[0].set_ylabel("observed rate", fontsize=7)
    return _save(fig, "reliability")


def main():
    made = [f() for f in (integration, rolling, crosscity, layers, impact_scatter, reliability)]
    print("written:", ", ".join(m for m in made if m))


if __name__ == "__main__":
    main()
