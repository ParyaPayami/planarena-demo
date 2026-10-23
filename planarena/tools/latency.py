"""How long the impact engine takes to score a change, as a planner experiences it in the Studio.

For the Manhattan demo twin and the largest city twin (New York, all boroughs): a one-parcel change and a 50-parcel
change in one district, each evaluated with the sensitivity ranges on, 20 times after a warm-up. Writes the medians
to results/validation/latency.json with the machine description and each twin's derived settings (unit size, targets).

Run:  python -I -m planarena.tools.latency
"""
from __future__ import annotations

import json
import platform
import time

import numpy as np

from planarena.cities.registry import city_dir
from planarena.config import PROCESSED, ROOT, USE_INDEX
from planarena.indicators.engine import load_engine

OUT = ROOT / "results" / "validation" / "latency.json"


def bench(engine, n_changes: int, reps: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cd = engine.p.cd.to_numpy()
    district = cd[rng.integers(len(cd))]
    lots = np.nonzero((cd == district) & (engine.base_use >= 0))[0]
    pick = rng.choice(lots, size=min(n_changes, len(lots)), replace=False)
    use = engine.base_use.copy()
    use[pick] = USE_INDEX["mix"]
    engine.evaluate(use, ranges=True)  # warm-up
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        engine.evaluate(use, ranges=True)
        ms.append(1000 * (time.perf_counter() - t0))
    return {"changes": int(len(pick)), "median_ms": round(float(np.median(ms)), 1), "max_ms": round(float(max(ms)), 1)}


def main() -> dict:
    out = {"machine": f"{platform.machine()} {platform.system()} {platform.release()}, Python {platform.python_version()}",
           "test": "ImpactEngine.evaluate with sensitivity ranges, after one warm-up; 20 repetitions", "twins": {}}
    for name, path in (("Manhattan", PROCESSED), ("New York City", city_dir("3651000"))):
        e = load_engine(path)
        out["twins"][name] = {"parcels": int(e.n), "one_change": bench(e, 1), "fifty_changes": bench(e, 50),
                              "derived": e.derived_summary() | {"sqft_per_unit_new": round(e.sqft_per_unit_new, 1)}}
        print(name, out["twins"][name], flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
