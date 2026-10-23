"""Compare the result files of a reproduction with the committed ones, number by number.

Run:  python -I -m planarena.tools.compare_results <reproduction folder> [<commit>]
The reproduction folder is a clean clone after `make reproduce`; the commit defaults to the clone's HEAD. Run times
(keys ending in seconds or s) are expected to differ and are not counted.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from planarena.config import ROOT

TIMING = re.compile(r"(seconds|/s|s_mean|/ms|build_seconds)$")


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def compare(clone: Path, commit: str | None = None) -> list[dict]:
    commit = commit or subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True,
                                      check=True).stdout.strip()
    out = []
    for f in sorted(p.relative_to(clone) for p in (clone / "results").rglob("*.json")):
        r = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{f}"], capture_output=True, text=True)
        if r.returncode:
            out.append({"file": str(f), "status": "not committed"})
            continue
        a, b = dict(leaves(json.loads(r.stdout))), dict(leaves(json.loads((clone / f).read_text())))
        keys = [k for k in a.keys() & b.keys() if not TIMING.search(k)]
        diff = [k for k in keys if a[k] != b[k]]
        worst = None
        for k in diff:
            x, y = a[k], b[k]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                rel = abs(x - y) / max(abs(x), abs(y), 1e-9)
                if worst is None or rel > worst[1]:
                    worst = (k, rel, x, y)
        out.append({"file": str(f), "values": len(keys), "different": len(diff),
                    "only_one_side": len(a.keys() ^ b.keys()),
                    "largest_relative_difference": None if worst is None else
                    {"key": worst[0], "relative": round(worst[1], 4), "committed": worst[2], "reproduced": worst[3]}})
    return out


if __name__ == "__main__":
    rows = compare(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
    for r in rows:
        print(f"{r['file']}: " + (r.get("status") or f"{r['values'] - r['different']}/{r['values']} identical"
                                  + (f"; largest difference {r['largest_relative_difference']}" if r["different"] else "")))
    sys.exit(1 if any(r.get("different") for r in rows) else 0)
