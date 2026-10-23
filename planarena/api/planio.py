"""Plan import (with a validation review) and export (CSV, GeoJSON, provenance JSON, staff report).

Import rules:
* Rows are matched to authoritative parcels by BBL or PlanArena index. Imported
  geometry never replaces parcel boundaries.
* Land use may be a PlanArena key, a label or a MapPLUTO LandUse code (01-11).
  FAR is optional and must be between 0 and 30.
* Shapefile ZIPs are extracted safely: no path traversal, size limits, exactly one
  .shp file. GeoPackages must have exactly one layer.
"""
from __future__ import annotations

import csv
import html
import io
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from planarena.config import LAND_USES, PLUTO_TO_USE, USE_INDEX, USE_KEYS

MAX_BYTES = 120 * 1024 * 1024
MAX_ROWS = 100_000
ID_KEYS = ("bbl", "i", "parcel_id", "id")
USE_KEYS_IN = ("land_use", "landuse", "use", "proposed_use", "lu")
FAR_KEYS = ("far", "proposed_far")


def parse_use(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if v in USE_INDEX:
        return v
    if v.isdigit() and v.zfill(2) in PLUTO_TO_USE:
        return USE_KEYS[PLUTO_TO_USE[v.zfill(2)]]
    for u in LAND_USES:
        if v.lower() in (u.label.lower(), u.key):
            return u.key
    return None


def read_rows(filename: str, raw: bytes) -> list[dict]:
    name = (filename or "").lower()
    if len(raw) > MAX_BYTES:
        raise ValueError("File exceeds 120 MB")
    if name.endswith((".geojson", ".json")):
        gj = json.loads(raw)
        feats = gj.get("features", []) if isinstance(gj, dict) else []
        return [f.get("properties") or {} for f in feats][: MAX_ROWS + 1]
    if name.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))[: MAX_ROWS + 1]
    if name.endswith((".zip", ".gpkg")):
        return _read_gis(name, raw)
    raise ValueError("Upload a .csv, .geojson/.json, Shapefile .zip or .gpkg file")


def _read_gis(name: str, raw: bytes) -> list[dict]:
    import geopandas as gpd

    with tempfile.TemporaryDirectory(prefix="planarena-import-") as tmp:
        root = Path(tmp).resolve()
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                entries = zf.infolist()
                if len(entries) > 100 or sum(e.file_size for e in entries) > 300 * 1024 * 1024:
                    raise ValueError("Archive exceeds extraction limits")
                ok = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
                for e in entries:
                    target = (root / e.filename).resolve()
                    if not target.is_relative_to(root):
                        raise ValueError("Unsafe path in archive")
                    if not e.is_dir() and Path(e.filename).suffix.lower() in ok:
                        zf.extract(e, root)
            shps = list(root.rglob("*.shp"))
            if len(shps) != 1:
                raise ValueError("The ZIP must contain exactly one Shapefile")
            path = shps[0]
        else:
            path = root / "plan.gpkg"
            path.write_bytes(raw)
            if len(gpd.list_layers(path)) != 1:
                raise ValueError("The GeoPackage must contain exactly one layer")
        frame = gpd.read_file(path, rows=MAX_ROWS + 1)
        frame = frame.drop(columns=frame.geometry.name)
        return json.loads(frame.to_json(orient="records"))


def review(rows: list[dict], bbls: np.ndarray, base_use: np.ndarray) -> dict:
    """Validate rows against the parcel register. Nothing is applied here."""
    if len(rows) > MAX_ROWS:
        raise ValueError(f"Plan exceeds {MAX_ROWS:,} rows")
    bbl_to_i = {int(b): i for i, b in enumerate(bbls)}
    n = len(bbls)
    changes, seen = {}, {}
    unmatched, duplicates, unknown_uses, far_issues, unchanged = [], [], [], [], 0
    for k, r in enumerate(rows):
        low = {str(a).strip().lower(): b for a, b in r.items()}
        idx = None
        if low.get("bbl") not in (None, ""):
            try:
                idx = bbl_to_i.get(int(float(low["bbl"])))
            except (TypeError, ValueError):
                idx = None
        elif low.get("i") not in (None, "") and str(low["i"]).strip().isdigit():
            idx = int(low["i"]) if 0 <= int(low["i"]) < n else None
        ident = low.get("bbl") or low.get("i")
        if idx is None:
            unmatched.append({"row": k + 1, "id": ident})
            continue
        if idx in seen:
            duplicates.append({"row": k + 1, "id": ident, "first_row": seen[idx]})
            continue
        seen[idx] = k + 1
        use_raw = next((low[c] for c in USE_KEYS_IN if c in low), None)
        use = parse_use(use_raw)
        if use is None:
            unknown_uses.append({"row": k + 1, "value": use_raw})
            continue
        far_raw = next((low[c] for c in FAR_KEYS if c in low and low[c] not in (None, "")), None)
        far = None
        if far_raw is not None:
            try:
                far = float(far_raw)
                if not (0 <= far <= 30) or not np.isfinite(far):
                    raise ValueError
            except (TypeError, ValueError):
                far_issues.append({"row": k + 1, "value": far_raw})
                continue
        if use == USE_KEYS[int(base_use[idx])] and far is None:
            unchanged += 1
            continue
        changes[str(idx)] = use if far is None else {"use": use, "far": far}
    return {
        "rows": len(rows), "valid_changes": len(changes), "unchanged_rows": unchanged,
        "unmatched": len(unmatched), "duplicates": len(duplicates), "unknown_uses": len(unknown_uses),
        "far_issues": len(far_issues),
        "issues": {"unmatched": unmatched[:50], "duplicates": duplicates[:50],
                   "unknown_uses": unknown_uses[:50], "far_issues": far_issues[:50]},
        "changes": changes,
    }


def export_csv(changes: dict, bbls: np.ndarray, addresses, base_use: np.ndarray) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["i", "bbl", "address", "current_use", "land_use", "far"])
    for k, v in sorted(changes.items(), key=lambda kv: int(kv[0])):
        i = int(k)
        use, far = (v, "") if isinstance(v, str) else (v["use"], v.get("far", ""))
        cur = USE_KEYS[int(base_use[i])] if base_use[i] >= 0 else ""
        w.writerow([i, int(bbls[i]), addresses[i], cur, use, far])
    return buf.getvalue()


def export_geojson(changes: dict, geoms, bbls, addresses) -> dict:
    feats = []
    for k, v in sorted(changes.items(), key=lambda kv: int(kv[0])):
        i = int(k)
        use, far = (v, None) if isinstance(v, str) else (v["use"], v.get("far"))
        feats.append({"type": "Feature", "geometry": geoms[i].__geo_interface__,
                      "properties": {"i": i, "bbl": int(bbls[i]), "address": addresses[i], "land_use": use, "far": far}})
    return {"type": "FeatureCollection", "features": feats}


def provenance(meta: dict, coef: dict, derived: dict, models: dict, engine_version: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": engine_version,
        "source": {"name": meta.get("source"), "sha256": meta.get("source_sha256"), "url": meta.get("source_url"),
                   "data_quality": meta.get("data_quality")},
        "method": ("Observed baseline (MapPLUTO for Manhattan; national open data for Tier A places, see "
                   "source.data_quality); only changed parcels are modelled. Ranges are a sensitivity envelope "
                   "from low/high coefficient variants (empirical Q25/Q75 or published ranges), not calibrated "
                   "prediction intervals. Zoning screen is simplified rules-as-code, not legal compliance."),
        "coefficients": coef, "derived_from_data": derived, "models": models,
    }


def _fmt(v: float, kind: str, key: str) -> str:
    if kind == "share":
        return f"{v * 100:+.2f} pts" if abs(v) >= 5e-5 else "no change"
    if key == "value":
        return f"{'+' if v >= 0 else '−'}${abs(v) / 1e6:,.1f}M"
    return f"{v:+,.0f}" if abs(v) >= 10 else f"{v:+,.1f}"


def staff_report(title: str, description: str, ev: dict, changes: dict, prov: dict,
                 vote: dict | None = None) -> str:
    """Printable staff-report draft. Deterministic text, built only from engine outputs."""
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(r['label'])}</td><td>{e(r['pillar'])}</td><td class=n>{e(_fmt(r['delta'], r['kind'], r['key']))}</td>"
        f"<td class=n>{e(_fmt(r['range'][0], r['kind'], r['key']) + ' to ' + _fmt(r['range'][1], r['kind'], r['key'])) if r.get('range') else ''}</td>"
        f"<td class=n>{r['score']:+.2f}</td><td>{'provisional' if r['provisional'] else ''}</td></tr>"
        for r in ev["indicators"])
    by_use = "".join(
        f"<tr><td>{e(r['label'])}</td><td class=n>{r['parcels']:+d}</td><td class=n>{r['acres']:+.1f}</td>"
        f"<td class=n>{r['homes']:+,.0f}</td><td class=n>{r['jobs']:+,.0f}</td></tr>"
        for r in ev["by_use"] if r["parcels"] or abs(r["homes"]) > 0.5)
    warn = "".join(f"<li>{e(w['message'])}</li>" for w in ev["warnings"]) or "<li>None detected by the screening rules.</li>"
    cap_b, cap_s = ev["capacity"]["baseline"], ev["capacity"]["scenario"]
    neg = ""
    if vote:
        d = vote.get("decision", {})
        rules = "".join(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>" for k, v in (vote.get("rules") or {}).items())
        lines = "".join(f"<li>{e(x)}</li>" for x in d.get("explanation", []))
        neg = (f"<h2>Stakeholder vote</h2><p>The stakeholder council voted over {int(vote.get('n_candidates', 0))} "
               f"candidate plans. Decision ({e(d.get('rule', 'Schulze'))} rule): <b>{e(d.get('title', ''))}</b>.</p>"
               f"<ul>{lines}</ul>"
               + (f"<table><tr><th>Rule</th><th>Council winner</th></tr>{rules}</table>" if rules else ""))
    p = ev["pillars"]
    return f"""<!doctype html><html><head><meta charset=utf-8><title>{e(title)} – staff report draft</title>
<style>body{{font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;max-width:860px;margin:32px auto;color:#111;padding:0 16px}}
h1{{font-size:24px;margin:0}}h2{{font-size:16px;margin-top:26px;border-bottom:1px solid #ccc;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}td,th{{border-bottom:1px solid #e3e3e3;padding:4px 6px;text-align:left}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}.muted{{color:#666}}.pill{{display:inline-block;padding:2px 9px;border-radius:12px;background:#eef;margin-right:6px}}
@media print{{body{{margin:0}}}}</style></head><body>
<div class=muted>DRAFT – generated by PlanArena {e(prov['engine_version'])} on {e(prov['generated_at'])}</div>
<h1>{e(title)}</h1><p>{e(description or '')}</p>
<h2>Summary</h2><p>The proposal changes <b>{ev['n_changed']}</b> parcel(s). Within the evaluated area ({ev['scope_parcels']:,} parcels),
the overall score changes by <b>{ev['overall']:+.2f}</b>:
<span class=pill>Social {p['social']:+.2f}</span><span class=pill>Economic {p['economic']:+.2f}</span><span class=pill>Environmental {p['environmental']:+.2f}</span></p>
<h2>Development capacity</h2><table><tr><th></th><th class=n>Existing</th><th class=n>Proposed</th></tr>
<tr><td>Built floor area (sq ft)</td><td class=n>{cap_b['built_sqft']:,.0f}</td><td class=n>{cap_s['built_sqft']:,.0f}</td></tr>
<tr><td>Zoning capacity (sq ft)</td><td class=n>{cap_b['zoning_capacity_sqft']:,.0f}</td><td class=n>{cap_s['zoning_capacity_sqft'] + cap_s['plan_added_capacity_sqft']:,.0f}</td></tr>
<tr><td>Soft sites</td><td class=n>{cap_b['soft_sites']}</td><td class=n>{cap_s['soft_sites']}</td></tr></table>
<h2>Impacts</h2><table><tr><th>Indicator</th><th>Pillar</th><th class=n>Change</th><th class=n>Sensitivity range</th><th class=n>Score</th><th></th></tr>{rows}</table>
<h2>Change by land use</h2><table><tr><th>Land use</th><th class=n>Parcels</th><th class=n>Acres</th><th class=n>Homes</th><th class=n>Jobs</th></tr>{by_use}</table>
<h2>Screening flags</h2><ul>{warn}</ul>{neg}
<h2>Method and limitations</h2><p class=muted>{e(prov['method'])} Source: {e(str(prov['source']['name']))}, SHA-256 {e(str(prov['source']['sha256']))}.
This draft supports staff analysis; it is not a determination, and all findings require professional review.</p>
</body></html>"""
