"""PlanArena API (FastAPI).

Run:  python -m planarena.api           (serves the built web app from web/dist, if present)
Dev:  uvicorn planarena.api.app:app --reload --port 8787
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from typing import Literal, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, ORJSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from planarena.agents.negotiation import Plan, generate_candidates, negotiate, price_of_anarchy
from planarena.agents.voting import hold_vote
from planarena.agents.stakeholders import DEFAULT_STAKEHOLDERS
from planarena.api import planio, store
from planarena.cities import registry, tiles
from planarena.config import LAND_USES, ROOT, USE_INDEX, USE_KEYS, WEB_DATA
from planarena.taxonomy import CLASSES
from planarena.indicators.engine import INDICATORS, ImpactEngine, load_engine
from planarena.optimize.pareto import pareto as run_pareto
from planarena.optimize.suggest import SuggestConfig, Suggester

ENGINE_VERSION = "1.0.0-rc.1"
app = FastAPI(title="PlanArena API", version=ENGINE_VERSION, default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=2000)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
STAKE = {s.key: s for s in DEFAULT_STAKEHOLDERS}
CITY: ContextVar[str] = ContextVar("city", default=registry.DEFAULT)


@app.middleware("http")
async def select_city(request: Request, call_next):
    """Every API call works on one city: the X-PlanArena-City header or ?city= (default Manhattan)."""
    city = request.headers.get("x-planarena-city") or request.query_params.get("city") or registry.DEFAULT
    if request.url.path.startswith("/api/") and city != registry.DEFAULT and not registry.exists(city):
        return ORJSONResponse({"detail": f"city not built: {city}"}, status_code=404)
    token = CITY.set(city)
    try:
        return await call_next(request)
    finally:
        CITY.reset(token)


# ----------------------------------------------------------------------------- state
_RUNTIMES: OrderedDict[str, dict] = OrderedDict()
_RT_LOCK = threading.Lock()
MAX_CITIES_LOADED = 3


def runtime(city: str | None = None) -> dict:
    city = city or CITY.get()
    with _RT_LOCK:
        if city in _RUNTIMES:
            _RUNTIMES.move_to_end(city)
            return _RUNTIMES[city]
        r = _load_runtime(city)
        _RUNTIMES[city] = r
        while len(_RUNTIMES) > MAX_CITIES_LOADED:
            _RUNTIMES.popitem(last=False)
        return r


def _load_runtime(city: str) -> dict:
    t0 = time.time()
    d = registry.city_dir(city)
    engine = load_engine(d)
    suit, model_info = None, {"gnn": False}
    if (d / "suitability.npy").exists():
        suit = np.load(d / "suitability.npy")
        model_info["gnn"] = True
        from planarena.models.gnn import attach_value_model  # lazy: torch loads only when a model exists

        attach_value_model(engine)
    if (d / "pattern.npy").exists():  # the ablation's best model replaces the GNN alone (planarena/models/ablation.py)
        suit = np.load(d / "pattern.npy")
        model_info["pattern_model"] = json.loads((d / "pattern_meta.json").read_text())
    if city == registry.DEFAULT:  # validation results exist for the Manhattan demo only
        for name in ("gnn_cv", "ensemble_cv"):
            f = ROOT / "results" / f"{name}.json"
            if f.exists():
                model_info[name.replace("gnn_cv", "cv_summary")] = json.loads(f.read_text()).get("cv_summary")
        vf = ROOT / "results" / "validation_emissions_ll84.json"
        if vf.exists():
            model_info["emissions_validation"] = json.loads(vf.read_text())
    redev = None
    if (d / "redevelopment.npz").exists():
        z = np.load(d / "redevelopment.npz")
        redev = {"p": z["p"], "q": z["q"]}
        model_info["redevelopment"] = json.loads((d / "redevelopment_meta.json").read_text())
    sug = Suggester(engine, suit)
    sug.per_use_values()
    engine.evaluate(engine.base_use, ranges=True)  # warms the sensitivity-variant caches
    with store.connect() as db:
        store.audit(db, "city_loaded", {"engine_version": ENGINE_VERSION, "city": city,
                                        "source_sha256": engine.meta.get("source_sha256")})
    return {"city": city, "engine": engine, "suggester": sug, "suitability": suit, "model_info": model_info,
            "soft_sites": engine.soft_site_mask(), "startup_seconds": round(time.time() - t0, 2), "geoms": None,
            "redevelopment": redev}


def _geoms():
    r = runtime()
    if r["geoms"] is None:
        import geopandas as gpd
        r["geoms"] = gpd.read_parquet(registry.city_dir(r["city"]) / "parcels.geoparquet").geometry.values
    return r["geoms"]


def _wgs84(geoms, e: ImpactEngine):
    """RFC 7946 GeoJSON is longitude/latitude: reproject from the twin's projected CRS."""
    import geopandas as gpd

    from planarena.config import PROJECTED_CRS, WEB_CRS

    return gpd.GeoSeries(geoms, crs=e.meta.get("crs", PROJECTED_CRS)).to_crs(WEB_CRS).values


Change = Union[str, dict]


def _arrays(changes: dict[str, Change] | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Changes {index: use_key | {use, far}} -> (use array, FAR array or None)."""
    e: ImpactEngine = runtime()["engine"]
    use = e.base_use.copy()
    far = None
    for k, v in (changes or {}).items():
        try:
            i = int(k)
        except ValueError:
            raise HTTPException(400, f"bad parcel index: {k}") from None
        if not 0 <= i < e.n:
            raise HTTPException(400, f"parcel index out of range: {k}")
        key, f = (v, None) if isinstance(v, str) else (v.get("use"), v.get("far"))
        if key not in USE_INDEX:
            raise HTTPException(400, f"unknown land use: {key}")
        use[i] = USE_INDEX[key]
        if f is not None:
            try:
                f = float(f)
            except (TypeError, ValueError):
                raise HTTPException(400, f"FAR must be a number (parcel {k})") from None
            if not (0 <= f <= 30) or not np.isfinite(f):
                raise HTTPException(400, f"FAR must be between 0 and 30 (parcel {k})")
            if far is None:
                far = np.full(e.n, np.nan)
            far[i] = f
    return use, far


def _changes_of(use: np.ndarray, far: np.ndarray | None = None) -> dict[str, Change]:
    e: ImpactEngine = runtime()["engine"]
    out: dict[str, Change] = {}
    for i in np.nonzero(e.changed_mask(use, far))[0]:
        u = USE_KEYS[int(use[i])]
        out[str(int(i))] = u if far is None or not np.isfinite(far[i]) else {"use": u, "far": float(far[i])}
    return out


def _audit(action: str, body: dict) -> None:
    with store.connect() as db:
        store.audit(db, action, body)


# ----------------------------------------------------------------------------- models
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalRequest(Strict):
    changes: dict[str, Change] = Field(default_factory=dict, description="parcel index -> land-use key or {use, far}")
    scope: str = "affected"
    pillar_weights: dict[str, float] | None = None
    ranges: bool = True


class SuggestRequest(Strict):
    scope: str = "city"
    mode: str = "balanced"  # balanced | consensus | <stakeholder key>
    pillar_weights: dict[str, float] | None = None
    max_change_share: float = Field(0.05, gt=0, le=0.5)
    respect_zoning: bool = True
    allow_demolition: bool = False
    parcels: list[int] | None = None
    changes: dict[str, Change] = Field(default_factory=dict)
    polish: int = Field(30, ge=0, le=200, description="exact re-scoring of the top-K moves (0 = first order only)")


class NegotiateRequest(Strict):
    scope: str = "cd:103"
    parcels: list[int] | None = None
    changes: dict[str, Change] = Field(default_factory=dict)
    max_change_share: float = Field(0.05, gt=0, le=0.5)


class PlanIn(Strict):
    title: str = Field(..., min_length=1, max_length=120)
    changes: dict[str, Change] = Field(default_factory=dict)


class VoteRequest(Strict):
    scope: str = "cd:103"
    parcels: list[int] | None = None
    changes: dict[str, Change] = Field(default_factory=dict, description="the user's plan, added as a candidate")
    max_change_share: float = Field(0.05, gt=0, le=0.5)
    seats: dict[str, float] | None = Field(None, description="stakeholder key -> seats (default 1 each)")
    residents: bool = True
    parcel_ballot: bool = True
    samples: int = Field(200, ge=0, le=2000)
    plans: dict[str, PlanIn] | None = Field(None, description="explicit candidates; skips plan generation")


class ParetoRequest(Strict):
    parcels: list[int] = Field(min_length=1, max_length=300)
    respect_zoning: bool = True
    allow_demolition: bool = False
    seed: int = Field(42, ge=0, le=2**31 - 1)
    generations: int = Field(120, ge=10, le=300)
    changes: dict[str, Change] = Field(default_factory=dict)


class SelectionRequest(Strict):
    parcels: list[int] = Field(min_length=1, max_length=1_000_000)


class RadiusRequest(Strict):
    parcels: list[int] = Field(min_length=1, max_length=5000)
    feet: float = Field(500, gt=0, le=5280)


class ScenarioIn(Strict):
    name: str = Field("Untitled study", min_length=1, max_length=120)
    description: str = Field("", max_length=4000)
    study_area: str = Field("", max_length=200)
    hearing_date: str | None = None
    changes: dict[str, Change] = Field(default_factory=dict)
    checklist: list[dict] | None = None


class ChecklistItem(Strict):
    label: str = Field(..., min_length=1, max_length=200)
    done: bool = False


class RezoningTestRequest(Strict):
    changes: dict[str, Change] = Field(default_factory=dict, description="parcels with a FAR = proposed zoning maximum")
    year: int = Field(2025, ge=2000, le=2100)


class ScenarioPatch(Strict):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=4000)
    study_area: str | None = Field(None, max_length=200)
    status: str | None = None
    hearing_date: str | None = None
    changes: dict[str, Change] | None = None
    checklist: list[ChecklistItem] | None = Field(None, max_length=50)


class ExportRequest(Strict):
    changes: dict[str, Change] = Field(default_factory=dict)
    title: str = Field("PlanArena study", max_length=120)
    description: str = Field("", max_length=4000)
    scope: str = "affected"
    vote: dict | None = Field(None, description="compact vote summary for the staff report")


# ----------------------------------------------------------------------------- core routes
@app.get("/api/health")
def health():
    r = runtime()
    with store.connect() as db:
        chain = store.verify(db)
    return {"ok": True, "version": ENGINE_VERSION, "startup_seconds": r["startup_seconds"],
            "gnn": r["model_info"]["gnn"], "source_sha256": r["engine"].meta.get("source_sha256"), "audit": chain}


@app.get("/api/meta")
def meta():
    r = runtime()
    e: ImpactEngine = r["engine"]
    coef = e.coef  # provisional defaults merged with planarena/indicators/calibrated.yaml
    base = e.evaluate(e.base_use, scope="city")
    return {
        "version": ENGINE_VERSION, "city": e.meta["city"], "city_id": r["city"], "tier": e.meta.get("tier", "B"),
        "map": tiles.map_mode(e.n), "tile_zoom": [tiles.MIN_ZOOM, tiles.MAX_ZOOM],
        "columns": e.meta.get("columns"), "missing_indicators": e.meta.get("missing_indicators", []),
        "source": e.meta["source"],
        "source_sha256": e.meta.get("source_sha256"), "source_url": e.meta.get("source_url"),
        "data_quality": e.meta.get("data_quality"), "n_parcels": e.n, "bbox": e.meta["bbox_wgs84"],
        "levels": e.meta["levels"],
        "land_uses": [{"key": u.key, "label": u.label, "color": u.color, "built": u.built} for u in LAND_USES],
        "classes": [{"key": c.key, "label": c.label, "group": c.group, "color": c.color, "built": c.built} for c in CLASSES],
        "layers": {k: bool((e.p[c] >= 0).any()) if c in e.p else False
                   for k, c in (("existing", "lu16"), ("plan", "plan16"), ("zoning", "zone16"))},
        "indicators": [{"key": i.key, "label": i.label, "unit": i.unit, "pillar": i.pillar,
                        "direction": i.direction, "kind": i.kind, "provisional": e.provisional[i.key]} for i in INDICATORS],
        "baseline_city": {row["key"]: row["baseline"] for row in base["indicators"]},
        "capacity_city": base["capacity"]["baseline"],
        "stakeholders": [s.to_dict() for s in DEFAULT_STAKEHOLDERS],
        "derived": e.derived_summary() | {"sensitivity_multipliers": {"unit_size": e.mult_unit, "value": e.mult_value}},
        "coefficients": coef, "zoning_rules": e.meta["zoning_rules"], "models": r["model_info"],
        "statuses": list(store.STATUSES),
    }


@app.get("/api/parcel/{i}")
def parcel(i: int):
    r = runtime()
    e: ImpactEngine = r["engine"]
    if not 0 <= i < e.n:
        raise HTTPException(404, "parcel not found")
    row = e.p.iloc[i]
    allowed = [USE_KEYS[k] for k in range(len(USE_KEYS)) if (int(row.allowed) >> k) & 1]
    pu = r["suggester"].per_use_values()
    out = {
        "i": i, "bbl": int(row.bbl), "address": row.address, "cd": int(row.cd), "block": int(row.block),
        "zone": row.zone, "overlay": row.overlay, "special_district": row.special,
        "use": USE_KEYS[int(row.use)] if row.use >= 0 else None, "bldg_class": row.bldg_class,
        "lot_area_sqft": float(row.lot_area), "bldg_area_sqft": float(row.bldg_area), "floors": float(row.floors),
        "units": float(row.units_res), "assessed_value": float(row.assess_tot), "year_built": int(row.year_built),
        "built_far": float(row.built_far), "max_far": float(row.max_far), "resid_far": float(row.resid_far),
        "comm_far": float(row.comm_far), "facil_far": float(row.facil_far),
        "landmark": bool(row.landmark), "historic_district": bool(row.historic), "soft_site": bool(r["soft_sites"][i]),
        "allowed_uses": allowed, **_official_layers(row),
        "per_use": {USE_KEYS[u]: {k: float(pu[k][i, u]) for k in ("homes", "jobs", "value", "ghg", "runoff", "green_space")}
                    for u in range(len(USE_KEYS))},
    }
    if r["suitability"] is not None:
        out["suitability"] = {USE_KEYS[u]: float(r["suitability"][i, u]) for u in range(len(USE_KEYS))}
    if r["redevelopment"] is not None:
        rd = r["redevelopment"]
        pct = float((rd["p"] < rd["p"][i]).mean())
        out["redevelopment"] = {"p": float(rd["p"][i]), "percentile": round(pct, 3),
                                "likely_use": USE_KEYS[int(rd["q"][i].argmax())],
                                "q": {USE_KEYS[u]: round(float(rd["q"][i, u]), 3) for u in range(len(USE_KEYS))}}
    return out


_WORD_NUM = {"FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4", "FIFTH": "5", "SIXTH": "6",
             "SEVENTH": "7", "EIGHTH": "8", "NINTH": "9", "TENTH": "10", "ELEVENTH": "11", "TWELFTH": "12"}
_ABBR = {"AVE": "AVENUE", "AV": "AVENUE", "ST": "STREET", "STR": "STREET", "E": "EAST", "W": "WEST",
         "N": "NORTH", "S": "SOUTH", "BLVD": "BOULEVARD", "PL": "PLACE", "DR": "DRIVE", "RD": "ROAD",
         "PKWY": "PARKWAY", "SQ": "SQUARE", "TER": "TERRACE", "LN": "LANE", "HWY": "HIGHWAY"}


def normalize_address(q: str) -> list[str]:
    """User address text -> MapPLUTO address tokens ("350 Fifth Ave" -> ["350", "5", "AVENUE"])."""
    t = re.sub(r"[.,#]", " ", q.upper())
    t = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", t)  # ordinals: 5th -> 5
    out = []
    for k, tok in enumerate(t.split()):
        tok = _WORD_NUM.get(tok, tok)
        # "ST" after a house number is "SAINT" in some names; elsewhere it is STREET
        tok = _ABBR.get(tok, tok) if not (tok == "ST" and k == 1) else "ST"
        out.append(tok)
    return out


@app.get("/api/search")
def search(q: str, limit: int = 12):
    """Find parcels by address fragment or BBL."""
    e: ImpactEngine = runtime()["engine"]
    q = q.strip()
    if len(q) < 2:
        return []
    if q.isdigit() and len(q) >= 6:
        hits = np.nonzero(e.p.bbl.astype(str).str.startswith(q).to_numpy())[0][:limit]
    else:
        addr = e.p.address.str.upper()
        tokens = normalize_address(q)
        m = np.ones(e.n, bool)
        for t in tokens:
            m &= addr.str.contains(rf"\b{re.escape(t)}\b", regex=True).to_numpy()
        hits = np.nonzero(m)[0][:limit]
        if not len(hits) and tokens and tokens[0].isdigit() and len(tokens) > 1:
            # No exact match: same street, nearest house number (lots are listed by their lowest address).
            street = " ".join(tokens[1:])
            same = addr.str.endswith(" " + street).to_numpy()
            nums = pd.to_numeric(addr.str.extract(r"^(\d+)")[0], errors="coerce").to_numpy()
            cand = np.nonzero(same & np.isfinite(nums))[0]
            hits = cand[np.argsort(np.abs(nums[cand] - int(tokens[0])))][:limit]
    return [{"i": int(i), "bbl": int(e.p.bbl.iloc[i]), "address": e.p.address.iloc[i],
             "lon": float(e.p.lon.iloc[i]), "lat": float(e.p.lat.iloc[i])} for i in hits]


@app.get("/api/block/{i}")
def block(i: int):
    """All parcels in the same tax block as parcel ``i``."""
    e: ImpactEngine = runtime()["engine"]
    if not 0 <= i < e.n:
        raise HTTPException(404, "parcel not found")
    b = e.p.block.iloc[i]
    return {"block": int(b), "parcels": np.nonzero((e.p.block == b).to_numpy())[0].tolist()}


@app.get("/api/suitability")
def suitability():
    """Compact arrays for the map overlay: top use and its probability for each parcel."""
    s = runtime()["suitability"]
    if s is None:
        raise HTTPException(404, "GNN not trained yet")
    return {"top": s.argmax(1).astype(int).tolist(), "p": np.round(s.max(1), 3).tolist()}


@app.get("/api/redevelopment")
def redevelopment():
    """Five-year redevelopment likelihood (transition model) and the most likely new use, per parcel."""
    rd = runtime()["redevelopment"]
    if rd is None:
        raise HTTPException(404, "transition model not trained (python -I -m planarena.models.transition all)")
    p = rd["p"]
    rank = (np.argsort(np.argsort(p)) / max(len(p) - 1, 1))
    return {"p": np.round(p, 4).tolist(), "rank": np.round(rank, 3).tolist(),
            "top": rd["q"].argmax(1).astype(int).tolist(), "meta": runtime()["model_info"]["redevelopment"]}


@app.get("/api/soft-sites")
def soft_sites():
    m = runtime()["soft_sites"]
    return {"parcels": np.nonzero(m)[0].tolist(),
            "rule": "built FAR <= 50% of max zoning FAR; lot >= 2,500 sq ft; not landmark/historic, park, civic or infrastructure"}


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    e: ImpactEngine = runtime()["engine"]
    t0 = time.time()
    use, far = _arrays(req.changes)
    res = e.evaluate(use, scope=req.scope, pillar_weights=req.pillar_weights, far=far, ranges=req.ranges)
    res["ms"] = round(1000 * (time.time() - t0), 1)
    return res


# ----------------------------------------------------------------------------- AI
def _suggest_config(req: SuggestRequest) -> SuggestConfig:
    cfg = SuggestConfig(scope=req.scope, max_change_share=req.max_change_share,
                        respect_zoning=req.respect_zoning, allow_demolition=req.allow_demolition,
                        candidate_parcels=np.array(req.parcels) if req.parcels else None, polish=req.polish)
    if req.pillar_weights:
        cfg.pillar_weights = req.pillar_weights
    if req.mode == "consensus":
        cfg.unanimity_weights = [s.normalized() for s in DEFAULT_STAKEHOLDERS]
    elif req.mode in STAKE:
        cfg.indicator_weights = STAKE[req.mode].normalized()
    elif req.mode != "balanced":
        raise HTTPException(400, f"unknown mode: {req.mode}")
    return cfg


@app.post("/api/suggest")
async def suggest(req: SuggestRequest):
    r = runtime()
    t0 = time.time()
    use, _ = _arrays(req.changes)
    out = await run_in_threadpool(r["suggester"].suggest, _suggest_config(req), use)
    res = {"changes": _changes_of(out["use"]), "evaluation": out["evaluation"], "history": out["history"],
           "budget": out["budget"], "eligible": out["eligible"], "ms": round(1000 * (time.time() - t0), 1)}
    _audit("optimizer_run", {"mode": req.mode, "scope": req.scope, "n_changes": len(res["changes"]),
                             "overall": out["evaluation"]["overall"]})
    return res


@app.post("/api/pareto")
async def pareto_route(req: ParetoRequest):
    r = runtime()
    use, _ = _arrays(req.changes)
    cfg = SuggestConfig(scope="affected", respect_zoning=req.respect_zoning, allow_demolition=req.allow_demolition)
    try:
        res = await run_in_threadpool(run_pareto, r["suggester"], req.parcels, cfg, use, req.seed, req.generations)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    _audit("pareto_run", {"n_parcels": len(req.parcels), "seed": req.seed, "generations": req.generations,
                          "front_size": len(res["front"])})
    return res


@app.post("/api/negotiate")
async def negotiate_route(req: NegotiateRequest):
    r = runtime()
    t0 = time.time()
    cur = _arrays(req.changes)[0] if req.changes else None

    def run():
        plans = generate_candidates(r["suggester"], DEFAULT_STAKEHOLDERS, scope=req.scope,
                                    max_change_share=req.max_change_share, current_use=cur,
                                    candidate_parcels=np.array(req.parcels) if req.parcels else None)
        return plans, negotiate(plans, DEFAULT_STAKEHOLDERS)

    plans, res = await run_in_threadpool(run)
    res["plan_changes"] = {p.id: _changes_of(p.use) for p in plans}
    res["plan_evaluations"] = {p.id: p.evaluation for p in plans}
    res["ms"] = round(1000 * (time.time() - t0), 1)
    _audit("negotiation", {"scope": req.scope, "agreed_plan": res["agreed_plan"], "rounds": res["rounds"],
                           "by_protocol": res["agreed_by_protocol"], "utilities": res["utilities"]})
    return res


@app.post("/api/vote")
async def vote_route(req: VoteRequest):
    """Comprehensive stakeholder vote: council (9 rules), residents' referendum, parcel-by-parcel ballot."""
    r = runtime()
    t0 = time.time()
    keys = {s.key for s in DEFAULT_STAKEHOLDERS}
    if req.seats and (set(req.seats) - keys or any(v < 0 for v in req.seats.values())
                      or sum(req.seats.values()) <= 0):
        raise HTTPException(400, f"seats must be non-negative, sum > 0, keys in {sorted(keys)}")
    cur = _arrays(req.changes)[0] if req.changes else None
    explicit = None
    if req.plans is not None:
        if not 1 <= len(req.plans) <= 30:
            raise HTTPException(400, "give 1 to 30 plans")
        explicit = [Plan(pid, p.title, "submitted", _arrays(p.changes)[0], {}) for pid, p in req.plans.items()]

    def run():
        return hold_vote(r["suggester"], DEFAULT_STAKEHOLDERS, seats=req.seats, scope=req.scope,
                         max_change_share=req.max_change_share, current_use=cur,
                         candidate_parcels=np.array(req.parcels) if req.parcels else None,
                         residents=req.residents, samples=req.samples, plans=explicit,
                         with_parcel_ballot=req.parcel_ballot)

    try:
        res = await run_in_threadpool(run)
    except ValueError as err:
        raise HTTPException(400, str(err)) from None
    plans = res.pop("plans")
    res["plan_changes"] = {p.id: _changes_of(p.use) for p in plans}
    res["plan_evaluations"] = {p.id: p.evaluation for p in plans}
    res["ms"] = round(1000 * (time.time() - t0), 1)
    _audit("vote", {"scope": req.scope, "seats": req.seats, "winner": res["decision"]["plan"],
                    "rule": res["decision"]["rule"], "double_majority": res["decision"]["double_majority"],
                    "candidates": [c["id"] for c in res["candidates"]]})
    return res


@app.post("/api/rezoning-test")
def rezoning_test(req: RezoningTestRequest):
    """CEQR-style development-site screen: FAR values in the plan are treated as the proposed zoning maximum."""
    from planarena.indicators import rwcds

    e: ImpactEngine = runtime()["engine"]
    use, far = _arrays(req.changes)
    if far is None or not np.isfinite(far).any():
        raise HTTPException(400, "set a floor area ratio on the parcels to rezone (Density: Set floor area ratio)")
    area = np.isfinite(far)
    new_max = np.where(area, far, e.p["max_far"].to_numpy())
    res_ok = np.isin(use, [USE_INDEX["res"], USE_INDEX["mix"]]) | np.array(
        [bool((a >> USE_INDEX["res"]) & 1) for a in e.allowed])
    sites = rwcds.screen(e, new_max, area, req.year)
    full = rwcds.project(e, sites, new_max, res_ok, 1.0)
    util = e.coef["assumptions"]["buildout_utilization"]["value"]
    part = rwcds.project(e, sites, new_max, res_ok, util)
    idx = np.nonzero(sites)[0]
    _audit("rezoning_test", {"rezoned": int(area.sum()), "sites": int(len(idx)), "homes": round(full["homes_increment"])})
    return {"rezoned_parcels": int(area.sum()), "sites": idx.tolist(), "n_sites": int(len(idx)),
            "homes_increment": round(full["homes_increment"]), "homes_increment_at_utilisation": round(part["homes_increment"]),
            "utilisation": util, "floor_area_increment_sqft": round(full["floor_area_increment_sqft"]),
            "criteria": rwcds.__doc__.strip(), "min_far_increase": rwcds.MIN_FAR_INCREASE}


@app.post("/api/price-of-anarchy")
async def poa(req: NegotiateRequest):
    return await run_in_threadpool(price_of_anarchy, runtime()["suggester"], DEFAULT_STAKEHOLDERS,
                                   req.scope, req.max_change_share)


# ----------------------------------------------------------------------------- planner's desk
@app.post("/api/zoning-summary")
def zoning_summary(req: SelectionRequest):
    """What zoning allows on a selection: maximum FAR, built FAR now, districts and the uses allowed on every parcel."""
    e: ImpactEngine = runtime()["engine"]
    idx = np.array([i for i in req.parcels if 0 <= i < e.n], dtype=int)
    if not len(idx):
        raise HTTPException(400, "no valid parcels")
    p = e.p.iloc[idx]
    q = lambda v: {"min": round(float(np.min(v)), 2), "median": round(float(np.median(v)), 2), "max": round(float(np.max(v)), 2)}
    allowed = np.bitwise_and.reduce(p["allowed"].to_numpy().astype("int64"))
    has_zoning = not str(e.meta.get("zoning_rules", "")).startswith("none")
    return {"n": int(len(idx)), "has_zoning": has_zoning, "zoning_rules": e.meta.get("zoning_rules"),
            "max_far": q(p["max_far"].to_numpy()), "built_far": q(p["built_far"].to_numpy()),
            "districts": {str(k): int(v) for k, v in p["zone"].value_counts().head(4).items()} if has_zoning else {},
            "allowed_on_all": [USE_KEYS[k] for k in range(len(USE_KEYS)) if (int(allowed) >> k) & 1]}


@app.post("/api/radius")
def radius(req: RadiusRequest):
    """Parcels within a notice radius of the selected parcels (polygon-to-polygon distance)."""
    from shapely import STRtree, union_all

    e: ImpactEngine = runtime()["engine"]
    g = _geoms()
    idx = [i for i in req.parcels if 0 <= i < e.n]
    zone = union_all(g[idx]).buffer(req.feet)
    hits = STRtree(g).query(zone, predicate="intersects")
    hits = sorted(set(int(h) for h in hits) - set(idx))
    rows = [{"i": h, "bbl": int(e.p.bbl.iloc[h]), "address": e.p.address.iloc[h],
             "use": USE_KEYS[int(e.base_use[h])] if e.base_use[h] >= 0 else None} for h in hits]
    _audit("notice_radius", {"n_selected": len(idx), "feet": req.feet, "n_notified": len(rows)})
    return {"feet": req.feet, "selected": len(idx), "count": len(rows), "parcels": rows}


# ----------------------------------------------------------------------------- scenarios & audit
def _study_city(s: dict | None) -> dict | None:
    return s if s is not None and s.get("city", registry.DEFAULT) == CITY.get() else None


@app.get("/api/scenarios")
def scenarios():
    with store.connect() as db:
        return [s for s in store.list_scenarios(db) if _study_city(s)]


@app.post("/api/scenarios", status_code=201)
def create_scenario(body: ScenarioIn):
    e: ImpactEngine = runtime()["engine"]
    use, far = _arrays(body.changes)
    data = body.model_dump() | {"city": CITY.get()}
    ev = e.evaluate(use, far=far)
    data["summary"] = {"overall": ev["overall"], "pillars": ev["pillars"], "n_changed": ev["n_changed"]}
    with store.connect() as db:
        return store.create(db, data, e.meta.get("source_sha256", ""))


@app.get("/api/scenarios/{sid}")
def get_scenario(sid: str):
    with store.connect() as db:
        s = _study_city(store.get(db, sid))
    if s is None:
        raise HTTPException(404, "scenario not found")
    return s


def _own_study(sid: str) -> None:
    with store.connect() as db:
        if _study_city(store.get(db, sid)) is None:
            raise HTTPException(404, "scenario not found")


@app.patch("/api/scenarios/{sid}")
def patch_scenario(sid: str, patch: ScenarioPatch):
    _own_study(sid)
    data = {k: v for k, v in patch.model_dump().items() if v is not None}
    if "changes" in data:
        use, far = _arrays(data["changes"])
        ev = runtime()["engine"].evaluate(use, far=far)
        data["summary"] = {"overall": ev["overall"], "pillars": ev["pillars"], "n_changed": ev["n_changed"]}
    try:
        with store.connect() as db:
            s = store.update(db, sid, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if s is None:
        raise HTTPException(404, "scenario not found")
    return s


@app.delete("/api/scenarios/{sid}")
def delete_scenario(sid: str):
    _own_study(sid)
    with store.connect() as db:
        if not store.delete(db, sid):
            raise HTTPException(404, "scenario not found")
    return {"deleted": sid}


@app.get("/api/scenarios/{sid}/history")
def scenario_history(sid: str):
    _own_study(sid)
    with store.connect() as db:
        return store.history(db, sid)


@app.get("/api/audit")
def audit_log(limit: int = 200):
    with store.connect() as db:
        rows = [{"id": i, "timestamp": t, "action": a, "body": json.loads(b), "row_hash": h}
                for i, t, a, b, h in db.execute(
                    "SELECT id, timestamp, action, body, row_hash FROM audit ORDER BY id DESC LIMIT ?", (limit,))]
        return {"verify": store.verify(db), "rows": rows}


# ----------------------------------------------------------------------------- import / export / report
@app.post("/api/import")
async def import_plan(file: UploadFile = File(...)):
    """Read and validate a plan. Returns a review and proposed changes; nothing is applied."""
    e: ImpactEngine = runtime()["engine"]
    raw = await file.read(planio.MAX_BYTES + 1)
    try:
        rows = await run_in_threadpool(planio.read_rows, file.filename or "", raw)
        rev = planio.review(rows, e.p.bbl.to_numpy(), e.base_use)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(422, f"Could not read plan: {str(exc)[:300]}") from exc
    except Exception as exc:  # malformed GIS files raise library-specific errors
        raise HTTPException(422, f"Could not read plan: {type(exc).__name__}: {str(exc)[:200]}") from exc
    rev["filename"] = file.filename
    _audit("plan_import_reviewed", {"filename": file.filename, "rows": rev["rows"], "valid_changes": rev["valid_changes"],
                                    "unmatched": rev["unmatched"], "duplicates": rev["duplicates"]})
    return rev


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), scope: Literal["affected", "city"] = "affected"):
    """One-step import and evaluate (kept for scripts). The UI uses /api/import, then confirmation."""
    rev = await import_plan(file)
    use, far = _arrays(rev["changes"])
    res = runtime()["engine"].evaluate(use, scope=scope, far=far, ranges=True)
    res["upload"] = {k: rev[k] for k in ("rows", "valid_changes", "unmatched", "duplicates", "unknown_uses", "far_issues")}
    res["upload"]["matched"] = rev["valid_changes"]
    res["changes"] = rev["changes"]
    return res


def _prov() -> dict:
    r = runtime()
    e: ImpactEngine = r["engine"]
    return planio.provenance(e.meta, e.coef,
                             e.derived_summary() | {"sensitivity_multipliers": {"unit_size": e.mult_unit, "value": e.mult_value}},
                             r["model_info"], ENGINE_VERSION)


@app.post("/api/export/{fmt}")
def export(fmt: Literal["csv", "geojson", "provenance"], req: ExportRequest):
    e: ImpactEngine = runtime()["engine"]
    _arrays(req.changes)  # validate
    _audit("export", {"format": fmt, "n_changes": len(req.changes)})
    if fmt == "csv":
        text = planio.export_csv(req.changes, e.p.bbl.to_numpy(), e.p.address.to_numpy(), e.base_use)
        return Response(text, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=planarena-plan.csv"})
    if fmt == "geojson":
        gj = planio.export_geojson(req.changes, _wgs84(_geoms(), e), e.p.bbl.to_numpy(), e.p.address.to_numpy())
        return Response(json.dumps(gj), media_type="application/geo+json",
                        headers={"Content-Disposition": "attachment; filename=planarena-plan.geojson"})
    use, far = _arrays(req.changes)
    body = _prov() | {"title": req.title, "changes": req.changes,
                      "evaluation": e.evaluate(use, scope=req.scope, far=far, ranges=True)}
    return Response(json.dumps(body, default=float), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=planarena-report.json"})


@app.post("/api/report", response_class=HTMLResponse)
def report(req: ExportRequest):
    e: ImpactEngine = runtime()["engine"]
    use, far = _arrays(req.changes)
    ev = e.evaluate(use, scope=req.scope, far=far, ranges=True)
    _audit("staff_report", {"title": req.title, "n_changes": ev["n_changed"]})
    return planio.staff_report(req.title, req.description, ev, req.changes, _prov(), req.vote)


@app.get("/api/sample-plan.csv")
def sample_plan():
    """Example upload: 40 Lower Manhattan parking lots turned residential at FAR 6."""
    e: ImpactEngine = runtime()["engine"]
    idx = np.nonzero((e.base_use == USE_INDEX["prk"]) & (e.cd == 103))[0][:40]
    lines = ["bbl,land_use,far"] + [f"{int(e.p.bbl.iloc[i])},res,6" for i in idx]
    return Response("\n".join(lines) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=planarena-sample-plan.csv"})


# ----------------------------------------------------------------------------- static web app
# ----------------------------------------------------------------------------- cities
@app.get("/api/cities")
def cities():
    """The PlanArena cities: the Manhattan demo and the ten cities with official parcel-level land use, built or not."""
    from planarena.cities.build import CITIES

    built = registry.cities()
    out = [built[registry.DEFAULT]] if registry.DEFAULT in built else []
    for geoid, (label, adapter) in CITIES.items():
        b = built.get(geoid)
        out.append({"id": geoid, "label": label, "built": b is not None,
                    "status": "built" if b else ("sources ready" if adapter else "sources being ingested")}
                   | ({k: v for k, v in b.items() if k not in ("id", "label")} if b else {}))
    return out


def _official_layers(row) -> dict:
    """The parcel's three official layers: existing use, adopted plan and zoning (16-class keys; None if missing)."""
    cls = lambda c: CLASSES[int(row[c])].key if c in row and int(row[c]) >= 0 else None
    ex, pl, zn = cls("lu16"), cls("plan16"), cls("zone16")
    group = lambda k: next(c.group for c in CLASSES if c.key == k) if k else None
    return {"existing_class": ex, "plan_class": pl, "zoning_class": zn,
            "plan_designation": row.get("plan_designation") or None,
            "plan_consistent": (group(ex) == group(pl)) if ex and pl else None}


@app.get("/api/layers")
def official_layers():
    """Little-endian int16 arrays of the current city: existing use, adopted plan and zoning class of every parcel
    (16-class index, -1 where the city has no such layer or the parcel no value)."""
    p = runtime()["engine"].p
    col = lambda c: (p[c] if c in p else pd.Series(-1, index=p.index)).to_numpy().astype("<i2").tobytes()
    return Response(col("lu16") + col("plan16") + col("zone16"), media_type="application/octet-stream")


@app.get("/api/points")
def parcel_points():
    """Compact parcel points of the current city (see planarena.cities.tiles.points)."""
    return Response(tiles.points(runtime()["engine"]), media_type="application/octet-stream")


@app.get("/city-tiles/{city}/{z}/{x}/{y}.pbf")
def city_tile(city: str, z: int, x: int, y: int):
    """Vector tile of a built city's parcels (zoom 13-15)."""
    if not registry.exists(city):
        raise HTTPException(404, "city not built")
    if not (tiles.MIN_ZOOM <= z <= tiles.MAX_ZOOM) or not (0 <= x < 1 << z) or not (0 <= y < 1 << z):
        raise HTTPException(404, "tile out of range")
    return Response(tiles.tile(city, z, x, y), media_type="application/x-protobuf",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/city-data/{geoid}/{path:path}")
def city_data(geoid: str, path: str):
    """Web layers of a built city other than Manhattan."""
    if not registry.exists(geoid) or geoid == registry.DEFAULT:
        raise HTTPException(404, "city not built")
    base = registry.web_dir(geoid).resolve()
    f = (base / path).resolve()
    if not f.is_file() or not f.is_relative_to(base):
        raise HTTPException(404, "not found")
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


DIST = ROOT / "web" / "dist"
if WEB_DATA.exists():
    app.mount("/data", StaticFiles(directory=WEB_DATA), name="data")
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = (DIST / path).resolve()
        if path and f.is_file() and f.is_relative_to(DIST.resolve()):
            return FileResponse(f, headers={"Cache-Control": "no-cache"})
        # The page names the hashed bundles, so it must be revalidated after every rebuild.
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
