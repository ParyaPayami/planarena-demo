"""Scenario store and tamper-evident audit log (SQLite).

Each scenario is a planning study record: name, description, study area, status
workflow, hearing date, checklist and plan changes.

Audit rows are hash-chained: row_hash = sha256(prev_hash + timestamp + action + body).
verify() recomputes the chain, so any edit or deletion of a past row is detectable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from planarena.config import DATA

DB = DATA / "planarena.sqlite3"
STATUSES = ("draft", "under review", "hearing scheduled", "adopted", "withdrawn")
DEFAULT_CHECKLIST = [
    "Define study area and purpose",
    "Document existing conditions (land use, zoning, capacity)",
    "Develop alternatives (optimizer / stakeholder vote)",
    "Evaluate impacts and uncertainty",
    "Review zoning conflicts and protected resources",
    "Prepare notice radius and mailing list",
    "Draft staff report",
    "Schedule hearing",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS scenarios (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, body TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, action TEXT NOT NULL,
        body TEXT NOT NULL, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL)""")
    return db


def _hash(prev: str, ts: str, action: str, body: str) -> str:
    return hashlib.sha256((prev + ts + action + body).encode()).hexdigest()


def audit(db: sqlite3.Connection, action: str, body: dict) -> None:
    row = db.execute("SELECT row_hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
    prev = row[0] if row else "0" * 64
    ts, b = _now(), json.dumps(body, sort_keys=True, default=str)
    db.execute("INSERT INTO audit(timestamp, action, body, prev_hash, row_hash) VALUES (?,?,?,?,?)",
               (ts, action, b, prev, _hash(prev, ts, action, b)))


def verify(db: sqlite3.Connection) -> dict:
    prev = "0" * 64
    n = 0
    for i, ts, action, body, p, h in db.execute("SELECT id, timestamp, action, body, prev_hash, row_hash FROM audit ORDER BY id"):
        if p != prev or _hash(p, ts, action, body) != h:
            return {"valid": False, "broken_at": i, "rows": n}
        prev, n = h, n + 1
    return {"valid": True, "rows": n, "head": prev}


def list_scenarios(db) -> list[dict]:
    return [json.loads(b) for (b,) in db.execute("SELECT body FROM scenarios ORDER BY updated_at DESC")]


def get(db, sid: str) -> dict | None:
    row = db.execute("SELECT body FROM scenarios WHERE id=?", (sid,)).fetchone()
    return json.loads(row[0]) if row else None


def create(db, data: dict, source_sha256: str) -> dict:
    now = _now()
    body = {
        "id": str(uuid.uuid4()), "name": data.get("name") or "Untitled study",
        "description": data.get("description", ""), "status": "draft", "hearing_date": data.get("hearing_date"),
        "study_area": data.get("study_area", ""), "changes": data.get("changes", {}),
        "checklist": data.get("checklist") or [{"label": c, "done": False} for c in DEFAULT_CHECKLIST],
        "summary": data.get("summary"), "created_at": now, "updated_at": now, "source_sha256": source_sha256,
        "city": data.get("city", "manhattan"),
    }
    db.execute("INSERT INTO scenarios VALUES (?,?,?,?,?)", (body["id"], body["name"], json.dumps(body), now, now))
    audit(db, "scenario_created", {"id": body["id"], "name": body["name"], "city": body["city"],
                                   "n_changes": len(body["changes"])})
    return body


def update(db, sid: str, patch: dict) -> dict | None:
    body = get(db, sid)
    if body is None:
        return None
    if "status" in patch and patch["status"] not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    before = {k: body.get(k) for k in patch}
    body.update({k: v for k, v in patch.items() if k in
                 ("name", "description", "status", "hearing_date", "study_area", "changes", "checklist", "summary")})
    body["updated_at"] = _now()
    db.execute("UPDATE scenarios SET name=?, body=?, updated_at=? WHERE id=?",
               (body["name"], json.dumps(body), body["updated_at"], sid))
    changed = {k: {"from": before[k] if k != "changes" else len(before[k] or {}),
                   "to": patch[k] if k != "changes" else len(patch[k] or {})} for k in patch}
    audit(db, "scenario_updated", {"id": sid, "changed": changed})
    return body


def delete(db, sid: str) -> bool:
    cur = db.execute("DELETE FROM scenarios WHERE id=?", (sid,))
    if cur.rowcount:
        audit(db, "scenario_deleted", {"id": sid})
    return bool(cur.rowcount)


def history(db, sid: str) -> list[dict]:
    out = []
    for i, ts, action, body in db.execute("SELECT id, timestamp, action, body FROM audit ORDER BY id"):
        b = json.loads(body)
        if b.get("id") == sid or b.get("scenario_id") == sid:
            out.append({"id": i, "timestamp": ts, "action": action, "body": b})
    return out
