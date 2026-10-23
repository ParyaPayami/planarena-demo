"""Parcel features and targets shared by the GNN and the baselines (numpy and pandas only)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

FAMILIES = ("R", "C12", "C3", "C456", "C8", "M1", "M23", "MX", "PARK", "BPC", "OTHER")


def zone_family(zone: str) -> str:
    z = (zone or "").upper()
    if z in ("PARK", "BPC"):
        return z
    if "/R" in z:
        return "MX"
    m = re.match(r"^([RCM])(\d+)", z)
    if not m:
        return "OTHER"
    k, n = m.group(1), int(m.group(2))
    if k == "R":
        return "R"
    if k == "C":
        return {1: "C12", 2: "C12", 3: "C3", 4: "C456", 5: "C456", 6: "C456", 7: "C3"}.get(n, "C8")
    return "M1" if n == 1 else "M23"


def context_features(df: pd.DataFrame) -> np.ndarray:
    """Context features that do not reveal the parcel's own land use."""
    yb = df.year_built.to_numpy()
    fam = df.zone.map(zone_family)
    onehot = np.stack([(fam == f).to_numpy() for f in FAMILIES], axis=1).astype("float32")
    cols = [
        np.log1p(df.lot_area.to_numpy()) / 12.0,
        np.log1p(df.bldg_area.to_numpy()) / 14.0,
        np.clip(df.floors.to_numpy(), 0, 100) / 50.0,
        np.where(yb > 1700, (yb - 1900) / 100.0, 0.0),
        (yb <= 1700).astype(float),
        df.resid_far.to_numpy() / 15.0, df.comm_far.to_numpy() / 15.0,
        df.facil_far.to_numpy() / 15.0, np.clip(df.built_far.to_numpy(), 0, 40) / 15.0,
        (df.overlay.str.len() > 0).to_numpy().astype(float),
        df.historic.to_numpy().astype(float), df.landmark.to_numpy().astype(float),
    ]
    return np.concatenate([np.stack(cols, axis=1).astype("float32"), onehot], axis=1)


def value_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    v = df.assess_tot.to_numpy() / np.maximum(df.lot_area.to_numpy(), 1.0)
    valid = (df.assess_tot.to_numpy() > 0) & (df.use.to_numpy() >= 0)
    return np.log1p(v).astype("float32"), valid
