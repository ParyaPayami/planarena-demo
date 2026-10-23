"""Stakeholder agents: each one values the impact indicators with its own weights.

The weights are editable modelling assumptions, not empirical estimates.
Utility is a weighted sum of the engine's signed indicator scores, so the
status quo has utility 0 and positive utility means the agent gains.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from planarena.indicators.engine import IND_KEYS


@dataclass(frozen=True)
class Stakeholder:
    key: str
    name: str
    icon: str
    color: str
    blurb: str
    weights: dict  # indicator key -> weight (normalized to sum 1)

    def normalized(self) -> dict:
        s = sum(self.weights.values()) or 1.0
        return {k: self.weights.get(k, 0.0) / s for k in IND_KEYS}

    def utility(self, evaluation: dict) -> float:
        w = self.normalized()
        return float(sum(w[r["key"]] * r["score"] for r in evaluation["indicators"]))

    def to_dict(self) -> dict:
        return asdict(self) | {"weights": self.normalized()}


DEFAULT_STAKEHOLDERS: tuple[Stakeholder, ...] = (
    Stakeholder("developer", "Developer", "🏗️", "#E07A1F",
                "Wants buildable homes and a strong return on land.",
                {"homes": 0.35, "value": 0.45, "jobs": 0.20}),
    Stakeholder("residents", "Residents' association", "🏘️", "#2E86AB",
                "Wants parks, shops, schools and clinics within walking distance.",
                {"park_access": 0.25, "services_access": 0.20, "facility_access": 0.20,
                 "mix": 0.10, "homes": 0.15, "green_cover": 0.10}),
    Stakeholder("environment", "Environmental advocate", "🌳", "#3FA34D",
                "Wants more green space, less runoff and lower building emissions.",
                {"green_space": 0.30, "runoff": 0.20, "ghg": 0.30, "green_cover": 0.20}),
    Stakeholder("city", "City budget office", "🏛️", "#6C4AB6",
                "Wants a growing tax base and jobs to fund services.",
                {"value": 0.55, "jobs": 0.25, "homes": 0.20}),
    Stakeholder("business", "Local business alliance", "🛍️", "#D7263D",
                "Wants customers nearby, active streets and jobs.",
                {"jobs": 0.40, "services_access": 0.35, "mix": 0.25}),
)


def utility_matrix(stakeholders, evaluations: list[dict]) -> np.ndarray:
    """Rows are candidate plans; columns are stakeholders."""
    return np.array([[s.utility(ev) for s in stakeholders] for ev in evaluations])
