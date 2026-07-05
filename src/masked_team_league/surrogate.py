from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .models import Team


@dataclass(frozen=True)
class SurrogatePrediction:
    win_prob: float
    uncertainty: float
    margin: float
    duration: float
    counter_residual: float

    def conservative(self, beta: float = 1.0) -> float:
        return min(max(self.win_prob - beta * self.uncertainty, 1e-4), 1.0 - 1e-4)

    def optimistic(self, beta: float = 1.0) -> float:
        return min(max(self.win_prob + beta * self.uncertainty, 1e-4), 1.0 - 1e-4)


class SurrogateScorer(Protocol):
    def predict(self, attack: Team, defense: Team) -> SurrogatePrediction:
        ...


class HeuristicSurrogateScorer:
    """Deterministic stand-in for the learned single-team model.

    It keeps the production interface intact while the real neural surrogate is
    trained later. The formula separates power baseline and a small structured
    residual so tests can validate ranking/search behavior deterministically.
    """

    def __init__(
        self,
        power_scale: float = 650.0,
        base_uncertainty: float = 0.08,
        *,
        use_equipment_star_features: bool = True,
        use_position_features: bool = True,
    ) -> None:
        self.power_scale = float(power_scale)
        self.base_uncertainty = float(base_uncertainty)
        self.use_equipment_star_features = bool(use_equipment_star_features)
        self.use_position_features = bool(use_position_features)

    def predict(self, attack: Team, defense: Team) -> SurrogatePrediction:
        attack_power = attack.total_power
        defense_power = defense.total_power
        power_delta = attack_power - defense_power
        attack_star = _average_star(attack) if self.use_equipment_star_features else 0.0
        defense_star = _average_star(defense) if self.use_equipment_star_features else 0.0
        star_delta = attack_star - defense_star
        position_delta = _position_profile(attack) - _position_profile(defense) if self.use_position_features else 0.0
        residual = 0.08 * math.tanh(star_delta / 2.0) + 0.04 * math.tanh(position_delta / 5.0)
        logit = power_delta / self.power_scale + residual
        win_prob = 1.0 / (1.0 + math.exp(-logit))
        uncertainty = min(0.35, self.base_uncertainty + 0.12 * math.exp(-abs(power_delta) / self.power_scale))
        margin = power_delta / 100.0 + residual * 10.0
        duration = max(10.0, 90.0 - abs(power_delta) / 80.0)
        return SurrogatePrediction(
            win_prob=win_prob,
            uncertainty=uncertainty,
            margin=margin,
            duration=duration,
            counter_residual=residual,
        )


def _average_star(team: Team) -> float:
    stars = [loadout.unique_equip_star for loadout in team.slots if loadout.unique_equip_star is not None]
    if not stars:
        return 0.0
    return sum(stars) / len(stars)


def _position_profile(team: Team) -> float:
    return sum((index + 1) * loadout.standing_rank for index, loadout in enumerate(team.slots)) / len(team.slots)
