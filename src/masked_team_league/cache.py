from __future__ import annotations

from dataclasses import dataclass

from .models import Team, canonical_hash
from .surrogate import SurrogateScorer


@dataclass(frozen=True)
class MatchupCacheKey:
    attack_team_hash: str
    defense_team_hash: str
    version: str
    simulator_version: str
    seed_policy: str

    @classmethod
    def from_teams(
        cls,
        attack: Team,
        defense: Team,
        *,
        version: str = "v4",
        simulator_version: str = "surrogate",
        seed_policy: str = "deterministic",
    ) -> "MatchupCacheKey":
        return cls(
            attack_team_hash=attack.hash(),
            defense_team_hash=defense.hash(),
            version=version,
            simulator_version=simulator_version,
            seed_policy=seed_policy,
        )

    def hash(self) -> str:
        cached = getattr(self, "_canonical_hash_cache", None)
        if cached is not None:
            return str(cached)
        value = canonical_hash(self)
        object.__setattr__(self, "_canonical_hash_cache", value)
        return value


@dataclass(frozen=True)
class SimulationResult:
    wins: int
    games: int
    mean_margin: float
    mean_duration: float

    @property
    def win_rate(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.wins / self.games


class SimulationCache:
    def __init__(self) -> None:
        self._items: dict[str, SimulationResult] = {}

    def get(self, key: MatchupCacheKey) -> SimulationResult | None:
        return self._items.get(key.hash())

    def put(self, key: MatchupCacheKey, result: SimulationResult) -> None:
        self._items[key.hash()] = result

    def get_or_run(self, key: MatchupCacheKey, runner) -> SimulationResult:
        existing = self.get(key)
        if existing is not None:
            return existing
        result = runner()
        self.put(key, result)
        return result

    def __len__(self) -> int:
        return len(self._items)


class SurrogateSimulator:
    def __init__(self, scorer: SurrogateScorer) -> None:
        self.scorer = scorer

    def run(self, attack: Team, defense: Team, *, games: int) -> SimulationResult:
        prediction = self.scorer.predict(attack, defense)
        wins = int(round(prediction.win_prob * games))
        return SimulationResult(
            wins=wins,
            games=games,
            mean_margin=prediction.margin,
            mean_duration=prediction.duration,
        )
