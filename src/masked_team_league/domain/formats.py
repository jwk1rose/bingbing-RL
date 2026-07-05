from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchFormat:
    n_teams: int
    team_size: int = 5
    win_required: int | None = None
    max_hidden_per_team: int = 2
    max_hidden_total: int = 10

    def __post_init__(self) -> None:
        if self.win_required is None:
            object.__setattr__(self, "win_required", self.n_teams // 2 + 1)
        if self.n_teams not in (3, 5):
            raise ValueError("n_teams must be 3 or 5")
        if self.team_size != 5:
            raise ValueError("team_size must be 5")
        if self.win_required is None or not (1 <= self.win_required <= self.n_teams):
            raise ValueError("win_required must be within the match length")
        if self.max_hidden_per_team < 0 or self.max_hidden_total < 0:
            raise ValueError("mask limits must be non-negative")
