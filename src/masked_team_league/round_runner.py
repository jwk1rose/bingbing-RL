from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any

from .attack_oracle import AttackOracle, AttackOracleConfig
from .constraints import ConstraintEngine
from .generation import LegalPlanGenerator
from .hyperband import HalvingStage
from .league import LeagueManager
from .models import AttackPlan, DefensePlan, MatchFormat
from .real_oracle import OracleBatchEvaluator, OracleEvaluationRecord


@dataclass(frozen=True)
class LeagueRoundConfig:
    teams: int = 3
    defenses: int = 20
    attacks_per_defense: int = 200
    oracle_top_k: int = 20
    seed: int = 0
    round_id: str = "round_0001"


@dataclass(frozen=True)
class LeagueRoundSummary:
    round_id: str
    out_dir: str
    defenses: int
    candidates: int
    oracle_pairs: int
    oracle_requests: int
    best_attack_success: float
    worst_defense_break_rate: float


class LeagueRoundRunner:
    def __init__(
        self,
        *,
        loadout_pool,
        evaluator: OracleBatchEvaluator,
        league: LeagueManager | None = None,
        config: LeagueRoundConfig | None = None,
    ) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.evaluator = evaluator
        self.league = league or LeagueManager()
        self.config = config or LeagueRoundConfig()
        self.engine = ConstraintEngine(self.loadout_pool)

    def run(self, out_dir: Path) -> LeagueRoundSummary:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        match_format = MatchFormat(self.config.teams)
        generator = LegalPlanGenerator(self.loadout_pool, seed=self.config.seed)
        attack_oracle = AttackOracle(
            loadout_pool=self.loadout_pool,
            constraint_engine=self.engine,
            seed=self.config.seed + 101,
            config=_attack_config(self.config),
        )
        defenses: list[tuple[str, DefensePlan]] = []
        seen_defenses: set[str] = set()
        attempts = 0
        while len(defenses) < self.config.defenses and attempts < max(self.config.defenses * 20, 20):
            attempts += 1
            defense = generator.generate_defense_plan(match_format, source="league_random_defense")
            if not self.engine.is_legal_defense(defense):
                continue
            digest = defense.hash()
            if digest in seen_defenses:
                continue
            seen_defenses.add(digest)
            defense_id = f"def-{len(defenses) + 1:05d}"
            defenses.append((defense_id, defense))

        candidate_rows: list[dict[str, Any]] = []
        pairs: list[tuple[str, AttackPlan, str, DefensePlan]] = []
        attack_by_hash: dict[str, str] = {}
        attack_counter = 0
        for defense_id, defense in defenses:
            output = attack_oracle.search(defense)
            for rank, attack in enumerate(output.ranked_attacks[: self.config.oracle_top_k], start=1):
                attack_hash = attack.hash()
                attack_id = attack_by_hash.get(attack_hash)
                if attack_id is None:
                    attack_counter += 1
                    attack_id = f"atk-{attack_counter:05d}"
                    attack_by_hash[attack_hash] = attack_id
                predicted = output.predicted_scores[rank - 1] if rank - 1 < len(output.predicted_scores) else None
                simulated = output.simulated_scores[rank - 1] if rank - 1 < len(output.simulated_scores) else None
                candidate_rows.append(
                    {
                        "round_id": self.config.round_id,
                        "defense_id": defense_id,
                        "attack_id": attack_id,
                        "rank": rank,
                        "attack_hash": attack_hash,
                        "defense_hash": defense.hash(),
                        "predicted_score": predicted,
                        "surrogate_score": simulated,
                    }
                )
                pairs.append((attack_id, attack, defense_id, defense))

        records = self.evaluator.evaluate_pairs(
            pairs,
            job_prefix=self.config.round_id,
            base_seed=self.config.seed,
            metadata={"round_id": self.config.round_id},
        )
        attack_records, defense_records = self._update_league(pairs, records)
        self._write_artifacts(out_dir, candidate_rows, records, attack_records, defense_records)
        best_attack_success = max((record.attack_success for record in records), default=0.0)
        defense_break_rates = _defense_break_rates(records)
        worst_defense_break_rate = max(defense_break_rates.values(), default=0.0)
        summary = LeagueRoundSummary(
            round_id=self.config.round_id,
            out_dir=str(out_dir),
            defenses=len(defenses),
            candidates=len(candidate_rows),
            oracle_pairs=len(records),
            oracle_requests=sum(len(record.oracle_request_ids) for record in records),
            best_attack_success=best_attack_success,
            worst_defense_break_rate=worst_defense_break_rate,
        )
        (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    def _update_league(
        self,
        pairs: list[tuple[str, AttackPlan, str, DefensePlan]],
        records: list[OracleEvaluationRecord],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pair_by_ids = {(attack_id, defense_id): (attack, defense) for attack_id, attack, defense_id, defense in pairs}
        defense_strength = {defense_id: 1.0 - rate for defense_id, rate in _defense_break_rates(records).items()}
        attack_rows: list[dict[str, Any]] = []
        defense_rows: list[dict[str, Any]] = []
        seen_attacks: set[str] = set()
        seen_defenses: set[str] = set()
        for record in records:
            attack, defense = pair_by_ids[(record.attack_id, record.defense_id)]
            if record.attack_id not in seen_attacks:
                seen_attacks.add(record.attack_id)
                league_record = self.league.add_attack(
                    attack,
                    role="candidate",
                    source="attack_oracle",
                    strength=record.attack_success,
                )
                attack_rows.append(
                    {
                        "attack_id": record.attack_id,
                        "league_id": league_record.strategy_id,
                        "attack_hash": record.attack_hash,
                        "strength": record.attack_success,
                    }
                )
            if record.defense_id not in seen_defenses:
                seen_defenses.add(record.defense_id)
                strength = defense_strength.get(record.defense_id, 0.0)
                league_record = self.league.add_defense(
                    defense,
                    role="candidate",
                    source="legal_random_defense",
                    strength=strength,
                )
                defense_rows.append(
                    {
                        "defense_id": record.defense_id,
                        "league_id": league_record.strategy_id,
                        "defense_hash": record.defense_hash,
                        "strength": strength,
                        "break_rate": 1.0 - strength,
                    }
                )
            self.league.record_payoff(
                record.attack_id,
                record.defense_id,
                attack_success=record.attack_success,
                games=len(record.round_win_rates),
            )
        return attack_rows, defense_rows

    def _write_artifacts(
        self,
        out_dir: Path,
        candidate_rows: list[dict[str, Any]],
        records: list[OracleEvaluationRecord],
        attack_records: list[dict[str, Any]],
        defense_records: list[dict[str, Any]],
    ) -> None:
        _write_jsonl(out_dir / "candidates.jsonl", candidate_rows)
        _write_jsonl(out_dir / "oracle_requests.jsonl", [request for record in records for request in record.requests])
        _write_jsonl(out_dir / "oracle_results.jsonl", [result for record in records for result in record.results])
        _write_jsonl(out_dir / "scored_attacks.jsonl", attack_records)
        _write_jsonl(out_dir / "scored_defenses.jsonl", defense_records)
        state = {
            "iteration": self.league.iteration,
            "attack_pool": [_jsonable(record) for _plan, record in self.league.attack_pool.values()],
            "defense_pool": [_jsonable(record) for _plan, record in self.league.defense_pool.values()],
            "payoffs": [_jsonable(entry) for entry in self.league.payoffs.values()],
        }
        (out_dir / "league_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _attack_config(config: LeagueRoundConfig) -> AttackOracleConfig:
    diversity_keep = min(config.attacks_per_defense, max(config.oracle_top_k * 4, config.oracle_top_k))
    first_keep = min(diversity_keep, max(config.oracle_top_k * 2, config.oracle_top_k))
    return AttackOracleConfig(
        candidate_count=config.attacks_per_defense,
        diversity_keep=diversity_keep,
        final_keep=config.oracle_top_k,
        halving_stages=(HalvingStage(games_each=2, keep=first_keep), HalvingStage(games_each=5, keep=config.oracle_top_k)),
    )


def _defense_break_rates(records: list[OracleEvaluationRecord]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record.defense_id, []).append(record.attack_success)
    return {defense_id: max(values) for defense_id, values in grouped.items() if values}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
