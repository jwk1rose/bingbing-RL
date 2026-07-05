from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

from ..constraints import ConstraintEngine
from ..domain import DefensePlan, Observation, Team, canonical_hash, observe_defense
from ..real_platform.calibration import RealMetaDB, time_decay_weight

BeliefRanker = Callable[[Observation, tuple[Team, ...], Mapping[str, float]], float]


@dataclass(frozen=True)
class BeliefOutput:
    candidates: tuple[tuple[Team, ...], ...]
    weights: tuple[float, ...]
    entropy: float
    feasible_count_estimate: int
    top1_top2_gap: float
    domain_stats: tuple[tuple[str, float], ...]


class BeliefEngine:
    def __init__(
        self,
        constraint_engine: ConstraintEngine,
        *,
        defense_pool: Sequence[DefensePlan | tuple[DefensePlan, float]] = (),
        real_meta_db: RealMetaDB | None = None,
        ranker: BeliefRanker | None = None,
        ranker_weight: float = 1.0,
        now: float | None = None,
        real_tau: float = 30.0 * 24.0 * 3600.0,
        real_frequency_alpha: float = 2.0,
        pool_frequency_alpha: float = 1.5,
        strength_beta: float = 0.4,
        recency_gamma: float = 1.0,
        similarity_eta: float = 0.5,
        unseen_real_frequency: float = 0.01,
        real_similarity_threshold: float = 0.25,
        real_similarity_max_records: int = 256,
        use_equipment_star_features: bool = True,
        use_position_features: bool = True,
    ) -> None:
        self.constraint_engine = constraint_engine
        self.defense_pool = tuple(defense_pool)
        self.real_meta_db = real_meta_db
        self.ranker = ranker
        self.ranker_weight = float(ranker_weight)
        self.now = now
        self.real_tau = real_tau
        self.real_frequency_alpha = real_frequency_alpha
        self.pool_frequency_alpha = pool_frequency_alpha
        self.strength_beta = strength_beta
        self.recency_gamma = recency_gamma
        self.similarity_eta = similarity_eta
        self.unseen_real_frequency = unseen_real_frequency
        self.real_similarity_threshold = float(real_similarity_threshold)
        self.real_similarity_max_records = int(real_similarity_max_records)
        self.use_equipment_star_features = bool(use_equipment_star_features)
        self.use_position_features = bool(use_position_features)

    def build(self, observation: Observation, *, max_k: int = 64) -> BeliefOutput:
        domains = self.constraint_engine.build_domains(observation)
        completions = self.constraint_engine.enumerate_completions(observation, max_k=max_k)
        if not completions:
            return BeliefOutput((), (), 0.0, 0, 0.0, tuple((f"domain_{slot}", float(len(domain))) for slot, domain in domains.items()))
        real_summary = self._real_candidate_stats(observation)
        real_stats = real_summary.stats
        pool_stats, pool_record_count = self._defense_pool_candidate_stats(observation)
        completion_by_key = {_roster_key(roster): roster for roster in completions}
        for key, stats in real_stats.items():
            if key in completion_by_key:
                continue
            completion_by_key[key] = stats.roster
        for key, stats in pool_stats.items():
            if key in completion_by_key:
                continue
            completion_by_key[key] = stats.roster
        candidates = tuple(completion_by_key.values())
        scores = [
            self._score_candidate(observation, roster, real_stats, real_summary, pool_stats, pool_record_count)
            for roster in candidates
        ]
        weights = _softmax(scores)
        entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
        ordered = sorted(zip(candidates, weights), key=lambda pair: pair[1], reverse=True)
        top_weights = [weight for _candidate, weight in ordered[:2]]
        top1_top2_gap = top_weights[0] - top_weights[1] if len(top_weights) > 1 else top_weights[0]
        domain_stats = _domain_stats(domains) + tuple((f"{slot[0]}:{slot[1]}", float(len(domain))) for slot, domain in domains.items()) + (
            ("real_candidate_count", float(len(real_stats))),
            ("real_record_count", float(real_summary.record_count)),
            ("real_exact_record_count", float(real_summary.exact_record_count)),
            ("real_similar_record_count", float(real_summary.similar_record_count)),
            ("real_similarity_mean", real_summary.similarity_mean),
            ("real_match_result_mean", real_summary.match_result_mean),
            ("defense_pool_candidate_count", float(len(pool_stats))),
            ("defense_pool_record_count", float(pool_record_count)),
            ("ranker_applied", 1.0 if self.ranker is not None else 0.0),
        ) + _weight_stats(tuple(weight for _candidate, weight in ordered), len(candidates))
        return BeliefOutput(
            candidates=tuple(candidate for candidate, _weight in ordered),
            weights=tuple(weight for _candidate, weight in ordered),
            entropy=entropy,
            feasible_count_estimate=len(candidates),
            top1_top2_gap=top1_top2_gap,
            domain_stats=domain_stats,
        )

    def _real_candidate_stats(self, observation: Observation) -> "_RealCandidateStatsSummary":
        if self.real_meta_db is None:
            return _RealCandidateStatsSummary(stats={})
        exact_records = tuple((record, 1.0, True) for record in self.real_meta_db.by_observation_hash(observation.hash()))
        similar_records = tuple(
            (match.record, float(match.similarity), False)
            for match in self.real_meta_db.similar_observations(
                observation,
                min_similarity=self.real_similarity_threshold,
                max_records=self.real_similarity_max_records,
                include_exact=False,
            )
        )
        weighted_records = exact_records + similar_records
        if not weighted_records:
            return _RealCandidateStatsSummary(stats={})
        now = self.now if self.now is not None else max(record.timestamp for record, _similarity, _exact in weighted_records)
        stats: dict[str, _RealCandidateStats] = {}
        used_records = 0
        exact_used = 0
        similar_used = 0
        similarity_sum = 0.0
        match_result_sum = 0.0
        weight_sum = 0.0
        for record, similarity, exact in weighted_records:
            defense = record.full_defense_if_available
            if defense is None:
                continue
            if observation.season != "unknown" and record.season != observation.season:
                continue
            if observation.rank_segment != "unknown" and record.rank_segment != observation.rank_segment:
                continue
            if not _defense_matches_observation(defense, observation):
                continue
            if not self.constraint_engine.is_legal_defense(defense):
                continue
            used_records += 1
            if exact:
                exact_used += 1
            else:
                similar_used += 1
            key = _roster_key(defense.teams)
            recency = time_decay_weight(now=float(now), timestamp=record.timestamp, tau=self.real_tau)
            weight = max(float(similarity), 0.0)
            similarity_sum += weight
            match_result_sum += weight * float(record.match_result)
            weight_sum += weight
            current = stats.get(key)
            if current is None:
                stats[key] = _RealCandidateStats(
                    roster=defense.teams,
                    count=1,
                    weight=weight,
                    recency=weight * recency,
                    similarity=weight,
                    match_result=weight * float(record.match_result),
                )
            else:
                stats[key] = _RealCandidateStats(
                    roster=current.roster,
                    count=current.count + 1,
                    weight=current.weight + weight,
                    recency=current.recency + weight * recency,
                    similarity=current.similarity + weight,
                    match_result=current.match_result + weight * float(record.match_result),
                )
        return _RealCandidateStatsSummary(
            stats=stats,
            record_count=used_records,
            exact_record_count=exact_used,
            similar_record_count=similar_used,
            similarity_sum=similarity_sum,
            match_result_sum=match_result_sum,
            weight_sum=weight_sum,
        )

    def _defense_pool_candidate_stats(self, observation: Observation) -> tuple[dict[str, "_DefensePoolCandidateStats"], int]:
        stats: dict[str, _DefensePoolCandidateStats] = {}
        used_records = 0
        for item in self.defense_pool:
            if isinstance(item, tuple):
                defense, weight = item
            else:
                defense, weight = item, 1.0
            if observe_defense(defense).hash() != observation.hash():
                continue
            if not self.constraint_engine.is_legal_defense(defense):
                continue
            key = _roster_key(defense.teams)
            used_records += 1
            current = stats.get(key)
            if current is None:
                stats[key] = _DefensePoolCandidateStats(roster=defense.teams, count=1, weight=float(weight))
            else:
                stats[key] = _DefensePoolCandidateStats(
                    roster=current.roster,
                    count=current.count + 1,
                    weight=current.weight + float(weight),
                )
        return stats, used_records

    def _score_candidate(
        self,
        observation: Observation,
        roster: tuple[Team, ...],
        real_stats: dict[str, "_RealCandidateStats"],
        real_summary: "_RealCandidateStatsSummary",
        pool_stats: dict[str, "_DefensePoolCandidateStats"],
        pool_record_count: int,
    ) -> float:
        if not real_stats and not pool_stats and self.ranker is None:
            return _roster_strength(
                roster,
                use_equipment_star_features=self.use_equipment_star_features,
                use_position_features=self.use_position_features,
            )
        key = _roster_key(roster)
        real_stat = real_stats.get(key)
        pool_stat = pool_stats.get(key)
        if real_stat is None:
            frequency = self.unseen_real_frequency
            recency = self.unseen_real_frequency
            similarity = self.unseen_real_frequency
            match_result = 0.5
        else:
            frequency = real_stat.weight / max(real_summary.weight_sum, 1e-12)
            recency = real_stat.recency / max(real_stat.weight, 1e-12)
            similarity = real_stat.similarity / max(1, real_stat.count)
            match_result = real_stat.match_result / max(real_stat.weight, 1e-12)
        if pool_stat is None:
            pool_frequency = self.unseen_real_frequency
        else:
            pool_total = sum(stat.weight for stat in pool_stats.values())
            pool_frequency = pool_stat.weight / max(pool_total, 1e-12)
        strength = _roster_strength(
            roster,
            use_equipment_star_features=self.use_equipment_star_features,
            use_position_features=self.use_position_features,
        )
        features = {
            "roster_strength": strength,
            "real_frequency": frequency,
            "pool_frequency": pool_frequency,
            "recency": recency,
            "real_similarity": similarity,
            "real_match_result": match_result,
            "compatible_visible_ratio": _compatible_visible_ratio(observation, roster),
            "hidden_slot_count": float(len(observation.hidden_slots)),
        }
        score = (
            self.real_frequency_alpha * math.log(max(frequency, 1e-12))
            + self.pool_frequency_alpha * math.log(max(pool_frequency, 1e-12))
            + self.strength_beta * math.log(max(strength, 1.0))
            + self.recency_gamma * math.log(max(recency, 1e-12))
            + self.similarity_eta * similarity
        )
        if self.ranker is not None:
            score += self.ranker_weight * float(self.ranker(observation, roster, features))
        return 500.0 * score


@dataclass(frozen=True)
class _RealCandidateStats:
    roster: tuple[Team, ...]
    count: int
    weight: float
    recency: float
    similarity: float
    match_result: float


@dataclass(frozen=True)
class _RealCandidateStatsSummary:
    stats: dict[str, _RealCandidateStats]
    record_count: int = 0
    exact_record_count: int = 0
    similar_record_count: int = 0
    similarity_sum: float = 0.0
    match_result_sum: float = 0.0
    weight_sum: float = 0.0

    @property
    def similarity_mean(self) -> float:
        if self.record_count <= 0:
            return 0.0
        return self.similarity_sum / self.record_count

    @property
    def match_result_mean(self) -> float:
        if self.weight_sum <= 0.0:
            return 0.0
        return self.match_result_sum / self.weight_sum


@dataclass(frozen=True)
class _DefensePoolCandidateStats:
    roster: tuple[Team, ...]
    count: int
    weight: float


def _roster_strength(
    roster: tuple[Team, ...],
    *,
    use_equipment_star_features: bool = True,
    use_position_features: bool = True,
) -> float:
    total = sum(team.total_power for team in roster)
    if use_equipment_star_features:
        total += sum(10.0 * sum(loadout.unique_equip_star or 0 for loadout in team.slots) for team in roster)
    if use_position_features:
        total += 0.1 * sum(_team_position_profile(team) for team in roster)
    return total


def _team_position_profile(team: Team) -> float:
    if not team.slots:
        return 0.0
    return sum((index + 1) * loadout.standing_rank for index, loadout in enumerate(team.slots)) / len(team.slots)


def _compatible_visible_ratio(observation: Observation, roster: tuple[Team, ...]) -> float:
    visible_count = 0
    compatible_count = 0
    for team_idx, row in enumerate(observation.slots, start=1):
        for slot_idx, visible in enumerate(row, start=1):
            if visible.is_hidden:
                continue
            visible_count += 1
            candidate = roster[team_idx - 1].slots[slot_idx - 1]
            if visible.hero_id == candidate.hero_id and visible.unique_equip_id == candidate.unique_equip_id:
                compatible_count += 1
    if visible_count == 0:
        return 1.0
    return compatible_count / visible_count


def _defense_matches_observation(defense: DefensePlan, observation: Observation) -> bool:
    if defense.format != observation.format:
        return False
    for team_idx, row in enumerate(observation.slots, start=1):
        for slot_idx, visible in enumerate(row, start=1):
            if visible.is_hidden:
                continue
            candidate = defense.teams[team_idx - 1].slots[slot_idx - 1]
            if visible.hero_id != candidate.hero_id:
                return False
            if visible.unique_equip_id != candidate.unique_equip_id:
                return False
            if visible.unique_equip_star != candidate.unique_equip_star:
                return False
    return True


def _roster_key(roster: tuple[Team, ...]) -> str:
    return canonical_hash(roster)


def _domain_stats(domains) -> tuple[tuple[str, float], ...]:
    counts = [float(len(domain)) for domain in domains.values()]
    if not counts:
        return (
            ("hidden_slot_count", 0.0),
            ("domain_count_min", 0.0),
            ("domain_count_max", 0.0),
            ("domain_count_mean", 0.0),
            ("domain_count_entropy", 0.0),
        )
    total = sum(counts)
    probabilities = [count / total for count in counts if total > 0.0 and count > 0.0]
    entropy = -sum(probability * math.log(max(probability, 1e-12)) for probability in probabilities)
    return (
        ("hidden_slot_count", float(len(counts))),
        ("domain_count_min", min(counts)),
        ("domain_count_max", max(counts)),
        ("domain_count_mean", total / len(counts)),
        ("domain_count_entropy", entropy),
    )


def _weight_stats(ordered_weights: tuple[float, ...], candidate_count: int) -> tuple[tuple[str, float], ...]:
    top1 = ordered_weights[0] if ordered_weights else 0.0
    top2 = ordered_weights[1] if len(ordered_weights) > 1 else 0.0
    entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in ordered_weights)
    max_entropy = math.log(candidate_count) if candidate_count > 1 else 0.0
    normalized = entropy / max_entropy if max_entropy > 0.0 else 0.0
    return (
        ("candidate_count", float(candidate_count)),
        ("top1_weight", top1),
        ("top2_weight", top2),
        ("weight_entropy", entropy),
        ("weight_entropy_normalized", normalized),
    )


def _softmax(scores: list[float]) -> tuple[float, ...]:
    if not scores:
        return ()
    center = max(scores)
    values = [math.exp((score - center) / 500.0) for score in scores]
    total = sum(values)
    return tuple(value / total for value in values)
