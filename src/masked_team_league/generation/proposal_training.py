from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..constraints import ConstraintEngine
from ..belief import BeliefOutput
from ..domain import AttackPlan, DefensePlan, Loadout, MatchFormat, Observation, Team, observe_defense
from ..training.checkpoints import CheckpointRegistry, ModelCheckpointRecord
from .legal_generator import GenerationGoal
from .proposal_networks import (
    AttackGenerationNetwork,
    DefenseRosterGenerationNetwork,
    GenerationContextEncoder,
    GenerationContextOutput,
    MaskSelectionNetwork,
    ProposalNetworkConfig,
    ProposalSequence,
    beam_search_proposal_tokens,
    proposal_distillation_loss,
    sample_proposal_tokens,
)


MASK_SLOT_FEATURE_NAMES = (
    "team_index_norm",
    "slot_index_norm",
    "standing_rank_norm",
    "final_power_norm",
    "unique_equip_star_norm",
    "has_unique_equip",
    "hidden_budget_fraction",
    "ambiguity_score",
    "estimated_break_rate",
    "meta_attack_success",
    "learned_mask_score",
    "counter_sensitivity",
)


@dataclass(frozen=True)
class ProposalTeacherSample:
    target_token_ids: tuple[int, ...]
    legal_action_masks: tuple[tuple[bool, ...], ...]
    value_target: float
    gap_target: float
    anti_meta_residual_target: float | None = None
    weight: float = 1.0
    source: str = "unknown"

    def selected_prefix(self, step: int) -> tuple[int, ...]:
        if step < 0 or step > len(self.target_token_ids):
            raise ValueError("step must be within the target sequence")
        return (0,) + tuple(token_id + 1 for token_id in self.target_token_ids[:step])


@dataclass(frozen=True)
class ProposalTrainingHistory:
    train_losses: tuple[float, ...]


@dataclass(frozen=True)
class MaskTrainingSample:
    slot_features: tuple[tuple[float, ...], ...]
    target_mask: tuple[float, ...]
    weight: float = 1.0
    source: str = "unknown"


@dataclass(frozen=True)
class MaskSelectionTrainingHistory:
    train_losses: tuple[float, ...]


@dataclass(frozen=True)
class MaskSelectionLosses:
    bce_loss: torch.Tensor
    ranking_loss: torch.Tensor
    total_loss: torch.Tensor


@dataclass(frozen=True)
class ProposalAttackCandidate:
    plan: AttackPlan
    sequence: ProposalSequence


@dataclass(frozen=True)
class ProposalDefenseRosterCandidate:
    roster: tuple[Team, ...]
    sequence: ProposalSequence


@dataclass(frozen=True)
class AttackProposalContextTensors:
    observation_token_ids: torch.Tensor
    observation_hidden_mask: torch.Tensor
    belief_token_ids: torch.Tensor
    belief_weights: torch.Tensor
    pool_token_ids: torch.Tensor
    numeric_features: torch.Tensor

    def encode(
        self,
        encoder: GenerationContextEncoder,
        *,
        device: torch.device | str | None = None,
    ) -> GenerationContextOutput:
        target_device = torch.device(device) if device is not None else next(encoder.parameters()).device
        encoder.to(target_device)
        encoder.eval()
        with torch.no_grad():
            return encoder(
                self.observation_token_ids.to(target_device),
                observation_hidden_mask=self.observation_hidden_mask.to(target_device),
                belief_token_ids=self.belief_token_ids.to(target_device),
                belief_weights=self.belief_weights.to(target_device),
                pool_token_ids=self.pool_token_ids.to(target_device),
                numeric_features=self.numeric_features.to(target_device),
            )


@dataclass(frozen=True)
class DefenseProposalContextTensors:
    observation_token_ids: torch.Tensor
    observation_hidden_mask: torch.Tensor
    belief_token_ids: torch.Tensor
    belief_weights: torch.Tensor
    pool_token_ids: torch.Tensor
    numeric_features: torch.Tensor

    def encode(
        self,
        encoder: GenerationContextEncoder,
        *,
        device: torch.device | str | None = None,
    ) -> GenerationContextOutput:
        target_device = torch.device(device) if device is not None else next(encoder.parameters()).device
        encoder.to(target_device)
        encoder.eval()
        with torch.no_grad():
            return encoder(
                self.observation_token_ids.to(target_device),
                observation_hidden_mask=self.observation_hidden_mask.to(target_device),
                belief_token_ids=self.belief_token_ids.to(target_device),
                belief_weights=self.belief_weights.to(target_device),
                pool_token_ids=self.pool_token_ids.to(target_device),
                numeric_features=self.numeric_features.to(target_device),
            )


def build_attack_teacher_sample(
    plan: AttackPlan,
    *,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    value_target: float,
    gap_target: float,
    weight: float = 1.0,
    source: str = "unknown",
) -> ProposalTeacherSample:
    if not constraint_engine.is_legal_attack(plan):
        raise ValueError("teacher attack plan must be legal")
    return _build_roster_teacher_sample(
        plan.teams,
        match_format=plan.format,
        loadout_pool=loadout_pool,
        constraint_engine=constraint_engine,
        value_target=value_target,
        gap_target=gap_target,
        weight=weight,
        source=source,
    )


def build_defense_teacher_sample(
    plan: DefensePlan,
    *,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    value_target: float,
    gap_target: float,
    anti_meta_residual_target: float | None = None,
    weight: float = 1.0,
    source: str = "unknown",
) -> ProposalTeacherSample:
    if not constraint_engine.is_legal_defense(plan):
        raise ValueError("teacher defense plan must be legal")
    return _build_roster_teacher_sample(
        plan.teams,
        match_format=plan.format,
        loadout_pool=loadout_pool,
        constraint_engine=constraint_engine,
        value_target=value_target,
        gap_target=gap_target,
        anti_meta_residual_target=anti_meta_residual_target,
        weight=weight,
        source=source,
    )


def _build_roster_teacher_sample(
    roster: tuple[Team, ...],
    *,
    match_format: MatchFormat,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    value_target: float,
    gap_target: float,
    weight: float,
    source: str,
    anti_meta_residual_target: float | None = None,
) -> ProposalTeacherSample:
    index_by_loadout = {loadout: index for index, loadout in enumerate(loadout_pool)}
    target_token_ids: list[int] = []
    legal_masks: list[tuple[bool, ...]] = []
    used_heroes: set[int] = set()
    used_equips: set[int] = set()
    current_team_slots: list[Loadout] = []
    flat_loadouts = tuple(loadout for team in roster for loadout in team.slots)
    if len(flat_loadouts) != match_format.n_teams * match_format.team_size:
        raise ValueError("teacher roster does not match the match format")
    for step, loadout in enumerate(flat_loadouts):
        slot_idx = step % match_format.team_size
        if slot_idx == 0:
            current_team_slots = []
        if loadout not in index_by_loadout:
            raise ValueError("teacher plan contains a loadout that is not in loadout_pool")
        target_index = index_by_loadout[loadout]
        mask = constraint_engine.legal_action_mask(
            loadout_pool,
            current_team_slots=tuple(current_team_slots),
            remaining_team_slots_after_candidate=match_format.team_size - slot_idx - 1,
            used_hero_ids=frozenset(used_heroes),
            used_unique_equip_ids=frozenset(used_equips),
        )
        if not mask[target_index]:
            raise ValueError("teacher target is illegal under the generated action mask")
        legal_masks.append(mask)
        target_token_ids.append(target_index)
        current_team_slots.append(loadout)
        used_heroes.add(loadout.hero_id)
        if loadout.unique_equip_id is not None:
            used_equips.add(loadout.unique_equip_id)
    return ProposalTeacherSample(
        target_token_ids=tuple(target_token_ids),
        legal_action_masks=tuple(legal_masks),
        value_target=float(value_target),
        gap_target=float(gap_target),
        anti_meta_residual_target=None if anti_meta_residual_target is None else float(anti_meta_residual_target),
        weight=float(weight),
        source=source,
    )


def load_attack_teacher_samples_jsonl(
    path: str | Path,
    *,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    candidate_weight_temperature: float = 0.25,
    min_candidate_weight: float = 0.05,
) -> tuple[ProposalTeacherSample, ...]:
    if candidate_weight_temperature <= 0.0:
        raise ValueError("candidate_weight_temperature must be positive")
    samples: list[ProposalTeacherSample] = []
    rows: list[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "attack_plan" not in row:
                raise ValueError(f"line {line_no}: missing attack_plan")
            rows.append(row)
    inferred_weights = _infer_candidate_group_weights(
        rows,
        temperature=candidate_weight_temperature,
        min_weight=min_candidate_weight,
    )
    for row_index, row in enumerate(rows):
        plan = _attack_plan_from_dict(row["attack_plan"])
        value_target = float(row.get("value_target", row.get("attack_success", row.get("surrogate_score", 0.0))))
        gap_target = float(row.get("gap_target", row.get("belief_top1_top2_gap", 0.0)))
        samples.append(
            build_attack_teacher_sample(
                plan,
                loadout_pool=loadout_pool,
                constraint_engine=constraint_engine,
                value_target=value_target,
                gap_target=gap_target,
                weight=float(row.get("weight", inferred_weights[row_index])) * float(row.get("role_weight", 1.0)),
                source=str(row.get("source", row.get("attack_role", "unknown"))),
            )
        )
    return tuple(samples)


def load_defense_teacher_samples_jsonl(
    path: str | Path,
    *,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    candidate_weight_temperature: float = 0.25,
    min_candidate_weight: float = 0.05,
) -> tuple[ProposalTeacherSample, ...]:
    if candidate_weight_temperature <= 0.0:
        raise ValueError("candidate_weight_temperature must be positive")
    samples: list[ProposalTeacherSample] = []
    rows: list[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "defense_plan" not in row:
                raise ValueError(f"line {line_no}: missing defense_plan")
            rows.append(row)
    inferred_weights = _infer_candidate_group_weights(
        rows,
        temperature=candidate_weight_temperature,
        min_weight=min_candidate_weight,
    )
    for row_index, row in enumerate(rows):
        plan = _defense_plan_from_dict(row["defense_plan"])
        samples.append(
            build_defense_teacher_sample(
                plan,
                loadout_pool=loadout_pool,
                constraint_engine=constraint_engine,
                value_target=_defense_value_target(row),
                gap_target=_defense_gap_target(row),
                anti_meta_residual_target=_defense_anti_meta_residual_target(row),
                weight=float(row.get("weight", inferred_weights[row_index])),
                source=str(row.get("source", row.get("defense_role", "unknown"))),
            )
        )
    return tuple(samples)


def build_mask_slot_features(
    roster: tuple[Team, ...],
    match_format: MatchFormat,
    *,
    ambiguity_score: float = 0.0,
    estimated_break_rate: float = 0.0,
    meta_attack_success: float = 0.0,
    learned_mask_score: float = 0.0,
    counter_sensitivity: float = 0.0,
) -> tuple[tuple[float, ...], ...]:
    if len(roster) != match_format.n_teams:
        raise ValueError("roster team count does not match format")
    total_slots = match_format.n_teams * match_format.team_size
    features: list[tuple[float, ...]] = []
    for team_idx, team in enumerate(roster):
        if len(team.slots) != match_format.team_size:
            raise ValueError("roster team size does not match format")
        for slot_idx, loadout in enumerate(team.slots):
            features.append(
                (
                    team_idx / max(match_format.n_teams - 1, 1),
                    slot_idx / max(match_format.team_size - 1, 1),
                    float(loadout.standing_rank) / 1000.0,
                    float(loadout.final_power) / 100000.0,
                    float(loadout.unique_equip_star or 0) / 5.0,
                    1.0 if loadout.unique_equip_id is not None else 0.0,
                    float(match_format.max_hidden_total) / max(total_slots, 1),
                    float(ambiguity_score) / 10.0,
                    float(estimated_break_rate),
                    float(meta_attack_success),
                    float(learned_mask_score) / max(total_slots, 1),
                    float(counter_sensitivity),
                )
            )
    return tuple(features)


def build_mask_training_sample(
    plan: DefensePlan,
    *,
    ambiguity_score: float = 0.0,
    estimated_break_rate: float = 0.0,
    meta_attack_success: float = 0.0,
    learned_mask_score: float = 0.0,
    counter_sensitivity: float = 0.0,
    weight: float = 1.0,
    source: str = "unknown",
) -> MaskTrainingSample:
    flat_mask = tuple(float(value) for row in plan.mask for value in row)
    expected = plan.format.n_teams * plan.format.team_size
    if len(flat_mask) != expected:
        raise ValueError("defense mask does not match format")
    return MaskTrainingSample(
        slot_features=build_mask_slot_features(
            plan.teams,
            plan.format,
            ambiguity_score=ambiguity_score,
            estimated_break_rate=estimated_break_rate,
            meta_attack_success=meta_attack_success,
            learned_mask_score=learned_mask_score,
            counter_sensitivity=counter_sensitivity,
        ),
        target_mask=flat_mask,
        weight=float(weight),
        source=source,
    )


def load_mask_training_samples_jsonl(path: str | Path) -> tuple[MaskTrainingSample, ...]:
    samples: list[MaskTrainingSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "defense_plan" not in row:
                raise ValueError(f"line {line_no}: missing defense_plan")
            plan = _defense_plan_from_dict(row["defense_plan"])
            risk_report = row.get("defense_risk_report")
            if not isinstance(risk_report, Mapping):
                risk_report = {}
            estimated_break_rate = float(
                row.get(
                    "estimated_break_rate",
                    risk_report.get("estimated_break_rate", row.get("break_rate", 0.0)),
                )
            )
            meta_attack_success = float(row.get("meta_attack_success", risk_report.get("meta_attack_success", 0.0)))
            learned_mask_score = float(row.get("learned_mask_score", risk_report.get("learned_mask_score", 0.0)))
            survival = _defense_value_target(row)
            weight = float(row.get("weight", 1.0 + max(0.0, survival)))
            samples.append(
                build_mask_training_sample(
                    plan,
                    ambiguity_score=float(row.get("ambiguity_score", 0.0) or 0.0),
                    estimated_break_rate=estimated_break_rate,
                    meta_attack_success=meta_attack_success,
                    learned_mask_score=learned_mask_score,
                    counter_sensitivity=_mask_counter_sensitivity(row),
                    weight=weight,
                    source=str(row.get("source", row.get("defense_role", "unknown"))),
                )
            )
    return tuple(samples)


def mask_selection_ranking_loss(
    slot_scores: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    ranking_weight: float = 1.0,
) -> MaskSelectionLosses:
    if slot_scores.shape != target_mask.shape:
        raise ValueError("slot_scores and target_mask must have the same shape")
    if slot_scores.ndim != 2:
        raise ValueError("slot_scores and target_mask must have shape [batch, slots]")
    target = target_mask.to(device=slot_scores.device, dtype=slot_scores.dtype)
    bce_per_slot = F.binary_cross_entropy_with_logits(slot_scores, target, reduction="none")
    bce_per_sample = bce_per_slot.mean(dim=1)
    ranking_per_sample: list[torch.Tensor] = []
    for scores, mask in zip(slot_scores, target):
        positive = scores[mask >= 0.5]
        negative = scores[mask < 0.5]
        if positive.numel() == 0 or negative.numel() == 0:
            ranking_per_sample.append(scores.new_tensor(0.0))
            continue
        ranking_per_sample.append(F.softplus(negative.unsqueeze(0) - positive.unsqueeze(1)).mean())
    ranking = torch.stack(ranking_per_sample)
    if sample_weights is not None:
        weights = sample_weights.to(device=slot_scores.device, dtype=slot_scores.dtype)
        if weights.shape != bce_per_sample.shape:
            raise ValueError("sample_weights must have shape [batch]")
        scale = weights / weights.mean().clamp_min(1e-12)
        bce = (bce_per_sample * scale).mean()
        ranking_loss = (ranking * scale).mean()
    else:
        bce = bce_per_sample.mean()
        ranking_loss = ranking.mean()
    total = bce + float(ranking_weight) * ranking_loss
    return MaskSelectionLosses(bce_loss=bce, ranking_loss=ranking_loss, total_loss=total)


def train_mask_selection_network(
    network: MaskSelectionNetwork,
    samples: Sequence[MaskTrainingSample],
    *,
    epochs: int,
    lr: float,
    batch_size: int = 16,
    ranking_weight: float = 1.0,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> MaskSelectionTrainingHistory:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not samples:
        raise ValueError("samples must be non-empty")
    model_device = torch.device(device) if device is not None else next(network.parameters()).device
    network.to(model_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    rng = random.Random(seed)
    rows = list(samples)
    losses: list[float] = []
    for _epoch in range(epochs):
        rng.shuffle(rows)
        network.train()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            slot_features = torch.tensor([sample.slot_features for sample in batch], dtype=torch.float32, device=model_device)
            target_mask = torch.tensor([sample.target_mask for sample in batch], dtype=torch.float32, device=model_device)
            weights = torch.tensor([sample.weight for sample in batch], dtype=torch.float32, device=model_device)
            slot_scores = network(slot_features)
            loss = mask_selection_ranking_loss(
                slot_scores,
                target_mask,
                sample_weights=weights,
                ranking_weight=ranking_weight,
            ).total_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        losses.append(total_loss / max(batches, 1))
    return MaskSelectionTrainingHistory(train_losses=tuple(losses))


def _defense_value_target(row: Mapping[str, Any]) -> float:
    if "value_target" in row:
        return float(row["value_target"])
    if "survival_rate" in row:
        return float(row["survival_rate"])
    risk_report = row.get("defense_risk_report")
    if isinstance(risk_report, Mapping) and "estimated_survival_rate" in risk_report:
        return float(risk_report["estimated_survival_rate"])
    if "strength" in row:
        return float(row["strength"])
    if "break_rate" in row:
        return 1.0 - float(row["break_rate"])
    if "estimated_attack_success" in row:
        return 1.0 - float(row["estimated_attack_success"])
    return 0.0


def _defense_gap_target(row: Mapping[str, Any]) -> float:
    if "gap_target" in row:
        return float(row["gap_target"])
    return float(row.get("ambiguity_score", 0.0))


def _defense_anti_meta_residual_target(row: Mapping[str, Any]) -> float | None:
    if "anti_meta_residual_target" in row:
        return float(row["anti_meta_residual_target"])
    risk_report = row.get("defense_risk_report")
    if isinstance(risk_report, Mapping):
        survival = risk_report.get("estimated_survival_rate")
        meta_attack_success = risk_report.get("meta_attack_success")
        if isinstance(survival, (int, float)) and isinstance(meta_attack_success, (int, float)):
            return float(survival) - float(meta_attack_success)
    return None


def _mask_counter_sensitivity(row: Mapping[str, Any]) -> float:
    if "counter_sensitivity" in row:
        return float(row["counter_sensitivity"])
    risk_report = row.get("defense_risk_report")
    if not isinstance(risk_report, Mapping):
        return 0.0
    estimated_break = float(risk_report.get("estimated_break_rate", row.get("break_rate", 0.0)) or 0.0)
    meta_attack = float(risk_report.get("meta_attack_success", row.get("meta_attack_success", 0.0)) or 0.0)
    backup_breaks = [
        float(value)
        for value in risk_report.get("backup_break_rates", ())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    backup_spread = max(backup_breaks) - min(backup_breaks) if backup_breaks else 0.0
    counter_risk = risk_report.get("counter_attack_risk_report")
    counter_spread = 0.0
    if isinstance(counter_risk, Mapping):
        expected = counter_risk.get("expected_match_win")
        worst = counter_risk.get("worst_case_match_win")
        if isinstance(expected, (int, float)) and isinstance(worst, (int, float)):
            counter_spread = abs(float(expected) - float(worst))
    return max(0.0, estimated_break - meta_attack) + max(0.0, backup_spread) + max(0.0, counter_spread)


def _infer_candidate_group_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
    min_weight: float,
) -> tuple[float, ...]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(_teacher_group_key(row, index), []).append(index)
    weights = [1.0] * len(rows)
    for indexes in groups.values():
        rows_without_explicit_weight = [index for index in indexes if "weight" not in rows[index]]
        if not rows_without_explicit_weight:
            continue
        scores = [_teacher_candidate_score(rows[index]) for index in rows_without_explicit_weight]
        max_score = max(scores, default=0.0)
        raw = [max(float(min_weight), math.exp((score - max_score) / temperature)) for score in scores]
        scale = len(raw) / max(sum(raw), 1e-12)
        for index, value in zip(rows_without_explicit_weight, raw):
            weights[index] = value * scale
    return tuple(weights)


def _teacher_group_key(row: Mapping[str, Any], index: int) -> str:
    for key in ("teacher_group_id", "defense_id", "target_hash", "observation_hash"):
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    return f"line:{index}"


def _teacher_candidate_score(row: Mapping[str, Any]) -> float:
    if "exploiter_residual_target" in row:
        return float(row.get("attack_success", row.get("value_target", 0.0))) + float(row["exploiter_residual_target"])
    for key in (
        "attack_success",
        "anti_meta_residual_target",
        "strength",
        "defense_strength",
        "survival_rate",
        "value_target",
        "oracle_score",
        "surrogate_score",
        "predicted_score",
    ):
        if key in row:
            return float(row[key])
    if "rank" in row:
        return -float(row["rank"])
    return 0.0


def _normalize_weighted_attack_meta(attack_meta: Sequence[tuple[AttackPlan, float]]) -> tuple[tuple[AttackPlan, float], ...]:
    positive = tuple((attack, max(float(weight), 0.0)) for attack, weight in attack_meta)
    total = sum(weight for _attack, weight in positive)
    if total <= 0.0:
        return tuple((attack, 0.0) for attack, _weight in positive)
    return tuple((attack, weight / total) for attack, weight in positive)


def build_attack_proposal_context_tensors(
    observation: Observation,
    belief: BeliefOutput,
    *,
    loadout_pool: tuple[Loadout, ...],
    max_belief_candidates: int = 8,
) -> AttackProposalContextTensors:
    if max_belief_candidates <= 0:
        raise ValueError("max_belief_candidates must be positive")
    index_by_loadout = {loadout: index + 1 for index, loadout in enumerate(loadout_pool)}
    observation_tokens: list[int] = []
    hidden_mask: list[bool] = []
    for row in observation.slots:
        for slot in row:
            hidden_mask.append(slot.is_hidden)
            if slot.is_hidden or slot.loadout is None:
                observation_tokens.append(0)
            else:
                observation_tokens.append(index_by_loadout.get(slot.loadout, 0))
    total_slots = observation.format.n_teams * observation.format.team_size
    belief_rows: list[list[int]] = []
    weights: list[float] = []
    for roster, weight in list(zip(belief.candidates, belief.weights))[:max_belief_candidates]:
        belief_rows.append([index_by_loadout.get(loadout, 0) for team in roster for loadout in team.slots])
        weights.append(float(weight))
    while len(belief_rows) < max_belief_candidates:
        belief_rows.append([0] * total_slots)
        weights.append(0.0)
    pool_tokens = [index + 1 for index, _loadout in enumerate(loadout_pool)]
    numeric_features = [
        float(belief.entropy),
        float(belief.top1_top2_gap),
        float(belief.feasible_count_estimate) / 1000.0,
        float(len(observation.hidden_slots)) / max(total_slots, 1),
    ]
    return AttackProposalContextTensors(
        observation_token_ids=torch.tensor([observation_tokens], dtype=torch.long),
        observation_hidden_mask=torch.tensor([hidden_mask], dtype=torch.bool),
        belief_token_ids=torch.tensor([belief_rows], dtype=torch.long),
        belief_weights=torch.tensor([weights], dtype=torch.float32),
        pool_token_ids=torch.tensor([pool_tokens], dtype=torch.long),
        numeric_features=torch.tensor([numeric_features], dtype=torch.float32),
    )


def build_defense_proposal_context_tensors(
    attack_meta: Sequence[tuple[AttackPlan, float]],
    *,
    match_format: MatchFormat,
    loadout_pool: tuple[Loadout, ...],
    max_attack_meta: int = 8,
) -> DefenseProposalContextTensors:
    if max_attack_meta <= 0:
        raise ValueError("max_attack_meta must be positive")
    index_by_loadout = {loadout: index + 1 for index, loadout in enumerate(loadout_pool)}
    total_slots = match_format.n_teams * match_format.team_size
    meta_rows: list[list[int]] = []
    weights: list[float] = []
    normalized = _normalize_weighted_attack_meta(attack_meta)
    for attack, weight in normalized[:max_attack_meta]:
        meta_rows.append([index_by_loadout.get(loadout, 0) for team in attack.teams for loadout in team.slots])
        weights.append(float(weight))
    while len(meta_rows) < max_attack_meta:
        meta_rows.append([0] * total_slots)
        weights.append(0.0)
    pool_tokens = [index + 1 for index, _loadout in enumerate(loadout_pool)]
    costs = [sum(team.total_cost for team in attack.teams) for attack, _weight in normalized]
    mean_cost = sum(costs) / len(costs) if costs else 0.0
    max_cost = max(costs, default=0.0)
    weighted_cost = sum(cost * weight for cost, (_attack, weight) in zip(costs, normalized)) if normalized else 0.0
    numeric_features = [
        float(len(normalized)) / max(max_attack_meta, 1),
        mean_cost / 100000.0,
        max_cost / 100000.0,
        weighted_cost / 100000.0,
    ]
    return DefenseProposalContextTensors(
        observation_token_ids=torch.zeros((1, total_slots), dtype=torch.long),
        observation_hidden_mask=torch.zeros((1, total_slots), dtype=torch.bool),
        belief_token_ids=torch.tensor([meta_rows], dtype=torch.long),
        belief_weights=torch.tensor([weights], dtype=torch.float32),
        pool_token_ids=torch.tensor([pool_tokens], dtype=torch.long),
        numeric_features=torch.tensor([numeric_features], dtype=torch.float32),
    )


def attack_legal_action_mask_fn(
    match_format: MatchFormat,
    *,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    goal: GenerationGoal | None = None,
    reference_cost: float | None = None,
    use_future_feasibility: bool = True,
):
    total_slots = match_format.n_teams * match_format.team_size
    budget = None if goal is None or reference_cost is None else float(goal.target_power_ratio) * float(reference_cost)

    def _mask(prefix: tuple[int, ...]) -> tuple[bool, ...]:
        if len(prefix) >= total_slots:
            return tuple(False for _loadout in loadout_pool)
        selected = _prefix_loadouts(prefix, loadout_pool)
        current_team_start = (len(prefix) // match_format.team_size) * match_format.team_size
        current_team_slots = selected[current_team_start:]
        slot_idx = len(prefix) % match_format.team_size
        used_heroes = frozenset(loadout.hero_id for loadout in selected)
        used_equips = frozenset(loadout.unique_equip_id for loadout in selected if loadout.unique_equip_id is not None)
        structural = constraint_engine.legal_action_mask(
            loadout_pool,
            current_team_slots=tuple(current_team_slots),
            remaining_team_slots_after_candidate=match_format.team_size - slot_idx - 1,
            used_hero_ids=used_heroes,
            used_unique_equip_ids=used_equips,
            use_future_feasibility=use_future_feasibility,
        )
        if budget is None:
            return structural
        current_cost = sum(loadout.cost for loadout in selected)
        return tuple(
            allowed and current_cost + loadout.cost <= budget
            for allowed, loadout in zip(structural, loadout_pool)
        )

    return _mask


def defense_legal_action_mask_fn(
    match_format: MatchFormat,
    *,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    goal: GenerationGoal | None = None,
    reference_cost: float | None = None,
    use_future_feasibility: bool = True,
):
    return attack_legal_action_mask_fn(
        match_format,
        loadout_pool=loadout_pool,
        constraint_engine=constraint_engine,
        goal=goal,
        reference_cost=reference_cost,
        use_future_feasibility=use_future_feasibility,
    )


def proposal_sequence_to_attack_plan(
    token_ids: Sequence[int] | ProposalSequence,
    *,
    loadout_pool: tuple[Loadout, ...],
    match_format: MatchFormat,
    constraint_engine: ConstraintEngine,
    source: str = "attack_proposal",
) -> AttackPlan:
    sequence = token_ids.token_ids if isinstance(token_ids, ProposalSequence) else tuple(int(token) for token in token_ids)
    expected = match_format.n_teams * match_format.team_size
    if len(sequence) != expected:
        raise ValueError(f"token sequence length must be {expected}")
    loadouts = _prefix_loadouts(sequence, loadout_pool)
    teams = tuple(
        Team(tuple(loadouts[start : start + match_format.team_size]))
        for start in range(0, len(loadouts), match_format.team_size)
    )
    plan = AttackPlan(format=match_format, teams=teams, source=source)
    if not constraint_engine.is_legal_attack(plan):
        raise ValueError("illegal attack plan generated from proposal tokens")
    return plan


def proposal_sequence_to_defense_plan(
    token_ids: Sequence[int] | ProposalSequence,
    *,
    loadout_pool: tuple[Loadout, ...],
    match_format: MatchFormat,
    constraint_engine: ConstraintEngine,
    mask: tuple[tuple[int, ...], ...] | None = None,
    source: str = "defense_proposal",
) -> DefensePlan:
    sequence = token_ids.token_ids if isinstance(token_ids, ProposalSequence) else tuple(int(token) for token in token_ids)
    expected = match_format.n_teams * match_format.team_size
    if len(sequence) != expected:
        raise ValueError(f"token sequence length must be {expected}")
    loadouts = _prefix_loadouts(sequence, loadout_pool)
    teams = tuple(
        Team(tuple(loadouts[start : start + match_format.team_size]))
        for start in range(0, len(loadouts), match_format.team_size)
    )
    if mask is None:
        mask = tuple((0, 0, 0, 0, 0) for _team in range(match_format.n_teams))
    plan = DefensePlan(format=match_format, teams=teams, mask=mask, source=source)
    if not constraint_engine.is_legal_defense(plan):
        raise ValueError("illegal defense plan generated from proposal tokens")
    return plan


def generate_attack_plan_candidates(
    network: nn.Module,
    *,
    match_format: MatchFormat,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    legal_action_mask_fn=None,
    beam_size: int = 8,
    source: str = "attack_proposal",
    context_vector: torch.Tensor | None = None,
    goal: GenerationGoal | None = None,
    reference_cost: float | None = None,
    use_future_feasibility: bool = True,
    device: torch.device | str | None = None,
) -> tuple[ProposalAttackCandidate, ...]:
    mask_fn = legal_action_mask_fn or attack_legal_action_mask_fn(
        match_format,
        loadout_pool=loadout_pool,
        constraint_engine=constraint_engine,
        goal=goal,
        reference_cost=reference_cost,
        use_future_feasibility=use_future_feasibility,
    )
    sequences = beam_search_proposal_tokens(
        network,
        legal_action_mask_fn=mask_fn,
        max_steps=match_format.n_teams * match_format.team_size,
        beam_size=beam_size,
        context_vector=context_vector,
        device=device,
    )
    candidates: list[ProposalAttackCandidate] = []
    for sequence in sequences:
        plan = proposal_sequence_to_attack_plan(
            sequence,
            loadout_pool=loadout_pool,
            match_format=match_format,
            constraint_engine=constraint_engine,
            source=source,
        )
        candidates.append(ProposalAttackCandidate(plan=plan, sequence=sequence))
    return tuple(candidates)


def generate_defense_roster_candidates(
    network: nn.Module,
    *,
    match_format: MatchFormat,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    legal_action_mask_fn=None,
    beam_size: int = 8,
    source: str = "defense_proposal",
    context_vector: torch.Tensor | None = None,
    goal: GenerationGoal | None = None,
    reference_cost: float | None = None,
    use_future_feasibility: bool = True,
    device: torch.device | str | None = None,
) -> tuple[ProposalDefenseRosterCandidate, ...]:
    mask_fn = legal_action_mask_fn or defense_legal_action_mask_fn(
        match_format,
        loadout_pool=loadout_pool,
        constraint_engine=constraint_engine,
        goal=goal,
        reference_cost=reference_cost,
        use_future_feasibility=use_future_feasibility,
    )
    sequences = beam_search_proposal_tokens(
        network,
        legal_action_mask_fn=mask_fn,
        max_steps=match_format.n_teams * match_format.team_size,
        beam_size=beam_size,
        context_vector=context_vector,
        device=device,
    )
    candidates: list[ProposalDefenseRosterCandidate] = []
    for sequence in sequences:
        plan = proposal_sequence_to_defense_plan(
            sequence,
            loadout_pool=loadout_pool,
            match_format=match_format,
            constraint_engine=constraint_engine,
            source=source,
        )
        candidates.append(ProposalDefenseRosterCandidate(roster=plan.teams, sequence=sequence))
    return tuple(candidates)


def sample_attack_plan_candidate(
    network: nn.Module,
    *,
    match_format: MatchFormat,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    legal_action_mask_fn=None,
    source: str = "attack_proposal",
    temperature: float = 1.0,
    context_vector: torch.Tensor | None = None,
    goal: GenerationGoal | None = None,
    reference_cost: float | None = None,
    use_future_feasibility: bool = True,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> ProposalAttackCandidate:
    mask_fn = legal_action_mask_fn or attack_legal_action_mask_fn(
        match_format,
        loadout_pool=loadout_pool,
        constraint_engine=constraint_engine,
        goal=goal,
        reference_cost=reference_cost,
        use_future_feasibility=use_future_feasibility,
    )
    sequence = sample_proposal_tokens(
        network,
        legal_action_mask_fn=mask_fn,
        max_steps=match_format.n_teams * match_format.team_size,
        temperature=temperature,
        context_vector=context_vector,
        device=device,
        seed=seed,
    )
    plan = proposal_sequence_to_attack_plan(
        sequence,
        loadout_pool=loadout_pool,
        match_format=match_format,
        constraint_engine=constraint_engine,
        source=source,
    )
    return ProposalAttackCandidate(plan=plan, sequence=sequence)


def load_defense_proposal_network(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> DefenseRosterGenerationNetwork:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if str(payload.get("model_type", "")) != "defense_proposal":
        raise ValueError("checkpoint is not a defense_proposal model")
    config = ProposalNetworkConfig(**payload["config"])
    network = DefenseRosterGenerationNetwork(config)
    _load_proposal_state_dict_compat(network, payload["state_dict"])
    if device is not None:
        network.to(torch.device(device))
    network.eval()
    return network


def load_defense_proposal_candidate_source(
    path: str | Path,
    *,
    beam_size: int = 8,
    context_encoder: GenerationContextEncoder | None = None,
    max_attack_meta: int = 8,
    use_future_feasibility: bool = True,
    device: torch.device | str | None = None,
    source: str = "defense_proposal",
):
    network = load_defense_proposal_network(path, device=device)

    def _source(**kwargs):
        loadout_pool = tuple(kwargs["loadout_pool"])
        if len(loadout_pool) != network.config.loadout_count:
            raise ValueError("loadout_pool size does not match defense proposal checkpoint")
        context_vector = None
        if context_encoder is not None:
            context_vector = build_defense_proposal_context_tensors(
                tuple(kwargs.get("attack_meta", ())),
                match_format=kwargs["match_format"],
                loadout_pool=loadout_pool,
                max_attack_meta=max_attack_meta,
            ).encode(context_encoder, device=device).context_vector
        candidates = generate_defense_roster_candidates(
            network,
            match_format=kwargs["match_format"],
            loadout_pool=loadout_pool,
            constraint_engine=kwargs["constraint_engine"],
            beam_size=beam_size,
            source=source,
            context_vector=context_vector,
            goal=kwargs.get("goal"),
            reference_cost=kwargs.get("reference_cost"),
            use_future_feasibility=use_future_feasibility,
            device=device,
        )
        return tuple(candidate.roster for candidate in candidates)

    return _source


def load_attack_proposal_network(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> AttackGenerationNetwork:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if str(payload.get("model_type", "")) != "attack_proposal":
        raise ValueError("checkpoint is not an attack_proposal model")
    config = ProposalNetworkConfig(**payload["config"])
    network = AttackGenerationNetwork(config)
    _load_proposal_state_dict_compat(network, payload["state_dict"])
    if device is not None:
        network.to(torch.device(device))
    network.eval()
    return network


def _load_proposal_state_dict_compat(network: nn.Module, state_dict: Mapping[str, Any]) -> None:
    result = network.load_state_dict(state_dict, strict=False)
    allowed_missing = {"anti_meta_residual_head.weight", "anti_meta_residual_head.bias"}
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            "incompatible proposal checkpoint state_dict: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )


def load_attack_proposal_candidate_source(
    path: str | Path,
    *,
    beam_size: int = 8,
    context_encoder: GenerationContextEncoder | None = None,
    max_belief_candidates: int = 8,
    use_future_feasibility: bool = True,
    device: torch.device | str | None = None,
    source: str = "attack_proposal",
):
    network = load_attack_proposal_network(path, device=device)

    def _source(**kwargs):
        loadout_pool = tuple(kwargs["loadout_pool"])
        if len(loadout_pool) != network.config.loadout_count:
            raise ValueError("loadout_pool size does not match attack proposal checkpoint")
        context_vector = None
        if context_encoder is not None:
            target = kwargs.get("target")
            belief = kwargs.get("belief")
            if target is None or belief is None:
                raise ValueError("target and belief are required when context_encoder is provided")
            observation = target if isinstance(target, Observation) else observe_defense(target)
            context_vector = build_attack_proposal_context_tensors(
                observation,
                belief,
                loadout_pool=loadout_pool,
                max_belief_candidates=max_belief_candidates,
            ).encode(context_encoder, device=device).context_vector
        candidates = generate_attack_plan_candidates(
            network,
            match_format=kwargs["match_format"],
            loadout_pool=loadout_pool,
            constraint_engine=kwargs["constraint_engine"],
            beam_size=beam_size,
            source=source,
            context_vector=context_vector,
            goal=kwargs.get("goal"),
            reference_cost=kwargs.get("reference_cost"),
            use_future_feasibility=use_future_feasibility,
            device=device,
        )
        return tuple(candidate.plan for candidate in candidates)

    return _source


def train_proposal_network(
    network: nn.Module,
    samples: Sequence[ProposalTeacherSample],
    *,
    epochs: int,
    lr: float,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> ProposalTrainingHistory:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    model_device = torch.device(device) if device is not None else next(network.parameters()).device
    network.to(model_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    rng = random.Random(seed)
    rows = list(samples)
    losses: list[float] = []
    for _epoch in range(epochs):
        rng.shuffle(rows)
        network.train()
        total_loss = 0.0
        total_steps = 0
        for sample in rows:
            for step, target_token_id in enumerate(sample.target_token_ids):
                selected = torch.tensor([sample.selected_prefix(step)], dtype=torch.long, device=model_device)
                legal = torch.tensor([sample.legal_action_masks[step]], dtype=torch.bool, device=model_device)
                target = torch.tensor([[target_token_id]], dtype=torch.long, device=model_device)
                value_target = torch.tensor([sample.value_target], dtype=torch.float32, device=model_device)
                gap_target = torch.tensor([sample.gap_target], dtype=torch.float32, device=model_device)
                residual_target = (
                    None
                    if sample.anti_meta_residual_target is None
                    else torch.tensor([sample.anti_meta_residual_target], dtype=torch.float32, device=model_device)
                )
                weight = torch.tensor([sample.weight], dtype=torch.float32, device=model_device)
                output = network(selected, legal)
                loss = proposal_distillation_loss(
                    output.logits.unsqueeze(1),
                    target,
                    legal_action_mask=legal.unsqueeze(1),
                    candidate_weights=weight,
                    value_estimates=output.value_estimates,
                    value_targets=value_target,
                    gap_estimates=output.gap_estimates,
                    gap_targets=gap_target,
                    anti_meta_residual_estimates=(
                        output.anti_meta_residual_estimates if residual_target is not None else None
                    ),
                    anti_meta_residual_targets=residual_target,
                ).total_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                total_steps += 1
        losses.append(total_loss / max(total_steps, 1))
    return ProposalTrainingHistory(train_losses=tuple(losses))


def save_proposal_network_checkpoint(
    path: str | Path,
    network: nn.Module,
    history: ProposalTrainingHistory,
    *,
    registry_path: str | Path | None = None,
    checkpoint_id: str | None = None,
    dataset_hash: str = "unknown",
    model_type: str = "attack_proposal",
    metadata: Mapping[str, Any] | None = None,
) -> ModelCheckpointRecord:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_loss = float(history.train_losses[-1]) if history.train_losses else 0.0
    payload = {
        "model_type": model_type,
        "config": asdict(network.config) if hasattr(network, "config") else {},
        "state_dict": network.state_dict(),
        "train_losses": list(history.train_losses),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_payload = {"train_loss": train_loss, "epochs": len(history.train_losses)}
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = ModelCheckpointRecord(
        checkpoint_id=checkpoint_id or output_path.stem,
        model_type=model_type,
        model_path=str(output_path),
        metrics_path=str(metrics_path),
        created_at=time.time(),
        dataset_hash=dataset_hash,
        metrics=metrics_payload,
    )
    if registry_path is not None:
        CheckpointRegistry(registry_path).add(record)
    return record


def save_mask_selection_checkpoint(
    path: str | Path,
    network: MaskSelectionNetwork,
    history: MaskSelectionTrainingHistory,
    *,
    registry_path: str | Path | None = None,
    checkpoint_id: str | None = None,
    dataset_hash: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
) -> ModelCheckpointRecord:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_loss = float(history.train_losses[-1]) if history.train_losses else 0.0
    payload = {
        "model_type": "mask_selection",
        "feature_dim": network.feature_dim,
        "hidden_dim": network.hidden_dim,
        "feature_names": list(MASK_SLOT_FEATURE_NAMES),
        "state_dict": network.state_dict(),
        "train_losses": list(history.train_losses),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_payload = {"train_loss": train_loss, "epochs": len(history.train_losses)}
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = ModelCheckpointRecord(
        checkpoint_id=checkpoint_id or output_path.stem,
        model_type="mask_selection",
        model_path=str(output_path),
        metrics_path=str(metrics_path),
        created_at=time.time(),
        dataset_hash=dataset_hash,
        metrics=metrics_payload,
    )
    if registry_path is not None:
        CheckpointRegistry(registry_path).add(record)
    return record


def load_mask_selection_network(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> MaskSelectionNetwork:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if str(payload.get("model_type", "")) != "mask_selection":
        raise ValueError("checkpoint is not a mask_selection model")
    network = MaskSelectionNetwork(
        feature_dim=int(payload["feature_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 128)),
    )
    network.load_state_dict(payload["state_dict"])
    if device is not None:
        network.to(torch.device(device))
    network.eval()
    return network


def load_mask_slot_score_provider(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
):
    network = load_mask_selection_network(path, device=device)

    def _provider(roster: tuple[Team, ...], match_format: MatchFormat) -> tuple[tuple[float, ...], ...]:
        model_device = torch.device(device) if device is not None else next(network.parameters()).device
        features = build_mask_slot_features(roster, match_format)
        with torch.no_grad():
            scores = network(torch.tensor([features], dtype=torch.float32, device=model_device))[0].detach().cpu().tolist()
        rows: list[tuple[float, ...]] = []
        index = 0
        for _team_idx in range(match_format.n_teams):
            row = tuple(float(value) for value in scores[index : index + match_format.team_size])
            rows.append(row)
            index += match_format.team_size
        return tuple(rows)

    return _provider


def _prefix_loadouts(prefix: Sequence[int], loadout_pool: tuple[Loadout, ...]) -> tuple[Loadout, ...]:
    loadouts: list[Loadout] = []
    for token in prefix:
        token_id = int(token)
        if token_id < 0 or token_id >= len(loadout_pool):
            raise ValueError("proposal token is out of loadout_pool range")
        loadouts.append(loadout_pool[token_id])
    return tuple(loadouts)


def _match_format_from_dict(data: Mapping[str, Any]) -> MatchFormat:
    return MatchFormat(
        n_teams=int(data["n_teams"]),
        team_size=int(data.get("team_size", 5)),
        win_required=None if data.get("win_required") is None else int(data["win_required"]),
        max_hidden_per_team=int(data.get("max_hidden_per_team", 2)),
        max_hidden_total=int(data.get("max_hidden_total", 10)),
    )


def _loadout_from_dict(data: Mapping[str, Any]) -> Loadout:
    return Loadout(
        hero_id=int(data["hero_id"]),
        unique_equip_id=None if data.get("unique_equip_id") is None else int(data["unique_equip_id"]),
        unique_equip_star=None if data.get("unique_equip_star") is None else int(data["unique_equip_star"]),
        normal_equip_ids=tuple(int(value) for value in data.get("normal_equip_ids", ())),
        normal_equip_features=_pairs(data.get("normal_equip_features", ())),
        level_features=_pairs(data.get("level_features", ())),
        final_stats=_pairs(data.get("final_stats", ())),
        final_power=float(data.get("final_power", 0.0)),
        standing_rank=float(data.get("standing_rank", 0.0)),
        standing_bucket=str(data.get("standing_bucket", "custom")),
    )


def _team_from_dict(data: Mapping[str, Any]) -> Team:
    return Team(tuple(_loadout_from_dict(item) for item in data["slots"]))


def _attack_plan_from_dict(data: Mapping[str, Any]) -> AttackPlan:
    return AttackPlan(
        format=_match_format_from_dict(data["format"]),
        teams=tuple(_team_from_dict(item) for item in data["teams"]),
        source=str(data.get("source", "artifact")),
        plan_id=None if data.get("plan_id") is None else str(data["plan_id"]),
        version=str(data.get("version", "v4")),
        season=str(data.get("season", "unknown")),
        rank_segment=str(data.get("rank_segment", "unknown")),
    )


def _defense_plan_from_dict(data: Mapping[str, Any]) -> DefensePlan:
    return DefensePlan(
        format=_match_format_from_dict(data["format"]),
        teams=tuple(_team_from_dict(item) for item in data["teams"]),
        mask=tuple(tuple(int(value) for value in row) for row in data["mask"]),
        source=str(data.get("source", "artifact")),
        plan_id=None if data.get("plan_id") is None else str(data["plan_id"]),
        version=str(data.get("version", "v4")),
        season=str(data.get("season", "unknown")),
        rank_segment=str(data.get("rank_segment", "unknown")),
    )


def _pairs(values: Any) -> tuple[tuple[str, float], ...]:
    return tuple((str(item[0]), float(item[1])) for item in values)
