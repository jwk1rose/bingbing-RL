from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..constraints import ConstraintEngine
from ..domain import DefensePlan, Loadout, MatchFormat, Observation, Team, observe_defense
from ..training.checkpoints import CheckpointRegistry, ModelCheckpointRecord
from ..training.model_selection import build_jsonl_split_manifest, write_split_manifest


@dataclass(frozen=True)
class BeliefRankerVocab:
    loadout_to_index: Mapping[Loadout, int]

    @classmethod
    def from_loadouts(cls, loadouts: tuple[Loadout, ...]) -> "BeliefRankerVocab":
        return cls(loadout_to_index={loadout: index + 1 for index, loadout in enumerate(loadouts)})

    @property
    def loadout_count(self) -> int:
        return max(self.loadout_to_index.values(), default=0) + 1

    def index(self, loadout: Loadout | None) -> int:
        if loadout is None:
            return 0
        return int(self.loadout_to_index.get(loadout, 0))

    def ordered_loadouts(self) -> tuple[Loadout, ...]:
        return tuple(loadout for loadout, _index in sorted(self.loadout_to_index.items(), key=lambda item: item[1]))

    def to_dict(self) -> dict[str, object]:
        return {"loadouts": [asdict(loadout) for loadout in self.ordered_loadouts()]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BeliefRankerVocab":
        rows = payload.get("loadouts", ())
        if not isinstance(rows, Sequence):
            raise ValueError("belief ranker vocab loadouts must be a sequence")
        return cls.from_loadouts(tuple(_loadout_from_dict(row) for row in rows if isinstance(row, Mapping)))


@dataclass(frozen=True)
class BeliefRankerTrainingSample:
    observation: Observation
    positive_roster: tuple[Team, ...]
    candidate_rosters: tuple[tuple[Team, ...], ...]


@dataclass(frozen=True)
class BeliefRankerTrainingHistory:
    train_losses: tuple[float, ...]


@dataclass(frozen=True)
class BeliefRankerDatasetBuildResult:
    train_jsonl: Path
    holdout_jsonl: Path
    manifest_json: Path
    total_rows: int
    train_rows: int
    holdout_rows: int


class TorchBeliefRanker(nn.Module):
    def __init__(self, *, loadout_count: int, model_dim: int = 128, feature_dim: int = 6) -> None:
        super().__init__()
        self.loadout_count = int(loadout_count)
        self.model_dim = int(model_dim)
        self.feature_dim = int(feature_dim)
        self.loadout_embedding = nn.Embedding(self.loadout_count, self.model_dim, padding_idx=0)
        self.hidden_embedding = nn.Parameter(torch.zeros(self.model_dim))
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.model_dim * 6, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )

    def forward(
        self,
        observation_token_ids: Tensor,
        observation_hidden_mask: Tensor,
        candidate_token_ids: Tensor,
        features: Tensor,
    ) -> Tensor:
        if observation_token_ids.ndim != 2:
            raise ValueError("observation_token_ids must have shape [batch, slots]")
        if candidate_token_ids.ndim != 2:
            raise ValueError("candidate_token_ids must have shape [batch, slots]")
        obs = self.loadout_embedding(observation_token_ids)
        obs = obs + observation_hidden_mask.bool().unsqueeze(-1).to(obs.dtype) * self.hidden_embedding
        cand = self.loadout_embedding(candidate_token_ids)
        obs_vec = torch.cat((obs.mean(dim=1), obs.max(dim=1).values), dim=-1)
        cand_vec = torch.cat((cand.mean(dim=1), cand.max(dim=1).values), dim=-1)
        feature_vec = self.feature_mlp(features.float())
        merged = torch.cat((obs_vec, cand_vec, feature_vec, obs_vec[:, : self.model_dim] * cand_vec[:, : self.model_dim]), dim=-1)
        return self.scorer(merged).squeeze(-1)


class TorchBeliefRankerAdapter:
    def __init__(
        self,
        model: TorchBeliefRanker,
        vocab: BeliefRankerVocab,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

    @classmethod
    def from_loadouts(
        cls,
        loadouts: tuple[Loadout, ...],
        *,
        model_dim: int = 128,
        device: torch.device | str | None = None,
    ) -> "TorchBeliefRankerAdapter":
        vocab = BeliefRankerVocab.from_loadouts(loadouts)
        model = TorchBeliefRanker(loadout_count=vocab.loadout_count, model_dim=model_dim)
        if device is not None:
            model.to(torch.device(device))
        return cls(model, vocab, device=device)

    def __call__(self, observation: Observation, roster: tuple[Team, ...], features: Mapping[str, float]) -> float:
        self.model.eval()
        with torch.no_grad():
            batch = encode_belief_ranker_batch(
                (observation,),
                (roster,),
                (features,),
                self.vocab,
                device=self.device,
            )
            score = self.model(**batch)
        return float(score.detach().cpu()[0])


def train_belief_ranker(
    model: TorchBeliefRanker,
    vocab: BeliefRankerVocab,
    samples: Sequence[BeliefRankerTrainingSample],
    *,
    epochs: int,
    lr: float,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> BeliefRankerTrainingHistory:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    rows = list(samples)
    losses: list[float] = []
    for _epoch in range(epochs):
        rng.shuffle(rows)
        total_loss = 0.0
        total_count = 0
        model.train()
        for sample in rows:
            if sample.positive_roster not in sample.candidate_rosters:
                raise ValueError("positive_roster must be included in candidate_rosters")
            features = tuple(_candidate_features(sample.observation, roster) for roster in sample.candidate_rosters)
            batch = encode_belief_ranker_batch(
                tuple(sample.observation for _roster in sample.candidate_rosters),
                sample.candidate_rosters,
                features,
                vocab,
                device=model_device,
            )
            logits = model(**batch).unsqueeze(0)
            target = torch.tensor([sample.candidate_rosters.index(sample.positive_roster)], dtype=torch.long, device=model_device)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_count += 1
        losses.append(total_loss / max(total_count, 1))
    return BeliefRankerTrainingHistory(train_losses=tuple(losses))


def evaluate_belief_ranker(
    model: TorchBeliefRanker,
    vocab: BeliefRankerVocab,
    samples: Sequence[BeliefRankerTrainingSample],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(model_device)
    rows = list(samples)
    if not rows:
        return {"samples": 0.0, "top1_accuracy": 0.0, "mean_rank": 0.0, "mrr": 0.0, "nll": 0.0}
    top1 = 0
    rank_sum = 0.0
    reciprocal_sum = 0.0
    nll_sum = 0.0
    model.eval()
    with torch.no_grad():
        for sample in rows:
            if sample.positive_roster not in sample.candidate_rosters:
                raise ValueError("positive_roster must be included in candidate_rosters")
            features = tuple(_candidate_features(sample.observation, roster) for roster in sample.candidate_rosters)
            batch = encode_belief_ranker_batch(
                tuple(sample.observation for _roster in sample.candidate_rosters),
                sample.candidate_rosters,
                features,
                vocab,
                device=model_device,
            )
            logits = model(**batch)
            target_index = sample.candidate_rosters.index(sample.positive_roster)
            target = torch.tensor([target_index], dtype=torch.long, device=model_device)
            nll = F.cross_entropy(logits.unsqueeze(0), target)
            order = torch.argsort(logits, descending=True).detach().cpu().tolist()
            rank = order.index(target_index) + 1
            top1 += 1 if rank == 1 else 0
            rank_sum += float(rank)
            reciprocal_sum += 1.0 / float(rank)
            nll_sum += float(nll.detach().cpu())
    count = float(len(rows))
    return {
        "samples": count,
        "top1_accuracy": top1 / count,
        "mean_rank": rank_sum / count,
        "mrr": reciprocal_sum / count,
        "nll": nll_sum / count,
    }


def load_belief_ranker_samples_jsonl(
    path: str | Path,
    *,
    loadout_pool: tuple[Loadout, ...],
    constraint_engine: ConstraintEngine,
    negative_candidates: int = 31,
    max_completions: int = 128,
) -> tuple[BeliefRankerTrainingSample, ...]:
    canonical_by_loadout = {loadout: loadout for loadout in loadout_pool}
    samples: list[BeliefRankerTrainingSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"belief ranker JSONL row {line_number} must be an object")
            defense_payload = row.get("defense_plan")
            if not isinstance(defense_payload, Mapping):
                raise ValueError(f"belief ranker JSONL row {line_number} must contain defense_plan")
            defense = _canonicalize_defense_plan(_defense_plan_from_dict(defense_payload), canonical_by_loadout)
            if not constraint_engine.is_legal_defense(defense):
                raise ValueError(f"belief ranker JSONL row {line_number} contains illegal defense_plan")
            observation = observe_defense(defense)
            candidate_rosters = _candidate_rosters_for_training(
                observation,
                defense.teams,
                constraint_engine=constraint_engine,
                negative_candidates=negative_candidates,
                max_completions=max_completions,
            )
            samples.append(
                BeliefRankerTrainingSample(
                    observation=observation,
                    positive_roster=defense.teams,
                    candidate_rosters=candidate_rosters,
                )
            )
    return tuple(samples)


def build_belief_ranker_dataset_from_rounds(
    round_dirs: Sequence[str | Path],
    *,
    out_dir: str | Path,
    holdout_fraction: float = 0.1,
    seed: int = 0,
    dataset_id: str = "belief-ranker-rounds",
) -> BeliefRankerDatasetBuildResult:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for round_dir in round_dirs:
        round_path = Path(round_dir)
        path = round_path / "scored_defenses.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                defense_payload = row.get("defense_plan")
                if not isinstance(defense_payload, Mapping):
                    raise ValueError(f"{path}:{line_no}: missing defense_plan")
                defense = _defense_plan_from_dict(defense_payload)
                key = defense.hash()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "round_id": str(row.get("round_id", round_path.name)),
                        "defense_id": row.get("defense_id"),
                        "source_path": str(path),
                        "defense_hash": key,
                        "defense_plan": defense_payload,
                    }
                )
    rng = random.Random(seed)
    rng.shuffle(rows)
    holdout_count = int(round(len(rows) * holdout_fraction))
    if holdout_fraction > 0.0 and rows and holdout_count == 0:
        holdout_count = 1
    holdout_rows = rows[:holdout_count]
    train_rows = rows[holdout_count:]
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "belief_ranker_train.jsonl"
    holdout_path = output_dir / "belief_ranker_holdout.jsonl"
    _write_rows_jsonl(train_path, train_rows)
    _write_rows_jsonl(holdout_path, holdout_rows)
    manifest = build_jsonl_split_manifest(
        {"train": (train_path,), "holdout": (holdout_path,)},
        dataset_id=dataset_id,
        version="v4",
        metadata={"round_dirs": [str(Path(item)) for item in round_dirs]},
    )
    manifest_path = output_dir / "split_manifest.json"
    write_split_manifest(manifest_path, manifest)
    return BeliefRankerDatasetBuildResult(
        train_jsonl=train_path,
        holdout_jsonl=holdout_path,
        manifest_json=manifest_path,
        total_rows=len(rows),
        train_rows=len(train_rows),
        holdout_rows=len(holdout_rows),
    )


def save_belief_ranker_checkpoint(
    path: str | Path,
    model: TorchBeliefRanker,
    vocab: BeliefRankerVocab,
    history: BeliefRankerTrainingHistory,
    *,
    metrics: Mapping[str, float] | None = None,
    registry_path: str | Path | None = None,
    checkpoint_id: str | None = None,
    dataset_hash: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
) -> ModelCheckpointRecord:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "train_loss": float(history.train_losses[-1]) if history.train_losses else 0.0,
        "epochs": float(len(history.train_losses)),
    }
    metrics_payload.update({str(key): float(value) for key, value in (metrics or {}).items()})
    torch.save(
        {
            "model_type": "belief_ranker",
            "model_dim": model.model_dim,
            "feature_dim": model.feature_dim,
            "vocab": vocab.to_dict(),
            "state_dict": model.state_dict(),
            "train_losses": list(history.train_losses),
            "metrics": metrics_payload,
            "metadata": dict(metadata or {}),
        },
        output_path,
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = ModelCheckpointRecord(
        checkpoint_id=checkpoint_id or output_path.stem,
        model_type="belief_ranker",
        model_path=str(output_path),
        metrics_path=str(metrics_path),
        created_at=time.time(),
        dataset_hash=dataset_hash,
        metrics=metrics_payload,
    )
    if registry_path is not None:
        CheckpointRegistry(registry_path).add(record)
    return record


def load_belief_ranker_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> TorchBeliefRankerAdapter:
    payload = torch.load(Path(path), map_location=device or "cpu")
    vocab = BeliefRankerVocab.from_dict(payload["vocab"])
    model = TorchBeliefRanker(
        loadout_count=vocab.loadout_count,
        model_dim=int(payload.get("model_dim", 128)),
        feature_dim=int(payload.get("feature_dim", 6)),
    )
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model.to(torch.device(device))
    model.eval()
    return TorchBeliefRankerAdapter(model, vocab, device=device)


def encode_belief_ranker_batch(
    observations: tuple[Observation, ...],
    rosters: tuple[tuple[Team, ...], ...],
    feature_rows: tuple[Mapping[str, float], ...],
    vocab: BeliefRankerVocab,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    if not (len(observations) == len(rosters) == len(feature_rows)):
        raise ValueError("observations, rosters, and feature_rows must have the same length")
    observation_token_ids = []
    observation_hidden_mask = []
    candidate_token_ids = []
    features = []
    for observation, roster, feature_row in zip(observations, rosters, feature_rows):
        obs_tokens: list[int] = []
        hidden_mask: list[bool] = []
        for row in observation.slots:
            for slot in row:
                hidden_mask.append(slot.is_hidden)
                obs_tokens.append(0 if slot.is_hidden else vocab.index(slot.loadout))
        observation_token_ids.append(obs_tokens)
        observation_hidden_mask.append(hidden_mask)
        candidate_token_ids.append([vocab.index(loadout) for team in roster for loadout in team.slots])
        features.append(_feature_vector(feature_row))
    return {
        "observation_token_ids": torch.tensor(observation_token_ids, dtype=torch.long, device=device),
        "observation_hidden_mask": torch.tensor(observation_hidden_mask, dtype=torch.bool, device=device),
        "candidate_token_ids": torch.tensor(candidate_token_ids, dtype=torch.long, device=device),
        "features": torch.tensor(features, dtype=torch.float32, device=device),
    }


def _candidate_features(observation: Observation, roster: tuple[Team, ...]) -> Mapping[str, float]:
    return {
        "roster_strength": sum(team.total_power for team in roster),
        "real_frequency": 0.0,
        "pool_frequency": 0.0,
        "recency": 0.0,
        "compatible_visible_ratio": _compatible_visible_ratio(observation, roster),
        "hidden_slot_count": float(len(observation.hidden_slots)),
    }


def _feature_vector(features: Mapping[str, float]) -> tuple[float, ...]:
    roster_strength = float(features.get("roster_strength", 0.0))
    hidden_slot_count = float(features.get("hidden_slot_count", 0.0))
    if roster_strength > 100.0:
        roster_strength /= 100000.0
    if hidden_slot_count > 1.0:
        hidden_slot_count /= 15.0
    return (
        roster_strength,
        float(features.get("real_frequency", 0.0)),
        float(features.get("pool_frequency", 0.0)),
        float(features.get("recency", 0.0)),
        float(features.get("compatible_visible_ratio", 0.0)),
        hidden_slot_count,
    )


def _compatible_visible_ratio(observation: Observation, roster: tuple[Team, ...]) -> float:
    visible_count = 0
    matched = 0
    for team_idx, row in enumerate(observation.slots, start=1):
        for slot_idx, slot in enumerate(row, start=1):
            if slot.is_hidden:
                continue
            visible_count += 1
            if roster[team_idx - 1].slots[slot_idx - 1].hero_id == slot.hero_id:
                matched += 1
    if visible_count == 0:
        return 1.0
    return matched / visible_count


def _candidate_rosters_for_training(
    observation: Observation,
    positive_roster: tuple[Team, ...],
    *,
    constraint_engine: ConstraintEngine,
    negative_candidates: int,
    max_completions: int,
) -> tuple[tuple[Team, ...], ...]:
    rosters: list[tuple[Team, ...]] = [positive_roster]
    seen = {_roster_key(positive_roster)}
    for roster in constraint_engine.enumerate_completions(observation, max_k=max_completions):
        key = _roster_key(roster)
        if key in seen:
            continue
        rosters.append(roster)
        seen.add(key)
        if len(rosters) >= max(1, negative_candidates + 1):
            break
    return tuple(rosters)


def _canonicalize_defense_plan(defense: DefensePlan, canonical_by_loadout: Mapping[Loadout, Loadout]) -> DefensePlan:
    teams = tuple(Team(tuple(canonical_by_loadout.get(loadout, loadout) for loadout in team.slots)) for team in defense.teams)
    return DefensePlan(
        format=defense.format,
        teams=teams,
        mask=defense.mask,
        source=defense.source,
        plan_id=defense.plan_id,
        version=defense.version,
        season=defense.season,
        rank_segment=defense.rank_segment,
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


def _match_format_from_dict(data: Mapping[str, Any]) -> MatchFormat:
    return MatchFormat(
        n_teams=int(data["n_teams"]),
        team_size=int(data.get("team_size", 5)),
        win_required=None if data.get("win_required") is None else int(data["win_required"]),
        max_hidden_per_team=int(data.get("max_hidden_per_team", 2)),
        max_hidden_total=int(data.get("max_hidden_total", 10)),
    )


def _team_from_dict(data: Mapping[str, Any]) -> Team:
    return Team(tuple(_loadout_from_dict(item) for item in data["slots"]))


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


def _pairs(values: Any) -> tuple[tuple[str, float], ...]:
    return tuple((str(item[0]), float(item[1])) for item in values)


def _roster_key(roster: tuple[Team, ...]) -> str:
    return "|".join(team.hash() for team in roster)


def _write_rows_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
