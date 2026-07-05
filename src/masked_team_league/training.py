from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .metrics import binary_auc, brier_score, expected_calibration_error, precision_at_k, recall_at_k
from .models import Loadout, Team
from .real_calibration import RealCalibrationModel
from .single_team_model import LoadoutVocab, SingleTeamWinrateModel, binomial_nll, encode_team_batch


@dataclass(frozen=True)
class SingleTeamMatchupSample:
    attack: Team
    defense: Team
    wins: int
    games: int
    mean_margin: float = 0.0
    mean_duration: float = 0.0

    @property
    def label(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.wins / self.games


@dataclass(frozen=True)
class TrainingHistory:
    train_losses: tuple[float, ...]


@dataclass(frozen=True)
class HoldoutCalibrationReport:
    samples: int
    raw_brier: float
    calibrated_brier: float
    brier_delta: float
    raw_ece: float
    calibrated_ece: float
    ece_delta: float
    improved_brier: bool
    improved_ece: bool

    def to_json_dict(self) -> dict[str, float | int | bool]:
        return {
            "samples": self.samples,
            "raw_brier": self.raw_brier,
            "calibrated_brier": self.calibrated_brier,
            "brier_delta": self.brier_delta,
            "raw_ece": self.raw_ece,
            "calibrated_ece": self.calibrated_ece,
            "ece_delta": self.ece_delta,
            "improved_brier": self.improved_brier,
            "improved_ece": self.improved_ece,
        }


def load_single_team_matchup_samples_jsonl(
    path: str | Path,
    by_hero_id: Mapping[int, Loadout],
) -> tuple[SingleTeamMatchupSample, ...]:
    samples: list[SingleTeamMatchupSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            attack_ids = row.get("attack_hero_ids") or row.get("attack_team")
            defense_ids = row.get("defense_hero_ids") or row.get("defense_team")
            if attack_ids is None or defense_ids is None:
                raise ValueError(f"line {line_no}: missing attack_hero_ids/defense_hero_ids")
            attack = Team(tuple(by_hero_id[int(hero_id)] for hero_id in attack_ids))
            defense = Team(tuple(by_hero_id[int(hero_id)] for hero_id in defense_ids))
            games = int(row.get("games") or row.get("m") or 1)
            wins = int(row.get("wins") if "wins" in row else round(float(row.get("win_rate", row.get("label", 0.0))) * games))
            samples.append(
                SingleTeamMatchupSample(
                    attack=attack,
                    defense=defense,
                    wins=wins,
                    games=games,
                    mean_margin=float(row.get("mean_margin", row.get("margin", 0.0))),
                    mean_duration=float(row.get("mean_duration", row.get("duration", 0.0))),
                )
            )
    return tuple(samples)


def build_holdout_calibration_report(metrics: Mapping[str, float]) -> HoldoutCalibrationReport:
    raw_brier = float(metrics.get("brier", 0.0))
    raw_ece = float(metrics.get("ece", 0.0))
    calibrated_brier = float(metrics.get("calibrated_brier", raw_brier))
    calibrated_ece = float(metrics.get("calibrated_ece", raw_ece))
    brier_delta = round(raw_brier - calibrated_brier, 12)
    ece_delta = round(raw_ece - calibrated_ece, 12)
    return HoldoutCalibrationReport(
        samples=int(metrics.get("samples", 0.0)),
        raw_brier=raw_brier,
        calibrated_brier=calibrated_brier,
        brier_delta=brier_delta,
        raw_ece=raw_ece,
        calibrated_ece=calibrated_ece,
        ece_delta=ece_delta,
        improved_brier=brier_delta > 0.0,
        improved_ece=ece_delta > 0.0,
    )


def evaluate_single_team_model(
    model: SingleTeamWinrateModel,
    vocab: LoadoutVocab,
    samples: Sequence[SingleTeamMatchupSample],
    *,
    calibrator: RealCalibrationModel | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    if not samples:
        return {"samples": 0.0, "auc": 0.0, "brier": 0.0, "ece": 0.0, "precision_at_1": 0.0, "recall_at_1": 0.0}
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    attacks = tuple(sample.attack for sample in samples)
    defenses = tuple(sample.defense for sample in samples)
    labels = tuple(sample.label for sample in samples)
    scores = _predict_scores(model, vocab, samples, device=model_device)
    binary_labels = tuple(1.0 if label >= 0.5 else 0.0 for label in labels)
    metrics = {
        "samples": float(len(samples)),
        "auc": binary_auc(binary_labels, scores),
        "brier": brier_score(labels, scores),
        "ece": expected_calibration_error(binary_labels, scores),
        "precision_at_1": precision_at_k(binary_labels, scores, k=1),
        "recall_at_1": recall_at_k(binary_labels, scores, k=1),
    }
    if calibrator is not None:
        calibrated = tuple(calibrator.calibrate(score) for score in scores)
        metrics.update(
            {
                "calibrated_auc": binary_auc(binary_labels, calibrated),
                "calibrated_brier": brier_score(labels, calibrated),
                "calibrated_ece": expected_calibration_error(binary_labels, calibrated),
                "calibrated_precision_at_1": precision_at_k(binary_labels, calibrated, k=1),
                "calibrated_recall_at_1": recall_at_k(binary_labels, calibrated, k=1),
            }
        )
    return metrics


def fit_single_team_calibrator(
    model: SingleTeamWinrateModel,
    vocab: LoadoutVocab,
    samples: Sequence[SingleTeamMatchupSample],
    *,
    device: torch.device | str | None = None,
) -> RealCalibrationModel:
    if not samples:
        return RealCalibrationModel()
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    scores = _predict_scores(model, vocab, samples, device=model_device)
    labels = tuple(sample.label for sample in samples)
    return RealCalibrationModel.fit_platt(scores, labels)


def train_single_team_winrate_model(
    model: SingleTeamWinrateModel,
    vocab: LoadoutVocab,
    samples: Sequence[SingleTeamMatchupSample],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> TrainingHistory:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not samples:
        return TrainingHistory(train_losses=tuple(0.0 for _ in range(epochs)))
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    losses: list[float] = []
    rows = list(samples)
    for _epoch in range(epochs):
        rng.shuffle(rows)
        model.train()
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(rows), batch_size):
            batch_samples = rows[start : start + batch_size]
            loss = _batch_loss(model, vocab, batch_samples, model_device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_samples)
            total_count += len(batch_samples)
        losses.append(total_loss / max(total_count, 1))
    return TrainingHistory(train_losses=tuple(losses))


def _batch_loss(
    model: SingleTeamWinrateModel,
    vocab: LoadoutVocab,
    samples: Sequence[SingleTeamMatchupSample],
    device: torch.device,
) -> Tensor:
    attacks = tuple(sample.attack for sample in samples)
    defenses = tuple(sample.defense for sample in samples)
    batch = encode_team_batch(attacks, defenses, vocab, device=device)
    output = model(batch)
    wins = torch.tensor([sample.wins for sample in samples], dtype=torch.float32, device=device)
    games = torch.tensor([sample.games for sample in samples], dtype=torch.float32, device=device)
    margin = torch.tensor([sample.mean_margin for sample in samples], dtype=torch.float32, device=device)
    duration = torch.tensor([sample.mean_duration for sample in samples], dtype=torch.float32, device=device)
    win_loss = binomial_nll(output["win_prob"], wins, games)
    margin_loss = F.mse_loss(output["margin"], margin)
    duration_loss = F.mse_loss(output["duration"], duration)
    return win_loss + 0.01 * margin_loss + 0.001 * duration_loss


def _predict_scores(
    model: SingleTeamWinrateModel,
    vocab: LoadoutVocab,
    samples: Sequence[SingleTeamMatchupSample],
    *,
    device: torch.device,
) -> tuple[float, ...]:
    model.eval()
    attacks = tuple(sample.attack for sample in samples)
    defenses = tuple(sample.defense for sample in samples)
    with torch.no_grad():
        batch = encode_team_batch(attacks, defenses, vocab, device=device)
        output = model(batch)
    return tuple(float(value) for value in output["win_prob"].detach().cpu())
