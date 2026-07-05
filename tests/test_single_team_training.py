from __future__ import annotations

import math
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from masked_team_league.domain import Team
from masked_team_league.training.single_team_model import LoadoutVocab, SingleTeamWinrateModel, SingleTeamWinrateModelConfig
from masked_team_league.training import (
    SingleTeamMatchupSample,
    build_holdout_calibration_report,
    evaluate_single_team_model,
    fit_single_team_calibrator,
    load_single_team_matchup_samples_jsonl,
    train_single_team_winrate_model,
)


def _samples(loadouts):
    return (
        SingleTeamMatchupSample(Team(loadouts[:5]), Team(loadouts[5:10]), wins=3, games=3, mean_margin=10.0, mean_duration=60.0),
        SingleTeamMatchupSample(Team(loadouts[5:10]), Team(loadouts[:5]), wins=0, games=3, mean_margin=-10.0, mean_duration=60.0),
        SingleTeamMatchupSample(Team(loadouts[10:15]), Team(loadouts[15:20]), wins=2, games=3, mean_margin=3.0, mean_duration=70.0),
        SingleTeamMatchupSample(Team(loadouts[15:20]), Team(loadouts[10:15]), wins=1, games=3, mean_margin=-3.0, mean_duration=70.0),
    )


def test_evaluate_single_team_model_returns_required_metrics(loadouts):
    samples = _samples(loadouts)
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    model = SingleTeamWinrateModel(vocab, config)

    metrics = evaluate_single_team_model(model, vocab, samples)

    assert {"auc", "brier", "ece", "precision_at_1", "recall_at_1", "samples"} <= set(metrics)
    assert metrics["samples"] == 4


def test_train_single_team_winrate_model_returns_finite_loss_history(loadouts):
    samples = _samples(loadouts)
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    model = SingleTeamWinrateModel(vocab, config)

    history = train_single_team_winrate_model(model, vocab, samples, epochs=2, batch_size=2, lr=1e-3)

    assert len(history.train_losses) == 2
    assert all(math.isfinite(value) for value in history.train_losses)


def test_fit_single_team_calibrator_adds_calibrated_metrics(loadouts):
    samples = _samples(loadouts)
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    model = SingleTeamWinrateModel(vocab, config)

    calibrator = fit_single_team_calibrator(model, vocab, samples)
    metrics = evaluate_single_team_model(model, vocab, samples, calibrator=calibrator)

    assert "calibrated_brier" in metrics
    assert "calibrated_ece" in metrics
    assert metrics["calibrated_brier"] <= metrics["brier"] + 1e-9


def test_load_single_team_matchup_samples_jsonl_streams_rows(tmp_path, loadouts):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"attack_hero_ids":[1,2,3,4,5],"defense_hero_ids":[6,7,8,9,10],"wins":2,"games":3,"margin":4.5,"duration":90}\n'
        '{"attack_team":[11,12,13,14,15],"defense_team":[16,17,18,19,20],"label":0.25,"m":4}\n',
        encoding="utf-8",
    )
    by_hero_id = {loadout.hero_id: loadout for loadout in loadouts}

    samples = load_single_team_matchup_samples_jsonl(path, by_hero_id)

    assert len(samples) == 2
    assert samples[0].wins == 2
    assert samples[0].games == 3
    assert samples[0].mean_margin == 4.5
    assert samples[1].wins == 1
    assert samples[1].games == 4


def test_build_holdout_calibration_report_summarizes_improvement():
    report = build_holdout_calibration_report(
        {
            "samples": 20.0,
            "brier": 0.30,
            "ece": 0.18,
            "calibrated_brier": 0.22,
            "calibrated_ece": 0.11,
        }
    )

    assert report.samples == 20
    assert report.brier_delta == 0.08
    assert report.ece_delta == 0.07
    assert report.improved_brier
    assert report.to_json_dict()["calibrated_ece"] == 0.11


def test_train_single_team_model_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.train_single_team_model", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--samples-jsonl" in result.stdout
    assert "--heroes-json" in result.stdout
    assert "--calibrate" in result.stdout
    assert "--holdout-jsonl" in result.stdout
    assert "--registry" in result.stdout
