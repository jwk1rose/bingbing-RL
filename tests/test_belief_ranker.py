from __future__ import annotations

from dataclasses import asdict
import json
import math
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from masked_team_league.training.checkpoints import CheckpointRegistry
from masked_team_league.belief import BeliefEngine
from masked_team_league.belief.ranker import (
    BeliefRankerTrainingSample,
    TorchBeliefRanker,
    TorchBeliefRankerAdapter,
    build_belief_ranker_dataset_from_rounds,
    evaluate_belief_ranker,
    load_belief_ranker_checkpoint,
    load_belief_ranker_samples_jsonl,
    save_belief_ranker_checkpoint,
    train_belief_ranker,
)
from masked_team_league.constraints import ConstraintEngine
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.domain import DefensePlan, Team, observe_defense


def test_torch_belief_ranker_adapter_scores_observation_candidate(loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=201).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    observation = observe_defense(defense)
    adapter = TorchBeliefRankerAdapter.from_loadouts(loadouts, model_dim=32)

    score = adapter(observation, defense.teams, {"roster_strength": 1.0})

    assert math.isfinite(score)


def test_train_belief_ranker_returns_finite_loss_and_belief_engine_accepts_adapter(loadouts, fmt3):
    positive = LegalPlanGenerator(loadouts, seed=202).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    observation = observe_defense(positive)
    negative_teams = (Team(loadouts[20:25]), Team(loadouts[25:30]), Team(loadouts[30:35]))
    negative = DefensePlan(fmt3, negative_teams, positive.mask, "negative")
    adapter = TorchBeliefRankerAdapter.from_loadouts(loadouts, model_dim=32)
    sample = BeliefRankerTrainingSample(
        observation=observation,
        positive_roster=positive.teams,
        candidate_rosters=(positive.teams, negative.teams),
    )

    history = train_belief_ranker(adapter.model, adapter.vocab, (sample,), epochs=1, lr=1e-3)
    belief = BeliefEngine(ConstraintEngine(loadouts), ranker=adapter, ranker_weight=1.0).build(observation, max_k=8)

    assert len(history.train_losses) == 1
    assert math.isfinite(history.train_losses[0])
    assert belief.candidates


def test_load_belief_ranker_samples_jsonl_from_defense_artifact(tmp_path, loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=203).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    path = tmp_path / "scored_defenses.jsonl"
    path.write_text(json.dumps({"defense_plan": asdict(defense)}, separators=(",", ":")) + "\n", encoding="utf-8")

    samples = load_belief_ranker_samples_jsonl(
        path,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        negative_candidates=3,
        max_completions=16,
    )

    assert len(samples) == 1
    assert samples[0].observation.hash() == observe_defense(defense).hash()
    assert samples[0].positive_roster == defense.teams
    assert samples[0].positive_roster in samples[0].candidate_rosters
    assert len(samples[0].candidate_rosters) >= 2


def test_build_belief_ranker_dataset_from_round_artifacts_writes_splits_and_manifest(tmp_path, loadouts, fmt3):
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    round_a.mkdir()
    round_b.mkdir()
    defense_a = LegalPlanGenerator(loadouts, seed=205).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    defense_b = LegalPlanGenerator(loadouts, seed=206).generate_defense_plan(
        fmt3,
        mask=((0, 1, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    for round_dir, defense in ((round_a, defense_a), (round_b, defense_b)):
        (round_dir / "scored_defenses.jsonl").write_text(
            json.dumps({"round_id": round_dir.name, "defense_id": "def-1", "defense_plan": asdict(defense)}, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )

    dataset = build_belief_ranker_dataset_from_rounds(
        (round_a, round_b),
        out_dir=tmp_path / "belief_dataset",
        holdout_fraction=0.5,
        seed=7,
    )
    train_samples = load_belief_ranker_samples_jsonl(
        dataset.train_jsonl,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        negative_candidates=2,
        max_completions=8,
    )
    holdout_samples = load_belief_ranker_samples_jsonl(
        dataset.holdout_jsonl,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        negative_candidates=2,
        max_completions=8,
    )
    manifest = json.loads(dataset.manifest_json.read_text(encoding="utf-8"))

    assert dataset.total_rows == 2
    assert dataset.train_rows == 1
    assert dataset.holdout_rows == 1
    assert len(train_samples) == 1
    assert len(holdout_samples) == 1
    assert manifest["dataset_id"] == "belief-ranker-rounds"
    assert manifest["split_counts"] == {"holdout": 1, "train": 1}


def test_belief_ranker_evaluation_and_checkpoint_round_trip(tmp_path, loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=204).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    negative_teams = (Team(loadouts[20:25]), Team(loadouts[25:30]), Team(loadouts[30:35]))
    sample = BeliefRankerTrainingSample(
        observation=observe_defense(defense),
        positive_roster=defense.teams,
        candidate_rosters=(defense.teams, negative_teams),
    )
    adapter = TorchBeliefRankerAdapter.from_loadouts(loadouts, model_dim=32)
    history = train_belief_ranker(adapter.model, adapter.vocab, (sample,), epochs=1, lr=1e-3)

    metrics = evaluate_belief_ranker(adapter.model, adapter.vocab, (sample,))
    record = save_belief_ranker_checkpoint(
        tmp_path / "belief_ranker.pt",
        adapter.model,
        adapter.vocab,
        history,
        metrics=metrics,
        registry_path=tmp_path / "registry.json",
        checkpoint_id="belief-ranker-unit",
        dataset_hash="belief-dataset",
    )
    loaded = load_belief_ranker_checkpoint(tmp_path / "belief_ranker.pt")

    assert metrics["samples"] == 1
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert (tmp_path / "belief_ranker.metrics.json").exists()
    assert CheckpointRegistry(tmp_path / "registry.json").latest("belief_ranker") == record
    assert math.isfinite(loaded(sample.observation, sample.positive_roster, {"roster_strength": 1.0}))


def test_train_belief_ranker_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.train_belief_ranker", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--samples-jsonl" in result.stdout
    assert "--heroes-json" in result.stdout
    assert "--out-checkpoint" in result.stdout
    assert "--registry" in result.stdout


def test_build_belief_ranker_dataset_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.build_belief_ranker_dataset", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--out-dir" in result.stdout
    assert "--holdout-fraction" in result.stdout
