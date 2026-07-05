from __future__ import annotations

from dataclasses import asdict
import math
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from masked_team_league.constraints import ConstraintEngine
from masked_team_league.training.checkpoints import CheckpointRegistry
from masked_team_league.generation import GenerationGoal, LegalPlanGenerator
from masked_team_league.domain import observe_defense
from masked_team_league.belief import BeliefEngine
from masked_team_league.generation.proposal_networks import (
    AttackGenerationNetwork,
    DefenseRosterGenerationNetwork,
    GenerationContextEncoder,
    MaskSelectionNetwork,
    ProposalNetworkConfig,
)
from masked_team_league.generation.proposal_training import (
    MASK_SLOT_FEATURE_NAMES,
    attack_legal_action_mask_fn,
    defense_legal_action_mask_fn,
    build_mask_training_sample,
    build_defense_teacher_sample,
    build_attack_proposal_context_tensors,
    build_defense_proposal_context_tensors,
    build_attack_teacher_sample,
    generate_defense_roster_candidates,
    generate_attack_plan_candidates,
    load_defense_proposal_candidate_source,
    load_defense_teacher_samples_jsonl,
    load_attack_proposal_candidate_source,
    load_attack_teacher_samples_jsonl,
    load_mask_slot_score_provider,
    load_mask_training_samples_jsonl,
    mask_selection_ranking_loss,
    proposal_sequence_to_defense_plan,
    proposal_sequence_to_attack_plan,
    save_mask_selection_checkpoint,
    save_proposal_network_checkpoint,
    train_mask_selection_network,
    train_proposal_network,
)


def test_build_attack_teacher_sample_records_autoregressive_targets_and_masks(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=101).generate_attack_plan(fmt3)
    sample = build_attack_teacher_sample(
        plan,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.75,
        gap_target=0.10,
        weight=2.0,
        source="attack_oracle",
    )

    assert len(sample.target_token_ids) == fmt3.n_teams * fmt3.team_size
    assert len(sample.legal_action_masks) == len(sample.target_token_ids)
    assert sample.selected_prefix(0) == (0,)
    assert sample.selected_prefix(2)[1:] == tuple(token + 1 for token in sample.target_token_ids[:2])
    assert all(mask[target] for mask, target in zip(sample.legal_action_masks, sample.target_token_ids))
    assert sample.value_target == 0.75
    assert sample.gap_target == 0.10
    assert sample.weight == 2.0
    assert sample.source == "attack_oracle"


def test_train_proposal_network_runs_optimizer_loop(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=102).generate_attack_plan(fmt3)
    sample = build_attack_teacher_sample(
        plan,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.8,
        gap_target=0.2,
        weight=1.0,
        source="attack_oracle",
    )
    config = ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=15)
    network = AttackGenerationNetwork(config)

    history = train_proposal_network(network, (sample,), epochs=1, lr=1e-3)

    assert len(history.train_losses) == 1
    assert math.isfinite(history.train_losses[0])


def test_load_attack_teacher_samples_jsonl_replays_attack_plan_artifacts(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=103).generate_attack_plan(fmt3, source="attack_oracle:main")
    path = tmp_path / "teacher.jsonl"
    path.write_text(
        __import__("json").dumps(
            {
                "attack_plan": asdict(plan),
                "value_target": 0.9,
                "gap_target": 0.25,
                "weight": 3.0,
                "source": "attack_oracle:main",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_attack_teacher_samples_jsonl(path, loadout_pool=loadouts, constraint_engine=ConstraintEngine(loadouts))

    assert len(samples) == 1
    assert samples[0].value_target == 0.9
    assert samples[0].gap_target == 0.25
    assert samples[0].weight == 3.0
    assert samples[0].source == "attack_oracle:main"
    assert len(samples[0].target_token_ids) == fmt3.n_teams * fmt3.team_size


def test_load_defense_teacher_samples_jsonl_replays_scored_defense_artifacts(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=203).generate_defense_plan(
        fmt3,
        source="defense_oracle:main",
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    path = tmp_path / "defense_teacher.jsonl"
    path.write_text(
        __import__("json").dumps(
            {
                "defense_plan": asdict(plan),
                "strength": 0.85,
                "break_rate": 0.15,
                "ambiguity_score": 1.25,
                "source": "defense_oracle:main",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_defense_teacher_samples_jsonl(path, loadout_pool=loadouts, constraint_engine=ConstraintEngine(loadouts))

    assert len(samples) == 1
    assert samples[0].value_target == 0.85
    assert samples[0].gap_target == 1.25
    assert samples[0].source == "defense_oracle:main"
    assert len(samples[0].target_token_ids) == fmt3.n_teams * fmt3.team_size


def test_load_defense_teacher_samples_jsonl_uses_anti_meta_residual_as_gap_target(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=204).generate_defense_plan(
        fmt3,
        source="defense_oracle:exploiter",
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    path = tmp_path / "defense_teacher.jsonl"
    path.write_text(
        __import__("json").dumps(
            {
                "defense_plan": asdict(plan),
                "value_target": 0.7,
                "ambiguity_score": 2.0,
                "anti_meta_residual_target": 0.35,
                "source": "defense_oracle:exploiter",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_defense_teacher_samples_jsonl(path, loadout_pool=loadouts, constraint_engine=ConstraintEngine(loadouts))

    assert len(samples) == 1
    assert samples[0].value_target == 0.7
    assert samples[0].gap_target == 2.0
    assert samples[0].anti_meta_residual_target == 0.35
    assert samples[0].source == "defense_oracle:exploiter"


def test_build_mask_training_sample_records_slot_features_and_targets(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=214).generate_defense_plan(
        fmt3,
        source="defense_oracle:main",
        mask=((1, 0, 0, 0, 1), (0, 1, 0, 0, 0), (0, 0, 0, 1, 0)),
    )

    sample = build_mask_training_sample(
        plan,
        ambiguity_score=2.5,
        estimated_break_rate=0.35,
        meta_attack_success=0.60,
        learned_mask_score=4.0,
        counter_sensitivity=0.40,
        weight=2.0,
        source="defense_oracle:main",
    )

    counter_index = MASK_SLOT_FEATURE_NAMES.index("counter_sensitivity")

    assert len(sample.slot_features) == fmt3.n_teams * fmt3.team_size
    assert len(sample.slot_features[0]) == len(MASK_SLOT_FEATURE_NAMES)
    assert sum(sample.target_mask) == 4.0
    assert sample.weight == 2.0
    assert sample.source == "defense_oracle:main"
    assert all(features[counter_index] == 0.40 for features in sample.slot_features)


def test_load_mask_training_samples_jsonl_reads_scored_defense_artifacts(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=215).generate_defense_plan(
        fmt3,
        source="defense_oracle:exploiter",
        mask=((1, 0, 0, 0, 1), (0, 1, 0, 0, 0), (0, 0, 0, 1, 0)),
    )
    path = tmp_path / "scored_defenses.jsonl"
    path.write_text(
        __import__("json").dumps(
            {
                "defense_plan": asdict(plan),
                "ambiguity_score": 1.5,
                "strength": 0.8,
                "defense_risk_report": {
                    "estimated_break_rate": 0.2,
                    "meta_attack_success": 0.55,
                    "learned_mask_score": 3.0,
                    "backup_break_rates": [0.2, 0.35, 0.1],
                    "counter_attack_risk_report": {"expected_match_win": 0.7, "worst_case_match_win": 0.4},
                },
                "source": "defense_oracle:exploiter",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_mask_training_samples_jsonl(path)

    counter_index = MASK_SLOT_FEATURE_NAMES.index("counter_sensitivity")
    assert len(samples) == 1
    assert samples[0].target_mask.count(1.0) == 4
    assert samples[0].slot_features[0][counter_index] > 0.0
    assert samples[0].weight > 1.0


def test_mask_selection_ranking_loss_prefers_hidden_slots():
    scores = torch.tensor([[0.0, 3.0, -1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    losses = mask_selection_ranking_loss(scores, target, ranking_weight=2.0)

    assert losses.bce_loss.item() > 0.0
    assert losses.ranking_loss.item() > 0.0
    assert torch.isclose(losses.total_loss, losses.bce_loss + 2.0 * losses.ranking_loss)
    losses.total_loss.backward()
    assert scores.grad is not None


def test_train_mask_selection_network_runs_optimizer_loop(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=216).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 1, 0, 0, 0), (0, 0, 0, 1, 0)),
    )
    sample = build_mask_training_sample(plan, ambiguity_score=1.0, estimated_break_rate=0.25)
    network = MaskSelectionNetwork(feature_dim=len(MASK_SLOT_FEATURE_NAMES), hidden_dim=16)

    history = train_mask_selection_network(network, (sample,), epochs=1, lr=1e-3)

    assert len(history.train_losses) == 1
    assert math.isfinite(history.train_losses[0])


def test_mask_selection_checkpoint_loads_slot_score_provider(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=217).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 1, 0, 0, 0), (0, 0, 0, 1, 0)),
    )
    sample = build_mask_training_sample(plan, ambiguity_score=1.0, estimated_break_rate=0.25)
    network = MaskSelectionNetwork(feature_dim=len(MASK_SLOT_FEATURE_NAMES), hidden_dim=16)
    history = train_mask_selection_network(network, (sample,), epochs=1, lr=1e-3)
    checkpoint = tmp_path / "mask_selection.pt"

    save_mask_selection_checkpoint(checkpoint, network, history)
    provider = load_mask_slot_score_provider(checkpoint)
    scores = provider(plan.teams, fmt3)

    assert len(scores) == fmt3.n_teams
    assert all(len(row) == fmt3.team_size for row in scores)
    assert all(isinstance(value, float) for row in scores for value in row)


def test_load_attack_teacher_samples_jsonl_weights_ranked_candidate_groups(tmp_path, loadouts, fmt3):
    strong = LegalPlanGenerator(loadouts, seed=113).generate_attack_plan(fmt3, source="attack_oracle:main")
    weak = LegalPlanGenerator(loadouts, seed=114).generate_attack_plan(fmt3, source="attack_oracle:main")
    path = tmp_path / "ranked_teacher.jsonl"
    rows = (
        {"defense_id": "def-1", "attack_plan": asdict(strong), "attack_success": 0.90, "rank": 1, "source": "attack_oracle"},
        {"defense_id": "def-1", "attack_plan": asdict(weak), "attack_success": 0.20, "rank": 2, "source": "attack_oracle"},
    )
    path.write_text(
        "".join(__import__("json").dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    samples = load_attack_teacher_samples_jsonl(
        path,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        candidate_weight_temperature=0.25,
    )

    assert len(samples) == 2
    assert samples[0].weight > samples[1].weight
    assert abs((samples[0].weight + samples[1].weight) / 2.0 - 1.0) < 1e-9
    assert samples[0].value_target == 0.90
    assert samples[1].value_target == 0.20


def test_load_attack_teacher_samples_jsonl_prioritizes_exploiter_residual_within_group(tmp_path, loadouts, fmt3):
    high_residual = LegalPlanGenerator(loadouts, seed=115).generate_attack_plan(fmt3, source="attack_oracle:exploiter")
    low_residual = LegalPlanGenerator(loadouts, seed=116).generate_attack_plan(fmt3, source="attack_oracle:exploiter")
    path = tmp_path / "exploiter_teacher.jsonl"
    rows = (
        {
            "teacher_group_id": "round_0001:def-1:exploiter",
            "defense_id": "def-1",
            "attack_plan": asdict(high_residual),
            "attack_success": 0.50,
            "exploiter_residual_target": 0.35,
            "source": "attack_oracle:exploiter",
        },
        {
            "teacher_group_id": "round_0001:def-1:exploiter",
            "defense_id": "def-1",
            "attack_plan": asdict(low_residual),
            "attack_success": 0.50,
            "exploiter_residual_target": 0.05,
            "source": "attack_oracle:exploiter",
        },
    )
    path.write_text(
        "".join(__import__("json").dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    samples = load_attack_teacher_samples_jsonl(
        path,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        candidate_weight_temperature=0.25,
    )

    assert len(samples) == 2
    assert samples[0].weight > samples[1].weight
    assert samples[0].value_target == 0.50
    assert samples[1].value_target == 0.50


def test_save_proposal_network_checkpoint_registers_training_metrics(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=104).generate_attack_plan(fmt3)
    sample = build_attack_teacher_sample(
        plan,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.8,
        gap_target=0.2,
        source="attack_oracle",
    )
    network = AttackGenerationNetwork(ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=15))
    history = train_proposal_network(network, (sample,), epochs=1, lr=1e-3)
    checkpoint_path = tmp_path / "attack_proposal.pt"
    registry_path = tmp_path / "registry.json"

    record = save_proposal_network_checkpoint(
        checkpoint_path,
        network,
        history,
        registry_path=registry_path,
        checkpoint_id="attack-proposal-r0001",
        dataset_hash="teacher-dataset",
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".metrics.json").exists()
    assert record.metrics["train_loss"] == history.train_losses[-1]
    assert CheckpointRegistry(registry_path).latest("attack_proposal") == record


def test_proposal_sequence_to_attack_plan_converts_tokens_to_legal_plan(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=105).generate_attack_plan(fmt3)
    index_by_loadout = {loadout: index for index, loadout in enumerate(loadouts)}
    sequence = tuple(index_by_loadout[loadout] for team in plan.teams for loadout in team.slots)

    reconstructed = proposal_sequence_to_attack_plan(
        sequence,
        loadout_pool=loadouts,
        match_format=fmt3,
        constraint_engine=ConstraintEngine(loadouts),
        source="attack_proposal",
    )

    assert reconstructed.teams == plan.teams
    assert reconstructed.source == "attack_proposal"


def test_proposal_sequence_to_attack_plan_rejects_incomplete_or_illegal_tokens(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)

    with pytest.raises(ValueError, match="token sequence length"):
        proposal_sequence_to_attack_plan((0, 1), loadout_pool=loadouts, match_format=fmt3, constraint_engine=engine)

    duplicate = tuple([0] * (fmt3.n_teams * fmt3.team_size))
    with pytest.raises(ValueError, match="illegal attack plan"):
        proposal_sequence_to_attack_plan(duplicate, loadout_pool=loadouts, match_format=fmt3, constraint_engine=engine)


def test_proposal_sequence_to_defense_plan_converts_tokens_to_legal_roster(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=205).generate_defense_plan(fmt3)
    index_by_loadout = {loadout: index for index, loadout in enumerate(loadouts)}
    sequence = tuple(index_by_loadout[loadout] for team in plan.teams for loadout in team.slots)

    reconstructed = proposal_sequence_to_defense_plan(
        sequence,
        loadout_pool=loadouts,
        match_format=fmt3,
        constraint_engine=ConstraintEngine(loadouts),
        source="defense_proposal",
    )

    assert reconstructed.teams == plan.teams
    assert reconstructed.source == "defense_proposal"
    assert ConstraintEngine(loadouts).is_legal_defense(reconstructed)


def test_attack_legal_action_mask_fn_matches_constraint_engine(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=106).generate_attack_plan(fmt3)
    index_by_loadout = {loadout: index for index, loadout in enumerate(loadouts)}
    target_tokens = tuple(index_by_loadout[loadout] for team in plan.teams for loadout in team.slots)
    mask_fn = attack_legal_action_mask_fn(fmt3, loadout_pool=loadouts, constraint_engine=ConstraintEngine(loadouts))

    first_mask = mask_fn(())
    second_mask = mask_fn(target_tokens[:1])

    assert first_mask[target_tokens[0]]
    assert second_mask[target_tokens[1]]
    assert not second_mask[target_tokens[0]]


def test_attack_legal_action_mask_fn_respects_underdog_budget(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    no_budget = attack_legal_action_mask_fn(fmt3, loadout_pool=loadouts, constraint_engine=engine)(())
    budget = loadouts[0].cost
    budgeted = attack_legal_action_mask_fn(
        fmt3,
        loadout_pool=loadouts,
        constraint_engine=engine,
        goal=GenerationGoal(target_power_ratio=1.0),
        reference_cost=budget,
    )(())
    expensive_legal = next(index for index, allowed in enumerate(no_budget) if allowed and loadouts[index].cost > budget)

    assert budgeted[0]
    assert not budgeted[expensive_legal]


def test_defense_legal_action_mask_fn_respects_underdog_budget(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    no_budget = defense_legal_action_mask_fn(fmt3, loadout_pool=loadouts, constraint_engine=engine)(())
    budget = loadouts[0].cost
    budgeted = defense_legal_action_mask_fn(
        fmt3,
        loadout_pool=loadouts,
        constraint_engine=engine,
        goal=GenerationGoal(target_power_ratio=1.0),
        reference_cost=budget,
    )(())
    expensive_legal = next(index for index, allowed in enumerate(no_budget) if allowed and loadouts[index].cost > budget)

    assert budgeted[0]
    assert not budgeted[expensive_legal]


def test_generate_attack_plan_candidates_from_proposal_network(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=107).generate_attack_plan(fmt3)
    index_by_loadout = {loadout: index for index, loadout in enumerate(loadouts)}
    forced = tuple(index_by_loadout[loadout] for team in plan.teams for loadout in team.slots)
    network = AttackGenerationNetwork(ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20))

    def legal_mask(prefix):
        mask = [False] * len(loadouts)
        mask[forced[len(prefix)]] = True
        return tuple(mask)

    candidates = generate_attack_plan_candidates(
        network,
        match_format=fmt3,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        legal_action_mask_fn=legal_mask,
        beam_size=2,
    )

    assert len(candidates) == 1
    assert candidates[0].plan.teams == plan.teams
    assert candidates[0].sequence.token_ids == forced


def test_generate_defense_roster_candidates_from_proposal_network(loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=207).generate_defense_plan(fmt3)
    index_by_loadout = {loadout: index for index, loadout in enumerate(loadouts)}
    forced = tuple(index_by_loadout[loadout] for team in plan.teams for loadout in team.slots)
    network = DefenseRosterGenerationNetwork(
        ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20)
    )

    def legal_mask(prefix):
        mask = [False] * len(loadouts)
        mask[forced[len(prefix)]] = True
        return tuple(mask)

    candidates = generate_defense_roster_candidates(
        network,
        match_format=fmt3,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        legal_action_mask_fn=legal_mask,
        beam_size=2,
    )

    assert len(candidates) == 1
    assert candidates[0].roster == plan.teams
    assert candidates[0].sequence.token_ids == forced


def test_train_attack_proposal_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.train_attack_proposal", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--teacher-jsonl" in result.stdout
    assert "--heroes-json" in result.stdout
    assert "--out-checkpoint" in result.stdout
    assert "--registry" in result.stdout
    assert "--candidate-weight-temperature" in result.stdout


def test_train_defense_proposal_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.train_defense_proposal", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--teacher-jsonl" in result.stdout
    assert "--heroes-json" in result.stdout
    assert "--out-checkpoint" in result.stdout
    assert "--registry" in result.stdout


def test_train_mask_selection_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.train_mask_selection", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--teacher-jsonl" in result.stdout
    assert "--out-checkpoint" in result.stdout
    assert "--ranking-weight" in result.stdout


def test_load_attack_proposal_candidate_source_from_checkpoint(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=108).generate_attack_plan(fmt3)
    sample = build_attack_teacher_sample(
        plan,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.8,
        gap_target=0.2,
        source="attack_oracle",
    )
    network = AttackGenerationNetwork(ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20))
    history = train_proposal_network(network, (sample,), epochs=1, lr=1e-3)
    checkpoint_path = tmp_path / "attack_proposal.pt"
    save_proposal_network_checkpoint(checkpoint_path, network, history)

    source = load_attack_proposal_candidate_source(checkpoint_path, beam_size=1)
    candidates = source(
        match_format=fmt3,
        belief=None,
        goal=None,
        reference_cost=0.0,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
    )

    assert len(candidates) == 1
    assert candidates[0].source == "attack_proposal"
    assert ConstraintEngine(loadouts).is_legal_attack(candidates[0])


def test_load_defense_proposal_candidate_source_from_checkpoint(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=208).generate_defense_plan(fmt3)
    sample = build_defense_teacher_sample(
        plan,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.8,
        gap_target=0.2,
        source="defense_oracle",
    )
    network = DefenseRosterGenerationNetwork(
        ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20)
    )
    history = train_proposal_network(network, (sample,), epochs=1, lr=1e-3)
    checkpoint_path = tmp_path / "defense_proposal.pt"
    save_proposal_network_checkpoint(checkpoint_path, network, history, model_type="defense_proposal")

    source = load_defense_proposal_candidate_source(checkpoint_path, beam_size=1)
    rosters = source(
        match_format=fmt3,
        attack_meta=(),
        goal=None,
        reference_cost=0.0,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
    )

    assert len(rosters) == 1
    assert ConstraintEngine(loadouts).is_legal_defense(
        proposal_sequence_to_defense_plan(
            tuple(loadouts.index(loadout) for team in rosters[0] for loadout in team.slots),
            loadout_pool=loadouts,
            match_format=fmt3,
            constraint_engine=ConstraintEngine(loadouts),
        )
    )


def test_load_defense_proposal_candidate_source_accepts_legacy_checkpoint_without_residual_head(tmp_path, loadouts, fmt3):
    plan = LegalPlanGenerator(loadouts, seed=213).generate_defense_plan(fmt3)
    sample = build_defense_teacher_sample(
        plan,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.8,
        gap_target=0.2,
    )
    network = DefenseRosterGenerationNetwork(
        ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20)
    )
    history = train_proposal_network(network, (sample,), epochs=1, lr=1e-3)
    checkpoint_path = tmp_path / "legacy_defense_proposal.pt"
    save_proposal_network_checkpoint(checkpoint_path, network, history, model_type="defense_proposal")
    payload = torch.load(checkpoint_path, weights_only=False)
    payload["state_dict"] = {
        key: value
        for key, value in payload["state_dict"].items()
        if not key.startswith("anti_meta_residual_head.")
    }
    torch.save(payload, checkpoint_path)

    source = load_defense_proposal_candidate_source(checkpoint_path, beam_size=1)
    rosters = source(
        match_format=fmt3,
        attack_meta=(),
        goal=None,
        reference_cost=0.0,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
    )

    assert len(rosters) == 1


def test_defense_proposal_context_tensors_encode_attack_meta(loadouts, fmt3):
    attack_a = LegalPlanGenerator(loadouts, seed=209).generate_attack_plan(fmt3)
    attack_b = LegalPlanGenerator(loadouts, seed=210).generate_attack_plan(fmt3)

    tensors = build_defense_proposal_context_tensors(
        ((attack_a, 0.75), (attack_b, 0.25)),
        match_format=fmt3,
        loadout_pool=loadouts,
        max_attack_meta=4,
    )

    assert tensors.observation_token_ids.shape == (1, fmt3.n_teams * fmt3.team_size)
    assert tensors.belief_token_ids.shape == (1, 4, fmt3.n_teams * fmt3.team_size)
    assert tensors.belief_weights.tolist()[0][:2] == [0.75, 0.25]
    assert tensors.pool_token_ids.shape == (1, len(loadouts))
    assert tensors.numeric_features.shape == (1, 4)
    assert tensors.belief_token_ids[0, 0].sum().item() > 0
    assert tensors.belief_token_ids[0, 2].sum().item() == 0


def test_defense_checkpoint_candidate_source_uses_attack_meta_context(tmp_path, loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=211).generate_defense_plan(fmt3)
    attack = LegalPlanGenerator(loadouts, seed=212).generate_attack_plan(fmt3)
    sample = build_defense_teacher_sample(
        defense,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.8,
        gap_target=0.2,
        anti_meta_residual_target=0.4,
    )
    config = ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20)
    network = DefenseRosterGenerationNetwork(config)
    history = train_proposal_network(network, (sample,), epochs=1, lr=1e-3)
    checkpoint_path = tmp_path / "defense_proposal.pt"
    save_proposal_network_checkpoint(checkpoint_path, network, history, model_type="defense_proposal")
    context_encoder = GenerationContextEncoder(config, numeric_feature_dim=4)

    source = load_defense_proposal_candidate_source(checkpoint_path, beam_size=1, context_encoder=context_encoder)
    rosters = source(
        match_format=fmt3,
        attack_meta=((attack, 1.0),),
        goal=None,
        reference_cost=attack.teams[0].total_cost,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
    )

    assert len(rosters) == 1
    assert len(rosters[0]) == fmt3.n_teams


def test_checkpoint_candidate_source_uses_observation_belief_context(tmp_path, loadouts, fmt3):
    attack = LegalPlanGenerator(loadouts, seed=111).generate_attack_plan(fmt3)
    defense = LegalPlanGenerator(loadouts, seed=112).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    sample = build_attack_teacher_sample(
        attack,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
        value_target=0.8,
        gap_target=0.2,
    )
    config = ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20)
    network = AttackGenerationNetwork(config)
    history = train_proposal_network(network, (sample,), epochs=1, lr=1e-3)
    checkpoint_path = tmp_path / "attack_proposal.pt"
    save_proposal_network_checkpoint(checkpoint_path, network, history)
    context_encoder = GenerationContextEncoder(config, numeric_feature_dim=4)
    observation = observe_defense(defense)
    belief = BeliefEngine(ConstraintEngine(loadouts)).build(observation, max_k=2)

    source = load_attack_proposal_candidate_source(checkpoint_path, beam_size=1, context_encoder=context_encoder)
    candidates = source(
        target=observation,
        match_format=fmt3,
        belief=belief,
        goal=None,
        reference_cost=0.0,
        loadout_pool=loadouts,
        constraint_engine=ConstraintEngine(loadouts),
    )

    assert len(candidates) == 1
    assert ConstraintEngine(loadouts).is_legal_attack(candidates[0])


def test_build_attack_proposal_context_tensors_encodes_observation_belief_and_pool(loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=109).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    observation = observe_defense(defense)
    belief = BeliefEngine(ConstraintEngine(loadouts)).build(observation, max_k=4)

    tensors = build_attack_proposal_context_tensors(
        observation,
        belief,
        loadout_pool=loadouts,
        max_belief_candidates=4,
    )

    assert tensors.observation_token_ids.shape == (1, fmt3.n_teams * fmt3.team_size)
    assert tensors.observation_hidden_mask.shape == tensors.observation_token_ids.shape
    assert tensors.observation_hidden_mask.sum().item() == 2
    assert tensors.observation_token_ids[0, 0].item() == 0
    assert tensors.observation_token_ids[0, 1].item() > 0
    assert tensors.belief_token_ids.shape == (1, 4, fmt3.n_teams * fmt3.team_size)
    assert tensors.belief_weights.shape == (1, 4)
    assert tensors.pool_token_ids.shape == (1, len(loadouts))
    assert tensors.numeric_features.shape == (1, 4)


def test_context_encoder_consumes_attack_proposal_context_tensors(loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=110).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    observation = observe_defense(defense)
    belief = BeliefEngine(ConstraintEngine(loadouts)).build(observation, max_k=2)
    config = ProposalNetworkConfig(loadout_count=len(loadouts), model_dim=32, heads=4, layers=1, max_slots=20)
    encoder = GenerationContextEncoder(config, numeric_feature_dim=4)
    tensors = build_attack_proposal_context_tensors(observation, belief, loadout_pool=loadouts, max_belief_candidates=2)

    output = tensors.encode(encoder)

    assert output.context_vector.shape == (1, 32)
    assert torch.isfinite(output.context_vector).all()
