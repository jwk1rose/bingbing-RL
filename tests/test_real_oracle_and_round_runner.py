from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from masked_team_league.belief_ranker import (
    BeliefRankerTrainingSample,
    TorchBeliefRankerAdapter,
    evaluate_belief_ranker,
    save_belief_ranker_checkpoint,
    train_belief_ranker,
)
from masked_team_league.cache import SimulationCache
from masked_team_league.data_tables import CORE_TABLE_SCHEMA_VERSION, load_table_jsonl
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.models import MatchFormat, Team, observe_defense
from masked_team_league.real_oracle import OracleBatchEvaluator
from masked_team_league.resources import load_hero_resource_bundle
from masked_team_league.round_runner import LeagueRoundConfig, LeagueRoundRunner
from masked_team_league.run_metadata import load_run_metadata_manifest


def _write_heroes(path: Path, count: int = 40) -> None:
    heroes = []
    for hero_id in range(1, count + 1):
        heroes.append(
            {
                "id": hero_id,
                "displayName": f"英雄{hero_id}",
                "level": 100,
                "stars": 5,
                "rank": 23,
                "equipIds": [6000 + hero_id, 6100 + hero_id],
                "stats": {"GS": 10000 + hero_id * 100},
                "positionType": "front" if hero_id <= 12 else "mid" if hero_id <= 26 else "back",
            }
        )
    path.write_text(json.dumps({"heroes": heroes}, ensure_ascii=False), encoding="utf-8")


class _FakeOracleClient:
    def __init__(self) -> None:
        self.submitted_requests: list[dict[str, object]] = []

    def submit_and_wait(self, requests, *, metadata=None):
        self.submitted_requests.extend(requests)
        return {"job_id": "job-unit", "status": "completed"}

    def read_results(self, job_id):
        rows = []
        for index, request in enumerate(self.submitted_requests):
            rows.append(
                {
                    "request_id": request["request_id"],
                    "status": "completed",
                    "battle_result": 0 if index % 3 != 1 else 1,
                }
            )
        return rows


def test_oracle_batch_evaluator_batches_plan_pairs_and_populates_cache(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    fmt = MatchFormat(3)
    generator = LegalPlanGenerator(resources.loadouts, seed=1)
    attack = generator.generate_attack_plan(fmt)
    defense = generator.generate_defense_plan(fmt)
    cache = SimulationCache()
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=cache)

    records = evaluator.evaluate_pairs(
        [("atk-1", attack, "def-1", defense)],
        job_prefix="unit",
        base_seed=123,
        metadata={"kind": "unit"},
    )

    assert len(records) == 1
    assert records[0].attack_id == "atk-1"
    assert records[0].defense_id == "def-1"
    assert records[0].round_win_rates == (1.0, 0.0, 1.0)
    assert records[0].attack_success > 0.99
    assert len(client.submitted_requests) == 3
    assert len(cache) == 3

    second = evaluator.evaluate_pairs(
        [("atk-1", attack, "def-1", defense)],
        job_prefix="unit-repeat",
        base_seed=456,
    )

    assert second[0].round_win_rates == records[0].round_win_rates
    assert len(client.submitted_requests) == 3


def test_league_round_runner_writes_artifacts_and_updates_league(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    out_dir = tmp_path / "round_0001"
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=LeagueRoundConfig(
            teams=3,
            defenses=2,
            attacks_per_defense=16,
            oracle_top_k=2,
            seed=7,
            attack_roles=("main",),
            defense_roles=("main",),
        ),
    )

    summary = runner.run(out_dir)

    assert summary.defenses == 2
    assert summary.oracle_pairs == 4
    assert summary.oracle_requests == 12
    assert (out_dir / "candidates.jsonl").exists()
    assert (out_dir / "oracle_results.jsonl").exists()
    assert (out_dir / "oracle_pairs.jsonl").exists()
    assert (out_dir / "scored_attacks.jsonl").exists()
    assert (out_dir / "scored_defenses.jsonl").exists()
    assert (out_dir / "active_queries.jsonl").exists()
    oracle_pairs = [json.loads(line) for line in (out_dir / "oracle_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert oracle_pairs[0]["attack_id"].startswith("atk-")
    assert oracle_pairs[0]["defense_id"].startswith("def-")
    assert "attack_success" in oracle_pairs[0]
    state = json.loads((out_dir / "league_state.json").read_text(encoding="utf-8"))
    assert len(state["attack_pool"]) == 4
    assert len(state["defense_pool"]) == 2
    assert len(state["payoffs"]) == 4
    manifest = load_run_metadata_manifest(out_dir / "run_metadata.json")
    payload = manifest.to_json_dict()
    artifact_paths = {Path(row["path"]).name for row in payload["output_artifacts"]}
    assert payload["run_id"] == "round_0001"
    assert payload["random_seed"] == 7
    assert payload["simulator_version"] == "oracle_backend"
    assert payload["league_iteration"] == 1
    assert payload["generation_config_hash"]
    assert {"summary.json", "oracle_results.jsonl", "candidates.jsonl", "plan_matches.jsonl"} <= artifact_paths
    table_dir = out_dir / "tables"
    expected_tables = {
        "loadouts.jsonl",
        "observations.jsonl",
        "single_matchups.jsonl",
        "plan_matches.jsonl",
        "league_strategies.jsonl",
    }
    assert expected_tables <= {path.name for path in table_dir.iterdir()}
    plan_rows = load_table_jsonl(table_dir / "plan_matches.jsonl")
    single_rows = load_table_jsonl(table_dir / "single_matchups.jsonl")
    strategy_rows = load_table_jsonl(table_dir / "league_strategies.jsonl")
    assert plan_rows[0]["schema_version"] == CORE_TABLE_SCHEMA_VERSION
    assert len(plan_rows) == summary.oracle_pairs
    assert len(single_rows) == summary.oracle_requests
    assert len(strategy_rows) == len(state["attack_pool"]) + len(state["defense_pool"])


def test_league_round_runner_uses_defense_oracle_and_mask_observation_path(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    out_dir = tmp_path / "round_doc_path"
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=LeagueRoundConfig(
            teams=3,
            defenses=1,
            attacks_per_defense=12,
            oracle_top_k=1,
            seed=17,
            defense_roster_candidates=2,
            defense_masks_per_roster=1,
            defense_max_masks_per_roster=8,
            attack_roles=("main",),
            defense_roles=("main",),
        ),
    )

    summary = runner.run(out_dir)
    candidates = [json.loads(line) for line in (out_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    defenses = [json.loads(line) for line in (out_dir / "scored_defenses.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary.defenses == 1
    assert candidates
    assert candidates[0]["target_kind"] == "mask_observation"
    assert candidates[0]["attack_plan"]["format"]["n_teams"] == 3
    assert len(candidates[0]["attack_plan"]["teams"]) == 3
    assert candidates[0]["belief_candidates"] >= 1
    assert "belief_entropy" in candidates[0]
    assert "belief_domain_stats" in candidates[0]
    belief_stats = dict(candidates[0]["belief_domain_stats"])
    assert "real_record_count" in belief_stats
    assert "real_similar_record_count" in belief_stats
    assert "defense_pool_record_count" in belief_stats
    assert "attack_risk_report" in candidates[0]
    assert len(candidates[0]["attack_risk_report"]["expected_lane_win_rates"]) == 3
    assert defenses[0]["defense_plan"]["format"]["n_teams"] == 3
    assert defenses[0]["source"] == "defense_oracle:main"
    assert defenses[0]["defense_role"] == "main"
    assert defenses[0]["hidden_count"] > 0
    assert "defense_risk_report" in defenses[0]
    assert defenses[0]["defense_risk_report"]["backup_defense_count"] >= 0
    active_queries = [json.loads(line) for line in (out_dir / "active_queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert active_queries
    assert active_queries[0]["queue"] == "sim"
    assert active_queries[0]["query_type"] in {"mask_observation", "underdog"}
    assert active_queries[0]["attack_id"] == candidates[0]["attack_id"]
    assert active_queries[0]["defense_id"] == candidates[0]["defense_id"]


def test_league_round_runner_emits_main_exploiter_underdog_roles(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    out_dir = tmp_path / "round_roles"
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=LeagueRoundConfig(
            teams=3,
            defenses=3,
            attacks_per_defense=4,
            oracle_top_k=1,
            seed=19,
            defense_roster_candidates=1,
            defense_masks_per_roster=1,
            defense_max_masks_per_roster=1,
        ),
    )

    summary = runner.run(out_dir)
    candidates = [json.loads(line) for line in (out_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    defenses = [json.loads(line) for line in (out_dir / "scored_defenses.jsonl").read_text(encoding="utf-8").splitlines()]
    state = json.loads((out_dir / "league_state.json").read_text(encoding="utf-8"))

    assert summary.defenses == 3
    assert {"main", "exploiter", "underdog"} <= {row["attack_role"] for row in candidates}
    assert {"main", "exploiter", "underdog"} <= {row["defense_role"] for row in defenses}
    assert {"main", "exploiter", "underdog"} <= {row["role"] for row in state["attack_pool"]}
    assert {"main", "exploiter", "underdog"} <= {row["role"] for row in state["defense_pool"]}


def test_league_round_runner_applies_pool_retention_and_clusters(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    out_dir = tmp_path / "round_retention"
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=LeagueRoundConfig(
            teams=3,
            defenses=3,
            attacks_per_defense=4,
            oracle_top_k=1,
            seed=29,
            defense_roster_candidates=1,
            defense_masks_per_roster=1,
            defense_max_masks_per_roster=1,
            attack_pool_max_active=2,
            defense_pool_max_active=2,
            historical_keep=1,
        ),
    )

    runner.run(out_dir)
    state = json.loads((out_dir / "league_state.json").read_text(encoding="utf-8"))
    active_attacks = [row for row in state["attack_pool"] if row["active"]]
    active_defenses = [row for row in state["defense_pool"] if row["active"]]

    assert len(state["attack_pool"]) > len(active_attacks)
    assert len(state["defense_pool"]) > len(active_defenses)
    assert len(active_attacks) <= 2
    assert len(active_defenses) <= 2
    assert all(row["diversity_cluster"] != "default" for row in state["attack_pool"] + state["defense_pool"])
    assert any(row["retired_reason"] == "retention" for row in state["attack_pool"] + state["defense_pool"])


def test_league_round_runner_selects_belief_ranker_from_registry(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    generator = LegalPlanGenerator(resources.loadouts, seed=37)
    defense = generator.generate_defense_plan(
        MatchFormat(3),
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    negative = (Team(resources.loadouts[20:25]), Team(resources.loadouts[25:30]), Team(resources.loadouts[30:35]))
    adapter = TorchBeliefRankerAdapter.from_loadouts(resources.loadouts, model_dim=32)
    sample = BeliefRankerTrainingSample(
        observation=observe_defense(defense),
        positive_roster=defense.teams,
        candidate_rosters=(defense.teams, negative),
    )
    history = train_belief_ranker(adapter.model, adapter.vocab, (sample,), epochs=1, lr=1e-3)
    metrics = evaluate_belief_ranker(adapter.model, adapter.vocab, (sample,))
    registry = tmp_path / "belief_registry.json"
    checkpoint = tmp_path / "belief_ranker.pt"
    save_belief_ranker_checkpoint(
        checkpoint,
        adapter.model,
        adapter.vocab,
        history,
        metrics={"holdout_top1_accuracy": metrics["top1_accuracy"]},
        registry_path=registry,
        checkpoint_id="belief-ranker-r0001",
        dataset_hash="belief-dataset",
    )
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    out_dir = tmp_path / "round_belief_ranker"
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=LeagueRoundConfig(
            teams=3,
            defenses=1,
            attacks_per_defense=4,
            oracle_top_k=1,
            seed=41,
            attack_roles=("main",),
            defense_roles=("main",),
            defense_roster_candidates=1,
            defense_masks_per_roster=1,
            defense_max_masks_per_roster=1,
            belief_ranker_registry=registry,
            belief_ranker_metric="holdout_top1_accuracy",
            belief_ranker_metric_mode="max",
            belief_ranker_weight=0.5,
        ),
    )

    runner.run(out_dir)
    candidates = [json.loads(line) for line in (out_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]

    assert candidates
    assert candidates[0]["belief_ranker_applied"] == 1
    assert candidates[0]["belief_ranker_checkpoint"] == str(checkpoint)


def test_run_league_round_script_has_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_league_round.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--backend" in result.stdout
    assert "--defenses" in result.stdout
    assert "--defense-roster-candidates" in result.stdout
    assert "--underdog-residual-weight" in result.stdout
