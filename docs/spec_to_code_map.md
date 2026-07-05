# Spec To Code Map

This document maps `docs/masked_team_league_system_v4_detailed.tex` to the implementation.

| Tex Range | Spec Concept | Code | CLI / Artifact | Tests | Notes |
|---|---|---|---|---|---|
| §142-331 | MatchFormat, HeroRecord, Loadout, AttackPlan, DefensePlan, Observation | `src/masked_team_league/domain/` | `core_tables.v1` | `tests/test_models_and_constraints.py`, `tests/test_architecture_boundaries.py` | Pure domain objects and canonical hashes. |
| §360-470 | ConstraintEngine, hidden domains, MRV, forward checking, future feasible masks | `src/masked_team_league/constraints/` | `legality_diagnostics.v1` | `tests/test_models_and_constraints.py`, `tests/test_legality_acceptance_pack.py` | Hard legality layer. |
| §472-556 | Match win probability, cost, underdog residual | `src/masked_team_league/scoring/` | risk report fields | `tests/test_belief_and_evaluation.py`, `tests/test_metrics_and_reports.py` | Utility and objective helpers. |
| §557-663 | SingleTeamWinrateModel | `src/masked_team_league/training/single_team_model.py` | `mtl-train single-team`, checkpoint registry, holdout reports | `tests/test_single_team_model.py`, `tests/test_single_team_training.py` | Torch optional. |
| §665-993 | Proposal and mask generation networks | `src/masked_team_league/generation/` | `mtl-train attack-proposal`, `mtl-train defense-proposal`, `mtl-train mask-selection`, teacher JSONL | `tests/test_proposal_networks.py`, `tests/test_proposal_training.py` | Legal masks are separate from causal masks. |
| §994-1053 | BeliefEngine | `src/masked_team_league/belief/` | `mtl-train build-belief-ranker-dataset`, `mtl-train belief-ranker`, belief domain stats | `tests/test_belief_and_evaluation.py`, `tests/test_belief_ranker.py` | Legal completion and real-distribution weighting. |
| §1054-1168 | AttackOracle and DefenseOracle | `src/masked_team_league/oracles/` | `attack_oracle_output.v1`, `defense_oracle_output.v1`, `mtl-report attack-oracle-failure` | `tests/test_oracles_and_league.py`, `tests/test_output_contract_registry.py` | Online search and risk reports. |
| §1169-1234 | ActivePerceptionScheduler | `src/masked_team_league/league/active_perception.py` | `active_queries.jsonl`, `mtl-calibrate dispatch-active-real` | `tests/test_active_query_dispatch.py`, `tests/test_oracles_and_league.py` | Query acquisition. |
| §1235-1283 | RealMetaDB and calibration | `src/masked_team_league/real_platform/calibration.py` | `mtl-calibrate ingest-real`, `mtl-calibrate build-real-samples`, real calibration reports | `tests/test_real_calibration.py`, `tests/test_real_resources_and_backend.py` | Real distribution and drift validation. |
| §1284-1380 | LeagueManager and PSRO | `src/masked_team_league/league/` | `mtl-run-round`, `mtl-run-selfplay`, league state, payoff rows | `tests/test_oracles_and_league.py`, `tests/test_selfplay_orchestration.py`, `tests/test_real_oracle_and_round_runner.py` | Main/exploiter/underdog pools. |
| §1380-1415 | Hashes, core tables, reproducibility | `src/masked_team_league/data_engineering/` | `mtl-train build-split-manifest`, `run_metadata.v1`, `core_tables.v1` | `tests/test_data_tables.py`, `tests/test_run_metadata.py` | Persistent joins and artifact hashes. |
| §1416-1467 | Training workflow and evaluation protocol | `src/masked_team_league/training/`, `src/masked_team_league/reporting/` | `mtl-train schedule`, `mtl-report ...`, `mtl-ablate ...`, validation reports | `tests/test_training_schedule.py`, `tests/test_round_reports.py`, `tests/test_ablation_reports.py` | Red-line gates and readiness reports. |

## Reading Order

1. Start with `domain/`, then `constraints/`, then `scoring/`.
2. Read `belief/` before `oracles/` because AttackOracle consumes belief outputs.
3. Read `league/` after oracles; it is the runtime loop, not the place for low-level rules.
4. Use `real_platform/`, `training/`, `reporting/`, and `cli/` to understand production feedback, validation, and runnable entry points.
