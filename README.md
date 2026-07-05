# Masked Team League System

This repository is a clean implementation target for
`docs/masked_team_league_system_v4_detailed.tex`.

The repository is organized around the tex specification. Use
`docs/spec_to_code_map.md` as the live map from tex sections to code, CLI
entry points, artifacts, and tests.

The first deliverable implements the strict MVP from the spec:

- position-aware `Loadout` as the atomic action object
- BO3/BO5 `MatchFormat`, `AttackPlan`, `DefensePlan`, and `Observation`
- hard legality through `ConstraintEngine`
- legal random proposal generation with future feasibility checks
- belief construction from masked observations
- surrogate scoring, BO3/BO5 match evaluation, diversity selection, and successive halving
- AttackOracle and DefenseOracle interfaces with explanations and reproducible metadata
- early League and ActivePerception scaffolding

The system intentionally does not start with neural training. The spec requires
data structures, legality, complete-defense AttackOracle, cache, belief, defense,
distillation, and league in that order.

## Repository Structure

- `src/masked_team_league/domain/`: MatchFormat, Loadout, Team, plans, observations, canonical hashes.
- `src/masked_team_league/constraints/`: hard legality, hidden-slot domains, MRV completion, legal action masks.
- `src/masked_team_league/scoring/`: BO3/BO5 utility, resource cost, surrogate scoring, cache, successive halving.
- `src/masked_team_league/belief/`: legal completions, real-distribution weighting, belief rankers.
- `src/masked_team_league/generation/`: legal proposal generation, proposal networks, teacher data, mask networks.
- `src/masked_team_league/oracles/`: AttackOracle, DefenseOracle, mask search, oracle output contracts.
- `src/masked_team_league/league/`: LeagueManager, round runner, self-play, active perception.
- `src/masked_team_league/real_platform/`: backend adapter, battle codec, resource loading, real calibration.
- `src/masked_team_league/training/`: single-team model, checkpoint selection, training schedule.
- `src/masked_team_league/reporting/`: reports, red-line gates, production readiness, ablations.
- `src/masked_team_league/data_engineering/`: core tables and reproducibility metadata.
- `src/masked_team_league/cli/`: package CLI entry points. Fine-grained command bodies live in
  `src/masked_team_league/cli/commands/`.

Run tests:

```bash
python3 -m pytest -q
```

Submit one generated legal smoke match to an already running oracle backend.
The backend must report a ready worker pool; this script does not start or stop
emulators.

```bash
python3 examples/backend_smoke.py \
  --backend http://127.0.0.1:18281 \
  --heroes-json /home/yons/game_apk_analysis/outputs/hero_stats_viewer_20260630_mumu/data/heroes.json \
  --teams 3
```

Run one small true-oracle league round:

```bash
python3 -m masked_team_league.cli.run_round \
  --backend http://127.0.0.1:18281 \
  --heroes-json /home/yons/game_apk_analysis/outputs/hero_stats_viewer_20260630_mumu/data/heroes.json \
  --decoded-dir /home/yons/game_apk_analysis/exports/current_ptr_hotpatch_after_login_20260626/decoded \
  --out-dir exports/masked_league/round_0001 \
  --round-id round_0001 \
  --teams 3 \
  --defenses 20 \
  --attacks-per-defense 200 \
  --oracle-top-k 20
```

After installation, the same entry points are available as `mtl-run-round`,
`mtl-run-selfplay`, `mtl-train`, `mtl-report`, `mtl-calibrate`, and `mtl-ablate`.
