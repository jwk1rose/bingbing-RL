# Masked Team League System

This repository is a clean implementation target for
`docs/masked_team_league_system_v4_detailed.tex`.

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
python3 scripts/run_league_round.py \
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
