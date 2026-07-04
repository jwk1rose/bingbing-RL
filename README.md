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
