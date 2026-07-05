# Spec-Aligned Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository around `docs/masked_team_league_system_v4_detailed.tex` so each spec concept maps directly to code, CLI, artifacts, and tests.

**Architecture:** Build the new package tree first, then migrate behavior in dependency order: domain and scoring before constraints, constraints before belief, belief before oracles, oracles before league, then real platform, training, reporting, and CLI. Keep each migration verifiable with focused imports/tests before moving to the next layer.

**Tech Stack:** Python 3.10+, setuptools `src/` layout, pytest, torch optional, JSON/JSONL artifacts, external oracle backend adapter.

---

### Task 1: Create Spec Navigation Documents

**Files:**
- Create: `docs/spec_to_code_map.md`
- Modify: `README.md`
- Modify: `docs/masked_team_league_system_v4_detailed.tex`

- [ ] **Step 1: Add the spec-to-code map**

Create `docs/spec_to_code_map.md` with this structure:

```md
# Spec To Code Map

This document maps `docs/masked_team_league_system_v4_detailed.tex` to the implementation.

| Tex Range | Spec Concept | Code | CLI / Artifact | Tests | Notes |
|---|---|---|---|---|---|
| §142-331 | MatchFormat, HeroRecord, Loadout, AttackPlan, DefensePlan, Observation | `src/masked_team_league/domain/` | `core_tables.v1` | `tests/domain/` | Pure domain objects and canonical hashes. |
| §360-470 | ConstraintEngine, hidden domains, MRV, forward checking, future feasible masks | `src/masked_team_league/constraints/` | `legality_diagnostics.v1` | `tests/constraints/` | Hard legality layer. |
| §472-556 | Match win probability, cost, underdog residual | `src/masked_team_league/scoring/` | risk report fields | `tests/scoring/` | Utility and objective helpers. |
| §557-663 | SingleTeamWinrateModel | `src/masked_team_league/training/single_team_model.py` | checkpoint registry, holdout reports | `tests/training/` | Torch optional. |
| §665-993 | Proposal and mask generation networks | `src/masked_team_league/generation/` | proposal checkpoints, teacher JSONL | `tests/generation/` | Legal masks are separate from causal masks. |
| §994-1053 | BeliefEngine | `src/masked_team_league/belief/` | belief domain stats | `tests/belief/` | Legal completion and real-distribution weighting. |
| §1054-1168 | AttackOracle and DefenseOracle | `src/masked_team_league/oracles/` | `attack_oracle_output.v1`, `defense_oracle_output.v1` | `tests/oracles/` | Online search and risk reports. |
| §1169-1234 | ActivePerceptionScheduler | `src/masked_team_league/league/active_perception.py` | `active_queries.jsonl` | `tests/league/` | Query acquisition. |
| §1235-1283 | RealMetaDB and calibration | `src/masked_team_league/real_platform/calibration.py` | real calibration reports | `tests/real_platform/` | Real distribution and drift validation. |
| §1284-1380 | LeagueManager and PSRO | `src/masked_team_league/league/` | league state, payoff rows | `tests/league/` | Main/exploiter/underdog pools. |
| §1380-1415 | Hashes, core tables, reproducibility | `src/masked_team_league/data_engineering/` | `run_metadata.v1`, `core_tables.v1` | `tests/data_engineering/` | Persistent joins and artifact hashes. |
| §1416-1467 | Training workflow and evaluation protocol | `src/masked_team_league/training/`, `src/masked_team_league/reporting/` | validation reports, schedules | `tests/training/`, `tests/reporting/` | Red-line gates and readiness reports. |
```

- [ ] **Step 2: Update README structure section**

Add a short "Repository Structure" section referencing `docs/spec_to_code_map.md` and listing the new package groups.

- [ ] **Step 3: Add a tex implementation appendix**

Append a small section near the end of the tex document:

```tex
\section{实现目录映射}
当前代码实现按本文概念分层组织。日常维护时，以 \texttt{docs/spec\_to\_code\_map.md} 为准查找 tex 章节、代码模块、CLI、artifact 和测试之间的对应关系。
```

- [ ] **Step 4: Verify documentation references**

Run: `rg -n "spec_to_code_map|实现目录映射|Repository Structure" README.md docs`

Expected: the new README, tex, and map references appear.

### Task 2: Build the New Package Skeleton

**Files:**
- Create directories under `src/masked_team_league/`
- Create package `__init__.py` files
- Create package `README.md` files or docstrings

- [ ] **Step 1: Create directories**

Run:

```bash
mkdir -p src/masked_team_league/{domain,constraints,scoring,belief,generation,oracles,league,real_platform,training,reporting/validation_reports,data_engineering,cli}
```

- [ ] **Step 2: Add package docstrings**

Each package `__init__.py` should contain a Chinese docstring with tex section and responsibility.

- [ ] **Step 3: Add import boundary test**

Create `tests/test_architecture_boundaries.py`:

```python
from pathlib import Path


def test_domain_layer_does_not_import_runtime_layers():
    root = Path("src/masked_team_league/domain")
    forbidden = ("oracles", "league", "training", "reporting", "real_platform")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"masked_team_league.{name}" not in text
            assert f"..{name}" not in text
```

- [ ] **Step 4: Run boundary test**

Run: `python3 -m pytest -q tests/test_architecture_boundaries.py`

Expected in a pytest-enabled environment: PASS.

### Task 3: Migrate Domain Objects

**Files:**
- Create: `src/masked_team_league/domain/hashing.py`
- Create: `src/masked_team_league/domain/formats.py`
- Create: `src/masked_team_league/domain/loadouts.py`
- Create: `src/masked_team_league/domain/plans.py`
- Create: `src/masked_team_league/domain/observations.py`
- Modify: imports in source, scripts, tests
- Delete after migration: old authoritative `src/masked_team_league/models.py`

- [ ] **Step 1: Move canonical hash helpers to `domain/hashing.py`**
- [ ] **Step 2: Move `MatchFormat` to `domain/formats.py`**
- [ ] **Step 3: Move `HeroRecord` and `Loadout` to `domain/loadouts.py`**
- [ ] **Step 4: Move `Team`, `AttackPlan`, `DefensePlan`, `ResultMetadata` to `domain/plans.py`**
- [ ] **Step 5: Move `VisibleSlot`, `Observation`, `observe_defense` to `domain/observations.py`**
- [ ] **Step 6: Export domain API from `domain/__init__.py`**
- [ ] **Step 7: Replace imports**
- [ ] **Step 8: Run domain tests**

### Task 4: Migrate Scoring and Utility Layer

**Files:**
- Move `evaluation.py` to `scoring/match.py`
- Move `surrogate.py` to `scoring/surrogate.py`
- Move `cache.py` to `scoring/cache.py`
- Move `hyperband.py` to `scoring/halving.py`

- [ ] **Step 1: Move files**
- [ ] **Step 2: Add `scoring/__init__.py` exports**
- [ ] **Step 3: Run scoring tests**

### Task 5: Migrate Constraints

**Files:**
- Split `constraints.py` into:
  - `constraints/diagnostics.py`
  - `constraints/completion.py`
  - `constraints/action_masks.py`
  - `constraints/engine.py`

- [ ] **Step 1: Move diagnostics**
- [ ] **Step 2: Keep `ConstraintEngine` as public facade**
- [ ] **Step 3: Add comments for MRV and future feasible**
- [ ] **Step 4: Export from `constraints/__init__.py`**
- [ ] **Step 5: Run legality tests**

### Task 6: Migrate Belief, Oracles, League, and Real Platform

**Files:**
- Move `belief.py` and `belief_ranker.py` under `belief/`
- Move `attack_oracle.py`, `defense_oracle.py`, `mask.py`, `output_contracts.py` under `oracles/` or `reporting/`
- Move `active.py`, `active_feedback.py`, `league.py`, `round_runner.py`, `selfplay.py` under `league/`
- Move `backend.py`, `backend_codec.py`, `real_oracle.py`, `resources.py`, `real_calibration.py` under `real_platform/`

- [ ] **Step 1: Move belief files and update imports**
- [ ] **Step 2: Move oracle files and update imports**
- [ ] **Step 3: Move league runtime files and update imports**
- [ ] **Step 4: Move real platform files and update imports**
- [ ] **Step 5: Run focused runtime tests**

### Task 7: Split Reporting and Training Monoliths

**Files:**
- Split `reports.py` into `reporting/round_reports.py` and `reporting/validation_reports/*.py`
- Move `ablation.py` to `reporting/ablation.py`
- Move `training_schedule.py` to `training/schedule.py`
- Move `training.py`, `single_team_model.py`, `checkpointing.py`, `model_selection.py` to `training/`
- Move `proposal_networks.py`, `proposal_training.py`, and `networks.py` to `generation/`

- [ ] **Step 1: Split reports by schema**
- [ ] **Step 2: Move training schedule**
- [ ] **Step 3: Move proposal training**
- [ ] **Step 4: Run report/training tests**

### Task 8: Replace Script Entry Points with CLI Package

**Files:**
- Create: `src/masked_team_league/cli/run_round.py`
- Create: `src/masked_team_league/cli/run_selfplay.py`
- Create: `src/masked_team_league/cli/train.py`
- Create: `src/masked_team_league/cli/report.py`
- Create: `src/masked_team_league/cli/calibrate.py`
- Create: `src/masked_team_league/cli/ablate.py`
- Modify or remove: legacy `scripts/*.py`

- [ ] **Step 1: Move CLI parser code into package modules**
- [ ] **Step 2: Update tests to call package CLI help**
- [ ] **Step 3: Update README command examples**
- [ ] **Step 4: Run CLI help tests**

### Task 9: Final Verification

- [ ] **Step 1: Check for stale old imports**
- [ ] **Step 2: Run full tests**
- [ ] **Step 3: Inspect git diff**
- [ ] **Step 4: Commit**
