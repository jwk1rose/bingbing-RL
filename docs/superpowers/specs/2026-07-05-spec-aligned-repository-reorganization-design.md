# Spec-Aligned Repository Reorganization Design

## Goal

把当前 `masked-team-league` 仓库整理成一个以 `docs/masked_team_league_system_v4_detailed.tex` 为主轴的清晰实现仓库。重组后，读者应能从 tex 的概念直接找到对应代码、CLI、artifact、测试和验证报告。

## Scope

本次整理允许破坏旧的内部 import 路径和旧脚本入口，但最终仓库必须保持可测试、可运行、可解释。tex 可以小幅修订，用来同步实现结构、命名和交付物映射；数学定义和核心设计原则不做无关改写。

## Package Structure

新的 `src/masked_team_league/` 按 tex 概念分层：

```text
domain/             # MatchFormat, Loadout, Team, AttackPlan, DefensePlan, Observation, canonical hash
constraints/        # legality diagnostics, hard constraints, hidden completion, action masks
scoring/            # match probability, resource cost, surrogate scoring, cache, successive halving
belief/             # BeliefEngine, real-distribution weighting, ranker, datasets
generation/         # legal generator, generation goals, proposal networks, proposal training, mask network
oracles/            # AttackOracle, DefenseOracle, mask search, oracle output contracts
league/             # LeagueManager, round runner, self-play, active perception, active feedback
real_platform/      # backend client, backend codec, oracle evaluator, resources, real calibration
training/           # single-team model/training, checkpoints, model selection, schedule, scheduler daemon
reporting/          # contract registry, round reports, validation reports, ablation reports
data_engineering/   # core tables, run metadata, reproducibility artifacts
cli/                # new primary command entry points
```

Old root-level modules such as `reports.py`, `training_schedule.py`, `proposal_training.py`, and `real_calibration.py` are not kept as the authoritative structure. They are split or moved into their owning package.

## Tex Alignment

Add `docs/spec_to_code_map.md` as the daily navigation document. Each row maps a tex range to:

- spec concept
- code location
- CLI or artifact
- tests
- status or notes

Add a compact implementation-structure appendix to `docs/masked_team_league_system_v4_detailed.tex`, pointing readers to `docs/spec_to_code_map.md` for the live map.

## Module Boundaries

- `domain/` is pure data: no imports from oracle, training, backend, or reporting.
- `constraints/` owns hard legality. Network output is never trusted without this layer.
- `scoring/` owns utilities and simulator abstractions. Oracles compose scoring functions but do not hide them.
- `belief/` owns incomplete-information completion and weighting. AttackOracle consumes `BeliefOutput`.
- `generation/` owns proposal models, teacher data, and legal generation adapters.
- `oracles/` owns online best-response search and stable oracle JSON outputs.
- `league/` owns long-running strategy pools and runtime orchestration.
- `real_platform/` owns external backend and real-meta calibration details.
- `training/` owns model training and recurring schedule execution.
- `reporting/` owns report builders, red-line gates, schema registry, and ablations.
- `data_engineering/` owns persistent tables and reproducibility manifests.

## Comments and Documentation

Code comments should be Chinese and should explain system semantics, tex correspondence, non-obvious invariants, and red-line logic. Avoid mechanical comments.

Good:

```python
# 对应 tex §403-410：可行补全集合与 MRV 搜索。
# 先选择候选域最小的隐藏槽位，减少后续 AllDifferent 冲突。
```

Bad:

```python
# 遍历候选
for candidate in candidates:
```

Each package should expose either a short `README.md` or a package docstring that states:

- corresponding tex sections
- public modules
- inputs and outputs
- artifacts produced or consumed
- tests that protect the package
- what does not belong in this package

## Migration Strategy

Use staged migration, not a single unverified rewrite.

1. Create the new package directories and documentation map.
2. Move low-risk pure modules first: domain, scoring utilities, data engineering.
3. Move constraint and belief layers, then update imports and tests.
4. Move oracles and league runtime.
5. Split report builders by schema into `reporting/validation_reports/`.
6. Move training/proposal/real-platform code.
7. Replace legacy scripts with `cli/` entry points.
8. Update tex and README to reference the new structure.
9. Run focused tests after every migration group, then the full test suite.

## Testing Strategy

Tests should be reorganized to mirror the new package structure:

```text
tests/domain/
tests/constraints/
tests/scoring/
tests/belief/
tests/generation/
tests/oracles/
tests/league/
tests/real_platform/
tests/training/
tests/reporting/
tests/data_engineering/
```

During migration, preserve existing behavioral assertions. Add import-boundary tests where useful:

- `domain` does not import high-level runtime packages.
- output contract registry covers every report schema.
- `docs/spec_to_code_map.md` references existing files.

## Acceptance Criteria

- The repository has a tex-aligned package layout.
- `docs/spec_to_code_map.md` maps all major tex concepts to code, tests, CLI, and artifacts.
- `README.md` explains the new structure and primary commands.
- Large monoliths are split into purpose-owned modules, especially reports, training schedule, proposal training, and real calibration.
- Important non-obvious logic has concise Chinese comments.
- Tests import the new paths.
- `python3 -m pytest -q` passes in an environment with pytest installed.
- No unrelated files such as `.idea/` are modified.
