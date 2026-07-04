# Task Plan

Goal: implement the v4 masked BO3/BO5 attack-defense league search spec in a new repository.

## Phase 1: Repository and Spec Baseline

Status: complete

- Initialize independent git repository at `/home/yons/masked-team-league-system`.
- Copy the detailed v4 tex spec into `docs/`.
- Create package metadata and test setup.

## Phase 2: Data Structures and Legality

Status: complete

- Implement `MatchFormat`, `HeroRecord`, `Loadout`, `Team`, `AttackPlan`, `DefensePlan`, `Observation`.
- Implement canonical hashing and metadata.
- Implement `ConstraintEngine` with hero uniqueness, unique equipment ID uniqueness, position ordering, mask limits, hidden domains, MRV completions, and action mask.
- Add unit tests for legal and illegal cases.

## Phase 3: MVP Oracle Stack

Status: complete

- Implement match probability DP, cost functions, surrogate scorer interface, deterministic heuristic scorer.
- Implement cache and successive halving.
- Implement legal proposal generator.
- Implement complete-defense AttackOracle with explanation.

## Phase 4: Mask and Defense

Status: complete

- Implement BeliefEngine over legal completions.
- Implement mask search and mask-aware AttackOracle path.
- Implement DefenseOracle skeleton: roster generation, mask search, attack counter evaluation.

## Phase 5: League and Active Perception

Status: complete

- Implement LeagueManager pool/payoff skeleton.
- Implement ActivePerceptionScheduler acquisition queue.
- Keep neural generation as interface/proposal layer until Oracle teachers exist.

## Phase 6: Verification and Initial Commit

Status: complete

- Run full tests.
- Commit repository baseline.

## Phase 7: Real Resources and Decoupled Oracle Backend

Status: complete

- Load real decoded `heroes.json` catalogs into `HeroRecord` and `Loadout` objects.
- Convert legal attack/defense plans into the existing oracle backend request shape.
- Add a thin HTTP client for the decoupled oracle backend.
- Add a backend smoke example that submits one BO3/BO5 plan only when the worker pool is ready.
- Verify request/result scoring with unit tests and the real decoded hero catalog.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `apply_patch` wrote files into the old repository | 1 | Copied generated files to the new repository, restored old tracked files, and removed accidental untracked code from the old repository. |
| One illegal mask test used a legal BO3 mask | 1 | Replaced it with invalid mask values to test rejection correctly. |
| Oracle backend was not ready during smoke verification | 1 | Did not submit real battle jobs; recorded readiness failure and left a smoke command for use after backend startup. |
