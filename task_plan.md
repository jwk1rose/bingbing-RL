# Task Plan

Goal: implement the v4 masked BO3/BO5 attack-defense league search spec in a new repository.

Current strict-conformance note: Phases 1-8 were an MVP/skeleton pass. After re-reading
`docs/masked_team_league_system_v4_detailed.tex`, full v4 compliance is not complete.
The authoritative checklist is now `docs/spec_conformance_matrix.md`.

## Phase 1: Repository and Spec Baseline

Status: complete

- Initialize independent git repository at `/home/yons/masked-team-league-system`.
- Copy the detailed v4 tex spec into `docs/`.
- Create package metadata and test setup.

## Phase 2: Data Structures and Legality

Status: complete

- Implement `MatchFormat`, `HeroRecord`, `Loadout`, `Team`, `AttackPlan`, `DefensePlan`, `Observation`.
- Implement canonical hashing and metadata. Core hashes, `ResultMetadata`, split manifests, structured legality diagnostics, `run_metadata.v1` reproducibility manifests, `core_tables.v1` persistent JSONL tables, a machine-readable runtime/training output contract registry, and scheduled `data_engineering_validation_report.v1` metadata/table/hash coverage gates are done; `LeagueRoundRunner` emits round tables and `run_metadata`, and `run_training_schedule()` emits run manifests. Longer production metadata/table stability validation remains pending.
- Implement `ConstraintEngine` with hero uniqueness, unique equipment ID uniqueness, position ordering, mask limits, hidden domains, MRV completions, action mask, and structured rejection diagnostics.
- Add unit tests for legal and illegal cases, including a formal 20 legal attacks / 20 legal defenses / 20 illegal inputs acceptance pack and JSON-serializable rejection diagnostics.

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

## Phase 8: Real Oracle League Round

Status: partial

- Add batch oracle evaluator for many attack/defense plan pairs in one backend job.
- Populate the simulation cache from true oracle round results.
- Add one-round league runner: generate defenses, search attack candidates, true-oracle top-K, write artifacts, update league pools/payoffs.
- Add CLI entry point `scripts/run_league_round.py`.
- Add decoded runtime rules so battle protos include camp-group filtering, legend equipment, shard, and astrolabe fields.
- Verify with a real 16-worker backend smoke round.

Gap: this phase currently runs random defenses and complete-defense AttackOracle search.
The v4 document requires DefenseOracle roster/mask generation and mask-aware AttackOracle
through `Observation`/`BeliefEngine` as the main runtime path.

## Phase 9: Strict v4 Conformance Runtime

Status: in_progress

- Replace the main round path with the documented loop: league meta, DefenseOracle, `observe_defense`, BeliefEngine, AttackOracle, true-oracle re-evaluation, pool/payoff updates. Done for one-round true-oracle runtime with iteration advancement, default role loop, active query artifact output, and run metadata.
- Add roles and mixed meta sampling to `LeagueManager`: main, exploiter, historical, underdog. Mixed meta sampling, role generation loop, active historical retention, strength-based exploiter target APIs, runner retention wiring, multi-round attack/defense self-play orchestration, target/residual teacher feedback for learned exploiters against main-attack baselines, exploiter effectiveness reporting, League/PSRO health reporting, and validation-gated self-play early-stop are implemented; production-scale learned exploiter runs that clear the gate remain pending.
- Integrate underdog target power ratios into attack, defense, active perception, and explanations. Attack/defense role generation has underdog goal and fallback; proposal-network action masks can filter by goal/reference-cost budget; AttackOracle and DefenseOracle apply explicit underdog residual objective bonuses with stable risk/report fields; scheduled `underdog_residual_validation_report.v1` now checks production residual coverage/bonus quality and stops recurring schedulers on underdog red lines; active perception/report integration exists via `active_queries.jsonl`, daily reports, live dispatch feedback, and `active_real_query_dispatch_validation.v1`.
- Expand runtime artifacts to include belief entropy, top belief candidates, per-lane scores, worst-case risk, mask explanation, and generation metadata. Attack/defense structured risk reports now include lane rates, worst-case belief completion, backup attack/defense scores, hidden count, counter-attack risk, learned mask/risk slot explanations, stable JSON output contracts, AttackOracle domain-specific failure annotations, and scheduled AttackOracle failure annotation validation.

## Phase 10: Strict v4 Neural Modules

Status: pending

- Implement torch `SingleTeamWinrateModel` with loadout encoder, ordered team encoder, 5x5 cross interaction, win/uncertainty/margin/time/residual heads, and scorer adapter. Model/scorer/ensemble/save-load/streaming JSONL loader/training/eval API/CLI/calibration pass/holdout report/checkpoint registry, JSONL split manifests, automated checkpoint selection job, v4 training schedule orchestration, CPU/memory/GPU resource snapshots, and recurring scheduler daemon state are done; long-running production validation pending.
- Implement causal proposal network classes for AttackGenerationNetwork and DefenseRosterGenerationNetwork with legal action masking. Base causal pointer modules, generic observation/belief/pool context encoder, BC/value/gap/anti-meta residual distillation loss, attack/defense teacher sample builders, JSONL artifact loaders with grouped candidate weighting plus attack target/residual role weighting against main baselines, optimizer loop, checkpoint/registry integration, mask-respecting beam/sample token API, generated-token-to-plan/roster adapters, attack/defense proposal training CLIs, AttackOracle/DefenseOracle candidate-source hooks, trained checkpoint candidate-source loaders, runtime observation/belief/pool context tensors, defense attack-meta context tensors, multi-round attack/defense proposal self-play feedback, per-round learned exploiter validation/early-stop, defense anti-meta residual teacher targets, schedule wiring, exploiter effectiveness report CLI, defense anti-meta effectiveness report CLI, and combined learned exploiter validation report/CLI are done; stronger production anti-meta/learned-exploiter runs pending.
- Implement MaskSelectionNetwork slot scorer and constrained selector. Base module, runtime `MaskSearcher`/`DefenseOracle` learned slot-score provider integration, counter-sensitivity feature builder, BCE+ranking-loss training pipeline, checkpoint provider, CLI, schedule job, slot-level learned mask/risk explanations, high-hidden-priority bounded mask enumeration, and scheduled `mask_explanation_validation_report.v1` red-line validation are done; longer production validation pending.
- Add network tests from Appendix G. Causal attention, legal action masks, attack/defense underdog budget masks, mask leakage sensitivity, belief entropy, and SingleTeam position/equipment-star sensitivity now have explicit tests; Appendix G system tests also cover AttackOracle full-defense top-5 legality, mask-observation belief compatibility, DefenseOracle counter-attack success reporting, ActivePerception high-info/high-impact query priority, and League 10-round attack/defense cluster diversity.

## Phase 11: Training, Calibration, and Evaluation

Status: pending

- Implement coverage data, single-team value training, teacher generation, defense teacher generation, mask selection training, league self-play, and real calibration stages. Single-team value training entry point with holdout and checkpoint registry is done; attack/defense/mask teacher artifact training, active real-query feedback teacher rows with scheduled `active_query_feedback_report.v1` red-line validation, data-engineering metadata/table validation, proposal distillation, checkpoint registry updates, multi-round self-play orchestration, dry-run/execute training schedule wiring, resource monitoring, recurring scheduler daemon, recurring mask explanation validation, and recurring real calibration ingestion/red-line checks exist; production long-run validation pending.
- Implement RealMetaDB schema and RealCalibrationModel. JSONL persistence, Platt calibrator, real-context feature extraction, feature-weight calibration fitting, feature calibration CLI/report, holdout Brier/ECE validation report, calibration sample builder from league-round and active-real artifacts, time decay, exact-observation RealMetaDB belief weighting, league-round artifact ingestion, active real-query feedback ingestion, season/version drift report, scheduler drift red-line stop, and scheduler real-calibration holdout red-line stop are done; production-scale real holdout validation pending.
- Add DefensePool retrieval into BeliefModel. Exact-observation compatible DefensePool retrieval, pool-weighted belief scoring, neural ranker adapter, artifact-backed ranker training/eval job, checkpoint loading, round-artifact ranker dataset builder, split manifest, registry-based runner injection, scheduled round-artifact ranker retraining, aggregate domain/weight stats, RealMetaDB similar-observation weighting, candidate artifact `belief_domain_stats`, and scheduled `belief_real_distribution_validation_report.v1` red-line validation are done; production-scale validation of real-distribution similarity weighting remains pending.
- Add evaluation reports, ablations, daily training report JSON, and red-line checks. Metric helpers, daily report schema, round report CLI, active-query feedback report CLI with coverage/error red lines, active-real dispatch validation report CLI, data-engineering validation report CLI, underdog residual validation report CLI, League/PSRO health report CLI, AttackOracle failure validation report CLI, exploiter effectiveness report CLI/trend report, defense anti-meta effectiveness report CLI/trend report, combined learned exploiter validation report CLI, real calibration validation report CLI, production readiness aggregate report CLI with same-schedule report auto-collection, v4 conformance validation matrix CLI/schedule gate, mask explanation validation report CLI, belief real-distribution validation report CLI, red-line checks, risk-aware report fields including underdog residual aggregates, scheduler report ingestion, fixed scheduler output for active-query/active-real-dispatch/data-engineering/underdog-residual/league-health/attack-failure/exploiter/defense-anti-meta/learned-exploiter/real-calibration/production-readiness/v4-conformance/mask/belief validation reports, ablation suite/report CLI, and v4 ablation experiment plan/execution CLI are done. All required v4 ablation variants now have executable controls, including position features, equipment-star features, future-feasibility action mask, underdog objective, mask ambiguity, real calibration, and active perception. Production-scale full-variant ablation and feedback validation remain pending.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `apply_patch` wrote files into the old repository | 1 | Copied generated files to the new repository, restored old tracked files, and removed accidental untracked code from the old repository. |
| One illegal mask test used a legal BO3 mask | 1 | Replaced it with invalid mask values to test rejection correctly. |
| Oracle backend was not ready during smoke verification | 1 | Did not submit real battle jobs; recorded readiness failure and left a smoke command for use after backend startup. |
| First real league round had 3 oracle errors from Lua battle scripts | 1 | Compared new request proto with old random-lineup proto; added decoded runtime fields and camp-group resource loading. |
| MVP plan incorrectly implied full v4 completion | 1 | Added strict conformance matrix and reopened runtime/neural/training phases. |
| Real strict-conformance smoke round failed 1/3 oracle requests with `datatable.lua:22` | 1 | Repeated the failing request, isolated the root cause to hero 45 via one-hero replacement, added `excluded_hero_ids` resource filtering, and made the round CLI exclude hero 45 by default. |
| Real role-loop smoke round failed 1/9 oracle requests with `datatable.lua:22` | 1 | Repeated the failing request, isolated the root cause to hero 75 via one-hero replacement, and added hero 75 to the default oracle exclusion list. |
| One full pytest run segfaulted inside CPython JSON encoding after 18 tests | 1 | Ran the failing test alone, the containing test file, and a preceding test combination successfully; then re-ran full suite with `-X faulthandler`, which passed 68 tests. Recorded as a transient interpreter/extension failure unless it recurs. |
| Belief-ranker round-artifact schedule failed select-best when `holdout_fraction=0.0` | 1 | Root cause was selection using `holdout_top1_accuracy` even though generated holdout rows were empty and training emitted only train metrics. Added a regression test and switched generated zero-holdout schedules to train metrics. |
| Manual mask enumeration inspection failed with `ModuleNotFoundError: masked_team_league` | 1 | The ad hoc command lacked `PYTHONPATH=src`; re-ran with `PYTHONPATH=src` and confirmed bounded mask enumeration now starts from hidden masks. |
| Full-pool learned self-play smoke stayed on local CPU before backend submission | 1 | Stopped the process before backend work was submitted; created a temporary top-50 camp-group hero subset to validate the learned self-play/report path without full-pool candidate generation cost. |
| Multi-role learned-checkpoint self-play stalled in round 2 local CPU after round 1 completed | 1 | Stopped the run after preserving round 1 artifacts. A separate 1-defense checkpoint repro completed, narrowing the hotspot to multi-defense/multi-role learned-checkpoint local search rather than checkpoint loading alone. |
| CPU-only Python environment prevented neural GPU use | 1 | Switched validation run to `/home/yons/anaconda3/envs/unsloth/bin/python`, verified CUDA Torch and two RTX 4090 D devices, and completed a 3-round top50 multi-role learned self-play smoke with `--proposal-device cuda:0 --belief-ranker-device cuda:0`. |
| CUDA top50 learned exploiter smoke still failed attack anti-meta gate | 1 | Final learned validation red lines were `attack_anti_meta_residual_non_positive` and `attack_anti_meta_positive_rate_low`; defense anti-meta passed. Next fix should strengthen attack best-response/exploiter candidate generation rather than treating this as a hardware issue. |
| Learned-checkpoint candidate generation was minute-scale before oracle submission | 1 | Profiled with a fake evaluator and found local retry spins in budgeted underdog generation plus repeated sorted-pool/hash work. Added regression tests, cached sorted pools and hashes, changed random generation to budget-aware low-cost sampling, and verified 3-defense / 36-pair local learned-checkpoint generation now completes in 14.71s; full suite passed 276 tests. |
