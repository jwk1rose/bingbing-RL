# Run Metadata Schema

`run_metadata.v1` is the persistent manifest schema for reproducible training, simulation, oracle, calibration, and league runs.

The manifest is deliberately separate from round summaries and model reports. A summary tells what happened; this manifest records the exact versions and artifacts needed to reproduce or audit the run.

## Manifest Object

```json
{
  "schema_version": "run_metadata.v1",
  "run_id": "selfplay-r0007",
  "created_at": 1783190000.0,
  "code_version": "git-sha-or-release",
  "model_version": "value-r0042",
  "data_version": "oracle-data-20260705",
  "simulator_version": "oracle_backend",
  "league_iteration": 7,
  "random_seed": 2026070501,
  "generation_config_hash": "sha256-of-canonical-config",
  "calibration_version": "real-cal-r0003",
  "input_artifacts": [],
  "output_artifacts": [],
  "metrics": {},
  "extra": {}
}
```

## Required Reproducibility Fields

- `schema_version`: always `run_metadata.v1`.
- `run_id`: stable run identifier.
- `created_at`: Unix timestamp.
- `code_version`: git SHA, release tag, or explicit local build version.
- `model_version`: value/proposal/ranker checkpoint version used by the run.
- `data_version`: dataset or resource bundle version.
- `simulator_version`: oracle backend, APK, or surrogate simulator version.
- `league_iteration`: league/self-play iteration number.
- `random_seed`: seed used by generation and sampling.
- `generation_config_hash`: SHA-256 of the canonical generation/search configuration.
- `calibration_version`: RealCalibrationModel or calibration bundle version.
- `input_artifacts`: hashed files consumed by the run.
- `output_artifacts`: hashed files produced by the run.
- `metrics`: scalar audit metrics.
- `extra`: string key/value metadata for rollout-specific lineage.

## Artifact Reference

Each artifact reference has:

- `path`: local or exported artifact path.
- `kind`: artifact semantic type, such as `single_matchup_jsonl`, `checkpoint`, `round_summary`, or `oracle_results`.
- `role`: `input`, `output`, `model`, `report`, or another stable role label.
- `sha256`: SHA-256 of the artifact bytes.
- `size_bytes`: artifact file size.

## Generation Config Hash

`generation_config_hash` must be computed from a canonical JSON representation with sorted keys and compact separators. This avoids accidental changes in whitespace or dict ordering producing different lineage.
