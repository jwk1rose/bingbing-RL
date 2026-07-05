# Legality Diagnostics Schema

`legality_diagnostics.v1` is the production-facing schema for rejected attack or defense requests. It is intentionally small and stable so backend workers, frontend dashboards, data quality checks, and training ingestion can all classify rejection causes without parsing free-form text.

## Report Object

```json
{
  "schema_version": "legality_diagnostics.v1",
  "legal": false,
  "reasons": ["team 1: mask entries must be 0 or 1"],
  "diagnostics": [
    {
      "code": "MASK_VALUE",
      "message": "team 1: mask entries must be 0 or 1",
      "path": ["mask", "team_1"],
      "severity": "error",
      "details": {
        "raw_reason": "team 1: mask entries must be 0 or 1"
      }
    }
  ]
}
```

## Required Fields

- `schema_version`: always `legality_diagnostics.v1`.
- `legal`: boolean legality result.
- `reasons`: backwards-compatible human-readable reason strings.
- `diagnostics`: structured rejection records.
- `diagnostics[].code`: stable machine-readable code.
- `diagnostics[].message`: human-readable message matching one entry in `reasons`.
- `diagnostics[].path`: JSON-path-like location segments, such as `["teams", "team_2"]` or `["mask", "team_1"]`.
- `diagnostics[].severity`: currently `error`.
- `diagnostics[].details`: optional key/value strings for debugging and lineage.

## Codes

- `DUPLICATE_HERO`: hero uniqueness violation within a team or across a roster.
- `DUPLICATE_UNIQUE_EQUIP`: unique legendary equipment reused where it must be unique.
- `STANDING_ORDER`: roster slot order violates increasing `standing_rank`.
- `UNIQUE_EQUIP_STAR`: unique equipment star is outside the allowed 3/4/5 range.
- `FORMAT_TEAM_COUNT`: attack or defense team count does not match `MatchFormat`.
- `MASK_PER_TEAM_LIMIT`: a mask row hides more slots than `max_hidden_per_team`.
- `MASK_VALUE`: a mask row contains a value other than 0 or 1.
- `MASK_GLOBAL_LIMIT`: the mask hides more total slots than `max_hidden_total`.
- `LEGALITY_VIOLATION`: fallback for uncategorized legality failures.

## Compatibility

Callers that only need old behavior may continue reading `LegalReport.reasons`. New code should prefer `LegalReport.to_json_dict()["diagnostics"]` and branch on `code`, not on `message`.
