#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masked_team_league.metrics import brier_score
from masked_team_league.real_calibration import RealCalibrationModel, RealMetaRecord, build_real_calibration_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a real-distribution feature calibrator for simulator probabilities.")
    parser.add_argument("--samples-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--feature", action="append", default=None, help="Feature to fit. Defaults to all feature keys found.")
    parser.add_argument("--now", type=float, default=None, help="Reference timestamp for RealMetaRecord recency features.")
    parser.add_argument("--recency-tau", type=float, default=7.0 * 24.0 * 60.0 * 60.0)
    parser.add_argument("--max-passes", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scores, labels, feature_rows = _load_samples(args.samples_jsonl, now=args.now, recency_tau=args.recency_tau)
    feature_names = tuple(args.feature) if args.feature else tuple(sorted({key for row in feature_rows for key in row.keys()}))
    base_model = RealCalibrationModel.fit_platt(scores, labels)
    model = RealCalibrationModel.fit_feature_calibrator(
        scores,
        labels,
        feature_rows,
        feature_names=feature_names,
        base_model=base_model,
        max_passes=args.max_passes,
    )
    base_predictions = tuple(base_model.calibrate(score) for score in scores)
    feature_predictions = tuple(model.calibrate(score, row) for score, row in zip(scores, feature_rows))
    payload = {
        "samples_jsonl": [str(path) for path in args.samples_jsonl],
        "features": list(feature_names),
        "base_model": base_model.to_json_dict(),
        "model": model.to_json_dict(),
        "metrics": {
            "samples": len(scores),
            "raw_brier": brier_score(labels, scores),
            "base_brier": brier_score(labels, base_predictions),
            "feature_brier": brier_score(labels, feature_predictions),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "samples": len(scores)}, ensure_ascii=False))
    return 0


def _load_samples(
    paths: tuple[Path, ...] | list[Path],
    *,
    now: float | None,
    recency_tau: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[dict[str, float], ...]]:
    scores: list[float] = []
    labels: list[float] = []
    feature_rows: list[dict[str, float]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: sample must be a JSON object")
                scores.append(_required_float(row, ("sim_probability", "sim_prob", "prediction", "score"), path, line_no))
                labels.append(_required_float(row, ("label", "match_result", "real_label", "win_rate"), path, line_no))
                feature_rows.append(_extract_features(row, path=path, line_no=line_no, now=now, recency_tau=recency_tau))
    return tuple(scores), tuple(labels), tuple(feature_rows)


def _extract_features(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_no: int,
    now: float | None,
    recency_tau: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    record_payload = row.get("real_meta_record", row.get("record"))
    if record_payload is not None:
        if not isinstance(record_payload, Mapping):
            raise ValueError(f"{path}:{line_no}: record must be a JSON object")
        features.update(build_real_calibration_features(RealMetaRecord.from_dict(record_payload), now=now, recency_tau=recency_tau))
    raw_features = row.get("features", {})
    if not isinstance(raw_features, Mapping):
        raise ValueError(f"{path}:{line_no}: features must be a JSON object")
    features.update({str(key): float(value) for key, value in raw_features.items()})
    return features


def _required_float(row: Mapping[str, Any], keys: tuple[str, ...], path: Path, line_no: int) -> float:
    for key in keys:
        if key in row:
            return float(row[key])
    raise ValueError(f"{path}:{line_no}: missing one of {', '.join(keys)}")


if __name__ == "__main__":
    raise SystemExit(main())
