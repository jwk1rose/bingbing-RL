#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from masked_team_league.reporting.ablation import (
    V4_REQUIRED_ABLATION_VARIANTS,
    build_ablation_suite_report,
    build_v4_ablation_experiment_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and optionally execute the v4 masked-team league ablation experiment plan."
    )
    parser.add_argument("--root-dir", type=Path, required=True, help="Directory containing one round dir per variant.")
    parser.add_argument("--out-plan", type=Path, required=True, help="JSON path for the generated execution plan.")
    parser.add_argument("--out-report", type=Path, default=None, help="Optional suite report JSON path after execution.")
    parser.add_argument("--backend", default="http://127.0.0.1:18281")
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--real-meta-db-jsonl", type=Path, default=None)
    parser.add_argument("--suite-id", default="v4_ablation_suite")
    parser.add_argument(
        "--variant",
        action="append",
        choices=V4_REQUIRED_ABLATION_VARIANTS,
        default=None,
        help="Run only this variant. Repeat to build a subset; omitted means all required v4 variants.",
    )
    parser.add_argument("--teams", type=int, choices=(3, 5), default=3)
    parser.add_argument("--defenses", type=int, default=20)
    parser.add_argument("--attacks-per-defense", type=int, default=200)
    parser.add_argument("--oracle-top-k", type=int, default=20)
    parser.add_argument("--defense-roster-candidates", type=int, default=8)
    parser.add_argument("--defense-masks-per-roster", type=int, default=2)
    parser.add_argument("--defense-max-masks-per-roster", type=int, default=128)
    parser.add_argument("--active-sim-keep", type=int, default=32)
    parser.add_argument("--active-real-keep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026070501)
    parser.add_argument("--date", default="unknown")
    parser.add_argument("--execute", action="store_true", help="Execute each planned run_league_round command.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue executing later variants if one variant command fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = build_v4_ablation_experiment_plan(
        args.root_dir,
        backend=args.backend,
        heroes_json=args.heroes_json,
        decoded_dir=args.decoded_dir,
        real_meta_db_jsonl=args.real_meta_db_jsonl,
        suite_id=args.suite_id,
        variants=tuple(args.variant) if args.variant else None,
        teams=args.teams,
        defenses=args.defenses,
        attacks_per_defense=args.attacks_per_defense,
        oracle_top_k=args.oracle_top_k,
        defense_roster_candidates=args.defense_roster_candidates,
        defense_masks_per_roster=args.defense_masks_per_roster,
        defense_max_masks_per_roster=args.defense_max_masks_per_roster,
        active_sim_keep=args.active_sim_keep,
        active_real_keep=args.active_real_keep,
        seed=args.seed,
    )
    plan_payload = plan.to_json_dict()
    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    args.out_plan.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    execution_results: list[dict[str, object]] = []
    if args.execute:
        for variant in plan.variants:
            result = subprocess.run(list(variant.command), check=False)
            execution_results.append(
                {
                    "variant_id": variant.variant_id,
                    "round_dir": variant.round_dir,
                    "returncode": result.returncode,
                }
            )
            if result.returncode != 0 and not args.continue_on_failure:
                _print_summary(args.out_plan, plan_payload, execution_results, args.out_report)
                return result.returncode

    if args.out_report is not None:
        variant_round_dirs = {variant.variant_id: Path(variant.round_dir) for variant in plan.variants}
        report = build_ablation_suite_report(
            variant_round_dirs,
            baseline_variant="baseline",
            date=args.date,
            suite_id=args.suite_id,
        ).to_json_dict()
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _print_summary(args.out_plan, plan_payload, execution_results, args.out_report)
    return 0


def _print_summary(
    out_plan: Path,
    plan_payload: dict[str, object],
    execution_results: list[dict[str, object]],
    out_report: Path | None,
) -> None:
    print(
        json.dumps(
            {
                "out_plan": str(out_plan),
                "out_report": str(out_report) if out_report is not None else None,
                "variants": [variant["variant_id"] for variant in plan_payload["variants"]],
                "missing_required_variants": plan_payload["missing_required_variants"],
                "executed": bool(execution_results),
                "execution_results": execution_results,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
