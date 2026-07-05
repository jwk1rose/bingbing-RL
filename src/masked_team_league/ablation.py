from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

from .reports import build_league_round_report, red_line_violations


V4_REQUIRED_ABLATION_VARIANTS = (
    "baseline",
    "no_position_features",
    "no_equipment_stars",
    "no_future_feasibility_mask",
    "no_underdog_objective",
    "no_mask_ambiguity",
    "no_real_calibration",
    "no_active_perception",
)
ABLATION_SUITE_REPORT_SCHEMA_VERSION = "ablation_suite_report.v1"


@dataclass(frozen=True)
class AblationVariantReport:
    variant_id: str
    round_dir: str
    key_metrics: Mapping[str, float | int]
    red_line_violations: tuple[str, ...]
    report: Mapping[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AblationSuiteReport:
    suite_id: str
    date: str
    baseline_variant: str
    variants: tuple[str, ...]
    missing_required_variants: tuple[str, ...]
    variant_reports: Mapping[str, AblationVariantReport]
    deltas_vs_baseline: Mapping[str, Mapping[str, float]]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = ABLATION_SUITE_REPORT_SCHEMA_VERSION
        payload["module"] = "AblationSuiteReport"
        return payload


@dataclass(frozen=True)
class AblationExperimentVariant:
    variant_id: str
    round_id: str
    round_dir: str
    command: tuple[str, ...]
    description: str
    control_status: str
    implemented_controls: tuple[str, ...] = ()
    metadata_only_controls: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "round_id": self.round_id,
            "round_dir": self.round_dir,
            "command": list(self.command),
            "description": self.description,
            "control_status": self.control_status,
            "implemented_controls": list(self.implemented_controls),
            "metadata_only_controls": list(self.metadata_only_controls),
        }


@dataclass(frozen=True)
class AblationExperimentPlan:
    schema_version: str
    suite_id: str
    root_dir: str
    baseline_variant: str
    required_variants: tuple[str, ...]
    missing_required_variants: tuple[str, ...]
    variants: tuple[AblationExperimentVariant, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "root_dir": self.root_dir,
            "baseline_variant": self.baseline_variant,
            "required_variants": list(self.required_variants),
            "missing_required_variants": list(self.missing_required_variants),
            "variants": [variant.to_json_dict() for variant in self.variants],
        }


def build_v4_ablation_experiment_plan(
    root_dir: Path | str,
    *,
    backend: str,
    heroes_json: Path | str,
    decoded_dir: Path | str | None = None,
    real_meta_db_jsonl: Path | str | None = None,
    suite_id: str = "v4_ablation_suite",
    variants: tuple[str, ...] | list[str] | None = None,
    teams: int = 3,
    defenses: int = 20,
    attacks_per_defense: int = 200,
    oracle_top_k: int = 20,
    defense_roster_candidates: int = 8,
    defense_masks_per_roster: int = 2,
    defense_max_masks_per_roster: int = 128,
    active_sim_keep: int = 32,
    active_real_keep: int = 0,
    seed: int = 2026070501,
    extra_args: tuple[str, ...] | list[str] = (),
) -> AblationExperimentPlan:
    selected_variants = tuple(variants) if variants else V4_REQUIRED_ABLATION_VARIANTS
    unknown = tuple(variant for variant in selected_variants if variant not in V4_REQUIRED_ABLATION_VARIANTS)
    if unknown:
        raise ValueError(f"unknown ablation variant(s): {', '.join(unknown)}")
    if len(set(selected_variants)) != len(selected_variants):
        raise ValueError("duplicate ablation variants are not allowed")

    root = Path(root_dir)
    plan_variants: list[AblationExperimentVariant] = []
    for index, variant_id in enumerate(selected_variants):
        spec = _variant_control_spec(variant_id)
        round_dir = root / variant_id
        command = _league_round_command(
            backend=backend,
            heroes_json=heroes_json,
            decoded_dir=decoded_dir,
            real_meta_db_jsonl=real_meta_db_jsonl,
            out_dir=round_dir,
            round_id=variant_id,
            teams=teams,
            defenses=defenses,
            attacks_per_defense=attacks_per_defense,
            oracle_top_k=oracle_top_k,
            defense_roster_candidates=_variant_int(
                variant_id,
                "defense_roster_candidates",
                defense_roster_candidates,
            ),
            defense_masks_per_roster=_variant_int(
                variant_id,
                "defense_masks_per_roster",
                defense_masks_per_roster,
            ),
            defense_max_masks_per_roster=_variant_int(
                variant_id,
                "defense_max_masks_per_roster",
                defense_max_masks_per_roster,
            ),
            active_sim_keep=_variant_int(variant_id, "active_sim_keep", active_sim_keep),
            active_real_keep=_variant_int(variant_id, "active_real_keep", active_real_keep),
            attack_roles=spec.get("attack_roles"),
            defense_roles=spec.get("defense_roles"),
            variant_args=tuple(spec.get("run_args", ())),
            seed=seed + index,
            extra_args=extra_args,
        )
        implemented = tuple(spec.get("implemented_controls", ()))
        metadata_only = tuple(spec.get("metadata_only_controls", ()))
        if implemented and metadata_only:
            control_status = "mixed"
        elif implemented:
            control_status = "implemented"
        elif metadata_only:
            control_status = "metadata_only"
        else:
            control_status = "baseline"
        plan_variants.append(
            AblationExperimentVariant(
                variant_id=variant_id,
                round_id=variant_id,
                round_dir=str(round_dir),
                command=command,
                description=str(spec["description"]),
                implemented_controls=implemented,
                metadata_only_controls=metadata_only,
                control_status=control_status,
            )
        )

    missing = tuple(variant for variant in V4_REQUIRED_ABLATION_VARIANTS if variant not in selected_variants)
    return AblationExperimentPlan(
        schema_version="v4_ablation_experiment_plan.v1",
        suite_id=suite_id,
        root_dir=str(root),
        baseline_variant="baseline",
        required_variants=V4_REQUIRED_ABLATION_VARIANTS,
        missing_required_variants=missing,
        variants=tuple(plan_variants),
    )


def build_ablation_suite_report(
    variant_round_dirs: Mapping[str, Path | str],
    *,
    baseline_variant: str,
    date: str = "unknown",
    suite_id: str = "v4_ablation_suite",
) -> AblationSuiteReport:
    if baseline_variant not in variant_round_dirs:
        raise ValueError(f"baseline variant {baseline_variant!r} is not present")
    variant_reports: dict[str, AblationVariantReport] = {}
    for variant_id, round_dir in variant_round_dirs.items():
        path = Path(round_dir)
        report_payload = build_league_round_report(path, date=date).to_json_dict()
        variant_reports[variant_id] = AblationVariantReport(
            variant_id=variant_id,
            round_dir=str(path),
            key_metrics=extract_ablation_key_metrics(report_payload),
            red_line_violations=tuple(red_line_violations(report_payload)),
            report=report_payload,
        )
    baseline_metrics = variant_reports[baseline_variant].key_metrics
    deltas: dict[str, dict[str, float]] = {}
    for variant_id, variant_report in variant_reports.items():
        if variant_id == baseline_variant:
            continue
        deltas[variant_id] = _metric_deltas(baseline_metrics, variant_report.key_metrics)
    missing = tuple(variant for variant in V4_REQUIRED_ABLATION_VARIANTS if variant not in variant_reports)
    return AblationSuiteReport(
        suite_id=suite_id,
        date=date,
        baseline_variant=baseline_variant,
        variants=tuple(variant_reports),
        missing_required_variants=missing,
        variant_reports=variant_reports,
        deltas_vs_baseline=deltas,
    )


def _league_round_command(
    *,
    backend: str,
    heroes_json: Path | str,
    decoded_dir: Path | str | None,
    real_meta_db_jsonl: Path | str | None,
    out_dir: Path,
    round_id: str,
    teams: int,
    defenses: int,
    attacks_per_defense: int,
    oracle_top_k: int,
    defense_roster_candidates: int,
    defense_masks_per_roster: int,
    defense_max_masks_per_roster: int,
    active_sim_keep: int,
    active_real_keep: int,
    attack_roles: object,
    defense_roles: object,
    variant_args: tuple[str, ...],
    seed: int,
    extra_args: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "scripts/run_league_round.py",
        "--backend",
        backend,
        "--heroes-json",
        str(heroes_json),
    ]
    if decoded_dir is not None:
        command.extend(["--decoded-dir", str(decoded_dir)])
    if real_meta_db_jsonl is not None:
        command.extend(["--real-meta-db-jsonl", str(real_meta_db_jsonl)])
    command.extend(
        [
            "--out-dir",
            str(out_dir),
            "--round-id",
            round_id,
            "--teams",
            str(teams),
            "--defenses",
            str(defenses),
            "--attacks-per-defense",
            str(attacks_per_defense),
            "--oracle-top-k",
            str(oracle_top_k),
            "--defense-roster-candidates",
            str(defense_roster_candidates),
            "--defense-masks-per-roster",
            str(defense_masks_per_roster),
            "--defense-max-masks-per-roster",
            str(defense_max_masks_per_roster),
            "--active-sim-keep",
            str(active_sim_keep),
            "--active-real-keep",
            str(active_real_keep),
            "--seed",
            str(seed),
        ]
    )
    for role in _string_tuple(attack_roles):
        command.extend(["--attack-role", role])
    for role in _string_tuple(defense_roles):
        command.extend(["--defense-role", role])
    command.extend(str(arg) for arg in variant_args)
    command.extend(str(arg) for arg in extra_args)
    return tuple(command)


def _variant_control_spec(variant_id: str) -> Mapping[str, Any]:
    specs: dict[str, Mapping[str, Any]] = {
        "baseline": {
            "description": "Full v4 masked-team league round with all currently wired runtime controls enabled.",
        },
        "no_position_features": {
            "description": "Ablate position-aware features from model/search metadata.",
            "implemented_controls": ("position_features_disabled",),
            "run_args": ("--disable-position-features",),
        },
        "no_equipment_stars": {
            "description": "Ablate unique legend equipment star features from model/search metadata.",
            "implemented_controls": ("equipment_star_features_disabled",),
            "run_args": ("--disable-equipment-star-features",),
        },
        "no_future_feasibility_mask": {
            "description": "Ablate future-feasibility action-mask constraints from generation metadata.",
            "implemented_controls": ("future_feasibility_action_mask_disabled",),
            "run_args": ("--disable-future-feasibility-mask",),
        },
        "no_underdog_objective": {
            "description": "Disable underdog/exploiter role loop by running only main attack and defense roles.",
            "implemented_controls": ("role_loop_without_underdog",),
            "attack_roles": ("main",),
            "defense_roles": ("main",),
        },
        "no_mask_ambiguity": {
            "description": "Disable mask ambiguity search pressure by keeping exactly one mask candidate per defense roster.",
            "implemented_controls": ("single_mask_per_defense_roster",),
            "defense_roster_candidates": 1,
            "defense_masks_per_roster": 1,
            "defense_max_masks_per_roster": 1,
        },
        "no_real_calibration": {
            "description": "Ablate real calibration and real-distribution belief weighting metadata.",
            "implemented_controls": ("real_calibration_disabled",),
            "run_args": ("--disable-real-calibration",),
        },
        "no_active_perception": {
            "description": "Disable active perception query scheduling.",
            "implemented_controls": ("active_perception_disabled",),
            "active_sim_keep": 0,
            "active_real_keep": 0,
        },
    }
    return specs[variant_id]


def _variant_int(variant_id: str, key: str, default: int) -> int:
    value = _variant_control_spec(variant_id).get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"variant {variant_id!r} control {key!r} must be int")
    return value


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def extract_ablation_key_metrics(report: Mapping[str, Any]) -> dict[str, float | int]:
    attack = _mapping(report.get("attack_oracle"))
    defense = _mapping(report.get("defense_oracle"))
    league = _mapping(report.get("league"))
    underdog = _mapping(report.get("underdog"))
    active_queries = report.get("active_queries")
    if not isinstance(active_queries, list):
        active_queries = []
    metrics: dict[str, float | int] = {
        "sim_games": int(report.get("sim_games", 0) or 0),
        "real_matches": int(report.get("real_matches", 0) or 0),
        "attack_top1": _float(attack.get("top1")),
        "attack_top5_hit": _float(attack.get("top5_hit")),
        "defense_attack_success": _float(defense.get("attack_success")),
        "defense_ambiguity": _float(defense.get("ambiguity")),
        "league_attack_pool": int(league.get("attack_pool", 0) or 0),
        "league_defense_pool": int(league.get("defense_pool", 0) or 0),
        "league_clusters": int(league.get("clusters", 0) or 0),
        "active_query_count": int(league.get("active_query_count", len(active_queries)) or 0),
        "underdog_samples": int(underdog.get("samples", 0) or 0),
        "underdog_success_rate": _float(underdog.get("success_rate")),
        "failure_case_count": len(report.get("failure_cases") or ()),
    }
    _copy_metric(metrics, attack, "belief_expected_mean", "attack_expected_match_win_mean")
    _copy_metric(metrics, attack, "belief_worst_case_mean", "attack_worst_case_match_win_mean")
    _copy_metric(metrics, attack, "belief_worst_case_min", "attack_worst_case_match_win_min")
    _copy_metric(metrics, attack, "backup_attack_mean", "attack_backup_mean")
    _copy_metric(metrics, attack, "belief_case_mean", "attack_belief_case_mean")
    _copy_metric(metrics, attack, "underdog_gap_mean", "attack_underdog_gap_mean")
    _copy_metric(metrics, attack, "underdog_gap_max", "attack_underdog_gap_max")
    _copy_metric(metrics, attack, "underdog_residual_bonus_mean", "attack_underdog_residual_bonus_mean")
    _copy_metric(metrics, defense, "estimated_break_rate", "defense_estimated_break_rate")
    _copy_metric(metrics, defense, "estimated_survival_rate", "defense_estimated_survival_rate")
    _copy_metric(metrics, defense, "best_response_break_rate", "defense_best_response_break_rate")
    _copy_metric(metrics, defense, "hidden_count_mean", "defense_hidden_count_mean")
    _copy_metric(metrics, defense, "backup_defense_mean", "defense_backup_mean")
    _copy_metric(metrics, defense, "underdog_gap_mean", "defense_underdog_gap_mean")
    _copy_metric(metrics, defense, "underdog_gap_max", "defense_underdog_gap_max")
    _copy_metric(metrics, defense, "underdog_residual_bonus_mean", "defense_underdog_residual_bonus_mean")
    return metrics


def _metric_deltas(
    baseline_metrics: Mapping[str, float | int],
    variant_metrics: Mapping[str, float | int],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, value in variant_metrics.items():
        baseline = baseline_metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(baseline, (int, float)):
            deltas[key] = round(float(value) - float(baseline), 12)
    return deltas


def _copy_metric(
    out: dict[str, float | int],
    source: Mapping[str, Any],
    source_key: str,
    target_key: str,
) -> None:
    value = source.get(source_key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out[target_key] = float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0
