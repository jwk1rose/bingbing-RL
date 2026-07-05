from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import sys

from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.reporting.metrics import brier_score
from masked_team_league.domain import observe_defense
from masked_team_league.real_platform.calibration import (
    RealCalibrationModel,
    RealMetaDB,
    RealMetaRecord,
    build_real_calibration_features,
    build_real_calibration_samples_from_artifacts,
    build_real_calibration_validation_report,
    build_version_drift_report,
    ingest_active_real_query_feedback,
    ingest_league_round_real_meta,
    time_decay_weight,
)


def test_real_meta_db_records_and_queries_observation(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=31)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    observation = observe_defense(defense)
    record = RealMetaRecord.from_match(
        observation=observation,
        full_defense_if_available=defense,
        attack_plan=attack,
        lane_results=(1.0, 0.0, 1.0),
        match_result=1.0,
        rank_segment="top",
        server="unit",
        season="S28",
        timestamp=1000.0,
    )
    db = RealMetaDB()

    db.add(record)
    rows = db.by_observation_hash(observation.hash())

    assert rows == (record,)
    assert rows[0].hidden_slots == observation.hidden_slots
    assert rows[0].unique_equip_stars


def test_real_meta_db_persists_jsonl_records(tmp_path, loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=32)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    observation = observe_defense(defense)
    record = RealMetaRecord.from_match(
        observation=observation,
        full_defense_if_available=defense,
        attack_plan=attack,
        lane_results=(0.0, 1.0, 1.0),
        match_result=1.0,
        rank_segment="top",
        server="unit",
        season="S28",
        timestamp=1234.5,
    )
    path = tmp_path / "real_meta.jsonl"

    db = RealMetaDB(path=path)
    db.add(record)
    reloaded = RealMetaDB.load(path)

    assert reloaded.all() == (record,)
    assert reloaded.by_observation_hash(observation.hash()) == (record,)
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_real_calibration_model_fit_improves_brier():
    sim_scores = (0.40, 0.45, 0.55, 0.60)
    labels = (0.0, 0.0, 1.0, 1.0)

    model = RealCalibrationModel.fit_platt(sim_scores, labels)
    calibrated = tuple(model.calibrate(score) for score in sim_scores)

    assert brier_score(labels, calibrated) < brier_score(labels, sim_scores)


def test_real_calibration_features_capture_real_context(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=3201)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    record = RealMetaRecord.from_match(
        observation=observe_defense(defense),
        full_defense_if_available=defense,
        attack_plan=attack,
        lane_results=(1.0, 0.0, 1.0),
        match_result=1.0,
        rank_segment="top",
        server="unit",
        season="S28",
        timestamp=900.0,
    )

    features = build_real_calibration_features(record, now=1000.0, recency_tau=100.0)

    assert features["hidden_fraction"] == 3 / 15
    assert features["hidden_slots_total"] == 3.0
    assert features["match_teams"] == 3.0
    assert features["visible_unique_star_mean"] > 0.0
    assert 0.0 < features["recency_weight"] < 1.0
    assert features["server:unit"] == 1.0
    assert features["season:S28"] == 1.0
    assert features["rank_segment:top"] == 1.0


def test_real_calibration_model_feature_fit_separates_same_sim_score():
    sim_scores = (0.5, 0.5, 0.5, 0.5)
    labels = (0.0, 0.0, 1.0, 1.0)
    feature_rows = (
        {"hidden_fraction": 0.0},
        {"hidden_fraction": 0.0},
        {"hidden_fraction": 1.0},
        {"hidden_fraction": 1.0},
    )

    base = RealCalibrationModel.fit_platt(sim_scores, labels)
    model = RealCalibrationModel.fit_feature_calibrator(
        sim_scores,
        labels,
        feature_rows,
        feature_names=("hidden_fraction",),
    )
    base_predictions = tuple(base.calibrate(score) for score in sim_scores)
    calibrated = tuple(model.calibrate(score, features) for score, features in zip(sim_scores, feature_rows))

    assert model.feature_weights is not None
    assert model.feature_weights["hidden_fraction"] > 0.0
    assert brier_score(labels, calibrated) < brier_score(labels, base_predictions)
    assert calibrated[0] < 0.25
    assert calibrated[-1] > 0.75


def test_build_real_calibration_validation_report_confirms_holdout_improvement(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"sim_probability": 0.8, "label": 0.0, "features": {"hidden_fraction": 0.0}},
                {"sim_probability": 0.8, "label": 0.0, "features": {"hidden_fraction": 0.0}},
                {"sim_probability": 0.8, "label": 1.0, "features": {"hidden_fraction": 1.0}},
                {"sim_probability": 0.8, "label": 1.0, "features": {"hidden_fraction": 1.0}},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    model_json = tmp_path / "calibration.json"
    model_json.write_text(
        json.dumps(
            {
                "model": RealCalibrationModel(
                    logit_scale=1.0,
                    bias=-5.0,
                    feature_weights={"hidden_fraction": 8.0},
                ).to_json_dict(),
                "base_model": RealCalibrationModel(logit_scale=1.0, bias=0.0).to_json_dict(),
            }
        ),
        encoding="utf-8",
    )

    report = build_real_calibration_validation_report(
        samples_jsonl=(holdout,),
        calibration_json=model_json,
        min_samples=4,
        min_brier_improvement=0.10,
        min_ece_improvement=0.10,
    )

    assert report["schema_version"] == "real_calibration_validation_report.v1"
    assert report["module"] == "RealCalibrationValidationReport"
    assert report["samples"] == 4
    assert report["raw_brier"] > report["calibrated_brier"]
    assert report["raw_ece"] > report["calibrated_ece"]
    assert report["brier_improvement"] > 0.10
    assert report["ece_improvement"] > 0.10
    assert report["red_line_violations"] == []
    assert report["production_ready"] is True


def test_build_real_calibration_validation_report_flags_no_holdout_improvement(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text(
        "\n".join(
            json.dumps({"sim_probability": 0.8, "label": label, "features": {"hidden_fraction": 0.0}})
            for label in (0.0, 0.0, 1.0, 1.0)
        )
        + "\n",
        encoding="utf-8",
    )
    model_json = tmp_path / "calibration.json"
    model_json.write_text(
        json.dumps({"model": RealCalibrationModel(logit_scale=1.0, bias=0.0).to_json_dict()}),
        encoding="utf-8",
    )

    report = build_real_calibration_validation_report(
        samples_jsonl=(holdout,),
        calibration_json=model_json,
        min_samples=8,
        min_brier_improvement=0.01,
        min_ece_improvement=0.01,
    )

    assert report["samples"] == 4
    assert "real_calibration_holdout_samples_low" in report["red_line_violations"]
    assert "real_calibration_brier_not_improved" in report["red_line_violations"]
    assert "real_calibration_ece_not_improved" in report["red_line_violations"]
    assert report["production_ready"] is False


def test_time_decay_weight_decreases_for_old_records():
    assert time_decay_weight(now=100.0, timestamp=95.0, tau=10.0) > time_decay_weight(now=100.0, timestamp=50.0, tau=10.0)


def test_ingest_league_round_real_meta_from_round_artifacts(tmp_path, loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=33)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    round_dir = tmp_path / "round_0001"
    round_dir.mkdir()
    (round_dir / "summary.json").write_text(json.dumps({"round_id": "round_0001"}), encoding="utf-8")
    (round_dir / "candidates.jsonl").write_text(
        json.dumps({"round_id": "round_0001", "attack_id": "atk-1", "defense_id": "def-1", "attack_plan": asdict(attack)})
        + "\n",
        encoding="utf-8",
    )
    (round_dir / "scored_defenses.jsonl").write_text(
        json.dumps({"defense_id": "def-1", "defense_plan": asdict(defense)}) + "\n",
        encoding="utf-8",
    )
    (round_dir / "oracle_results.jsonl").write_text(
        "".join(
            json.dumps({"request_id": f"round_0001-p000001-r{idx}", "status": "completed", "battle_result": result}) + "\n"
            for idx, result in enumerate((0, 1, 0), start=1)
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "real_meta.jsonl"

    summary = ingest_league_round_real_meta(
        round_dir,
        db_path,
        rank_segment="unit",
        server="oracle",
        season="S28",
        timestamp=123.0,
    )
    db = RealMetaDB.load(db_path)
    record = db.all()[0]

    assert summary.records_added == 1
    assert record.lane_results == (1.0, 0.0, 1.0)
    assert record.match_result > 0.99
    assert record.season == "S28"
    assert record.full_defense_if_available == defense


def test_ingest_active_real_query_feedback_into_real_meta_db(tmp_path, loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=3301)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    feedback_dir = tmp_path / "active_real_feedback"
    feedback_dir.mkdir()
    (feedback_dir / "summary.json").write_text(
        json.dumps({"round_dir": str(tmp_path / "round_0007"), "dispatched_pairs": 1}),
        encoding="utf-8",
    )
    (feedback_dir / "real_query_pairs.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q-real-1",
                "query_type": "underdog",
                "queue": "real",
                "attack_id": "atk-real-1",
                "defense_id": "def-real-1",
                "attack_success": 1.0,
                "round_win_rates": [1.0, 0.0, 1.0],
                "oracle_request_ids": ["rq-1", "rq-2", "rq-3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (feedback_dir / "attack_teacher.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q-real-1",
                "attack_id": "atk-real-1",
                "defense_id": "def-real-1",
                "attack_plan": asdict(attack),
                "source": "active_real_query",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (feedback_dir / "defense_teacher.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q-real-1",
                "defense_id": "def-real-1",
                "defense_plan": asdict(defense),
                "source": "active_real_query",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "real_meta.jsonl"

    summary = ingest_active_real_query_feedback(
        feedback_dir,
        db_path,
        rank_segment="top",
        server="oracle_backend",
        season="S29",
        timestamp=456.0,
    )
    db = RealMetaDB.load(db_path)
    record = db.all()[0]

    assert summary.records_added == 1
    assert summary.round_id == "round_0007"
    assert summary.source_kind == "active_real_query_feedback"
    assert record.full_defense_if_available == defense
    assert record.attack_plan == attack
    assert record.observation_hash == observe_defense(defense).hash()
    assert record.lane_results == (1.0, 0.0, 1.0)
    assert record.match_result == 1.0
    assert record.rank_segment == "top"
    assert record.server == "oracle_backend"
    assert record.season == "S29"


def test_build_real_calibration_samples_from_round_artifacts(tmp_path, loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=3302)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    round_dir = tmp_path / "round_0008"
    round_dir.mkdir()
    (round_dir / "summary.json").write_text(json.dumps({"round_id": "round_0008"}), encoding="utf-8")
    (round_dir / "candidates.jsonl").write_text(
        json.dumps(
            {
                "attack_id": "atk-8",
                "defense_id": "def-8",
                "predicted_score": 0.35,
                "surrogate_score": 0.45,
                "attack_plan": asdict(attack),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (round_dir / "scored_defenses.jsonl").write_text(
        json.dumps({"defense_id": "def-8", "defense_plan": asdict(defense)}) + "\n",
        encoding="utf-8",
    )
    (round_dir / "oracle_pairs.jsonl").write_text(
        json.dumps(
            {
                "attack_id": "atk-8",
                "defense_id": "def-8",
                "attack_success": 1.0,
                "round_win_rates": [1.0, 0.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "real_calibration_samples.jsonl"

    summary = build_real_calibration_samples_from_artifacts(
        out_jsonl=out_jsonl,
        round_dirs=(round_dir,),
        rank_segment="top",
        server="oracle_backend",
        season="S29",
        timestamp=789.0,
    )
    row = json.loads(out_jsonl.read_text(encoding="utf-8"))
    record = RealMetaRecord.from_dict(row["real_meta_record"])

    assert summary.samples_written == 1
    assert summary.skipped_pairs == 0
    assert summary.mean_label == 1.0
    assert summary.mean_sim_probability == 0.35
    assert row["sim_probability"] == 0.35
    assert row["label"] == 1.0
    assert row["prediction_source"] == "predicted_score"
    assert row["source_kind"] == "league_round_artifact"
    assert row["round_id"] == "round_0008"
    assert record.full_defense_if_available == defense
    assert record.attack_plan == attack
    assert record.lane_results == (1.0, 0.0, 1.0)


def test_build_real_calibration_samples_from_active_real_feedback_uses_source_round_prediction(tmp_path, loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=3303)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    round_dir = tmp_path / "round_0009"
    round_dir.mkdir()
    (round_dir / "summary.json").write_text(json.dumps({"round_id": "round_0009"}), encoding="utf-8")
    (round_dir / "candidates.jsonl").write_text(
        json.dumps(
            {
                "attack_id": "atk-real-9",
                "defense_id": "def-real-9",
                "predicted_score": 0.25,
                "attack_plan": asdict(attack),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (round_dir / "scored_defenses.jsonl").write_text(
        json.dumps({"defense_id": "def-real-9", "defense_plan": asdict(defense)}) + "\n",
        encoding="utf-8",
    )
    feedback_dir = tmp_path / "real_query_feedback"
    feedback_dir.mkdir()
    (feedback_dir / "summary.json").write_text(json.dumps({"round_dir": str(round_dir)}), encoding="utf-8")
    (feedback_dir / "real_query_pairs.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q-real-9",
                "attack_id": "atk-real-9",
                "defense_id": "def-real-9",
                "attack_success": 0.0,
                "round_win_rates": [0.0, 1.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (feedback_dir / "attack_teacher.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q-real-9",
                "attack_id": "atk-real-9",
                "defense_id": "def-real-9",
                "attack_plan": asdict(attack),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (feedback_dir / "defense_teacher.jsonl").write_text(
        json.dumps({"query_id": "q-real-9", "defense_id": "def-real-9", "defense_plan": asdict(defense)}) + "\n",
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "real_calibration_samples.jsonl"

    summary = build_real_calibration_samples_from_artifacts(
        out_jsonl=out_jsonl,
        active_real_feedback_dirs=(feedback_dir,),
        rank_segment="top",
        server="oracle_backend",
        season="S29",
        timestamp=790.0,
    )
    row = json.loads(out_jsonl.read_text(encoding="utf-8"))

    assert summary.samples_written == 1
    assert row["sim_probability"] == 0.25
    assert row["label"] == 0.0
    assert row["source_kind"] == "active_real_query_feedback"
    assert row["round_id"] == "round_0009"
    assert row["query_id"] == "q-real-9"


def test_build_version_drift_report_detects_shift(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=34)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))
    attack = generator.generate_attack_plan(fmt3)
    observation = observe_defense(defense)
    db = RealMetaDB()
    for season, result in (("S28", 1.0), ("S28", 1.0), ("S29", 0.0), ("S29", 0.0)):
        db.add(
            RealMetaRecord.from_match(
                observation=observation,
                full_defense_if_available=defense,
                attack_plan=attack,
                lane_results=(result, result, result),
                match_result=result,
                rank_segment="top",
                server="unit",
                season=season,
                timestamp=100.0,
            )
        )

    report = build_version_drift_report(db.all(), baseline_season="S28", current_season="S29", delta_threshold=0.5)

    assert report.baseline_records == 2
    assert report.current_records == 2
    assert report.match_result_delta == -1.0
    assert report.drift_detected


def test_ingest_real_calibration_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.ingest_real_calibration", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--active-real-feedback-dir" in result.stdout
    assert "--db-jsonl" in result.stdout
    assert "--drift-baseline-season" in result.stdout


def test_build_real_calibration_samples_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.build_real_calibration_samples", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--active-real-feedback-dir" in result.stdout
    assert "--out-jsonl" in result.stdout


def test_fit_real_feature_calibration_script_writes_model(tmp_path):
    samples_path = tmp_path / "real_feature_samples.jsonl"
    samples_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"sim_probability": 0.5, "label": 0.0, "features": {"hidden_fraction": 0.0}},
                {"sim_probability": 0.5, "label": 0.0, "features": {"hidden_fraction": 0.0}},
                {"sim_probability": 0.5, "label": 1.0, "features": {"hidden_fraction": 1.0}},
                {"sim_probability": 0.5, "label": 1.0, "features": {"hidden_fraction": 1.0}},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    out_json = tmp_path / "real_feature_calibrator.json"

    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.fit_real_feature_calibration",
            "--samples-jsonl",
            str(samples_path),
            "--out-json",
            str(out_json),
            "--feature",
            "hidden_fraction",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert payload["model"]["feature_weights"]["hidden_fraction"] > 0.0
    assert payload["metrics"]["feature_brier"] < payload["metrics"]["base_brier"]


def test_report_real_calibration_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.report_real_calibration_validation", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--samples-jsonl" in result.stdout
    assert "--calibration-json" in result.stdout
    assert "--out-report" in result.stdout
