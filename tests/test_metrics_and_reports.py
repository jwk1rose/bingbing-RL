from __future__ import annotations

from masked_team_league.metrics import (
    DailyTrainingReport,
    binary_auc,
    brier_score,
    expected_calibration_error,
    precision_at_k,
    recall_at_k,
)


def test_binary_metrics_match_expected_values():
    labels = (0.0, 0.0, 1.0, 1.0)
    scores = (0.1, 0.4, 0.35, 0.9)

    assert round(binary_auc(labels, scores), 6) == 0.75
    assert round(brier_score(labels, scores), 6) == 0.150625
    assert round(precision_at_k(labels, scores, k=2), 6) == 0.5
    assert round(recall_at_k(labels, scores, k=2), 6) == 0.5


def test_expected_calibration_error_uses_confidence_bins():
    labels = (0.0, 1.0, 1.0, 0.0)
    scores = (0.1, 0.8, 0.9, 0.2)

    assert expected_calibration_error(labels, scores, bins=2) < 0.25


def test_daily_training_report_json_contains_required_fields():
    report = DailyTrainingReport(
        date="2026-07-05",
        sim_games=100,
        real_matches=10,
        single_model={"brier": 0.1, "ece": 0.05, "auc": 0.8},
        attack_oracle={"top1": 0.7, "top5_hit": 0.9},
        defense_oracle={"attack_success": 0.4, "ambiguity": 1.2},
        league={"attack_pool": 4, "defense_pool": 5, "clusters": 3},
        underdog={"samples": 8, "success_rate": 0.25},
        active_queries=[{"query_id": "q1"}],
        failure_cases=[],
    )

    payload = report.to_json_dict()

    assert payload["date"] == "2026-07-05"
    assert payload["single_model"]["ece"] == 0.05
    assert payload["active_queries"][0]["query_id"] == "q1"
