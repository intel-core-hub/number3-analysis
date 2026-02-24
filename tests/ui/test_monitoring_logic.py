"""tests/ui/test_monitoring_logic.py — monitoring logic unit tests."""
from __future__ import annotations

import pandas as pd

from src.ui.monitoring_logic import (
    build_schema_remediation_steps,
    extract_remediation_commands,
    order_remediation_commands,
    MONITORING_THRESHOLDS,
    calculate_core_kpis,
    calculate_recent_alert_metrics,
    filter_results_since_days,
    summarize_schema_diagnostics,
)


def test_calculate_core_kpis_empty_inputs():
    results = pd.DataFrame()
    draws = pd.DataFrame()

    kpis = calculate_core_kpis(results, draws)

    assert kpis["total_rounds"] == 0
    assert kpis["n_draws"] == 0
    assert kpis["hit_rate"] == 0.0
    assert kpis["total_profit"] == 0
    assert kpis["roi"] == 0.0


def test_calculate_core_kpis_with_data():
    results = pd.DataFrame(
        {
            "n_hits": [1, 0, 0],
            "profit": [100, -200, 50],
            "total_cost": [300, 300, 300],
        }
    )
    draws = pd.DataFrame({"round_no": [1, 2, 3, 4]})

    kpis = calculate_core_kpis(results, draws)

    assert kpis["total_rounds"] == 3
    assert kpis["n_draws"] == 4
    assert kpis["hit_rate"] == (1 / 3) * 100
    assert kpis["total_profit"] == -50
    assert kpis["roi"] == (-50 / 900) * 100


def test_calculate_recent_alert_metrics_counts_and_weekly_profit():
    now = pd.Timestamp.now()
    results = pd.DataFrame(
        {
            "round_no": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
            "n_hits": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "profit": [-5000, -4000, 200, 100, -100, 50, 0, 0, 0, 0],
            "created_at": [
                now,
                now,
                now,
                now,
                now,
                now,
                now,
                now - pd.Timedelta(days=10),
                now - pd.Timedelta(days=12),
                now - pd.Timedelta(days=15),
            ],
        }
    )

    metrics = calculate_recent_alert_metrics(results)

    assert metrics["consecutive_losses"] == 9
    assert metrics["weekly_profit"] == -8750
    assert MONITORING_THRESHOLDS.hit_rate_window == 10


def test_calculate_recent_alert_metrics_without_round_no_is_safe():
    results = pd.DataFrame(
        {
            "n_hits": [0, 1, 0, 0],
            "profit": [100, -200, 50, -30],
        }
    )

    metrics = calculate_recent_alert_metrics(results)

    assert metrics["consecutive_losses"] == 3
    assert isinstance(metrics["weekly_profit"], float)


def test_summarize_schema_diagnostics_normal():
    summary = summarize_schema_diagnostics({"critical": [], "warning": []})

    assert summary["status"] == "normal"
    assert summary["total"] == 0
    assert summary["critical"] == 0
    assert summary["warning"] == 0


def test_summarize_schema_diagnostics_warning_only():
    summary = summarize_schema_diagnostics({"critical": [], "warning": ["a", "b"]})

    assert summary["status"] == "warning"
    assert summary["total"] == 2
    assert summary["critical"] == 0
    assert summary["warning"] == 2


def test_summarize_schema_diagnostics_with_critical():
    summary = summarize_schema_diagnostics({"critical": ["x"], "warning": ["y"]})

    assert summary["status"] == "critical"
    assert summary["total"] == 2
    assert summary["critical"] == 1
    assert summary["warning"] == 1


def test_build_schema_remediation_steps_empty_for_no_critical():
    steps = build_schema_remediation_steps({"critical": [], "warning": ["a"]})

    assert steps == []


def test_build_schema_remediation_steps_for_missing_tables():
    steps = build_schema_remediation_steps(
        {
            "critical": [
                "prediction_results テーブルが見つかりません。",
                "numbers3_draws テーブルが見つかりません。",
            ],
            "warning": [],
        }
    )

    assert len(steps) == 2
    assert any("prediction_results" in step for step in steps)
    assert any("numbers3_draws" in step for step in steps)
    assert any("python main_automation.py" in step for step in steps)


def test_build_schema_remediation_steps_for_predictions_has_command():
    steps = build_schema_remediation_steps(
        {
            "critical": ["predictions テーブルが見つかりません。"],
            "warning": [],
        }
    )

    assert len(steps) == 1
    assert "python scripts/run_predict_notify.py" in steps[0]


def test_extract_remediation_commands_deduplicates_and_filters():
    steps = [
        "`prediction_results` の復旧: `python main_automation.py` を実行してください。",
        "`numbers3_draws` の復旧: `python main_automation.py` を実行してください。",
        "`predictions` の復旧: `python scripts/run_predict_notify.py` を実行してください。",
        "補足: DB確認のみ",
    ]

    commands = extract_remediation_commands(steps)

    assert commands == [
        "python main_automation.py",
        "python scripts/run_predict_notify.py",
    ]


def test_order_remediation_commands_prefers_main_then_predict():
    commands = [
        "python scripts/run_predict_notify.py",
        "python main_automation.py",
    ]

    ordered = order_remediation_commands(commands)

    assert ordered == [
        "python main_automation.py",
        "python scripts/run_predict_notify.py",
    ]


def test_order_remediation_commands_keeps_unknown_after_preferred():
    commands = [
        "python custom_repair.py",
        "python scripts/run_predict_notify.py",
    ]

    ordered = order_remediation_commands(commands)

    assert ordered == [
        "python scripts/run_predict_notify.py",
        "python custom_repair.py",
    ]


def test_filter_results_since_days_with_invalid_timestamps_is_safe():
    results = pd.DataFrame(
        {
            "created_at": ["not-a-date", None, "2026-02-20T00:00:00"],
            "profit": [1, 2, 3],
        }
    )

    filtered = filter_results_since_days(results, days=30)

    assert isinstance(filtered, pd.DataFrame)
    assert len(filtered) <= len(results)
