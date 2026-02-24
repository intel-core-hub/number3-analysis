"""Monitoring dashboard logic helpers (UI-independent)."""
from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class MonitoringThresholds:
    hit_rate_window: int = 10
    severe_consecutive_losses_in_window: int = 8
    warning_consecutive_losses_in_window: int = 5
    weekly_loss_warning_yen: int = -10_000
    weekly_window_days: int = 7
    critical_roi_pct: float = -30.0
    theoretical_hit_rate_pct: float = 1.0
    backtest_warning_hit_rate_pct: float = 0.5


MONITORING_THRESHOLDS = MonitoringThresholds()


def calculate_core_kpis(results: pd.DataFrame, draws: pd.DataFrame) -> dict[str, float | int]:
    total_rounds = len(results)
    n_draws = len(draws)

    if not results.empty and "n_hits" in results.columns:
        hit_rate = float((results["n_hits"] > 0).mean() * 100)
    else:
        hit_rate = 0.0

    total_profit = int(results["profit"].sum()) if not results.empty and "profit" in results.columns else 0
    total_cost = int(results["total_cost"].sum()) if not results.empty and "total_cost" in results.columns else 0
    roi = float(total_profit / total_cost * 100) if total_cost > 0 else 0.0

    return {
        "total_rounds": total_rounds,
        "n_draws": n_draws,
        "hit_rate": hit_rate,
        "total_profit": total_profit,
        "total_cost": total_cost,
        "roi": roi,
    }


def filter_results_since_days(results: pd.DataFrame, days: int) -> pd.DataFrame:
    if results.empty or "created_at" not in results.columns:
        return results

    recent = results.copy()
    recent["created_at"] = pd.to_datetime(recent["created_at"], errors="coerce", format="ISO8601")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    return recent[recent["created_at"] >= cutoff]


def calculate_recent_alert_metrics(
    results: pd.DataFrame,
    thresholds: MonitoringThresholds = MONITORING_THRESHOLDS,
) -> dict[str, float | int]:
    if results.empty or "n_hits" not in results.columns:
        return {
            "consecutive_losses": 0,
            "weekly_profit": 0.0,
        }

    if "round_no" in results.columns:
        recent_window = results.sort_values("round_no", ascending=False).head(thresholds.hit_rate_window)
    else:
        recent_window = results.tail(thresholds.hit_rate_window)
    consecutive_losses = int((recent_window["n_hits"] == 0).sum())

    recent_week = filter_results_since_days(results, thresholds.weekly_window_days)
    weekly_profit = float(recent_week["profit"].sum()) if not recent_week.empty and "profit" in recent_week.columns else 0.0

    return {
        "consecutive_losses": consecutive_losses,
        "weekly_profit": weekly_profit,
    }


def summarize_schema_diagnostics(schema_diagnostics: dict[str, list[str]]) -> dict[str, int | str]:
    critical = schema_diagnostics.get("critical", [])
    warning = schema_diagnostics.get("warning", [])
    total = len(critical) + len(warning)

    if total == 0:
        status = "normal"
    elif len(critical) > 0:
        status = "critical"
    else:
        status = "warning"

    return {
        "status": status,
        "total": total,
        "critical": len(critical),
        "warning": len(warning),
    }


def build_schema_remediation_steps(schema_diagnostics: dict[str, list[str]]) -> list[str]:
    critical = schema_diagnostics.get("critical", [])
    if not critical:
        return []

    steps: list[str] = []

    if any("prediction_results" in message for message in critical):
        steps.append(
            "`prediction_results` の復旧: `python main_automation.py` を実行して日次処理を再実行し、結果テーブルを再生成してください。"
        )

    if any("predictions" in message for message in critical):
        steps.append(
            "`predictions` の復旧: `python scripts/run_predict_notify.py` を実行して予測生成を再実行してください。"
        )

    if any("numbers3_draws" in message for message in critical):
        steps.append(
            "`numbers3_draws` の復旧: `python main_automation.py` を実行してスクレイピング更新を再実行してください。"
        )

    if not steps:
        steps.append("DB スキーマを確認し、不足テーブルを復旧したうえで監視画面を再読み込みしてください。")

    return steps


def extract_remediation_commands(steps: list[str]) -> list[str]:
    commands: list[str] = []
    pattern = re.compile(r"`([^`]+)`")

    for step in steps:
        for token in pattern.findall(step):
            candidate = token.strip()
            if candidate.startswith("python ") and candidate not in commands:
                commands.append(candidate)

    return commands


def order_remediation_commands(commands: list[str]) -> list[str]:
    preferred_order = [
        "python main_automation.py",
        "python scripts/run_predict_notify.py",
    ]

    ordered: list[str] = []
    for preferred in preferred_order:
        if preferred in commands and preferred not in ordered:
            ordered.append(preferred)

    for command in commands:
        if command not in ordered:
            ordered.append(command)

    return ordered
