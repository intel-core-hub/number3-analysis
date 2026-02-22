"""Monitoring data access helpers (UI-independent)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def has_table(db_path: Path, table_name: str) -> bool:
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def has_column(db_path: Path, table_name: str, column_name: str) -> bool:
    if not has_table(db_path, table_name):
        return False
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cur.fetchall()]
            return column_name in columns
    except Exception:
        return False


def get_schema_diagnostics(db_path: Path) -> dict[str, list[str]]:
    diagnostics: dict[str, list[str]] = {
        "critical": [],
        "warning": [],
    }

    # results table checks
    if not has_table(db_path, "prediction_results"):
        diagnostics["critical"].append("prediction_results テーブルが見つかりません。")
    else:
        for col in ("round_no", "created_at"):
            if not has_column(db_path, "prediction_results", col):
                diagnostics["warning"].append(
                    f"prediction_results.{col} が不足しています（表示は継続します）。"
                )

    # predictions table checks
    if not has_table(db_path, "predictions"):
        diagnostics["critical"].append("predictions テーブルが見つかりません。")
    else:
        for col in ("id", "created_at"):
            if not has_column(db_path, "predictions", col):
                diagnostics["warning"].append(
                    f"predictions.{col} が不足しています（表示は継続します）。"
                )

    if not has_table(db_path, "numbers3_draws"):
        diagnostics["critical"].append("numbers3_draws テーブルが見つかりません。")

    return diagnostics


def get_schema_warnings(db_path: Path) -> list[str]:
    diagnostics = get_schema_diagnostics(db_path)
    return diagnostics["critical"] + diagnostics["warning"]


def _load_ordered_table(
    db_path: Path,
    table_name: str,
    order_by: str,
    start_iso: Optional[str] = None,
) -> pd.DataFrame:
    if not has_table(db_path, table_name):
        return pd.DataFrame()

    if has_column(db_path, table_name, order_by):
        query = f"SELECT * FROM {table_name} ORDER BY {order_by} ASC"
    else:
        query = f"SELECT * FROM {table_name}"

    params: tuple[str, ...] | tuple[()] = ()
    if start_iso and has_column(db_path, table_name, "created_at"):
        query = (
            f"SELECT * FROM {table_name} "
            "WHERE created_at >= ? "
            + (f"ORDER BY {order_by} ASC" if has_column(db_path, table_name, order_by) else "")
        )
        params = (start_iso,)

    try:
        with _connect(db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def load_results(db_path: Path, start_iso: Optional[str] = None) -> pd.DataFrame:
    return _load_ordered_table(
        db_path=db_path,
        table_name="prediction_results",
        order_by="round_no",
        start_iso=start_iso,
    )


def load_predictions(db_path: Path, start_iso: Optional[str] = None) -> pd.DataFrame:
    return _load_ordered_table(
        db_path=db_path,
        table_name="predictions",
        order_by="id",
        start_iso=start_iso,
    )


def load_draws(db_path: Path) -> pd.DataFrame:
    if not has_table(db_path, "numbers3_draws"):
        return pd.DataFrame()

    try:
        with _connect(db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM numbers3_draws ORDER BY round_no ASC",
                conn,
            )
    except Exception:
        return pd.DataFrame()


def load_backtest_tracking(path: Path = Path("results/auto_backtest_tracking.csv")) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_schema_diagnostics_history(
    path: Path = Path("results/schema_diagnostics_history.csv"),
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "critical_count", "warning_count", "total_count", "status"])
    try:
        history = pd.read_csv(path)
        expected = {"date", "critical_count", "warning_count", "total_count", "status"}
        if not expected.issubset(set(history.columns)):
            return pd.DataFrame(columns=["date", "critical_count", "warning_count", "total_count", "status"])
        return history
    except Exception:
        return pd.DataFrame(columns=["date", "critical_count", "warning_count", "total_count", "status"])


def record_schema_diagnostics(
    schema_diagnostics: dict[str, list[str]],
    path: Path = Path("results/schema_diagnostics_history.csv"),
) -> pd.DataFrame:
    critical = len(schema_diagnostics.get("critical", []))
    warning = len(schema_diagnostics.get("warning", []))
    total = critical + warning
    status = "normal" if total == 0 else ("critical" if critical > 0 else "warning")
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    history = load_schema_diagnostics_history(path)
    if history.empty:
        history = pd.DataFrame(columns=["date", "critical_count", "warning_count", "total_count", "status"])

    today_mask = history["date"] == today
    if today_mask.any():
        history.loc[today_mask, "critical_count"] = critical
        history.loc[today_mask, "warning_count"] = warning
        history.loc[today_mask, "total_count"] = total
        history.loc[today_mask, "status"] = status
    else:
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "date": today,
                            "critical_count": critical,
                            "warning_count": warning,
                            "total_count": total,
                            "status": status,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    history = history.sort_values("date").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False, encoding="utf-8-sig")
    return history


def apply_schema_auto_fix(db_path: Path) -> dict[str, object]:
    """Apply automatic schema repair for supported tables.

    Returns a dict that includes repair report and post-fix diagnostics.
    """
    from src.database import Numbers3Database

    db = Numbers3Database(db_path=db_path)
    repair_report = db.auto_repair_all_schemas()
    diagnostics_after = get_schema_diagnostics(db_path)

    failed_tables = repair_report.get("failed_tables", [])
    repaired_tables = repair_report.get("repaired_tables", [])
    applied = bool(repaired_tables) or not diagnostics_after.get("critical", [])

    return {
        "applied": applied,
        "repair_report": repair_report,
        "diagnostics_after": diagnostics_after,
        "failed_tables": failed_tables,
        "repaired_tables": repaired_tables,
    }
