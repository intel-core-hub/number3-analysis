"""tests/ui/test_monitoring_data.py — monitoring data layer tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ui import monitoring_data


def _prepare_db(db_path: Path) -> None:
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_results (
                round_no INTEGER,
                n_hits INTEGER,
                profit INTEGER,
                total_cost INTEGER,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE predictions (
                id INTEGER,
                target_round INTEGER,
                predicted_number TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE numbers3_draws (
                round_no INTEGER,
                winning_number TEXT
            )
            """
        )

        conn.executemany(
            "INSERT INTO prediction_results VALUES (?, ?, ?, ?, ?)",
            [
                (1, 0, -200, 200, "2026-01-01T12:00:00"),
                (2, 1, 500, 200, "2026-02-01T12:00:00"),
            ],
        )
        conn.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?, ?)",
            [
                (1, 3, "123", "2026-01-15T12:00:00"),
                (2, 4, "456", "2026-02-15T12:00:00"),
            ],
        )
        conn.executemany(
            "INSERT INTO numbers3_draws VALUES (?, ?)",
            [
                (1, "321"),
                (2, "654"),
            ],
        )


def test_load_results_with_filter(tmp_path):
    db_path = tmp_path / "monitoring.db"
    _prepare_db(db_path)

    df = monitoring_data.load_results(db_path, start_iso="2026-02-01T00:00:00")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert int(df.iloc[0]["round_no"]) == 2


def test_load_predictions_without_filter(tmp_path):
    db_path = tmp_path / "monitoring.db"
    _prepare_db(db_path)

    df = monitoring_data.load_predictions(db_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["id"]) == [1, 2]


def test_load_draws_missing_table_returns_empty(tmp_path):
    db_path = tmp_path / "empty.db"
    import sqlite3
    with sqlite3.connect(db_path):
        pass

    df = monitoring_data.load_draws(db_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_backtest_tracking_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "not_found.csv"

    df = monitoring_data.load_backtest_tracking(missing)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_results_without_created_at_still_returns_rows(tmp_path):
    import sqlite3

    db_path = tmp_path / "schema_drift.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_results (
                round_no INTEGER,
                n_hits INTEGER,
                profit INTEGER,
                total_cost INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO prediction_results VALUES (?, ?, ?, ?)",
            [
                (1, 0, -200, 200),
                (2, 1, 500, 200),
            ],
        )

    df = monitoring_data.load_results(db_path, start_iso="2026-02-01T00:00:00")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_predictions_without_order_column_still_returns_rows(tmp_path):
    import sqlite3

    db_path = tmp_path / "schema_drift.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE predictions (
                target_round INTEGER,
                predicted_number TEXT,
                created_at TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?)",
            [
                (3, "123", "2026-01-15T12:00:00"),
                (4, "456", "2026-02-15T12:00:00"),
            ],
        )

    df = monitoring_data.load_predictions(db_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_backtest_tracking_malformed_csv_returns_empty(tmp_path):
    malformed = tmp_path / "broken.csv"
    malformed.write_bytes(b"\xff\xfe\x00\x00")

    df = monitoring_data.load_backtest_tracking(malformed)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_schema_warnings_empty_db_reports_missing_tables(tmp_path):
    db_path = tmp_path / "empty.db"
    import sqlite3
    with sqlite3.connect(db_path):
        pass

    warnings = monitoring_data.get_schema_warnings(db_path)

    assert isinstance(warnings, list)
    assert any("prediction_results" in w for w in warnings)
    assert any("predictions" in w for w in warnings)
    assert any("numbers3_draws" in w for w in warnings)


def test_get_schema_warnings_missing_columns_reports_degraded_mode(tmp_path):
    import sqlite3

    db_path = tmp_path / "schema_drift.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE prediction_results (profit INTEGER)")
        conn.execute("CREATE TABLE predictions (target_round INTEGER)")
        conn.execute("CREATE TABLE numbers3_draws (round_no INTEGER)")

    warnings = monitoring_data.get_schema_warnings(db_path)

    assert any("prediction_results.round_no" in w for w in warnings)
    assert any("prediction_results.created_at" in w for w in warnings)
    assert any("predictions.id" in w for w in warnings)
    assert any("predictions.created_at" in w for w in warnings)


def test_get_schema_warnings_with_full_schema_returns_empty(tmp_path):
    import sqlite3

    db_path = tmp_path / "ok_schema.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_results (
                round_no INTEGER,
                created_at TEXT,
                n_hits INTEGER,
                profit INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE predictions (
                id INTEGER,
                created_at TEXT,
                target_round INTEGER,
                predicted_number TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE numbers3_draws (
                round_no INTEGER,
                winning_number TEXT
            )
            """
        )

    warnings = monitoring_data.get_schema_warnings(db_path)

    assert warnings == []


def test_get_schema_diagnostics_empty_db_classifies_as_critical(tmp_path):
    db_path = tmp_path / "empty.db"
    import sqlite3
    with sqlite3.connect(db_path):
        pass

    diagnostics = monitoring_data.get_schema_diagnostics(db_path)

    assert isinstance(diagnostics, dict)
    assert len(diagnostics["critical"]) >= 3
    assert diagnostics["warning"] == []


def test_get_schema_diagnostics_missing_columns_classifies_as_warning(tmp_path):
    import sqlite3

    db_path = tmp_path / "schema_drift.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE prediction_results (profit INTEGER)")
        conn.execute("CREATE TABLE predictions (target_round INTEGER)")
        conn.execute("CREATE TABLE numbers3_draws (round_no INTEGER)")

    diagnostics = monitoring_data.get_schema_diagnostics(db_path)

    assert diagnostics["critical"] == []
    assert any("prediction_results.round_no" in w for w in diagnostics["warning"])
    assert any("prediction_results.created_at" in w for w in diagnostics["warning"])
    assert any("predictions.id" in w for w in diagnostics["warning"])
    assert any("predictions.created_at" in w for w in diagnostics["warning"])


def test_record_schema_diagnostics_creates_history_file(tmp_path):
    history_path = tmp_path / "schema_history.csv"

    history = monitoring_data.record_schema_diagnostics(
        {"critical": ["x"], "warning": ["y"]},
        path=history_path,
    )

    assert history_path.exists()
    assert isinstance(history, pd.DataFrame)
    assert len(history) == 1
    assert int(history.iloc[0]["critical_count"]) == 1
    assert int(history.iloc[0]["warning_count"]) == 1
    assert history.iloc[0]["status"] == "critical"


def test_record_schema_diagnostics_upserts_same_day(tmp_path):
    history_path = tmp_path / "schema_history.csv"

    monitoring_data.record_schema_diagnostics(
        {"critical": ["x"], "warning": []},
        path=history_path,
    )
    history = monitoring_data.record_schema_diagnostics(
        {"critical": [], "warning": ["w1", "w2"]},
        path=history_path,
    )

    assert len(history) == 1
    assert int(history.iloc[0]["critical_count"]) == 0
    assert int(history.iloc[0]["warning_count"]) == 2
    assert int(history.iloc[0]["total_count"]) == 2
    assert history.iloc[0]["status"] == "warning"


def test_load_schema_diagnostics_history_invalid_schema_returns_empty(tmp_path):
    history_path = tmp_path / "schema_history.csv"
    pd.DataFrame({"foo": [1]}).to_csv(history_path, index=False)

    history = monitoring_data.load_schema_diagnostics_history(history_path)

    assert isinstance(history, pd.DataFrame)
    assert history.empty
    assert list(history.columns) == ["date", "critical_count", "warning_count", "total_count", "status"]


def test_apply_schema_auto_fix_repairs_missing_tables(tmp_path):
    db_path = tmp_path / "empty.db"
    import sqlite3
    with sqlite3.connect(db_path):
        pass

    result = monitoring_data.apply_schema_auto_fix(db_path)

    assert isinstance(result, dict)
    assert "repair_report" in result
    assert "diagnostics_after" in result
    assert "failed_tables" in result
    assert "repaired_tables" in result
    assert result["diagnostics_after"]["critical"] == []


def test_apply_schema_auto_fix_reports_after_warning_only(tmp_path):
    import sqlite3

    db_path = tmp_path / "schema_drift.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE prediction_results (profit INTEGER)")
        conn.execute("CREATE TABLE predictions (target_round INTEGER)")
        conn.execute("CREATE TABLE numbers3_draws (round_no INTEGER)")

    result = monitoring_data.apply_schema_auto_fix(db_path)

    assert isinstance(result, dict)
    assert result["diagnostics_after"]["critical"] == []
