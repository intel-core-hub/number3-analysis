"""
src.database - SQLite persistence for Numbers3 draws.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import COLUMN_MAP, DB_PATH, RAW_COLUMNS, REVERSE_COLUMN_MAP
from src.data.loader import normalize_numbers3_columns
from src.helpers import get_logger

logger = get_logger(__name__)


class Numbers3Database:
    """SQLite database adapter for Numbers3 draws."""

    EXPECTED_TABLE = "numbers3_draws"
    EXPECTED_COLUMNS = {
        "round_no": "INTEGER",
        "draw_date": "TEXT",
        "winning_number": "TEXT",
        "straight": "INTEGER",
        "box": "INTEGER",
        "set_straight": "INTEGER",
        "set_box": "INTEGER",
        "mini": "INTEGER",
        "sales": "INTEGER",
    }
    SUPPORTED_TABLE_SCHEMAS = {
        "numbers3_draws": {
            "round_no": "INTEGER",
            "draw_date": "TEXT",
            "winning_number": "TEXT",
            "straight": "INTEGER",
            "box": "INTEGER",
            "set_straight": "INTEGER",
            "set_box": "INTEGER",
            "mini": "INTEGER",
            "sales": "INTEGER",
        },
        "predictions": {
            "id": "INTEGER",
            "created_at": "TEXT",
            "target_round": "INTEGER",
            "source": "TEXT",
            "predicted_number": "TEXT",
            "probability": "REAL",
            "purchase_amount": "INTEGER",
            "kelly_ev": "REAL",
        },
        "prediction_results": {
            "id": "INTEGER",
            "round_no": "INTEGER",
            "actual_number": "TEXT",
            "n_predictions": "INTEGER",
            "n_hits": "INTEGER",
            "total_cost": "INTEGER",
            "total_return": "INTEGER",
            "profit": "INTEGER",
            "roi": "REAL",
            "details": "TEXT",
            "created_at": "TEXT",
        },
    }
    TABLE_DDLS = {
        "numbers3_draws": """
            CREATE TABLE IF NOT EXISTS numbers3_draws (
                round_no INTEGER PRIMARY KEY,
                draw_date TEXT,
                winning_number TEXT,
                straight INTEGER,
                box INTEGER,
                set_straight INTEGER,
                set_box INTEGER,
                mini INTEGER,
                sales INTEGER
            )
        """,
        "predictions": """
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    NOT NULL,
                target_round    INTEGER,
                source          TEXT    NOT NULL,
                predicted_number TEXT   NOT NULL,
                probability     REAL,
                purchase_amount INTEGER DEFAULT 200,
                kelly_ev        REAL
            )
        """,
        "prediction_results": """
            CREATE TABLE IF NOT EXISTS prediction_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                round_no        INTEGER NOT NULL,
                actual_number   TEXT    NOT NULL,
                n_predictions   INTEGER DEFAULT 0,
                n_hits          INTEGER DEFAULT 0,
                total_cost      INTEGER DEFAULT 0,
                total_return    INTEGER DEFAULT 0,
                profit          INTEGER DEFAULT 0,
                roi             REAL    DEFAULT 0.0,
                details         TEXT,
                created_at      TEXT    NOT NULL
            )
        """,
    }
    REQUIRED_NON_NULL_COLUMNS = {
        "numbers3_draws": set(),
        "predictions": {"created_at", "source", "predicted_number"},
        "prediction_results": {"round_no", "actual_number", "created_at"},
    }
    MIGRATION_DEFAULTS = {
        "predictions": {
            "created_at": "1970-01-01T00:00:00",
            "source": "migration",
            "predicted_number": "000",
        },
        "prediction_results": {
            "round_no": 0,
            "actual_number": "000",
            "created_at": "1970-01-01T00:00:00",
        },
    }

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        """Create schema if it does not exist."""
        with self._connect() as conn:
            conn.execute(self.TABLE_DDLS[self.EXPECTED_TABLE])

    def initialize_support_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(self.TABLE_DDLS["predictions"])
            conn.execute(self.TABLE_DDLS["prediction_results"])

    def initialize_all_tables(self) -> None:
        with self._connect() as conn:
            for table_name in self.TABLE_DDLS:
                conn.execute(self.TABLE_DDLS[table_name])

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None

    def _current_columns(self, conn: sqlite3.Connection, table_name: str) -> dict[str, str]:
        if not self._table_exists(conn, table_name):
            return {}
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        rows = cur.fetchall()
        return {str(row[1]): str(row[2]).upper() for row in rows}

    def detect_schema_mismatch(self) -> dict[str, object]:
        """Detect schema mismatch against expected numbers3_draws definition."""
        return self.detect_table_schema_mismatch(self.EXPECTED_TABLE)

    def detect_table_schema_mismatch(self, table_name: str) -> dict[str, object]:
        """Detect schema mismatch for a supported table."""
        if table_name not in self.SUPPORTED_TABLE_SCHEMAS:
            raise ValueError(f"Unsupported table for schema check: {table_name}")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            current = self._current_columns(conn, table_name)

        expected = self.SUPPORTED_TABLE_SCHEMAS[table_name]
        missing_columns = sorted([c for c in expected if c not in current])
        extra_columns = sorted([c for c in current if c not in expected])
        type_mismatches = sorted(
            [
                c for c in expected
                if c in current and expected[c].upper() != current[c].upper()
            ]
        )

        if not current:
            return {
                "table_name": table_name,
                "table_exists": False,
                "is_healthy": False,
                "missing_columns": list(expected.keys()),
                "extra_columns": [],
                "type_mismatches": [],
            }

        is_healthy = not missing_columns and not extra_columns and not type_mismatches
        return {
            "table_name": table_name,
            "table_exists": True,
            "is_healthy": is_healthy,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "type_mismatches": type_mismatches,
        }

    def detect_all_schema_mismatches(self) -> dict[str, dict[str, object]]:
        report: dict[str, dict[str, object]] = {}
        for table_name in self.SUPPORTED_TABLE_SCHEMAS:
            report[table_name] = self.detect_table_schema_mismatch(table_name)
        return report

    def auto_repair_schema(self) -> dict[str, object]:
        """Repair numbers3_draws schema by recreating table and restoring common columns.

        Returns:
            dict with repair status and migrated row count.
        """
        return self.auto_repair_table_schema(self.EXPECTED_TABLE)

    def auto_repair_table_schema(self, table_name: str) -> dict[str, object]:
        if table_name not in self.SUPPORTED_TABLE_SCHEMAS:
            raise ValueError(f"Unsupported table for auto-repair: {table_name}")

        report = self.detect_table_schema_mismatch(table_name)
        if report.get("is_healthy"):
            return {
                "table_name": table_name,
                "repaired": False,
                "reason": "already_healthy",
                "migrated_rows": 0,
            }

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                expected_columns = self.SUPPORTED_TABLE_SCHEMAS[table_name]
                old_exists = self._table_exists(conn, table_name)
                current_cols = self._current_columns(conn, table_name)
                common_cols = [c for c in expected_columns if c in current_cols]

                backup_table = f"{table_name}_backup_tmp"
                if self._table_exists(conn, backup_table):
                    conn.execute(f"DROP TABLE {backup_table}")

                if old_exists:
                    conn.execute(f"ALTER TABLE {table_name} RENAME TO {backup_table}")

                conn.execute(self.TABLE_DDLS[table_name])

                migrated_rows = 0
                if old_exists and common_cols:
                    defaults = self.MIGRATION_DEFAULTS.get(table_name, {})
                    target_columns = list(expected_columns.keys())
                    insert_columns: list[str] = []
                    select_exprs: list[str] = []

                    for column in target_columns:
                        if column in current_cols:
                            insert_columns.append(column)
                            select_exprs.append(column)
                        elif column in defaults:
                            value = defaults[column]
                            insert_columns.append(column)
                            if isinstance(value, str):
                                select_exprs.append(f"'{value}'")
                            else:
                                select_exprs.append(str(value))

                    required = self.REQUIRED_NON_NULL_COLUMNS.get(table_name, set())
                    if required.issubset(set(insert_columns)) and insert_columns:
                        conn.execute(
                            f"INSERT OR REPLACE INTO {table_name} ({', '.join(insert_columns)}) "
                            f"SELECT {', '.join(select_exprs)} FROM {backup_table}"
                        )
                        cur = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                        migrated_rows = int(cur.fetchone()[0])

                if self._table_exists(conn, backup_table):
                    conn.execute(f"DROP TABLE {backup_table}")

                conn.commit()
                return {
                    "table_name": table_name,
                    "repaired": True,
                    "reason": "schema_migrated",
                    "migrated_rows": migrated_rows,
                }
            except Exception as exc:
                conn.rollback()
                logger.exception("Schema auto-repair failed for %s: %s", table_name, exc)
                return {
                    "table_name": table_name,
                    "repaired": False,
                    "reason": f"failed: {exc}",
                    "migrated_rows": 0,
                }

    def auto_repair_all_schemas(self) -> dict[str, object]:
        per_table: dict[str, dict[str, object]] = {}
        repaired_tables: list[str] = []
        failed_tables: list[str] = []

        for table_name in self.SUPPORTED_TABLE_SCHEMAS:
            result = self.auto_repair_table_schema(table_name)
            per_table[table_name] = result
            if result.get("repaired"):
                repaired_tables.append(table_name)
            if str(result.get("reason", "")).startswith("failed"):
                failed_tables.append(table_name)

        return {
            "repaired_tables": repaired_tables,
            "failed_tables": failed_tables,
            "results": per_table,
        }

    def _normalize_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        df = normalize_numbers3_columns(df)
        df = df.copy()
        df = df.rename(columns=COLUMN_MAP)
        keep_cols = [c for c in COLUMN_MAP.values() if c in df.columns]
        df = df[keep_cols]
        if "round_no" in df.columns:
            df["round_no"] = pd.to_numeric(df["round_no"], errors="coerce")
            df = df.dropna(subset=["round_no"])
            df["round_no"] = df["round_no"].astype(int)
        if "winning_number" in df.columns:
            df["winning_number"] = df["winning_number"].astype(str).str.zfill(3)
        numeric_cols = [
            col
            for col in [
                "straight",
                "box",
                "set_straight",
                "set_box",
                "mini",
                "sales",
            ]
            if col in df.columns
        ]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = df.where(pd.notna(df), None)
        return df

    def _denormalize_from_db(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=REVERSE_COLUMN_MAP)
        ordered = [c for c in RAW_COLUMNS if c in df.columns]
        if ordered:
            df = df[ordered]
        return df

    def upsert_draws(self, df: pd.DataFrame) -> int:
        """Insert or update draw records.

        Args:
            df: Raw or normalized draws dataframe.

        Returns:
            Number of rows upserted.
        """
        if df is None or df.empty:
            return 0

        self.auto_repair_schema()
        self.initialize_all_tables()
        df_db = self._normalize_for_db(df)
        if df_db.empty:
            return 0

        columns = list(df_db.columns)
        placeholders = ", ".join(["?"] * len(columns))
        sql = (
            f"INSERT OR REPLACE INTO numbers3_draws ({', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )

        rows = df_db.itertuples(index=False, name=None)
        with self._connect() as conn:
            conn.executemany(sql, list(rows))
            conn.commit()
        return len(df_db)

    def load_draws(self) -> pd.DataFrame:
        """Load all draws from the database."""
        self.initialize()
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM numbers3_draws ORDER BY round_no ASC", conn
            )
        return self._denormalize_from_db(df)

    def is_empty(self) -> bool:
        """Return True if the database has no draws."""
        self.initialize()
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM numbers3_draws")
            count = cur.fetchone()[0]
        return count == 0

    def get_latest_round(self) -> int | None:
        """Return the latest round number, if any."""
        self.initialize()
        with self._connect() as conn:
            cur = conn.execute("SELECT MAX(round_no) FROM numbers3_draws")
            value = cur.fetchone()[0]
        return int(value) if value is not None else None

    def get_latest_draw_date(self) -> str | None:
        """Return the latest draw date, if any."""
        self.initialize()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT draw_date FROM numbers3_draws ORDER BY round_no DESC LIMIT 1"
            )
            row = cur.fetchone()
        return row[0] if row else None

    def migrate_from_csv(self, csv_path: Path | str) -> int:
        """Migrate legacy CSV data into SQLite.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Number of rows migrated.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        return self.upsert_draws(df)
