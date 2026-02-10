"""
src.data.fetcher — スクレイピング

責務:
    - 楽天宝くじ Web サイトからの HTML テーブル取得
    - HTML レコードの正規化
    - 月単位の一括取得 (fetch_numbers3_by_month)
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.utils.config import (
    ALIAS_MAP,
    COLUMNS,
    DEFAULT_SOURCE_TEMPLATES,
    PAYOUT_COLUMNS,
    SCRAPING_BACKOFF,
    SCRAPING_RETRIES,
    SCRAPING_SLEEP_SECONDS,
    SCRAPING_TIMEOUT,
)


# =====================================================================
# 内部ヘルパー
# =====================================================================


def _clean_digits(value: Any) -> Optional[str]:
    """数値以外の文字を除去して数字文字列を返す."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    return digits if digits else None


def _col_contains(columns: Sequence, text: str) -> bool:
    return any(text in str(c) for c in columns)


def _extract_round_from_columns(columns: Sequence) -> Optional[str]:
    for col in columns[1:]:
        digits = _clean_digits(col)
        if digits:
            return digits
    return None


def _normalize_record(record: Dict) -> Dict:
    row: Dict[str, Any] = {}

    round_raw = record.get("回号")
    round_digits = _clean_digits(round_raw)
    if round_digits:
        row["回号"] = round_digits.zfill(4)

    draw_date = record.get("抽せん日") or record.get("抽選日")
    if draw_date:
        row["抽せん日"] = str(draw_date).strip()

    win_raw = record.get("当選番号") or record.get("当せん番号")
    win_digits = _clean_digits(win_raw)
    if win_digits:
        row["当選番号"] = win_digits.zfill(3)

    for key in PAYOUT_COLUMNS:
        val = record.get(key)
        digits = _clean_digits(val)
        if digits is not None:
            row[key] = int(digits)

    return row


def _table_to_records(df: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    df_work = df.copy()
    df_work.columns = [str(c).strip() for c in df_work.columns]

    if not df_work.empty and _col_contains(df_work.columns, "回号"):
        if _col_contains(df_work.columns, "当選番号") or _col_contains(
            df_work.columns, "当せん番号"
        ):
            for _, row in df_work.iterrows():
                records.append(row.to_dict())
            return records

    if not df_work.empty:
        header = [str(v).strip() for v in df_work.iloc[0].tolist()]
        if _col_contains(header, "回号") and (
            _col_contains(header, "当選番号") or _col_contains(header, "当せん番号")
        ):
            df_work = df_work.iloc[1:].copy()
            df_work.columns = header
            for _, row in df_work.iterrows():
                records.append(row.to_dict())
            return records

    if df_work.shape[1] >= 2:
        record = dict(zip(df_work.iloc[:, 0], df_work.iloc[:, 1]))
        round_digits = _extract_round_from_columns(df_work.columns)
        if round_digits:
            record["回号"] = round_digits
        records.append(record)
    return records


def _read_html_tables(
    url: str,
    timeout: int = SCRAPING_TIMEOUT,
    retries: int = SCRAPING_RETRIES,
    backoff: float = SCRAPING_BACKOFF,
) -> List[pd.DataFrame]:
    """URL から HTML テーブルを読み込む (リトライ付き)."""
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return pd.read_html(url, header=0)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_error  # type: ignore[misc]


# =====================================================================
# 公開 API
# =====================================================================


def fetch_numbers3_by_month(
    start_date: datetime,
    end_date: datetime,
    sleep_seconds: float = SCRAPING_SLEEP_SECONDS,
    source_templates: Optional[List[str]] = None,
    fail_fast: bool = True,
    timeout: int = SCRAPING_TIMEOUT,
    retries: int = SCRAPING_RETRIES,
    backoff: float = SCRAPING_BACKOFF,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """月単位でスクレイピングして (DataFrame, failures) を返す."""
    rows: List[Dict] = []
    failures: List[Dict] = []

    current = datetime(start_date.year, start_date.month, 1)
    last_month = datetime(end_date.year, end_date.month, 1)

    if source_templates is None:
        source_templates = list(DEFAULT_SOURCE_TEMPLATES)

    while current <= last_month:
        yyyymm = current.strftime("%Y%m")
        success = False
        last_error: Optional[Exception] = None

        for template in source_templates:
            url = template.format(yyyymm=yyyymm)
            try:
                tables = _read_html_tables(
                    url, timeout=timeout, retries=retries, backoff=backoff
                )
                for tbl in tables:
                    records = _table_to_records(tbl)
                    for record in records:
                        if "当選番号" not in record and "当せん番号" not in record:
                            continue
                        row = _normalize_record(record)
                        if row.get("回号") and row.get("当選番号"):
                            rows.append(row)
                success = True
                break
            except Exception as e:
                if fail_fast:
                    raise
                last_error = e
                print(f"[WARN] {yyyymm} failed: {e}")

        if not success:
            if last_error is not None:
                failures.append({"month": yyyymm, "error": str(last_error)})
            print(f"[SKIP] {yyyymm}")

        time.sleep(sleep_seconds)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    return pd.DataFrame(rows), failures
