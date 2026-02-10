"""
src.data.loader — データ取得・クレンジング

責務:
    - CSV の読み書き・正規化
    - バリデーション
    - スクレイピング関数は src.data.fetcher に分離
      (後方互換のため fetch_numbers3_by_month を re-export)
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.config import ALIAS_MAP, COLUMNS, PAYOUT_COLUMNS

# --- fetcher から re-export (後方互換) ---
from src.data.fetcher import (  # noqa: F401
    fetch_numbers3_by_month,
    _read_html_tables,
    _clean_digits,
)


# =====================================================================
# 内部ヘルパー
# =====================================================================


def _normalize_winning_number_series(series: pd.Series) -> pd.Series:
    cleaned = series.apply(_clean_digits)
    return cleaned.apply(lambda x: x.zfill(3) if isinstance(x, str) else None)


# =====================================================================
# 公開関数
# =====================================================================

def normalize_numbers3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名エイリアスの正規化."""
    if df is None:
        raise ValueError("normalize_numbers3_columns: df is None")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"normalize_numbers3_columns: expected DataFrame, got {type(df)}"
        )
    df = df.copy()
    for old, new in ALIAS_MAP.items():
        if old in df.columns:
            if new in df.columns:
                df[new] = df[new].fillna(df[old])
                df = df.drop(columns=[old])
            else:
                df = df.rename(columns={old: new})
    return df


def add_digit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """当選番号から百・十・一の位を分解して列に追加する."""
    df = df.copy()
    if "当選番号" not in df.columns:
        if "当せん番号" in df.columns:
            df["当選番号"] = df["当せん番号"]
        else:
            raise KeyError("当選番号（または当せん番号）列が必要です。")
    num = df["当選番号"].astype(str).str.zfill(3)
    df["digit_h"] = num.str[0].astype(int)
    df["digit_t"] = num.str[1].astype(int)
    df["digit_o"] = num.str[2].astype(int)
    return df


def _coverage_ratio(df: pd.DataFrame) -> Optional[float]:
    if "回号" not in df.columns:
        return None
    rounds = pd.to_numeric(df["回号"], errors="coerce").dropna().astype(int)
    if rounds.empty:
        return None
    min_r, max_r = rounds.min(), rounds.max()
    total_range = max_r - min_r + 1
    if total_range <= 0:
        return None
    return rounds.nunique() / total_range


def update_numbers3_clean(
    clean_path: str = "numbers3_clean.csv",
    backup: bool = True,
    sleep_seconds: float = 1,
    force_full: bool = False,
) -> pd.DataFrame:
    """差分取得してクリーンCSVを更新する."""
    path = Path(clean_path)

    if path.exists() and not force_full:
        df_existing = pd.read_csv(path)
        df_existing = normalize_numbers3_columns(df_existing)
        if "当選番号" in df_existing.columns:
            df_existing["当選番号"] = _normalize_winning_number_series(
                df_existing["当選番号"]
            )
    else:
        df_existing = pd.DataFrame(columns=COLUMNS)

    if force_full or df_existing.empty:
        start_date = datetime(1994, 10, 1)
    else:
        dates = pd.to_datetime(df_existing.get("抽せん日"), errors="coerce")
        last_date = dates.max()
        if pd.isna(last_date):
            start_date = datetime(1994, 10, 1)
        else:
            start_date = datetime(last_date.year, last_date.month, 1)

    new_df, _failures = fetch_numbers3_by_month(
        start_date,
        datetime.now(),
        sleep_seconds=sleep_seconds,
    )

    if new_df is None or new_df.empty:
        print("新規データなし。既存データを返します。")
        return df_existing

    new_df = normalize_numbers3_columns(new_df)
    if "当選番号" in new_df.columns:
        new_df["当選番号"] = _normalize_winning_number_series(new_df["当選番号"])

    combined = pd.concat([df_existing, new_df], ignore_index=True)
    combined = normalize_numbers3_columns(combined)

    if "回号" in combined.columns:
        combined["回号"] = combined["回号"].astype(str).str.zfill(4)
        combined = combined.drop_duplicates(subset=["回号"], keep="last")
        combined = combined.sort_values("回号", key=lambda s: s.astype(int))

    ordered_cols = [c for c in COLUMNS if c in combined.columns]
    combined = combined[ordered_cols]

    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(".csv.bak"))

    combined.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"更新完了: {len(combined)} 件")
    return combined


def backfill_numbers3_range(
    clean_path: str,
    start_date: datetime,
    end_date: datetime,
    backup: bool = True,
    sleep_seconds: float = 1,
) -> pd.DataFrame:
    """指定期間のデータをバックフィルする."""
    path = Path(clean_path)
    if path.exists():
        df_existing = pd.read_csv(path)
        df_existing = normalize_numbers3_columns(df_existing)
        if "当選番号" in df_existing.columns:
            df_existing["当選番号"] = _normalize_winning_number_series(
                df_existing["当選番号"]
            )
    else:
        df_existing = pd.DataFrame(columns=COLUMNS)

    new_df, _failures = fetch_numbers3_by_month(
        start_date, end_date, sleep_seconds=sleep_seconds
    )
    new_df = normalize_numbers3_columns(new_df)
    if "当選番号" in new_df.columns:
        new_df["当選番号"] = _normalize_winning_number_series(new_df["当選番号"])

    combined = pd.concat([df_existing, new_df], ignore_index=True)
    combined = normalize_numbers3_columns(combined)
    if "当選番号" in combined.columns:
        combined["当選番号"] = _normalize_winning_number_series(combined["当選番号"])
    if "回号" in combined.columns:
        combined["回号"] = combined["回号"].astype(str).str.zfill(4)
        combined = combined.drop_duplicates(subset=["回号"], keep="last")
        combined = combined.sort_values(by="回号", key=lambda s: s.astype(int))

    ordered_cols = [c for c in COLUMNS if c in combined.columns]
    if ordered_cols:
        combined = combined[ordered_cols]

    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)
        print(f"バックアップ保存: {backup_path}")

    combined.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"更新完了: {len(combined)}件を {path.name} に保存しました。")
    return combined


def validate_numbers3(
    df: pd.DataFrame,
    strict: bool = True,
    allow_missing_rounds: bool = True,
) -> List[str]:
    """データの整合性チェック."""
    errors: List[str] = []
    warnings: List[str] = []
    df = normalize_numbers3_columns(df)
    if "当選番号" in df.columns:
        df["当選番号"] = _normalize_winning_number_series(df["当選番号"])

    if "回号" in df.columns:
        rounds = df["回号"].astype(str)
        round_nums = pd.to_numeric(rounds, errors="coerce")
        if round_nums.isna().any():
            errors.append("回号に数値化できない値があります。")
        else:
            round_nums = round_nums.astype(int)
            duplicates = rounds[rounds.duplicated()].unique().tolist()
            if duplicates:
                errors.append(f"回号の重複: {duplicates[:10]}")

            if not round_nums.empty:
                min_r, max_r = round_nums.min(), round_nums.max()
                missing = sorted(set(range(min_r, max_r + 1)) - set(round_nums))
                if missing:
                    message = f"回号の欠番: {missing[:10]}"
                    if allow_missing_rounds:
                        warnings.append(message)
                    else:
                        errors.append(message)

    if "当選番号" in df.columns:
        invalid = df[
            df["当選番号"].isna()
            | ~df["当選番号"].astype(str).str.fullmatch(r"\d{3}")
        ]
        if not invalid.empty:
            errors.append("当選番号が3桁でない行があります。")

    if errors and strict:
        raise ValueError(" / ".join(errors))

    if errors:
        print("警告:", " / ".join(errors))
    if warnings:
        print("注意:", " / ".join(warnings))

    return errors + warnings
