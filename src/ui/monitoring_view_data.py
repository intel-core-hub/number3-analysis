"""View-data preparation helpers for monitoring dashboard."""
from __future__ import annotations

import pandas as pd


def select_existing_columns(candidates: list[str], available: list[str]) -> list[str]:
    available_set = set(available)
    return [column for column in candidates if column in available_set]


def prepare_cumulative_profit(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "profit" not in results.columns or "round_no" not in results.columns:
        return pd.DataFrame()

    prepared = results.sort_values("round_no").copy()
    prepared["cumulative_profit"] = prepared["profit"].cumsum()
    return prepared


def prepare_hit_rate_ma(results: pd.DataFrame, window: int) -> pd.DataFrame:
    if (
        results.empty
        or "n_hits" not in results.columns
        or "round_no" not in results.columns
        or len(results) < window
    ):
        return pd.DataFrame()

    prepared = results.sort_values("round_no").copy()
    prepared["hit_flag"] = (prepared["n_hits"] > 0).astype(int)
    prepared["hit_rate_ma"] = prepared["hit_flag"].rolling(window).mean() * 100
    return prepared


def recent_rows(df: pd.DataFrame, sort_col: str, limit: int) -> pd.DataFrame:
    if df.empty or sort_col not in df.columns:
        return pd.DataFrame()
    return df.sort_values(sort_col, ascending=False).head(limit)
