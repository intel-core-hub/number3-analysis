"""tests/ui/test_monitoring_view_data.py — monitoring view-data helpers tests."""
from __future__ import annotations

import pandas as pd

from src.ui.monitoring_view_data import (
    prepare_cumulative_profit,
    prepare_hit_rate_ma,
    recent_rows,
    select_existing_columns,
)


def test_select_existing_columns_returns_intersection_ordered():
    cols = select_existing_columns(["a", "x", "b"], ["b", "a", "c"])
    assert cols == ["a", "b"]


def test_prepare_cumulative_profit_basic():
    df = pd.DataFrame({"round_no": [2, 1, 3], "profit": [10, -5, 20]})
    out = prepare_cumulative_profit(df)

    assert list(out["round_no"]) == [1, 2, 3]
    assert list(out["cumulative_profit"]) == [-5, 5, 25]


def test_prepare_cumulative_profit_missing_columns_returns_empty():
    out = prepare_cumulative_profit(pd.DataFrame({"profit": [1, 2]}))
    assert out.empty


def test_prepare_hit_rate_ma_basic():
    df = pd.DataFrame({"round_no": [1, 2, 3, 4], "n_hits": [1, 0, 0, 1]})
    out = prepare_hit_rate_ma(df, window=2)

    assert "hit_rate_ma" in out.columns
    assert len(out) == 4


def test_prepare_hit_rate_ma_insufficient_rows_returns_empty():
    df = pd.DataFrame({"round_no": [1], "n_hits": [1]})
    out = prepare_hit_rate_ma(df, window=3)
    assert out.empty


def test_recent_rows_sort_desc_limit():
    df = pd.DataFrame({"id": [1, 3, 2], "value": [10, 30, 20]})
    out = recent_rows(df, sort_col="id", limit=2)

    assert list(out["id"]) == [3, 2]
    assert len(out) == 2
