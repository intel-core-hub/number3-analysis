"""
src.ui.analysis_helpers — 分析計算ヘルパー関数

責務:
    - ストレステスト実行
    - レジーム検出
    - 資産曲線のメトリクス計算
"""
import pandas as pd
import streamlit as st

from src.analysis.regime_detector import detect_current_regime
from src.analysis.stress_test import StressTestConfig, StressTestResult, run_stress_test


@st.cache_data(show_spinner=False)
def cached_stress_test(cfg: StressTestConfig) -> StressTestResult:
    """ストレステストをキャッシュ付きで実行する

    同じ設定に対するストレステストの重複実行を防ぐ。

    Args:
        cfg: ストレステスト設定

    Returns:
        ストレステスト結果
    """
    return run_stress_test(cfg)


@st.cache_data(show_spinner=False)
def cached_regime_detection(df: pd.DataFrame):
    """レジーム検出をキャッシュ付きで実行する

    同じデータに対するレジーム検出の重複実行を防ぐ。

    Args:
        df: 抽選履歴の DataFrame

    Returns:
        レジーム検出結果
    """
    return detect_current_regime(df, n_regimes=3, window=60)


def compute_equity_metrics(df: pd.DataFrame) -> dict:
    """損益データから各種メトリクスを計算する

    Args:
        df: profitカラムを持つDataFrame

    Returns:
        メトリクスの辞書 {total_profit, max_drawdown, sharpe_ratio, win_rate, profit_factor}
    """
    # 空または不正なDataFrameの処理
    if df.empty or "profit" not in df.columns:
        return {
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    # profitカラムからNaNを除去
    profits = df["profit"].dropna()

    if profits.empty:
        return {
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    # 基本指標
    total_profit = float(profits.sum())

    # 勝率計算
    win_count = (profits > 0).sum()
    total_count = len(profits)
    win_rate = float(win_count / total_count) if total_count > 0 else 0.0

    # プロフィットファクター
    gross_profit = profits[profits > 0].sum()
    gross_loss = -profits[profits < 0].sum()
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

    # シャープレシオ
    mean_return = profits.mean()
    std_return = profits.std()
    sharpe_ratio = float(mean_return / std_return) if std_return > 0 else 0.0

    # 最大ドローダウン
    equity_curve = profits.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    return {
        "total_profit": total_profit,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
