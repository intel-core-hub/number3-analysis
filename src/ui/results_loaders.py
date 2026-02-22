"""
src.ui.results_loaders — 結果ファイル読み込み関数

責務:
    - バックテスト結果の読み込み
    - 最適化結果の読み込み
    - 推奨パラメータの取得
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.helpers import get_logger

logger = get_logger(__name__)


def load_latest_backtest_results(results_dir: str) -> pd.DataFrame | None:
    """最新のバックテスト結果を読み込む

    Args:
        results_dir: 結果ディレクトリのパス（文字列またはPath）

    Returns:
        バックテスト結果の DataFrame、存在しない場合は None
    """
    dir_path = Path(results_dir)
    if not dir_path.exists():
        return None

    candidates = list(dir_path.glob("backtest_results_*.csv"))
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return pd.read_csv(latest, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("Failed to load backtest results: %s", exc)
        return None


def recommend_ev_threshold(df: pd.DataFrame) -> float:
    """バックテスト結果から最適なEV閾値を推奨する

    Args:
        df: バックテスト結果のDataFrame（expected_value, profit等を含む）

    Returns:
        推奨EV閾値（デフォルト: 1.05）
    """
    # DataFrameが空または必要なカラムがない場合はデフォルト値
    if df.empty or "expected_value" not in df.columns or "profit" not in df.columns:
        return 1.05

    # 利益が出ているエントリのEV閾値を分析
    profitable = df[df["profit"] > 0]
    if profitable.empty:
        return 1.05

    # 利益が出ているケースのEVの中央値を推奨値とする
    recommended_threshold = profitable["expected_value"].median()

    # 範囲制限（0.8 ~ 2.0）
    recommended_threshold = max(0.8, min(2.0, recommended_threshold))

    return float(recommended_threshold)


def load_feature_importance(file_path: str) -> pd.DataFrame | None:
    """特徴量重要度ファイルを読み込む

    Args:
        file_path: 特徴量重要度ファイルのパス

    Returns:
        特徴量重要度の DataFrame、存在しない場合は None
    """
    path = Path(file_path)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("Failed to load feature importance: %s", exc)
        return None

    return df


def load_optimization_results(file_path: str) -> dict | None:
    """多目的最適化結果を読み込む

    Args:
        file_path: 結果JSONファイルのパス

    Returns:
        最適化結果の辞書、存在しない場合は None
    """
    path = Path(file_path)
    if not path.exists():
        return None

    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load optimization results: %s", exc)
        return None


def load_latest_selective_summary(results_dir: str) -> pd.DataFrame | None:
    """最新の選択的購入サマリーを読み込む

    Args:
        results_dir: 結果ディレクトリのパス

    Returns:
        サマリーのDataFrame、存在しない場合は None
    """
    dir_path = Path(results_dir)
    if not dir_path.exists():
        return None

    candidates = list(dir_path.glob("selective_summary_*.csv"))
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return pd.read_csv(latest, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("Failed to load selective summary: %s", exc)
        return None
