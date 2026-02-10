"""
src.features.engineer — 特徴量生成

責務:
    - 共通特徴量計算 (_compute_common_features)
    - Numbers3FeatureEngineer (ルールベース予測向け)
    - ML 用拡張特徴量 (ラグ、頻度、ギャップ)
    - マルチウィンドウ特徴量 (短期5, 中期50)
    - カレンダー特徴量 (曜日, 月初/月末フラグ)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data.loader import normalize_numbers3_columns

# =====================================================================
# 共通特徴量
# =====================================================================


def compute_common_features(
    df: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 10,
) -> pd.DataFrame:
    """FeatureEngineer と MLPredictor で共有する基本特徴量を計算する.

    Parameters
    ----------
    df : DataFrame
        当選番号を含むデータ
    window_short : int
        短期ローリングウィンドウ (default 5)
    window_long : int
        長期ローリングウィンドウ (default 10)

    Returns
    -------
    DataFrame  特徴量を追加した DataFrame
    """
    df = normalize_numbers3_columns(df)
    df = df.copy().reset_index(drop=True)
    if "当選番号" not in df.columns:
        return df

    num = df["当選番号"].astype(str).str.zfill(3)
    df["n1"] = num.str[0].astype(int)
    df["n2"] = num.str[1].astype(int)
    df["n3"] = num.str[2].astype(int)

    # 後方互換エイリアス
    df["digit_h"] = df["n1"]
    df["digit_t"] = df["n2"]
    df["digit_o"] = df["n3"]

    # 基本統計
    df["digit_sum"] = df[["n1", "n2", "n3"]].sum(axis=1)
    df["odd_count"] = (df[["n1", "n2", "n3"]] % 2 == 1).sum(axis=1)
    df["big_count"] = (df[["n1", "n2", "n3"]] >= 5).sum(axis=1)
    df["even_count"] = 3 - df["odd_count"]
    df["small_count"] = 3 - df["big_count"]

    # カレンダー特徴量
    df["dt"] = pd.to_datetime(df.get("抽せん日"), errors="coerce")
    df["weekday"] = df["dt"].dt.weekday
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # >>>>>>> 新規: 月初・月末フラグ <<<<<<<<
    df["is_month_start"] = df["dt"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["dt"].dt.is_month_end.astype(int)
    df["day_of_month"] = df["dt"].dt.day.fillna(0).astype(int)

    # >>>>>>> 新規: 給料日フラグ (25日) <<<<<<<<
    df["is_payday"] = (df["day_of_month"] == 25).astype(int)

    # >>>>>>> 新規: 回号サイクル特徴量 (Sin/Cos) <<<<<<<<
    if "回号" in df.columns:
        draw_num = pd.to_numeric(df["回号"], errors="coerce").fillna(0)
    else:
        draw_num = pd.Series(np.arange(len(df)), dtype=float)
    cycle_period = 100  # 100 回を 1 周期と仮定
    df["draw_cycle_sin"] = np.sin(2 * np.pi * draw_num / cycle_period)
    df["draw_cycle_cos"] = np.cos(2 * np.pi * draw_num / cycle_period)

    # 直前値 (lag-1)
    for col in ["n1", "n2", "n3"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df["prev_sum"] = df["digit_sum"].shift(1)
    df["prev_odd_count"] = df["odd_count"].shift(1)
    df["prev_big_count"] = df["big_count"].shift(1)
    df["prev_even_count"] = df["even_count"].shift(1)
    df["prev_small_count"] = df["small_count"].shift(1)

    # ローリング統計
    df["sum_ma_5"] = df["digit_sum"].rolling(window_short, min_periods=1).mean()
    df["sum_ma_10"] = df["digit_sum"].rolling(window_long, min_periods=1).mean()
    df["sum_std_5"] = (
        df["digit_sum"].rolling(window_short, min_periods=1).std().fillna(0)
    )
    df["sum_std_10"] = (
        df["digit_sum"].rolling(window_long, min_periods=1).std().fillna(0)
    )

    return df


# =====================================================================
# FeatureEngineer (ルールベース予測用)
# =====================================================================


class Numbers3FeatureEngineer:
    """ルールベース予測 (Numbers3Predictor) 向けの特徴量エンジニアリング."""

    def __init__(self, window: int = 20):
        self.window = window

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_common_features(df, window_short=5, window_long=10)
        if "当選番号" not in df.columns:
            return df

        # FeatureEngineer 固有の特徴量
        df["delta_prev"] = df["当選番号"].astype(int).diff()
        df["sum_ma_20"] = (
            df["digit_sum"].rolling(self.window, min_periods=1).mean()
        )

        rolling_prob = (
            df["digit_h"]
            .rolling(self.window, min_periods=1)
            .apply(lambda x: (x == x.iloc[-1]).mean(), raw=False)
        )
        df["hundreds_prob_window"] = rolling_prob

        # マルチウィンドウ頻度 (短期 5回 vs 長期 60回)
        for w in [5, 60]:
            for col in ["digit_h", "digit_t", "digit_o"]:
                for n in range(10):
                    df[f"{col}_freq_{n}_w{w}"] = (
                        df[col]
                        .rolling(w, min_periods=1)
                        .apply(lambda x, _n=n: (x == _n).sum(), raw=False)
                    )

        # 合計値トレンド
        df["sum_diff"] = df["digit_sum"] - df["sum_ma_10"]
        df["draw_dow"] = df["weekday"]

        return df.fillna(0)


# =====================================================================
# ML 用拡張特徴量
# =====================================================================


def ml_feature_columns() -> List[str]:
    """ML モデルで使用する特徴量名の一覧を返す."""
    base = [
        "weekday",
        "weekday_sin",
        "weekday_cos",
        "is_month_start",
        "is_month_end",
        "day_of_month",
        # >>>>>>> 新規: 給料日フラグ <<<<<<<<
        "is_payday",
        # >>>>>>> 新規: 回号サイクル特徴量 <<<<<<<<
        "draw_cycle_sin",
        "draw_cycle_cos",
        "prev_n1",
        "prev_n2",
        "prev_n3",
        "prev_sum",
        "prev_odd_count",
        "prev_big_count",
        "prev_even_count",
        "prev_small_count",
        # リーク防止: 当日値ではなく1回遅れ(履歴)で計算したローリングを使う
        "sum_ma_5_hist",
        "sum_ma_10_hist",
        "sum_std_5_hist",
        "sum_std_10_hist",
        "sum_diff_hist",
        # >>>>>>> 新規: マルチウィンドウ移動平均 <<<<<<<<
        "sum_ma_50_hist",
        # >>>>>>> 新規: 30回偏差 <<<<<<<<
        "sum_dev_30_hist",
    ]
    # 追加ラグ特徴量 (lag-2, lag-3, lag-4, lag-5)
    for lag in [2, 3, 4, 5]:
        for col in ["n1", "n2", "n3"]:
            base.append(f"prev_{lag}_{col}")
    # 各桁×各数字(0-9) の直近20回出現頻度
    for col in ["n1", "n2", "n3"]:
        for n in range(10):
            base.append(f"{col}_freq_{n}")
    # 各桁×各数字(0-9) のハマリ回数
    for col in ["n1", "n2", "n3"]:
        for n in range(10):
            base.append(f"{col}_gap_{n}")
    return base


def add_enhanced_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """ML 固有の拡張特徴量を追加する.

    共通特徴量が計算済みの DataFrame を受け取る前提。
    """
    # リーク防止: 1回遅れ(履歴)でローリング統計を作る
    hist_sum = df["digit_sum"].shift(1)
    df["sum_ma_5_hist"] = hist_sum.rolling(5, min_periods=1).mean().fillna(0)
    df["sum_ma_10_hist"] = hist_sum.rolling(10, min_periods=1).mean().fillna(0)
    df["sum_std_5_hist"] = (
        hist_sum.rolling(5, min_periods=1).std().fillna(0)
    )
    df["sum_std_10_hist"] = (
        hist_sum.rolling(10, min_periods=1).std().fillna(0)
    )
    df["sum_diff_hist"] = (df["prev_sum"] - df["sum_ma_10_hist"]).fillna(0)

    # >>>>>>> 新規: 中期 (50回) 移動平均 (履歴ベース) <<<<<<<<
    df["sum_ma_50_hist"] = hist_sum.rolling(50, min_periods=1).mean().fillna(0)

    # >>>>>>> 新規: 30回偏差 (履歴ベース) <<<<<<<<
    sum_ma_30_hist = hist_sum.rolling(30, min_periods=1).mean().fillna(0)
    df["sum_dev_30_hist"] = (df["digit_sum"].shift(1) - sum_ma_30_hist).fillna(0)

    # 追加ラグ (lag-2, lag-3, lag-4, lag-5)
    for lag in [2, 3, 4, 5]:
        for col in ["n1", "n2", "n3"]:
            df[f"prev_{lag}_{col}"] = df[col].shift(lag)

    # 各桁×各数字の直近20回出現頻度（履歴ベース: shift(1)）
    for col in ["n1", "n2", "n3"]:
        shifted = df[col].shift(1)
        for n in range(10):
            df[f"{col}_freq_{n}"] = (
                (shifted == n).astype(float).rolling(20, min_periods=1).mean()
            )

    # 各桁×各数字のハマリ回数（履歴ベース: shift(1)）
    # 直近出現からの経過回数を逐次計算する（未来参照なし）
    for col in ["n1", "n2", "n3"]:
        shifted = df[col].shift(1)
        for n in range(10):
            last_seen: int | None = None
            gaps: list[float] = []
            for i, v in enumerate(shifted.tolist()):
                if pd.isna(v):
                    gaps.append(np.nan)
                    continue
                if int(v) == n:
                    last_seen = i
                    gaps.append(0.0)
                else:
                    if last_seen is None:
                        gaps.append(np.nan)
                    else:
                        gaps.append(float(i - last_seen))
            df[f"{col}_gap_{n}"] = gaps

    return df


def compute_all_ml_features(
    df: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 10,
) -> pd.DataFrame:
    """共通 + ML 拡張特徴量をワンショットで計算する."""
    df = compute_common_features(df, window_short, window_long)
    df = add_enhanced_ml_features(df)
    return df
