"""
src.features.transformer — スケーリング・正規化

責務:
    - 特徴量のスケーリング (StandardScaler / MinMaxScaler)
    - 正規化前後の列名保持
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


class FeatureScaler:
    """ML 特徴量のスケーリングを担当するクラス.

    sklearn に依存せず、DataFrame 単位で StandardScaling /
    MinMaxScaling を行う軽量な実装。
    """

    def __init__(self, method: str = "standard"):
        """
        Parameters
        ----------
        method : str
            'standard' (z-score) or 'minmax' (0-1)
        """
        if method not in ("standard", "minmax"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.min_: Optional[pd.Series] = None
        self.max_: Optional[pd.Series] = None
        self.columns_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> "FeatureScaler":
        """統計量を学習する."""
        cols = columns if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
        self.columns_ = cols
        subset = df[cols].astype(float)

        if self.method == "standard":
            self.mean_ = subset.mean()
            self.std_ = subset.std().replace(0, 1.0)
        else:
            self.min_ = subset.min()
            self.max_ = subset.max()

        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit 済みの統計量を使って変換する."""
        if self.columns_ is None:
            raise RuntimeError("fit() を先に呼び出してください。")
        out = df.copy()
        if self.method == "standard":
            assert self.mean_ is not None and self.std_ is not None
            out[self.columns_] = (out[self.columns_].astype(float) - self.mean_) / self.std_
        else:
            assert self.min_ is not None and self.max_ is not None
            range_ = (self.max_ - self.min_).replace(0, 1.0)
            out[self.columns_] = (out[self.columns_].astype(float) - self.min_) / range_
        return out

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """fit + transform を一括で実行する."""
        return self.fit(df, columns).transform(df)

    # ------------------------------------------------------------------
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """スケーリングを元に戻す."""
        if self.columns_ is None:
            raise RuntimeError("fit() を先に呼び出してください。")
        out = df.copy()
        if self.method == "standard":
            assert self.mean_ is not None and self.std_ is not None
            out[self.columns_] = out[self.columns_].astype(float) * self.std_ + self.mean_
        else:
            assert self.min_ is not None and self.max_ is not None
            range_ = (self.max_ - self.min_).replace(0, 1.0)
            out[self.columns_] = out[self.columns_].astype(float) * range_ + self.min_
        return out
