"""
src.models.base — 予測器の抽象基底クラス

責務:
    - 予測器の共通インターフェース定義
    - ユーティリティ関数 (ボックスキー, 損益評価)
"""
from __future__ import annotations

import abc
from typing import Tuple

import pandas as pd

from src.utils.config import SET_BOX_PRIZE, SET_STRAIGHT_PRIZE


# =====================================================================
# ユーティリティ
# =====================================================================


def box_key(number_str: str) -> str:
    """番号をボックスキー (ソート済み文字列) に変換する."""
    return "".join(sorted(number_str))


def box_type(number_str: str) -> Tuple[str, int]:
    """番号のタイプ (トリプル/ダブル/シングル) と通り数を返す."""
    digits = list(number_str)
    unique = len(set(digits))
    if unique == 1:
        return "トリプル", 1
    if unique == 2:
        return "ダブル", 3
    return "シングル", 6


def evaluate_set_profit(actual_num: str, predicted_num: str) -> Tuple[int, str]:
    """セット購入時の賞金と当落判定を返す."""
    if actual_num == predicted_num:
        return SET_STRAIGHT_PRIZE, "セット・ストレート"
    if box_key(actual_num) == box_key(predicted_num):
        return SET_BOX_PRIZE, "セット・ボックス"
    return 0, "外れ"


# =====================================================================
# 抽象基底クラス
# =====================================================================


class BasePredictor(abc.ABC):
    """全予測器が実装すべき共通インターフェース."""

    @abc.abstractmethod
    def predict(self, **kwargs) -> pd.DataFrame:
        """予測結果を DataFrame で返す."""
        ...

    @abc.abstractmethod
    def predict_next(self) -> str:
        """次回の予測番号 (3桁文字列) を返す."""
        ...


# 後方互換エイリアス (内部で使っていた private 名)
_box_key = box_key
_box_type = box_type
