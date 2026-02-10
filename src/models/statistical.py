"""
src.models.statistical — ルールベース予測器

責務:
    - Numbers3Predictor (遷移行列・合計確率・インターバル等に基づく統計予測)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.loader import normalize_numbers3_columns
from src.models.base import BasePredictor, _box_type
from src.utils.config import PREDICTION_MODELS


class Numbers3Predictor(BasePredictor):
    """統計ベースの予測器."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_features()
        if "回号" in self.df.columns:
            self.df["回号"] = pd.to_numeric(self.df["回号"], errors="coerce")
        self.df = (
            self.df.sort_values("回号").dropna(subset=["回号"]).reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def _ensure_features(self):
        if "当選番号" not in self.df.columns:
            if "当せん番号" in self.df.columns:
                self.df["当選番号"] = self.df["当せん番号"]
            else:
                raise KeyError("当選番号（または当せん番号）列が必要です。")

        num = self.df["当選番号"].astype(str).str.zfill(3)
        self.df["digit_h"] = num.str[0].astype(int)
        self.df["digit_t"] = num.str[1].astype(int)
        self.df["digit_o"] = num.str[2].astype(int)

        self.df["odd_count"] = (
            (self.df[["digit_h", "digit_t", "digit_o"]] % 2 == 1).sum(axis=1)
        )
        self.df["big_count"] = (
            (self.df[["digit_h", "digit_t", "digit_o"]] >= 5).sum(axis=1)
        )
        self.df["digit_sum"] = self.df[["digit_h", "digit_t", "digit_o"]].sum(
            axis=1
        )

    # ------------------------------------------------------------------
    def _get_transition_matrix(self, column: str) -> pd.DataFrame:
        current = self.df[column]
        nxt = self.df[column].shift(-1)
        return pd.crosstab(current, nxt, normalize="index")

    def _get_digit_intervals(self) -> Dict[str, Dict[int, int]]:
        intervals: Dict[str, Dict[int, int]] = {
            "digit_h": {},
            "digit_t": {},
            "digit_o": {},
        }
        last_index = self.df.index[-1]
        for col in ["digit_h", "digit_t", "digit_o"]:
            for n in range(10):
                series = self.df[col]
                locs = series[series == n].index
                if len(locs) > 0:
                    gap = last_index - locs[-1]
                else:
                    gap = len(self.df)
                intervals[col][n] = gap
        return intervals

    def _get_sum_probabilities(self) -> np.ndarray:
        p = np.ones(10) / 10.0
        p_sum = np.convolve(p, p)
        p_sum = np.convolve(p_sum, p)
        return p_sum

    def _get_digit_frequencies(
        self, window: int = 200
    ) -> Dict[str, Dict[int, float]]:
        df_slice = self.df if (window is None or window <= 0) else self.df.tail(window)
        freqs: Dict[str, Dict[int, float]] = {}
        for col in ["digit_h", "digit_t", "digit_o"]:
            counts = df_slice[col].value_counts(normalize=True)
            freqs[col] = {i: float(counts.get(i, 0.0)) for i in range(10)}
        return freqs

    def validate_interval_correlation(self, column: str = "digit_h") -> Dict:
        """ハマリ回数と次回出現の相関を検証（ギャンブラーの誤謬チェック）."""
        try:
            from scipy.stats import pearsonr
        except ImportError:
            return {"error": "scipy が必要です: pip install scipy"}

        results: Dict = {}
        for n in range(10):
            is_n = self.df[column] == n
            gap_list: List[int] = []
            gap = 0
            for val in is_n:
                if val:
                    gap_list.append(gap)
                    gap = 0
                else:
                    gap += 1
            if len(gap_list) < 20:
                continue
            gaps = np.array(gap_list)
            corr, pval = pearsonr(gaps[:-1], gaps[1:])
            results[n] = {
                "correlation": round(float(corr), 4),
                "p_value": round(float(pval), 4),
            }

        significant = sum(1 for r in results.values() if r["p_value"] < 0.05)
        total = len(results)
        return {
            "column": column,
            "per_digit": results,
            "significant_count": significant,
            "total_tested": total,
            "conclusion": (
                "相関なし（ギャンブラーの誤謬）"
                if significant <= max(1, int(total * 0.1))
                else f"{significant}/{total} 個の数字で有意な相関あり"
            ),
        }

    # ------------------------------------------------------------------
    def predict(
        self,
        top_n: int = 20,
        model: str = "hybrid",
        recent_window: int = 200,
        interval_weight: float = 0.0,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """統計ベース予測."""
        model = model.lower()
        supported = {
            "hybrid",
            "transition_sum",
            "interval_sum",
            "sum_only",
            "frequency",
            "recent_frequency",
            "interval_boost",
        }
        if model not in supported:
            raise ValueError(f"Unsupported model: {model}")

        use_transition = model in {"hybrid", "transition_sum"}
        use_interval = model in {"hybrid", "interval_sum", "interval_boost"}
        use_sum = True
        use_frequency = model in {"frequency", "recent_frequency"}
        if model == "interval_boost":
            interval_weight = max(interval_weight, 0.3)

        if use_interval and interval_weight > 0 and verbose:
            print(
                "⚠️ 注意: ハマリ係数はギャンブラーの誤謬に該当し得ます。"
                "validate_interval_correlation() で効果を検証してください。"
            )

        trans_odd = (
            self._get_transition_matrix("odd_count") if use_transition else None
        )
        trans_big = (
            self._get_transition_matrix("big_count") if use_transition else None
        )

        last_row = self.df.iloc[-1]
        current_odd = last_row["odd_count"]
        current_big = last_row["big_count"]
        current_num = last_row["当選番号"]
        if verbose:
            print(
                f"直近の結果: 回号={last_row['回号']}, 番号={current_num}, "
                f"奇数={current_odd}個, Big={current_big}個"
            )

        intervals = self._get_digit_intervals() if use_interval else None
        sum_probs = self._get_sum_probabilities() if use_sum else None
        freqs = (
            self._get_digit_frequencies(window=recent_window)
            if use_frequency
            else None
        )

        candidates: List[Dict] = []
        for n in range(1000):
            s_num = f"{n:03d}"
            h, t, o = int(s_num[0]), int(s_num[1]), int(s_num[2])

            c_sum = h + t + o
            c_odd = (h % 2) + (t % 2) + (o % 2)
            c_big = (1 if h >= 5 else 0) + (1 if t >= 5 else 0) + (1 if o >= 5 else 0)

            if use_transition:
                try:
                    prob_odd_trans = trans_odd.loc[current_odd, c_odd]
                except KeyError:
                    prob_odd_trans = 0
                try:
                    prob_big_trans = trans_big.loc[current_big, c_big]
                except KeyError:
                    prob_big_trans = 0
                pattern_score = prob_odd_trans * prob_big_trans
            else:
                pattern_score = 1.0

            sum_score = sum_probs[c_sum] if use_sum else 1.0

            if use_interval:
                total_gap = (
                    intervals["digit_h"][h]
                    + intervals["digit_t"][t]
                    + intervals["digit_o"][o]
                )
                interval_bonus = 1.0 + np.log1p(total_gap) * interval_weight
            else:
                total_gap = 0
                interval_bonus = 1.0

            if use_frequency:
                freq_score = (
                    freqs["digit_h"][h] * freqs["digit_t"][t] * freqs["digit_o"][o]
                )
            else:
                freq_score = 1.0

            final_score = pattern_score * sum_score * interval_bonus * freq_score
            box_t, box_count = _box_type(s_num)

            candidates.append(
                {
                    "モデル": model,
                    "予測番号": s_num,
                    "総合スコア": final_score,
                    "パターン確率": pattern_score if use_transition else 0.0,
                    "合計値確率": sum_score if use_sum else 0.0,
                    "頻度スコア": freq_score if use_frequency else 0.0,
                    "ハマリ係数": interval_bonus,
                    "直前ギャップ和": total_gap,
                    "合計": c_sum,
                    "奇数数": c_odd,
                    "Big数": c_big,
                    "タイプ": box_t,
                    "ボックス通り数": box_count,
                }
            )

        df_pred = pd.DataFrame(candidates)
        df_pred = df_pred.sort_values("総合スコア", ascending=False).reset_index(
            drop=True
        )

        if not df_pred.empty:
            max_score = df_pred["総合スコア"].max()
            if max_score > 0:
                df_pred["総合スコア"] = (
                    df_pred["総合スコア"] / max_score * 100.0
                ).round(1)

        if verbose and not df_pred.empty:
            top = df_pred.iloc[0]
            print(
                f"予測根拠: 番号={top['予測番号']}, スコア={top['総合スコア']}"
                f", パターン={top['パターン確率']:.3f}"
                f", 合計={top['合計値確率']:.3f}"
                f", ハマリ={top['ハマリ係数']:.2f}"
                f", 奇数={top['奇数数']}, Big={top['Big数']}"
                f", {top['タイプ']}({top['ボックス通り数']}通り)"
            )

        return df_pred.head(top_n)

    # ------------------------------------------------------------------
    def predict_next(self) -> str:
        """Top-1 の予測番号を返す."""
        result = self.predict(top_n=1, model="hybrid", verbose=False)
        if result.empty:
            return "000"
        return str(result.iloc[0]["予測番号"])
