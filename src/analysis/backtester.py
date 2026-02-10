"""
src.analysis.backtester — バックテスト・収支シミュレーション

責務:
    - Numbers3Backtester          (ルールベースバックテスト)
    - Numbers3MLBacktester        (ML バックテスト + 精度追跡)
    - 収支シミュレーション        (ストレートのみ / セット / ボックスのみ)
    - 最大ドローダウン計算
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.features.engineer import Numbers3FeatureEngineer
from src.models.base import _box_key, _box_type, evaluate_set_profit
from src.models.ml_wrapper import Numbers3MLPredictor, extract_feature_importance
from src.models.statistical import Numbers3Predictor
from src.utils.config import (
    BOX_PRIZE_DOUBLE,
    BOX_PRIZE_SINGLE,
    FEATURE_IMPORTANCE_CSV_PATH,
    PERFORMANCE_CSV_PATH,
    SET_BOX_PRIZE,
    SET_STRAIGHT_PRIZE,
    STRAIGHT_PRIZE,
    TICKET_COST,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =====================================================================
# 最大ドローダウン (Prompt C)
# =====================================================================


def max_drawdown(cumulative_profits: pd.Series) -> Dict[str, Any]:
    """累積損益から最大ドローダウンを計算する.

    Returns
    -------
    Dict
        max_drawdown : float
        peak_idx : int
        trough_idx : int
    """
    cum = cumulative_profits.values.astype(float)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    trough_idx = int(np.argmin(dd))
    peak_idx = int(np.argmax(cum[:trough_idx + 1])) if trough_idx > 0 else 0
    return {
        "max_drawdown": float(dd.min()),
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
    }


# =====================================================================
# 収支シミュレーション (Prompt C)
# =====================================================================


def simulate_revenue(
    result_df: pd.DataFrame,
    strategy: str = "set",
) -> pd.DataFrame:
    """バックテスト結果から収支シミュレーションを計算する.

    Parameters
    ----------
    result_df : DataFrame
        バックテスト結果 (actual, set_pick 列を含む)
    strategy : str
        'straight' / 'set' / 'box'

    Returns
    -------
    DataFrame  各ラウンドの損益を追加したテーブル
    """
    records: List[Dict] = []
    cumulative = 0

    for _, row in result_df.iterrows():
        actual = str(row.get("actual", "")).zfill(3)
        predicted = str(row.get("set_pick", "")).zfill(3)

        if strategy == "straight":
            prize = STRAIGHT_PRIZE if actual == predicted else 0
            cost = TICKET_COST
        elif strategy == "box":
            if _box_key(actual) == _box_key(predicted):
                _, box_count = _box_type(actual)
                if box_count == 6:
                    prize = BOX_PRIZE_SINGLE
                elif box_count == 3:
                    prize = BOX_PRIZE_DOUBLE
                else:
                    prize = STRAIGHT_PRIZE  # トリプル = ストレート同額
            else:
                prize = 0
            cost = TICKET_COST
        else:  # set (default)
            prize, _ = evaluate_set_profit(actual, predicted)
            cost = TICKET_COST

        profit = prize - cost
        cumulative += profit
        records.append(
            {
                "target_round": row.get("target_round"),
                "actual": actual,
                "predicted": predicted,
                "strategy": strategy,
                "prize": prize,
                "cost": cost,
                "profit": profit,
                "cumulative_profit": cumulative,
            }
        )

    return pd.DataFrame(records)


# =====================================================================
# ルールベース バックテスター
# =====================================================================


class Numbers3Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        engineer: Numbers3FeatureEngineer,
        window: int = 200,
        top_n: int = 50,
    ):
        self.df = df.sort_values("回号").reset_index(drop=True)
        self.fe = engineer
        self.window = window
        self.top_n = top_n

    def _prepare_features(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        if "digit_h" in df_slice.columns and "sum_ma_20" in df_slice.columns:
            return df_slice
        return self.fe.add_features(df_slice)

    def run(
        self,
        test_rounds: int = 50,
        top_k_list: Tuple[int, ...] = (10, 50, 100),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        results: List[Dict] = []
        total_return = 0
        total_cost = 0
        set_straight_hits = 0
        set_box_hits = 0
        total = min(test_rounds, len(self.df) - 2)

        for i in range(total):
            train_end = len(self.df) - 2 - i
            train_start = max(0, train_end - self.window)
            train_df = self.df.iloc[train_start : train_end + 1].copy()
            train_df = self._prepare_features(train_df)

            actual_row = self.df.iloc[train_end + 1]
            actual_num = str(actual_row["当選番号"]).zfill(3)
            actual_box = _box_key(actual_num)

            predictor = Numbers3Predictor(train_df)
            pred_df = predictor.predict(top_n=self.top_n, verbose=False)
            ranked_nums = pred_df["予測番号"].tolist()
            ranked_boxes = [_box_key(n) for n in ranked_nums]

            top_pick = ranked_nums[0] if ranked_nums else None
            if top_pick is None:
                prize, hit_type = 0, "外れ"
            else:
                prize, hit_type = evaluate_set_profit(actual_num, top_pick)
            profit = prize - TICKET_COST
            total_return += prize
            total_cost += TICKET_COST
            if hit_type == "セット・ストレート":
                set_straight_hits += 1
            elif hit_type == "セット・ボックス":
                set_box_hits += 1

            rank_st = (
                ranked_nums.index(actual_num) + 1
                if actual_num in ranked_nums
                else None
            )
            rank_box = (
                ranked_boxes.index(actual_box) + 1
                if actual_box in ranked_boxes
                else None
            )

            hit_topk = {
                f"st_top{k}": (rank_st is not None and rank_st <= k)
                for k in top_k_list
            }
            hit_topk.update(
                {
                    f"box_top{k}": (rank_box is not None and rank_box <= k)
                    for k in top_k_list
                }
            )

            results.append(
                {
                    "target_round": (
                        int(actual_row["回号"])
                        if pd.notna(actual_row["回号"])
                        else None
                    ),
                    "actual": actual_num,
                    "rank_st": rank_st,
                    "rank_box": rank_box,
                    "set_pick": top_pick,
                    "set_hit": hit_type,
                    "set_prize": prize,
                    "set_profit": profit,
                    **hit_topk,
                }
            )

        result_df = pd.DataFrame(results)
        summary = {
            "tested": len(result_df),
            "set_straight_hits": set_straight_hits,
            "set_box_hits": set_box_hits,
            "total_return": total_return,
            "total_cost": total_cost,
            "total_profit": total_return - total_cost,
            "roi_pct": (
                round((total_return - total_cost) / total_cost * 100.0, 2)
                if total_cost > 0
                else 0.0
            ),
            **{
                k: int(result_df[k].sum())
                for k in result_df.columns
                if k.startswith("st_top")
            },
            **{
                k: int(result_df[k].sum())
                for k in result_df.columns
                if k.startswith("box_top")
            },
        }
        return result_df, pd.DataFrame([summary])


# =====================================================================
# ML バックテスター (精度追跡対応)
# =====================================================================


class Numbers3MLBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 300,
        test_rounds: int = 50,
        valid_size: int = 180,
        tune_every: Optional[int] = None,
        param_grid: Optional[Dict] = None,
        num_boost_round: int = 120,
        random_state: int = 42,
        track_performance: bool = True,
        performance_csv: str = PERFORMANCE_CSV_PATH,
    ):
        self.df = df.sort_values("回号").reset_index(drop=True)
        self.window = window
        self.test_rounds = test_rounds
        self.valid_size = valid_size
        self.tune_every = tune_every
        self.param_grid = param_grid
        self.num_boost_round = num_boost_round
        self.random_state = random_state
        self.track_performance = track_performance
        self.performance_csv = performance_csv

    # ------------------------------------------------------------------
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        results: List[Dict] = []
        performance_records: List[Dict] = []
        total_return = 0
        total_cost = 0
        set_straight_hits = 0
        set_box_hits = 0

        backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        total = min(self.test_rounds, len(self.df) - 2)

        print(f"\n{'=' * 60}")
        print(f"Starting ML Backtest: (ID:{backtest_id})")
        print(
            f"Test rounds: {total}, Window: {self.window}, "
            f"Boost rounds: {self.num_boost_round}"
        )
        print(f"{'=' * 60}\n")

        predictor: Optional[Numbers3MLPredictor] = None

        for i in range(total):
            train_end = len(self.df) - 2 - i
            train_start = max(0, train_end - self.window)
            train_df = self.df.iloc[train_start : train_end + 1].copy()

            actual_row = self.df.iloc[train_end + 1]
            actual_num = str(actual_row["当選番号"]).zfill(3)
            actual_box = _box_key(actual_num)

            best_params = None
            if self.tune_every and i % self.tune_every == 0:
                print(f"[Round {i+1}/{total}] Tuning hyperparameters...")
                predictor = Numbers3MLPredictor(
                    train_df,
                    num_boost_round=self.num_boost_round,
                    random_state=self.random_state,
                )
                tune_result = predictor.tune_hyperparams(
                    param_grid=self.param_grid,
                    valid_size=self.valid_size,
                )
                best_params = tune_result["best_params"]
                print(f"  Best params: {best_params}")
                predictor.best_params = best_params
                predictor.train()
            else:
                predictor = Numbers3MLPredictor(
                    train_df,
                    num_boost_round=self.num_boost_round,
                    random_state=self.random_state,
                )
                predictor.train()

            predicted_num = predictor.predict_next()

            prize, hit_type = evaluate_set_profit(actual_num, predicted_num)
            profit = prize - TICKET_COST
            total_return += prize
            total_cost += TICKET_COST

            straight_hit = 1 if actual_num == predicted_num else 0
            box_hit = 1 if actual_box == _box_key(predicted_num) else 0

            if hit_type == "セット・ストレート":
                set_straight_hits += 1
            elif hit_type == "セット・ボックス":
                set_box_hits += 1

            results.append(
                {
                    "target_round": (
                        int(actual_row["回号"])
                        if pd.notna(actual_row["回号"])
                        else None
                    ),
                    "actual": actual_num,
                    "set_pick": predicted_num,
                    "set_hit": hit_type,
                    "set_prize": prize,
                    "set_profit": profit,
                    "match_exact": int(actual_num == predicted_num),
                    "match_box": int(actual_box == _box_key(predicted_num)),
                }
            )

            if self.track_performance:
                X_test = predictor._build_latest_features()
                actual_digits = {
                    "n1": int(actual_num[0]),
                    "n2": int(actual_num[1]),
                    "n3": int(actual_num[2]),
                }
                digit_accuracy = self._calculate_digit_accuracy(
                    actual_num, predicted_num
                )
                logloss_confidence = self._calculate_logloss_and_confidence(
                    predictor, X_test, actual_digits
                )

                used_params = best_params if best_params else {}
                learning_rate = used_params.get("learning_rate", 0.1)
                num_leaves = used_params.get("num_leaves", 31)
                max_depth_val = used_params.get("max_depth", -1)

                cumulative_profit = total_return - total_cost
                roi_percent = (
                    (cumulative_profit / total_cost * 100) if total_cost > 0 else 0
                )

                performance_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "backtest_id": backtest_id,
                    "target_round": (
                        int(actual_row["回号"])
                        if pd.notna(actual_row["回号"])
                        else None
                    ),
                    "actual_number": actual_num,
                    "predicted_number": predicted_num,
                    "model_type": "lightgbm",
                    "window_size": self.window,
                    "num_boost_round": self.num_boost_round,
                    "learning_rate": learning_rate,
                    "num_leaves": num_leaves,
                    "max_depth": max_depth_val,
                    "straight_hit": straight_hit,
                    "box_hit": box_hit,
                    **digit_accuracy,
                    **logloss_confidence,
                    "prize_amount": prize,
                    "profit": profit,
                    "cumulative_profit": cumulative_profit,
                    "roi_percent": round(roi_percent, 2),
                }
                performance_records.append(performance_record)

                if (i + 1) % 10 == 0 or i == 0:
                    print(
                        f"[Round {i+1}/{total}] Target: {int(actual_row['回号'])} | "
                        f"Predicted: {predicted_num} | Actual: {actual_num} | "
                        f"Hit: {hit_type} | Cumulative: {cumulative_profit:+,}円"
                    )

        result_df = pd.DataFrame(results)

        summary: Dict[str, Any] = {
            "tested": len(result_df),
            "set_straight_hits": set_straight_hits,
            "set_box_hits": set_box_hits,
            "total_return": total_return,
            "total_cost": total_cost,
            "total_profit": total_return - total_cost,
            "roi_pct": (
                round((total_return - total_cost) / total_cost * 100.0, 2)
                if total_cost > 0
                else 0.0
            ),
            "exact_hits": (
                int(result_df["match_exact"].sum()) if not result_df.empty else 0
            ),
            "box_hits": (
                int(result_df["match_box"].sum()) if not result_df.empty else 0
            ),
        }

        # ランダムベースラインとの比較
        random_straight_rate = 1 / 1000.0
        random_logloss = -np.log(0.1)
        random_brier = 0.09

        model_avg_logloss = None
        model_avg_brier = None
        p_value_straight = None

        if performance_records:
            model_avg_logloss = float(
                np.mean(
                    [
                        r.get("logloss_avg", random_logloss)
                        for r in performance_records
                    ]
                )
            )
            model_avg_brier = float(
                np.mean(
                    [r.get("brier_avg", random_brier) for r in performance_records]
                )
            )

        try:
            from scipy.stats import binomtest

            binom_result = binomtest(
                set_straight_hits, total, random_straight_rate, alternative="greater"
            )
            p_value_straight = float(binom_result.pvalue)
        except Exception:
            pass

        summary.update(
            {
                "random_straight_rate": random_straight_rate,
                "random_logloss": random_logloss,
                "random_brier": random_brier,
                "model_avg_logloss": model_avg_logloss,
                "model_avg_brier": model_avg_brier,
                "p_value_vs_random": p_value_straight,
            }
        )

        # --- 最大ドローダウン (Prompt C) ---
        if not result_df.empty:
            cum_profits = pd.Series(
                [r["set_profit"] for r in results]
            ).cumsum()
            dd_info = max_drawdown(cum_profits)
            summary["max_drawdown"] = dd_info["max_drawdown"]

        print(f"\n{'=' * 60}")
        print("📊 ランダムベースライン比較")
        print(f"{'=' * 60}")
        print(
            f"  モデル的中率 (ストレート): {set_straight_hits}/{total} "
            f"= {set_straight_hits / max(total, 1) * 100:.2f}%"
        )
        print(f"  ランダム期待的中率: {random_straight_rate * 100:.2f}%")
        if model_avg_logloss is not None:
            print(f"  モデル平均 Log-Loss: {model_avg_logloss:.4f}")
            print(f"  ランダム Log-Loss: {random_logloss:.4f}")
            improvement = (
                (random_logloss - model_avg_logloss) / random_logloss * 100
            )
            if improvement > 0:
                print(f"  ✅ ランダムより {improvement:.1f}% 改善")
            else:
                print(f"  ⚠️ ランダムより {-improvement:.1f}% 悪化")
        if model_avg_brier is not None:
            print(f"  モデル平均 Brier Score: {model_avg_brier:.4f}")
            print(f"  ランダム Brier Score: {random_brier:.4f}")
        if p_value_straight is not None:
            print(f"  二項検定 p値: {p_value_straight:.4f}")
            if p_value_straight < 0.05:
                print("  ✅ 統計的に有意にランダムを上回っています")
            else:
                print("  ⚠️ ランダムとの有意差なし")
        if "max_drawdown" in summary:
            print(f"  最大ドローダウン: {summary['max_drawdown']:+,.0f}円")
        print(f"{'=' * 60}")

        if self.track_performance and performance_records:
            performance_df = pd.DataFrame(performance_records)
            self._ensure_csv_file(self.performance_csv, performance_df.columns)

            if os.path.exists(self.performance_csv):
                performance_df.to_csv(
                    self.performance_csv,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                performance_df.to_csv(
                    self.performance_csv,
                    index=False,
                    encoding="utf-8-sig",
                )

            print(f"\n✓ Performance tracking saved to: {self.performance_csv}")
            print(f"  Total records added: {len(performance_records)}")

        # 特徴量重要度を保存
        if self.track_performance and predictor is not None and predictor.models:
            try:
                extract_feature_importance(
                    predictor, save_path=FEATURE_IMPORTANCE_CSV_PATH
                )
                print(
                    f"\n✓ Feature importance saved to: {FEATURE_IMPORTANCE_CSV_PATH}"
                )
            except Exception as e:
                print(f"\n⚠️  Could not save feature importance: {e}")

        return result_df, pd.DataFrame([summary])

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_csv_file(filepath: str, columns: pd.Index) -> None:
        if not filepath:
            return
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, "rb") as f:
                header = f.read(4)
        except Exception:
            return

        if not header.startswith(b"PK\x03\x04"):
            return

        backup_path = filepath + ".xlsx"
        if not os.path.exists(backup_path):
            shutil.move(filepath, backup_path)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = filepath + f".{ts}.xlsx"
            shutil.move(filepath, backup_path)

        try:
            df = pd.read_excel(backup_path)
            schema_cols = {"カラム名", "データ型", "説明", "例"}
            if schema_cols.issubset(set(df.columns)):
                pd.DataFrame(columns=columns).to_csv(
                    filepath, index=False, encoding="utf-8-sig"
                )
            else:
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
        except Exception:
            pd.DataFrame(columns=columns).to_csv(
                filepath, index=False, encoding="utf-8-sig"
            )

    @staticmethod
    def _calculate_digit_accuracy(
        actual_str: str, predicted_str: str
    ) -> Dict[str, int]:
        actual = str(actual_str).zfill(3)
        predicted = str(predicted_str).zfill(3)
        return {
            "digit1_correct": 1 if actual[0] == predicted[0] else 0,
            "digit2_correct": 1 if actual[1] == predicted[1] else 0,
            "digit3_correct": 1 if actual[2] == predicted[2] else 0,
        }

    @staticmethod
    def _calculate_logloss_and_confidence(
        predictor: Numbers3MLPredictor,
        X_test: pd.DataFrame,
        actual_digits: Dict[str, int],
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        digit_names = ["n1", "n2", "n3"]

        for i, digit in enumerate(digit_names, 1):
            proba = predictor._predict_proba(digit, X_test)[0]
            actual_digit = actual_digits[digit]
            y_true = [actual_digit]
            y_pred_proba = [proba]
            logloss_val = log_loss(
                y_true, y_pred_proba, labels=list(range(10))
            )
            predicted_digit = int(np.argmax(proba))
            confidence = proba[predicted_digit]

            actual_one_hot = np.zeros(10)
            actual_one_hot[actual_digit] = 1.0
            brier = float(np.mean((proba - actual_one_hot) ** 2))

            results[f"logloss_digit{i}"] = logloss_val
            results[f"confidence_digit{i}"] = confidence
            results[f"brier_digit{i}"] = brier

        results["logloss_avg"] = float(
            np.mean([results[f"logloss_digit{i}"] for i in range(1, 4)])
        )
        results["confidence_avg"] = float(
            np.mean([results[f"confidence_digit{i}"] for i in range(1, 4)])
        )
        results["brier_avg"] = float(
            np.mean([results[f"brier_digit{i}"] for i in range(1, 4)])
        )
        return results

    @staticmethod
    def save_performance_record(
        record: Dict,
        filepath: str = PERFORMANCE_CSV_PATH,
        mode: str = "append",
    ) -> None:
        df = pd.DataFrame([record])
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        file_exists = os.path.exists(filepath)

        if file_exists:
            Numbers3MLBacktester._ensure_csv_file(filepath, df.columns)

        if mode == "append" and os.path.exists(filepath):
            df.to_csv(
                filepath,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig",
            )
        else:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
