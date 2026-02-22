"""
src.analysis.parallel_backtester — 並列バックテスト実装

責務:
    - CPU並列処理によるバックテスト高速化
    - バッチ単位での分散処理
    - プロセスプールを使った予測並列化

使い方:
    backtester = ParallelMLBacktester(df, workers=4)
    results, perf = backtester.run()
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.backtester import Numbers3MLBacktester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelMLBacktester(Numbers3MLBacktester):
    """並列処理対応のMLバックテスター.

    Numbers3MLBacktesterを継承し、予測ループを並列化する。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 300,
        test_rounds: int = 50,
        workers: int | None = None,
        batch_size: int = 10,
        **kwargs
    ):
        """初期化.

        Parameters
        ----------
        df : DataFrame
            元データ
        window : int
            学習ウィンドウ
        test_rounds : int
            テストラウンド数
        workers : int, optional
            並列ワーカー数 (Noneの場合はCPU数-1)
        batch_size : int
            バッチサイズ (大きいほど並列効率向上、小さいほど進捗更新頻繁)
        **kwargs : その他のNumbers3MLBacktesterパラメータ
        """
        super().__init__(df, window, test_rounds, **kwargs)

        if workers is None:
            cpu_count = os.cpu_count() or 1
            workers = max(1, cpu_count - 1)

        self.workers = workers
        self.batch_size = batch_size

        logger.info(f"並列バックテスター初期化: workers={self.workers}, batch_size={self.batch_size}")

    def run_parallel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """並列バックテストを実行.

        Returns
        -------
        results_df : DataFrame
            各ラウンドの結果
        performance_df : DataFrame
            パフォーマンストラッキング
        """
        total = min(self.test_rounds, len(self.df) - 2)

        print(f"\n{'=' * 60}")
        print("並列MLバックテスト開始")
        print(f"Test rounds: {total}, Window: {self.window}")
        print(f"Workers: {self.workers}, Batch size: {self.batch_size}")
        print(f"{'=' * 60}\n")

        # バッチに分割
        batches = [
            list(range(i, min(i + self.batch_size, total)))
            for i in range(0, total, self.batch_size)
        ]

        all_results = []
        all_performance = []

        # 並列実行
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # 各バッチをサブミット
            future_to_batch = {
                executor.submit(
                    self._run_batch,
                    batch_indices,
                    self.df.copy(),
                    self.window,
                    self.num_boost_round,
                    self.track_performance,
                ): batch_idx
                for batch_idx, batch_indices in enumerate(batches)
            }

            # 結果を収集
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results, batch_perf = future.result()
                    all_results.extend(batch_results)
                    all_performance.extend(batch_perf)

                    completed = len(all_results)
                    pct = completed / total * 100
                    logger.info(f"[Batch {batch_idx + 1}/{len(batches)}] 完了: {completed}/{total} ({pct:.1f}%)")

                except Exception as e:
                    logger.error(f"Batch {batch_idx} でエラー: {e}")

        # 累積損益を再計算 (バッチ並列後に順序を復元)
        all_results = sorted(all_results, key=lambda x: x.get("round_index", 0))
        cumulative = 0
        for r in all_results:
            cumulative += r["set_profit"]
            r["cumulative_profit"] = cumulative

        results_df = pd.DataFrame(all_results)
        performance_df = pd.DataFrame(all_performance) if all_performance else pd.DataFrame()

        # サマリ表示
        self._print_summary(results_df)

        return results_df, performance_df

    @staticmethod
    def _run_batch(
        batch_indices: list[int],
        df: pd.DataFrame,
        window: int,
        num_boost_round: int,
        track_performance: bool,
    ) -> tuple[list[dict], list[dict]]:
        """バッチ単位でバックテストを実行 (プロセス内).

        Parameters
        ----------
        batch_indices : list of int
            処理するラウンドインデックス
        df : DataFrame
            元データ (コピー)
        window : int
            学習ウィンドウ
        num_boost_round : int
            ブースティングラウンド数
        track_performance : bool
            パフォーマンストラッキング有効化

        Returns
        -------
        results : list of dict
            各ラウンドの結果
        performance : list of dict
            パフォーマンス記録
        """
        from src.models.base import _box_key, evaluate_set_profit
        from src.models.ml_wrapper import Numbers3MLPredictor
        from src.utils.config import TICKET_COST

        results = []
        performance = []

        for i in batch_indices:
            train_end = len(df) - 2 - i
            train_start = max(0, train_end - window)
            train_df = df.iloc[train_start : train_end + 1].copy()

            actual_row = df.iloc[train_end + 1]
            actual_num = str(actual_row["当選番号"]).zfill(3)

            # 予測
            predictor = Numbers3MLPredictor(
                train_df,
                num_boost_round=num_boost_round,
                random_state=42,
            )
            predictor.train()
            predicted_num = predictor.predict_next()

            # 評価
            prize, hit_type = evaluate_set_profit(actual_num, predicted_num)
            profit = prize - TICKET_COST

            straight_hit = 1 if actual_num == predicted_num else 0
            box_hit = 1 if _box_key(actual_num) == _box_key(predicted_num) else 0

            results.append({
                "round_index": i,  # ソート用
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
                "match_exact": straight_hit,
                "match_box": box_hit,
            })

            if track_performance:
                # パフォーマンス記録 (簡易版)
                X_test = predictor._build_latest_features()
                actual_digits = {
                    "n1": int(actual_num[0]),
                    "n2": int(actual_num[1]),
                    "n3": int(actual_num[2]),
                }

                # LogLossと信頼度を計算
                try:
                    from sklearn.metrics import log_loss
                    np.array([actual_digits["n1"]])
                    np.array([actual_digits["n2"]])
                    np.array([actual_digits["n3"]])

                    probs_n1 = predictor.model_n1.predict_proba(X_test)[0]
                    probs_n2 = predictor.model_n2.predict_proba(X_test)[0]
                    probs_n3 = predictor.model_n3.predict_proba(X_test)[0]

                    ll_n1 = log_loss([actual_digits["n1"]], [probs_n1], labels=list(range(10)))
                    ll_n2 = log_loss([actual_digits["n2"]], [probs_n2], labels=list(range(10)))
                    ll_n3 = log_loss([actual_digits["n3"]], [probs_n3], labels=list(range(10)))
                    avg_logloss = (ll_n1 + ll_n2 + ll_n3) / 3

                    confidence_score = float(
                        probs_n1[actual_digits["n1"]]
                        * probs_n2[actual_digits["n2"]]
                        * probs_n3[actual_digits["n3"]]
                    )
                except Exception:
                    avg_logloss = None
                    confidence_score = None

                performance.append({
                    "round_index": i,
                    "target_round": (
                        int(actual_row["回号"])
                        if pd.notna(actual_row["回号"])
                        else None
                    ),
                    "actual_number": actual_num,
                    "predicted_number": predicted_num,
                    "straight_hit": straight_hit,
                    "box_hit": box_hit,
                    "avg_logloss": avg_logloss,
                    "confidence_score": confidence_score,
                })

        return results, performance

    def _print_summary(self, results_df: pd.DataFrame):
        """結果サマリを表示."""
        total_cost = len(results_df) * 200
        total_return = results_df["set_prize"].sum()
        total_profit = total_return - total_cost
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        straight_hits = results_df["match_exact"].sum()
        box_hits = results_df["match_box"].sum()

        print(f"\n{'=' * 60}")
        print("📊 バックテスト結果")
        print(f"{'=' * 60}")
        print(f"  総ラウンド: {len(results_df)}")
        print(f"  ストレート的中: {straight_hits}")
        print(f"  ボックス的中: {box_hits}")
        print(f"  投資総額: ¥{total_cost:,}")
        print(f"  払戻総額: ¥{total_return:,}")
        print(f"  損益: ¥{total_profit:+,}")
        print(f"  ROI: {roi:+.2f}%")
        print(f"{'=' * 60}\n")


# 便利関数
def run_parallel_backtest(
    df: pd.DataFrame,
    test_rounds: int = 100,
    workers: int | None = None,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """並列バックテストを実行する便利関数.

    Parameters
    ----------
    df : DataFrame
        元データ
    test_rounds : int
        テストラウンド数
    workers : int, optional
        並列ワーカー数
    **kwargs : その他のパラメータ

    Returns
    -------
    results_df : DataFrame
        各ラウンドの結果
    performance_df : DataFrame
        パフォーマンス記録
    """
    backtester = ParallelMLBacktester(
        df,
        test_rounds=test_rounds,
        workers=workers,
        **kwargs
    )
    return backtester.run_parallel()
