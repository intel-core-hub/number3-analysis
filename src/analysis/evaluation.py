"""
src.analysis.evaluation — 多角的評価指標モジュール

責務:
    - Top-K 的中率
    - ニアミス率 (tolerance ±1, ±2)
    - Brier Score (確率予測精度)
    - 数字別 Recall / Precision
    - ボックス的中率 vs 理論値の統計検定
    - 時系列安定性評価 (期間別精度 + 信頼区間)
    - ベースライン比較 (ランダム / 頻度 / ナイーブ)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =====================================================================
# Top-K 的中率
# =====================================================================


def topk_hit_rate(
    actuals: list[str],
    candidates_list: list[list[str]],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """実際の当選番号が上位 K 件の予測に含まれる率を算出.

    Parameters
    ----------
    actuals : list of str
        実際の当選番号 (3桁文字列)
    candidates_list : list of list of str
        各ラウンドの予測候補リスト (各リストは確率降順で並んだ予測番号)
    k_values : list of int, optional
        評価する K 値。デフォルト [5, 10, 20, 50, 100]

    Returns
    -------
    Dict[int, float]  K -> 的中率 (0.0-1.0)
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100]
    n = len(actuals)
    if n == 0:
        return {k: 0.0 for k in k_values}

    results: dict[int, float] = {}
    for k in k_values:
        hits = sum(
            1
            for actual, cands in zip(actuals, candidates_list)
            if _box_key(actual) in [_box_key(c) for c in cands[:k]]
        )
        results[k] = hits / n
    return results


def _box_key(num_str: str) -> str:
    """ボックス照合用キー (ソート済み)."""
    return "".join(sorted(str(num_str).zfill(3)))


# =====================================================================
# ニアミス率
# =====================================================================


def near_miss_rate(
    actuals: list[str],
    predictions: list[str],
    tolerances: list[int] | None = None,
) -> dict[str, Any]:
    """各桁の差が tolerance 以内のニアミス率を算出.

    Parameters
    ----------
    actuals : list of str
        実際の当選番号
    predictions : list of str
        予測番号
    tolerances : list of int, optional
        許容差 (デフォルト [1, 2])

    Returns
    -------
    Dict  tolerance ごとの的中率と詳細
    """
    if tolerances is None:
        tolerances = [1, 2]
    n = len(actuals)
    if n == 0:
        return {}

    results: dict[str, Any] = {"exact_rate": 0.0, "total": n}
    exact = 0
    for tol in tolerances:
        hits = 0
        for actual, pred in zip(actuals, predictions):
            a = str(actual).zfill(3)
            p = str(pred).zfill(3)
            if a == p:
                hits += 1
                if tol == tolerances[0]:
                    exact += 1
            elif all(abs(int(ai) - int(pi)) <= tol for ai, pi in zip(a, p)):
                hits += 1
        results[f"near_miss_rate_tol{tol}"] = hits / n
    results["exact_rate"] = exact / n
    return results


# =====================================================================
# Brier Score
# =====================================================================


def brier_score(
    actuals: list[str],
    proba_matrices: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """各桁の Brier Score を算出.

    Parameters
    ----------
    actuals : list of str
        実際の当選番号 (3桁文字列)
    proba_matrices : list of dict
        各ラウンドの {"n1": np.array(10,), "n2": ..., "n3": ...}

    Returns
    -------
    Dict  桁ごとの Brier Score と平均
    """
    n = len(actuals)
    if n == 0:
        return {"n1": np.nan, "n2": np.nan, "n3": np.nan, "avg": np.nan}

    brier: dict[str, float] = {}
    for pos_idx, pos_name in enumerate(["n1", "n2", "n3"]):
        total = 0.0
        for actual, proba_dict in zip(actuals, proba_matrices):
            actual_digit = int(str(actual).zfill(3)[pos_idx])
            proba = proba_dict.get(pos_name, np.ones(10) / 10)
            # Brier Score = (1/N) Σ (p_j - y_j)^2 for j=0..9
            y_one_hot = np.zeros(10)
            y_one_hot[actual_digit] = 1.0
            total += np.sum((proba - y_one_hot) ** 2)
        brier[pos_name] = total / n
    brier["avg"] = float(np.mean([brier["n1"], brier["n2"], brier["n3"]]))
    return brier


# =====================================================================
# 数字別 Recall / Precision
# =====================================================================


def digit_recall_precision(
    actuals: list[str],
    predictions: list[str],
) -> pd.DataFrame:
    """0-9 各数字の Recall / Precision を算出.

    Parameters
    ----------
    actuals : list of str
        実際の当選番号
    predictions : list of str
        予測番号

    Returns
    -------
    DataFrame  columns: digit, recall, precision, f1, support
    """
    records: list[dict] = []
    for d in range(10):
        tp = 0
        fp = 0
        fn = 0
        for actual, pred in zip(actuals, predictions):
            a_digits = set(str(actual).zfill(3))
            p_digits = set(str(pred).zfill(3))
            d_str = str(d)
            if d_str in a_digits and d_str in p_digits:
                tp += 1
            elif d_str in p_digits and d_str not in a_digits:
                fp += 1
            elif d_str in a_digits and d_str not in p_digits:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        records.append(
            {
                "digit": d,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "support": tp + fn,
            }
        )
    return pd.DataFrame(records)


# =====================================================================
# ボックス的中率の統計検定 (二項検定)
# =====================================================================


def box_hit_significance(
    hits: int,
    trials: int,
    expected_rate: float = 1 / 167.0,
) -> dict[str, Any]:
    """ボックス的中率がランダムと有意に異なるか検定 (二項検定).

    Parameters
    ----------
    hits : int
        ボックス的中数
    trials : int
        テストラウンド数
    expected_rate : float
        理論的中率 (デフォルト: 1/167 ≈ 0.6%)

    Returns
    -------
    Dict  p_value, significant_5pct, observed_rate, expected_rate, multiplier
    """
    if trials <= 0:
        return {
            "p_value": np.nan,
            "significant_5pct": False,
            "observed_rate": 0.0,
            "expected_rate": expected_rate,
            "multiplier": 0.0,
        }
    result = scipy_stats.binomtest(hits, trials, expected_rate, alternative="two-sided")
    observed = hits / trials
    multiplier = observed / expected_rate if expected_rate > 0 else 0.0
    return {
        "p_value": round(result.pvalue, 6),
        "significant_5pct": result.pvalue < 0.05,
        "observed_rate": round(observed, 6),
        "expected_rate": round(expected_rate, 6),
        "multiplier": round(multiplier, 2),
        "confidence_interval_95": (
            round(result.proportion_ci(confidence_level=0.95).low, 6),
            round(result.proportion_ci(confidence_level=0.95).high, 6),
        ),
    }


# =====================================================================
# 時系列安定性評価
# =====================================================================


def temporal_stability(
    result_df: pd.DataFrame,
    metric_col: str = "mlabel_hit",
    n_splits: int = 3,
) -> dict[str, Any]:
    """バックテスト結果を複数期間に分割して精度のばらつきを評価.

    Parameters
    ----------
    result_df : DataFrame
        バックテスト結果 (各行 = 1ラウンド)
    metric_col : str
        評価する列名 (0/1 の的中フラグ)
    n_splits : int
        分割数

    Returns
    -------
    Dict  period_rates, mean, std, ci_95
    """
    n = len(result_df)
    if n < n_splits:
        return {"period_rates": [], "mean": np.nan, "std": np.nan, "ci_95": (np.nan, np.nan)}

    chunk_size = n // n_splits
    rates: list[float] = []
    for i in range(n_splits):
        start = i * chunk_size
        end = start + chunk_size if i < n_splits - 1 else n
        chunk = result_df.iloc[start:end]
        rate = chunk[metric_col].mean() if metric_col in chunk.columns else 0.0
        rates.append(rate)

    mean_rate = np.mean(rates)
    std_rate = np.std(rates, ddof=1) if len(rates) > 1 else 0.0
    # 95% CI (t分布)
    if len(rates) > 1:
        t_val = scipy_stats.t.ppf(0.975, df=len(rates) - 1)
        margin = t_val * std_rate / np.sqrt(len(rates))
        ci = (mean_rate - margin, mean_rate + margin)
    else:
        ci = (mean_rate, mean_rate)

    return {
        "period_rates": [round(r, 4) for r in rates],
        "mean": round(mean_rate, 4),
        "std": round(std_rate, 4),
        "ci_95": (round(ci[0], 4), round(ci[1], 4)),
    }


# =====================================================================
# ベースライン比較
# =====================================================================


def baseline_comparison(
    actuals: list[str],
    model_predictions: list[str],
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """モデル予測を 3 つのベースラインと比較する.

    ベースライン:
        1. ランダム予測 (一様分布)
        2. 頻度モデル (直近50回の最頻数字)
        3. ナイーブモデル (前回の番号をそのまま予測)

    Parameters
    ----------
    actuals : list of str
        実際の当選番号
    model_predictions : list of str
        モデルの予測番号
    df : DataFrame, optional
        過去データ (ナイーブ・頻度ベースライン用)

    Returns
    -------
    DataFrame  model 名ごとの的中率比較
    """
    n = len(actuals)
    if n == 0:
        return pd.DataFrame()

    rng = np.random.RandomState(42)

    def _box_hit(actual: str, pred: str) -> bool:
        return _box_key(actual) == _box_key(pred)

    # モデル
    model_hits = sum(1 for a, p in zip(actuals, model_predictions) if _box_hit(a, p))

    # ランダム
    random_preds = [
        "".join(str(d) for d in rng.randint(0, 10, size=3)) for _ in range(n)
    ]
    random_hits = sum(1 for a, p in zip(actuals, random_preds) if _box_hit(a, p))

    # ナイーブ (前回番号をそのまま)
    naive_preds: list[str] = []
    for i, actual in enumerate(actuals):
        if i == 0:
            naive_preds.append(actual)  # 初回は自分自身
        else:
            naive_preds.append(actuals[i - 1])
    naive_hits = sum(1 for a, p in zip(actuals, naive_preds) if _box_hit(a, p))

    # 頻度モデル (直近50回の最頻3数字)
    freq_preds: list[str] = []
    for i in range(n):
        if df is not None and len(df) > 50:
            # バックテスト内でのインデックス計算は省略、全体頻度で代替
            all_digits = []
            for col in ["n1", "n2", "n3"]:
                if col in df.columns:
                    all_digits.extend(df[col].iloc[max(0, len(df) - 50 - n + i) : len(df) - n + i].tolist())
            if all_digits:
                counter = pd.Series(all_digits).value_counts().head(3)
                freq_preds.append("".join(str(d) for d in sorted(counter.index.tolist())))
            else:
                freq_preds.append("012")
        else:
            freq_preds.append("012")
    freq_hits = sum(1 for a, p in zip(actuals, freq_preds) if _box_hit(a, p))

    records = [
        {"model": "Current Model", "box_hits": model_hits, "box_hit_rate": round(model_hits / n * 100, 2)},
        {"model": "Random Baseline", "box_hits": random_hits, "box_hit_rate": round(random_hits / n * 100, 2)},
        {"model": "Naive (Previous)", "box_hits": naive_hits, "box_hit_rate": round(naive_hits / n * 100, 2)},
        {"model": "Frequency Top-3", "box_hits": freq_hits, "box_hit_rate": round(freq_hits / n * 100, 2)},
    ]
    return pd.DataFrame(records)


# =====================================================================
# 統合評価レポート
# =====================================================================


def generate_evaluation_report(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    save_path: str = "results/evaluation_report.csv",
) -> dict[str, Any]:
    """バックテスト結果から包括的な評価レポートを生成.

    Parameters
    ----------
    result_df : DataFrame
        バックテスト詳細結果
    summary_df : DataFrame
        バックテストサマリー
    save_path : str
        レポートの保存先

    Returns
    -------
    Dict  評価指標の集約
    """
    report: dict[str, Any] = {}

    # 基本サマリー
    if len(summary_df) > 0:
        s = summary_df.iloc[0]
        report["basic"] = {
            "tested": int(s.get("tested", 0)),
            "box_hit_rate": float(s.get("box_hit_rate", 0)),
            "mlabel_hit_rate": float(s.get("mlabel_hit_rate", 0)),
            "avg_digit_recall": float(s.get("avg_digit_recall", 0)),
            "avg_sum_mae": float(s.get("avg_sum_mae", 0)),
        }

    # ボックス有意性検定
    if "mlabel_hit" in result_df.columns:
        hits = int(result_df["mlabel_hit"].sum())
        trials = len(result_df)
        report["box_significance"] = box_hit_significance(hits, trials)

    # 時系列安定性
    if "mlabel_hit" in result_df.columns:
        report["stability"] = temporal_stability(result_df, "mlabel_hit", n_splits=3)

    # 数字別 Recall/Precision
    if "actual" in result_df.columns and "mlabel_predicted" in result_df.columns:
        actuals = result_df["actual"].tolist()
        preds = result_df["mlabel_predicted"].tolist()
        report["digit_metrics"] = digit_recall_precision(actuals, preds).to_dict("records")

    # ニアミス率
    if "actual" in result_df.columns and "box_predicted" in result_df.columns:
        actuals = result_df["actual"].tolist()
        preds = result_df["box_predicted"].tolist()
        report["near_miss"] = near_miss_rate(actuals, preds, tolerances=[1, 2])

    # CSV 保存
    if save_path:
        flat_rows: list[dict] = []
        for section, data in report.items():
            if isinstance(data, dict):
                for k, v in data.items():
                    flat_rows.append({"section": section, "metric": k, "value": str(v)})
            elif isinstance(data, list):
                for item in data:
                    flat_rows.append({"section": section, "metric": str(item.get("digit", "")), "value": str(item)})
        if flat_rows:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            pd.DataFrame(flat_rows).to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info("評価レポートを保存: %s", save_path)

    return report
