"""
src.analysis.independence_test — Numbers3 の統計的独立性検定

責務:
    - 各数字出現の自己相関 (ACF/PACF) 解析
    - Ljung-Box 検定による時系列依存性の定量評価
    - カイ二乗独立性検定 (曜日・月 × 数字)
    - コルモゴロフ・スミルノフ検定 (一様分布との比較)
    - 検定結果の集約レポート生成
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.data.loader import normalize_numbers3_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =====================================================================
# ACF / Ljung-Box
# =====================================================================


def compute_acf(series: pd.Series, nlags: int = 50) -> np.ndarray:
    """自己相関関数 (ACF) を手動計算する.

    statsmodels に依存しない純粋な NumPy 実装。

    Parameters
    ----------
    series : pd.Series
        入力系列 (欠損は内部で除去)
    nlags : int
        最大ラグ数

    Returns
    -------
    np.ndarray  shape (nlags+1,) — lag=0..nlags の ACF 値
    """
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < nlags + 2:
        nlags = max(1, n - 2)
    x_centered = x - x.mean()
    c0 = np.dot(x_centered, x_centered) / n
    if c0 == 0:
        return np.zeros(nlags + 1)
    acf_vals = np.zeros(nlags + 1)
    acf_vals[0] = 1.0
    for lag in range(1, nlags + 1):
        acf_vals[lag] = np.dot(x_centered[:-lag], x_centered[lag:]) / (n * c0)
    return acf_vals


def ljung_box_test(
    series: pd.Series,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Ljung-Box 検定を実施する.

    Parameters
    ----------
    series : pd.Series
        入力系列
    lags : list of int, optional
        検定するラグ。デフォルトは [5, 10, 20, 50]

    Returns
    -------
    DataFrame  columns: lag, lb_stat, p_value, significant_5pct
    """
    if lags is None:
        lags = [5, 10, 20, 50]

    x = series.dropna().values.astype(float)
    n = len(x)
    if n < 10:
        logger.warning("データ不足 (n=%d) — Ljung-Box テストをスキップ", n)
        return pd.DataFrame(columns=["lag", "lb_stat", "p_value", "significant_5pct"])

    acf_vals = compute_acf(series, nlags=max(lags))

    records: list[dict[str, Any]] = []
    for lag in lags:
        if lag >= n:
            continue
        # Q = n(n+2) * Σ r(k)^2 / (n-k)  for k=1..lag
        q_stat = 0.0
        for k in range(1, lag + 1):
            q_stat += (acf_vals[k] ** 2) / (n - k)
        q_stat *= n * (n + 2)
        p_value = 1.0 - scipy_stats.chi2.cdf(q_stat, df=lag)
        records.append(
            {
                "lag": lag,
                "lb_stat": round(q_stat, 4),
                "p_value": round(p_value, 6),
                "significant_5pct": p_value < 0.05,
            }
        )
    return pd.DataFrame(records)


# =====================================================================
# カイ二乗独立性検定
# =====================================================================


def chi2_independence_test(
    df: pd.DataFrame,
    group_col: str,
    digit_col: str,
) -> dict[str, Any]:
    """カイ二乗独立性検定 (group × digit).

    Parameters
    ----------
    df : DataFrame
        group_col と digit_col を含むデータ
    group_col : str
        グループ変数 (例: 'weekday', 'month')
    digit_col : str
        数字変数 (例: 'n1', 'n2', 'n3')

    Returns
    -------
    Dict  chi2, p_value, dof, cramers_v, significant_5pct
    """
    clean = df[[group_col, digit_col]].dropna()
    if len(clean) < 20:
        return {
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": 0,
            "cramers_v": np.nan,
            "significant_5pct": False,
        }
    ct = pd.crosstab(clean[group_col], clean[digit_col])
    chi2, p_value, dof, _ = scipy_stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": int(dof),
        "cramers_v": round(cramers_v, 4),
        "significant_5pct": p_value < 0.05,
    }


# =====================================================================
# コルモゴロフ・スミルノフ検定
# =====================================================================


def ks_uniformity_test(series: pd.Series) -> dict[str, Any]:
    """KS 検定で離散一様分布 (0-9) との乖離を検定する.

    Parameters
    ----------
    series : pd.Series
        0-9 の数字列

    Returns
    -------
    Dict  ks_stat, p_value, significant_5pct
    """
    x = series.dropna().values.astype(float)
    if len(x) < 10:
        return {"ks_stat": np.nan, "p_value": np.nan, "significant_5pct": False}
    # 離散一様分布の CDF: (floor(x)+1)/10 for x in [0, 9]
    stat, p_value = scipy_stats.kstest(x, lambda t: np.clip((np.floor(t) + 1) / 10, 0, 1))
    return {
        "ks_stat": round(stat, 6),
        "p_value": round(p_value, 6),
        "significant_5pct": p_value < 0.05,
    }


# =====================================================================
# 検出された依存性に基づく推奨
# =====================================================================


def recommend_feature_usage(report: dict[str, Any]) -> list[str]:
    """検定結果を受け取り、特徴量の有効/無効を推奨する.

    Parameters
    ----------
    report : Dict
        run_full_independence_test() の戻り値

    Returns
    -------
    List[str]  推奨メッセージのリスト
    """
    recommendations: list[str] = []

    # Ljung-Box
    lb = report.get("ljung_box", {})
    any_significant = False
    for col_name, lb_df in lb.items():
        if isinstance(lb_df, pd.DataFrame) and lb_df["significant_5pct"].any():
            any_significant = True
            sig_lags = lb_df[lb_df["significant_5pct"]]["lag"].tolist()
            recommendations.append(
                f"[ACF] {col_name}: ラグ {sig_lags} で有意な自己相関 → ラグ特徴量を活用"
            )
    if not any_significant:
        recommendations.append(
            "[ACF] すべての桁で有意な自己相関なし → ハマリ係数 (interval_boost) の除外を推奨"
        )

    # カイ二乗
    chi2 = report.get("chi2_independence", {})
    for test_name, result in chi2.items():
        if result.get("significant_5pct"):
            recommendations.append(
                f"[Chi2] {test_name}: 有意な関連 (V={result['cramers_v']:.3f}) → 特徴量として有効"
            )
        else:
            recommendations.append(
                f"[Chi2] {test_name}: 有意な関連なし → 特徴量としての価値は低い"
            )

    # KS
    ks = report.get("ks_uniformity", {})
    for col_name, result in ks.items():
        if result.get("significant_5pct"):
            recommendations.append(
                f"[KS] {col_name}: 一様分布から有意に乖離 → 頻度偏り特徴量が有効"
            )
        else:
            recommendations.append(
                f"[KS] {col_name}: 一様分布と整合 → 過去頻度特徴量の限界に注意"
            )

    return recommendations


# =====================================================================
# 統合検定ランナー
# =====================================================================


def run_full_independence_test(
    df: pd.DataFrame,
    save_path: str | None = "results/independence_test_report.csv",
) -> dict[str, Any]:
    """Numbers3 データに対して包括的な独立性検定を実施する.

    Parameters
    ----------
    df : DataFrame
        当選番号列を含むデータ
    save_path : str, optional
        レポート CSV の保存先

    Returns
    -------
    Dict  検定結果の集約辞書
    """
    df = normalize_numbers3_columns(df).copy()
    if "当選番号" in df.columns:
        num = df["当選番号"].astype(str).str.zfill(3)
        df["n1"] = num.str[0].astype(int)
        df["n2"] = num.str[1].astype(int)
        df["n3"] = num.str[2].astype(int)
    df["dt"] = pd.to_datetime(df.get("抽せん日"), errors="coerce")
    df["weekday"] = df["dt"].dt.weekday
    df["month"] = df["dt"].dt.month

    report: dict[str, Any] = {}

    # 1) Ljung-Box (ACF)
    logger.info("Ljung-Box 検定を実施中...")
    lb_results: dict[str, pd.DataFrame] = {}
    for col in ["n1", "n2", "n3"]:
        lb_results[col] = ljung_box_test(df[col], lags=[5, 10, 20, 50])
    report["ljung_box"] = lb_results

    # 2) カイ二乗独立性検定
    logger.info("カイ二乗独立性検定を実施中...")
    chi2_results: dict[str, dict] = {}
    for digit_col in ["n1", "n2", "n3"]:
        for group_col in ["weekday", "month"]:
            key = f"{group_col}_x_{digit_col}"
            chi2_results[key] = chi2_independence_test(df, group_col, digit_col)
    report["chi2_independence"] = chi2_results

    # 3) KS 一様性検定
    logger.info("KS 一様性検定を実施中...")
    ks_results: dict[str, dict] = {}
    for col in ["n1", "n2", "n3"]:
        ks_results[col] = ks_uniformity_test(df[col])
    report["ks_uniformity"] = ks_results

    # 4) 推奨事項
    report["recommendations"] = recommend_feature_usage(report)

    # CSV 保存
    if save_path:
        rows: list[dict] = []
        # Ljung-Box
        for col, lb_df in lb_results.items():
            for _, row in lb_df.iterrows():
                rows.append(
                    {
                        "test_type": "ljung_box",
                        "target": col,
                        "detail": f"lag={row['lag']}",
                        "statistic": row["lb_stat"],
                        "p_value": row["p_value"],
                        "significant_5pct": row["significant_5pct"],
                    }
                )
        # Chi2
        for key, res in chi2_results.items():
            rows.append(
                {
                    "test_type": "chi2",
                    "target": key,
                    "detail": f"dof={res['dof']}, V={res.get('cramers_v', '')}",
                    "statistic": res["chi2"],
                    "p_value": res["p_value"],
                    "significant_5pct": res.get("significant_5pct", False),
                }
            )
        # KS
        for col, res in ks_results.items():
            rows.append(
                {
                    "test_type": "ks_uniformity",
                    "target": col,
                    "detail": "",
                    "statistic": res["ks_stat"],
                    "p_value": res["p_value"],
                    "significant_5pct": res.get("significant_5pct", False),
                }
            )
        report_df = pd.DataFrame(rows)
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        report_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        logger.info("検定レポートを保存: %s", save_path)

    # サマリ出力
    print("\n" + "=" * 70)
    print("Numbers3 独立性検定レポート")
    print("=" * 70)
    for rec in report["recommendations"]:
        print(f"  {rec}")
    print("=" * 70 + "\n")

    return report
