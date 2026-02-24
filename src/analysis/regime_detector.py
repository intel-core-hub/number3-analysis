"""
src.analysis.regime_detector — レジーム検知エンジン

目的:
    - 当選番号の統計量からレジームを分類
    - 現在のレジームと確信度を推定

軽量なK-meansを使用し、CPUでミリ秒判定を想定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeResult:
    regime_id: int
    regime_label: str
    confidence: float
    features: dict[str, float]


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = np.sum(counts)
    if total <= 0:
        return 0.0
    probs = counts / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _compute_window_features(window: pd.Series) -> dict[str, float]:
    numbers = window.astype(str).str.zfill(3)

    # Digit frequency stats
    counts = np.zeros(10, dtype=float)
    for d in range(10):
        counts[d] = (numbers.str.contains(str(d))).sum()
    freq_var = float(np.var(counts))
    entropy = _entropy_from_counts(counts)

    # Repetition rate (e.g. 112, 222)
    rep_count = 0
    for n in numbers:
        if len(set(n)) < 3:
            rep_count += 1
    repetition_rate = float(rep_count / max(len(numbers), 1))

    # Sum of digits stats
    digit_sums = numbers.apply(lambda x: int(x[0]) + int(x[1]) + int(x[2]))
    sum_mean = float(digit_sums.mean())
    sum_var = float(digit_sums.var())

    # Unique digit ratio
    uniq_ratio = float(numbers.apply(lambda x: len(set(x)) / 3.0).mean())

    return {
        "freq_var": freq_var,
        "entropy": entropy,
        "repetition_rate": repetition_rate,
        "sum_mean": sum_mean,
        "sum_var": sum_var,
        "uniq_ratio": uniq_ratio,
    }


def _build_feature_matrix(df: pd.DataFrame, window: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    if "当選番号" not in df.columns:
        raise ValueError("Missing column: 当選番号")

    values = df["当選番号"].astype(str).str.zfill(3)
    features_list: list[dict[str, float]] = []

    for i in range(window - 1, len(values)):
        window_series = values.iloc[i - window + 1 : i + 1]
        features = _compute_window_features(window_series)
        features_list.append(features)

    if not features_list:
        raise ValueError("Not enough data for regime detection")

    matrix = np.array([
        [f["freq_var"], f["entropy"], f["repetition_rate"], f["sum_mean"], f["sum_var"], f["uniq_ratio"]]
        for f in features_list
    ])

    return matrix, features_list


def _label_regimes(centers: np.ndarray) -> dict[int, str]:
    labels: dict[int, str] = {}
    if centers.shape[0] == 3:
        for idx, center in enumerate(centers):
            freq_var, entropy, rep_rate, sum_mean, sum_var, uniq_ratio = center
            if entropy >= np.percentile(centers[:, 1], 66) and freq_var <= np.percentile(centers[:, 0], 33):
                labels[idx] = "安定期"
            elif entropy <= np.percentile(centers[:, 1], 33) and freq_var >= np.percentile(centers[:, 0], 66):
                labels[idx] = "偏り期"
            else:
                labels[idx] = "遷移期"
    else:
        for idx in range(centers.shape[0]):
            labels[idx] = f"レジーム{idx + 1}"
    return labels


def detect_current_regime(
    df: pd.DataFrame,
    n_regimes: int = 3,
    window: int = 60,
    random_state: int = 42,
) -> RegimeResult:
    """
    現在のレジームを検知

    Args:
        df: 当選番号を含むデータフレーム
        n_regimes: レジーム数 (3〜5推奨)
        window: 特徴量計算ウィンドウサイズ
        random_state: KMeansのseed

    Returns:
        RegimeResult
    """
    if len(df) < window:
        window = max(10, len(df))

    feature_matrix, features_list = _build_feature_matrix(df, window)

    kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    kmeans.fit(feature_matrix)

    # 最新ウィンドウを推定
    current_features = feature_matrix[-1].reshape(1, -1)
    distances = kmeans.transform(current_features).flatten()
    regime_id = int(np.argmin(distances))

    # Confidence as softmax on negative distances
    exp_scores = np.exp(-distances)
    confidence = float(exp_scores[regime_id] / np.sum(exp_scores)) if np.sum(exp_scores) > 0 else 0.0

    labels = _label_regimes(kmeans.cluster_centers_)
    regime_label = labels.get(regime_id, f"レジーム{regime_id + 1}")

    features = features_list[-1]
    logger.info("Regime detected: %s (id=%d, conf=%.2f)", regime_label, regime_id, confidence)

    return RegimeResult(
        regime_id=regime_id,
        regime_label=regime_label,
        confidence=confidence,
        features=features,
    )


def regime_to_weight_file(regime_id: int) -> str:
    """レジームIDに対応するGA重みファイル名を返す"""
    return f"ga_weights_regime_{regime_id + 1}.json"
