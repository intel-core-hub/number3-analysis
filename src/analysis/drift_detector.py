"""
src.analysis.drift_detector - Adversarial validation drift detector.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_drift_auc(
    df: pd.DataFrame,
    feature_cols: list[str],
    recent_window: int = 200,
    min_samples: int = 200,
) -> float:
    """Estimate drift by classifying recent vs past samples.

    Returns AUC where 0.5 ~= no drift, 1.0 ~= strong drift.
    """
    if df is None or df.empty or not feature_cols:
        return 0.5
    valid = df[feature_cols].dropna()
    if len(valid) < max(min_samples, recent_window * 2):
        return 0.5

    recent = valid.tail(recent_window)
    past = valid.iloc[-(recent_window * 2) : -recent_window]
    if past.empty or recent.empty:
        return 0.5

    X = pd.concat([past, recent], axis=0)
    y = np.array([0] * len(past) + [1] * len(recent))
    if len(np.unique(y)) < 2:
        return 0.5

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        clf = LogisticRegression(max_iter=500, solver="liblinear")
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        return float(auc)
    except Exception as exc:
        logger.warning("Drift detection failed: %s", exc)
        return 0.5
