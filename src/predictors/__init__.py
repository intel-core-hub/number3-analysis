from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.loader import normalize_numbers3_columns
from src.models.ml_wrapper import Numbers3MLPredictor


class PatternPredictor:
    """Simple pattern predictor compatible with UI expectations."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = normalize_numbers3_columns(df).copy()
        self.last_hl_probs: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.last_oe_probs: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

    @staticmethod
    def _transition_probs(
        series: pd.Series[Any], num_states: int = 4
    ) -> np.ndarray[Any, Any]:
        values = pd.to_numeric(series, errors="coerce").dropna().astype(int).to_numpy()
        if len(values) < 2:
            return np.full((num_states, num_states), 1.0 / num_states)

        mat = np.ones((num_states, num_states), dtype=float)  # Laplace smoothing
        for prev, curr in zip(values[:-1], values[1:]):
            if 0 <= prev < num_states and 0 <= curr < num_states:
                mat[prev, curr] += 1.0
        mat /= mat.sum(axis=1, keepdims=True)
        return mat

    def train(self) -> None:
        if self.df.empty:
            return

        df_local = self.df.copy()
        if not {"n1", "n2", "n3"}.issubset(df_local.columns):
            if "当選番号" in df_local.columns:
                num = df_local["当選番号"].astype(str).str.zfill(3)
                df_local["n1"] = pd.to_numeric(num.str[0], errors="coerce")
                df_local["n2"] = pd.to_numeric(num.str[1], errors="coerce")
                df_local["n3"] = pd.to_numeric(num.str[2], errors="coerce")
            else:
                return

        digits = df_local[["n1", "n2", "n3"]].apply(pd.to_numeric, errors="coerce")
        odd_count = (digits % 2 == 1).sum(axis=1)
        high_count = (digits >= 5).sum(axis=1)

        odd_tm = self._transition_probs(odd_count)
        high_tm = self._transition_probs(high_count)

        curr_odd = int(odd_count.iloc[-1]) if len(odd_count) > 0 else 1
        curr_high = int(high_count.iloc[-1]) if len(high_count) > 0 else 1

        self.last_oe_probs = {
            state: float(odd_tm[curr_odd, state]) for state in range(4)
        }
        self.last_hl_probs = {
            state: float(high_tm[curr_high, state]) for state in range(4)
        }

    def predict_top_patterns(
        self, top_k: int = 2
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        hl = sorted(self.last_hl_probs.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        oe = sorted(self.last_oe_probs.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        return hl, oe

    def generate_candidates(self, top_k: int = 20) -> pd.DataFrame:
        ml = Numbers3MLPredictor(self.df)
        ml.train()
        return ml.predict_topk_combinations(top_k=max(2, min(6, top_k)))


class MultiLabelPredictor:
    """Wrapper around Numbers3MLPredictor exposing digit-level probabilities."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = normalize_numbers3_columns(df).copy()
        self.model = Numbers3MLPredictor(self.df)

    def train(self) -> None:
        self.model.train()

    def predict_digit_probabilities(self) -> Dict[int, float]:
        if not self.model.models:
            raise RuntimeError("Model is not trained")
        X = self.model._build_latest_features()
        all_proba = self.model._predict_all_digit_proba(X)
        p_n1 = all_proba["n1"][0]
        p_n2 = all_proba["n2"][0]
        p_n3 = all_proba["n3"][0]
        avg = (p_n1 + p_n2 + p_n3) / 3.0
        return {digit: float(avg[digit]) for digit in range(10)}


__all__ = ["PatternPredictor", "MultiLabelPredictor"]


class LGBMPredictor:
    """Minimal LGBM predictor stub for type-checking and imports."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = normalize_numbers3_columns(df).copy()

        class _Backend:
            def tune_with_optuna(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

        self._predictor = _Backend()
        self._models: dict[str, Any] = {"n1": None, "n2": None, "n3": None}

    def train(self) -> None:
        return None

    def predict_topk_combinations(self, top_k: int = 10) -> pd.DataFrame:
        return pd.DataFrame()

    def models(self) -> dict[str, Any]:
        return self._models

    def feature_columns(self) -> list[str]:
        from src.features.engineer import ml_feature_columns

        return ml_feature_columns()

    def latest_features(self) -> pd.DataFrame:
        return pd.DataFrame()


class EnsemblePredictor:
    """Minimal ensemble predictor stub exposing `.lgbm`."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.lgbm = LGBMPredictor(df)

    def latest_features(self) -> pd.DataFrame:
        return pd.DataFrame()
