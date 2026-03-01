from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class EnsemblePredictor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # provide a lightweight lgbm attribute expected by other modules
        class _LGBMStub:
            def models(self) -> dict[str, Any]:
                return {"n1": None, "n2": None, "n3": None}

            def feature_columns(self) -> list[str]:
                from src.features.engineer import ml_feature_columns

                return ml_feature_columns()

            def latest_features(self) -> pd.DataFrame:
                return pd.DataFrame()

        self.lgbm = _LGBMStub()

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, Any]:
        return {}
