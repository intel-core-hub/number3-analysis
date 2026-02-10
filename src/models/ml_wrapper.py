"""
src.models.ml_wrapper — LightGBM 予測器

責務:
    - Numbers3MLPredictor (LightGBM による 3 桁予測)
    - 温度スケーリングによる確率補正
    - Optuna によるハイパーパラメータ最適化
    - TimeSeriesSplit による時系列交差検証
    - 特徴量重要度の自動枝刈り
    - ニアミス率評価メトリクス
    - 特徴量重要度抽出ユーティリティ
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.data.loader import normalize_numbers3_columns
from src.features.engineer import (
    Numbers3FeatureEngineer,
    compute_all_ml_features,
    compute_common_features,
    ml_feature_columns,
)
from src.models.base import BasePredictor, _box_key, _box_type
from src.utils.config import (
    DEFAULT_LGB_PARAMS,
    DEFAULT_NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    OPTUNA_CV_FOLDS,
    OPTUNA_N_TRIALS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =====================================================================
# ML 予測器 (LightGBM)
# =====================================================================


class Numbers3MLPredictor(BasePredictor):
    """LightGBM による Numbers3 予測器.

    Prompt B 拡張:
        - TimeSeriesSplit 5-fold 交差検証
        - Optuna によるハイパーパラメータ最適化
        - 特徴量重要度による自動枝刈り
        - ニアミス率 (near-miss rate) 評価メトリクス
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_short: int = 5,
        window_long: int = 10,
        num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
        random_state: int = 42,
        calibrate_proba: bool = True,
    ):
        self.df = normalize_numbers3_columns(df).copy()
        self.window_short = window_short
        self.window_long = window_long
        self.num_boost_round = num_boost_round
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.models_raw: Dict[str, Any] = {}
        self.best_params: Optional[Dict] = None
        self.calibrate_proba = calibrate_proba
        self.temperature: Dict[str, float] = {}
        self.pruned_features: Optional[List[str]] = None

    def _default_lgb_params(self) -> Dict[str, Any]:
        """小規模データ向けに過学習を抑えるデフォルトパラメータ."""
        params = dict(DEFAULT_LGB_PARAMS)
        params["n_estimators"] = self.num_boost_round
        params["random_state"] = self.random_state
        params["n_jobs"] = -1
        params["verbose"] = -1
        return params

    # =====================================================================
    # 温度スケーリング
    # =====================================================================

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _apply_temperature(
        self, proba: np.ndarray, temperature: float
    ) -> np.ndarray:
        if temperature <= 0:
            return proba
        eps = 1e-9
        logits = np.log(np.clip(proba, eps, 1.0)) / temperature
        return self._softmax(logits)

    def _find_temperature(
        self, proba: np.ndarray, y_true: np.ndarray
    ) -> float:
        temps = np.linspace(0.5, 2.5, 21)
        best_temp = 1.0
        best_ll = float("inf")
        for temp in temps:
            scaled = self._apply_temperature(proba, temp)
            ll = log_loss(y_true, scaled, labels=list(range(10)))
            if ll < best_ll:
                best_ll = ll
                best_temp = float(temp)
        return best_temp

    # =====================================================================
    # 特徴量
    # =====================================================================

    def _feature_columns(self) -> List[str]:
        if self.pruned_features is not None:
            return self.pruned_features
        return ml_feature_columns()

    def _compute_all_features(self) -> pd.DataFrame:
        return compute_all_ml_features(
            self.df, self.window_short, self.window_long
        )

    def _prepare_training_data(self) -> Tuple[pd.DataFrame, List[str]]:
        df = self._compute_all_features()
        df["target_n1"] = df["n1"].shift(-1)
        df["target_n2"] = df["n2"].shift(-1)
        df["target_n3"] = df["n3"].shift(-1)
        features = self._feature_columns()
        train_df = df.dropna(
            subset=features + ["target_n1", "target_n2", "target_n3"]
        )
        return train_df, features

    # =====================================================================
    # 特徴量重要度の自動枝刈り (Prompt B)
    # =====================================================================

    def auto_prune_features(
        self,
        importance_threshold: float = 0.01,
        min_features: int = 10,
    ) -> List[str]:
        """特徴量重要度が閾値以下の特徴量を自動的に除外する.

        Parameters
        ----------
        importance_threshold : float
            重要度の相対閾値 (最大値に対する比率)
        min_features : int
            最低限残す特徴量数

        Returns
        -------
        List[str]  枝刈り後の特徴量リスト
        """
        if not self.models_raw:
            logger.warning("モデル未学習のため枝刈りできません。先に train() を実行してください。")
            return ml_feature_columns()

        all_features = ml_feature_columns()
        total_importance = np.zeros(len(all_features))

        for digit in ["n1", "n2", "n3"]:
            model = self.models_raw.get(digit)
            if model is not None and hasattr(model, "feature_importances_"):
                total_importance += model.feature_importances_

        avg_importance = total_importance / 3.0
        max_imp = avg_importance.max() if avg_importance.max() > 0 else 1.0
        relative_importance = avg_importance / max_imp

        # 閾値以上の特徴量を残す
        keep_mask = relative_importance >= importance_threshold
        kept_features = [f for f, k in zip(all_features, keep_mask) if k]

        # 最低限の数は保証
        if len(kept_features) < min_features:
            sorted_idx = np.argsort(-avg_importance)
            kept_features = [all_features[i] for i in sorted_idx[:min_features]]

        self.pruned_features = kept_features
        logger.info(
            f"特徴量枝刈り: {len(all_features)} → {len(kept_features)} 特徴量"
        )
        return kept_features

    # =====================================================================
    # ニアミス率 (Prompt B)
    # =====================================================================

    @staticmethod
    def near_miss_rate(
        actual_list: List[str],
        predicted_list: List[str],
        tolerance: int = 1,
    ) -> Dict[str, Any]:
        """ニアミス率を計算する.

        Parameters
        ----------
        actual_list : List[str]
            実際の当選番号リスト
        predicted_list : List[str]
            予測番号リスト
        tolerance : int
            各桁のずれ許容値

        Returns
        -------
        Dict  ニアミス率および詳細
        """
        total = len(actual_list)
        exact_hits = 0
        near_miss_hits = 0
        details: List[Dict] = []

        for actual, predicted in zip(actual_list, predicted_list):
            actual = str(actual).zfill(3)
            predicted = str(predicted).zfill(3)

            is_exact = actual == predicted
            if is_exact:
                exact_hits += 1
                near_miss_hits += 1
                details.append({"actual": actual, "predicted": predicted, "type": "exact"})
                continue

            is_near = all(
                abs(int(a) - int(p)) <= tolerance
                for a, p in zip(actual, predicted)
            )
            if is_near:
                near_miss_hits += 1
                details.append({"actual": actual, "predicted": predicted, "type": "near_miss"})
            else:
                details.append({"actual": actual, "predicted": predicted, "type": "miss"})

        return {
            "total": total,
            "exact_hits": exact_hits,
            "near_miss_hits": near_miss_hits,
            "exact_rate": exact_hits / max(total, 1),
            "near_miss_rate": near_miss_hits / max(total, 1),
            "details": details,
        }

    # =====================================================================
    # Optuna ハイパーパラメータ最適化 (Prompt B)
    # =====================================================================

    def tune_with_optuna(
        self,
        n_trials: int = OPTUNA_N_TRIALS,
        n_splits: int = OPTUNA_CV_FOLDS,
        valid_size: int = 180,
    ) -> Dict:
        """Optuna + TimeSeriesSplit でハイパーパラメータを最適化.

        Parameters
        ----------
        n_trials : int
            Optuna の試行回数
        n_splits : int
            TimeSeriesSplit の分割数
        valid_size : int
            フォールバック用の検証サイズ

        Returns
        -------
        Dict  best_params と valid_logloss
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning(
                "optuna がインストールされていません。"
                "グリッドサーチにフォールバックします。"
            )
            return self.tune_hyperparams(valid_size=valid_size)

        df = self._compute_all_features()
        df["target_n1"] = df["n1"].shift(-1)
        df["target_n2"] = df["n2"].shift(-1)
        df["target_n3"] = df["n3"].shift(-1)
        features = self._feature_columns()
        full_df = df.dropna(
            subset=features + ["target_n1", "target_n2", "target_n3"]
        ).reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 50
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }

            fold_losses: List[float] = []
            for train_idx, val_idx in tscv.split(full_df):
                train_part = full_df.iloc[train_idx]
                val_part = full_df.iloc[val_idx]
                fold_ll = 0.0
                for digit in ["n1", "n2", "n3"]:
                    m = lgb.LGBMClassifier(
                        objective="multiclass",
                        num_class=10,
                        n_estimators=self.num_boost_round,
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbose=-1,
                        **params,
                    )
                    m.fit(
                        train_part[features],
                        train_part[f"target_{digit}"].astype(int),
                        eval_set=[
                            (
                                val_part[features],
                                val_part[f"target_{digit}"].astype(int),
                            )
                        ],
                        callbacks=[
                            lgb.early_stopping(
                                stopping_rounds=EARLY_STOPPING_ROUNDS,
                                verbose=False,
                            )
                        ],
                    )
                    proba = m.predict_proba(val_part[features])
                    ll = log_loss(
                        val_part[f"target_{digit}"].astype(int),
                        proba,
                        labels=list(range(10)),
                    )
                    fold_ll += ll
                fold_losses.append(fold_ll / 3.0)

            return float(np.mean(fold_losses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        logger.info(f"Optuna best params: {study.best_params}")
        logger.info(f"Optuna best logloss: {study.best_value:.4f}")

        return {
            "best_params": study.best_params,
            "valid_logloss": study.best_value,
        }

    # =====================================================================
    # グリッドサーチ (従来互換)
    # =====================================================================

    def tune_hyperparams(
        self,
        param_grid: Optional[Dict] = None,
        valid_size: int = 180,
    ) -> Dict:
        """グリッドサーチでハイパーパラメータを最適化する."""
        from itertools import product as _product

        if param_grid is None:
            param_grid = {
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 6],
                "min_child_samples": [10, 20],
            }

        df = self._compute_all_features()
        df["target_n1"] = df["n1"].shift(-1)
        df["target_n2"] = df["n2"].shift(-1)
        df["target_n3"] = df["n3"].shift(-1)
        features = self._feature_columns()
        full_df = df.dropna(
            subset=features + ["target_n1", "target_n2", "target_n3"]
        )

        if len(full_df) <= valid_size:
            valid_size = max(1, len(full_df) // 5)

        train_part = full_df.iloc[:-valid_size]
        valid_part = full_df.iloc[-valid_size:]

        best_ll = float("inf")
        best_params: Dict = {}

        keys = sorted(param_grid.keys())
        for combo in _product(*(param_grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            total_ll = 0.0
            for digit in ["n1", "n2", "n3"]:
                m = lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=10,
                    n_estimators=self.num_boost_round,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                    **params,
                )
                m.fit(
                    train_part[features],
                    train_part[f"target_{digit}"].astype(int),
                )
                proba = m.predict_proba(valid_part[features])
                ll = log_loss(
                    valid_part[f"target_{digit}"].astype(int),
                    proba,
                    labels=list(range(10)),
                )
                total_ll += ll
            avg_ll = total_ll / 3
            if avg_ll < best_ll:
                best_ll = avg_ll
                best_params = params

        self.best_params = best_params
        return {"best_params": best_params, "valid_logloss": best_ll}

    # =====================================================================
    # 学習 (TimeSeriesSplit 対応 — Prompt B)
    # =====================================================================

    def train(
        self,
        use_cv: bool = False,
        n_splits: int = OPTUNA_CV_FOLDS,
    ) -> "Numbers3MLPredictor":
        """モデルを学習する.

        Parameters
        ----------
        use_cv : bool
            True の場合 TimeSeriesSplit で交差検証評価も行う
        n_splits : int
            TimeSeriesSplit の分割数
        """
        train_df, features = self._prepare_training_data()

        # 時系列を崩さない validation（末尾を検証に回す）
        valid_size = min(180, max(1, len(train_df) // 5))
        use_valid = len(train_df) >= (valid_size + 30)
        if use_valid:
            fit_train = train_df.iloc[:-valid_size]
            fit_valid = train_df.iloc[-valid_size:]
        else:
            fit_train = train_df
            fit_valid = None

        # --- TimeSeriesSplit 交差検証 (オプション) ---
        if use_cv and len(train_df) > 100:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_df)):
                fold_train = train_df.iloc[tr_idx]
                fold_val = train_df.iloc[val_idx]
                fold_ll = 0.0
                for digit in ["n1", "n2", "n3"]:
                    params = self._default_lgb_params()
                    params.update(self.best_params or {})
                    m = lgb.LGBMClassifier(
                        objective="multiclass", num_class=10, **params
                    )
                    m.fit(
                        fold_train[features],
                        fold_train[f"target_{digit}"].astype(int),
                        eval_set=[
                            (
                                fold_val[features],
                                fold_val[f"target_{digit}"].astype(int),
                            )
                        ],
                        callbacks=[
                            lgb.early_stopping(
                                stopping_rounds=EARLY_STOPPING_ROUNDS,
                                verbose=False,
                            )
                        ],
                    )
                    proba = m.predict_proba(fold_val[features])
                    ll = log_loss(
                        fold_val[f"target_{digit}"].astype(int),
                        proba,
                        labels=list(range(10)),
                    )
                    fold_ll += ll
                cv_scores.append(fold_ll / 3.0)
            logger.info(
                f"TimeSeriesSplit CV: mean={np.mean(cv_scores):.4f} "
                f"std={np.std(cv_scores):.4f}"
            )

        # --- 最終学習 ---
        for digit in ["n1", "n2", "n3"]:
            params = self._default_lgb_params()
            params.update(self.best_params or {})

            model_raw = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=10,
                **params,
            )

            if use_valid and fit_valid is not None:
                model_raw.fit(
                    fit_train[features],
                    fit_train[f"target_{digit}"].astype(int),
                    eval_set=[
                        (
                            fit_valid[features],
                            fit_valid[f"target_{digit}"].astype(int),
                        )
                    ],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=EARLY_STOPPING_ROUNDS,
                            verbose=False,
                        )
                    ],
                )
            else:
                model_raw.fit(
                    fit_train[features],
                    fit_train[f"target_{digit}"].astype(int),
                )

            self.models_raw[digit] = model_raw

            # confidence の自信過剰を緩和: validation で温度スケーリング
            if self.calibrate_proba and use_valid and fit_valid is not None:
                valid_proba = model_raw.predict_proba(fit_valid[features])
                y_true = fit_valid[f"target_{digit}"].astype(int).to_numpy()
                self.temperature[digit] = self._find_temperature(
                    valid_proba, y_true
                )
            else:
                self.temperature[digit] = 1.0

            self.models[digit] = model_raw
        return self

    def _predict_proba(self, digit: str, X: pd.DataFrame) -> np.ndarray:
        model = self.models_raw.get(digit) or self.models.get(digit)
        if model is None:
            raise RuntimeError("Models are not trained")
        proba = model.predict_proba(X)
        temp = self.temperature.get(digit, 1.0)
        if temp != 1.0:
            proba = self._apply_temperature(proba, temp)
        return proba

    # ------------------------------------------------------------------
    def _build_latest_features(self) -> pd.DataFrame:
        df = self._compute_all_features()
        features = self._feature_columns()
        return df[features].iloc[[-1]].fillna(0)

    def predict_next(self) -> str:
        X = self._build_latest_features()
        return "".join(
            str(np.argmax(self._predict_proba(d, X))) for d in ["n1", "n2", "n3"]
        )

    # ------------------------------------------------------------------
    def predict(self, top_n: int = 20, **kwargs) -> pd.DataFrame:
        """Top-N 予測番号を DataFrame で返す (BasePredictor 準拠)."""
        return self.predict_topk_combinations(top_k=3).head(top_n)

    # ------------------------------------------------------------------
    def predict_next_proba(self, top_k: int = 5) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Models are not trained")
        X = self._build_latest_features()
        records: List[Dict] = []
        for digit_key, name in zip(
            ["n1", "n2", "n3"], ["hundreds", "tens", "ones"]
        ):
            proba = self._predict_proba(digit_key, X)[0]
            ranked = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            for rank, (num, p) in enumerate(ranked, start=1):
                records.append(
                    {
                        "digit_position": name,
                        "predicted_number": num,
                        "probability": float(p),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    def predict_topk_combinations(self, top_k: int = 3) -> pd.DataFrame:
        """各桁の Top-K 候補を組み合わせてボックス買い候補を生成する."""
        if not self.models:
            raise RuntimeError("モデルが未学習です")
        X = self._build_latest_features()
        digit_candidates: Dict[str, list] = {}
        for digit_key in ["n1", "n2", "n3"]:
            proba = self._predict_proba(digit_key, X)[0]
            ranked = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            digit_candidates[digit_key] = ranked

        combinations: List[Dict] = []
        for h, p_h in digit_candidates["n1"]:
            for t, p_t in digit_candidates["n2"]:
                for o, p_o in digit_candidates["n3"]:
                    number = f"{h}{t}{o}"
                    combined_prob = p_h * p_t * p_o
                    box_t, box_c = _box_type(number)
                    combinations.append(
                        {
                            "予測番号": number,
                            "結合確率": combined_prob,
                            "百の位確率": float(p_h),
                            "十の位確率": float(p_t),
                            "一の位確率": float(p_o),
                            "タイプ": box_t,
                            "ボックス通り数": box_c,
                        }
                    )

        df_comb = pd.DataFrame(combinations)
        df_comb = df_comb.sort_values("結合確率", ascending=False).reset_index(
            drop=True
        )
        return df_comb

    # ------------------------------------------------------------------
    def get_feature_importance(self) -> pd.DataFrame:
        """学習済みモデルから各桁の特徴量重要度を取得する."""
        importance_df_list: List[pd.DataFrame] = []
        for digit in ["n1", "n2", "n3"]:
            model = self.models_raw.get(digit) or self.models.get(digit)
            if model is None:
                continue
            if not hasattr(model, "feature_importances_"):
                continue
            imp = pd.DataFrame(
                {
                    "feature": self._feature_columns(),
                    "importance": model.feature_importances_,
                    "digit": digit,
                }
            )
            importance_df_list.append(imp)
        return pd.concat(importance_df_list, ignore_index=True)


# =====================================================================
# 特徴量重要度抽出 (スタンドアロン関数)
# =====================================================================


def extract_feature_importance(
    predictor: Numbers3MLPredictor,
    save_path: str = "results/feature_importance.csv",
) -> pd.DataFrame:
    """学習済み MLPredictor からの特徴量重要度を抽出して CSV に保存する."""
    if not predictor.models:
        raise ValueError("モデルが学習されていません。")

    feature_names = predictor._feature_columns()
    importance_data: List[Dict[str, Any]] = []

    for digit in ["n1", "n2", "n3"]:
        model = predictor.models_raw.get(digit) or predictor.models[digit]
        importances = model.feature_importances_
        for feature_name, importance in zip(feature_names, importances):
            importance_data.append(
                {"digit": digit, "feature": feature_name, "importance": importance}
            )

    df_importance = pd.DataFrame(importance_data)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df_importance.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df_importance
