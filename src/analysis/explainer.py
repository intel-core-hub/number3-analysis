"""
Phase 29: SHAP-based Explainable AI Module

SHAPを使用してモデル予測の根拠を明示化します。
- LGBMPredictor: Tree SHAP による特徴量の寄与度分析
- EnsemblePredictor: 各モデルの重み付けスコアと特徴量寄与の統合

軽量化モード:
- デフォルトでは最新の1件（次回予測）のみを計算します。
- 必要に応じて履歴全体の分析も可能です。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from src.predictors import LGBMPredictor
from src.predictors.ensemble import EnsemblePredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) ベースの説明器

    LGBMPredictorやEnsemblePredictorに対してTreeExplainerを適用し、
    予測の根拠となる特徴量の寄与度を算出します。
    """

    def __init__(
        self,
        predictor: LGBMPredictor | EnsemblePredictor,
        feature_columns: list[str] | None = None,
    ):
        """
        Args:
            predictor: 説明対象のPredictor (LGBMPredictorまたはEnsemblePredictor)
            feature_columns: 特徴量カラムのリスト（指定しない場合は自動判定）
        """
        self.predictor = predictor
        self.feature_columns = feature_columns
        self.explainers: dict[str, shap.TreeExplainer] = {}
        self._initialize_explainers()

    def _initialize_explainers(self) -> None:
        """
        各モデル（n1, n2, n3）に対してTreeExplainerを初期化します。
        """
        if isinstance(self.predictor, LGBMPredictor):
            models = self.predictor.models()
            if self.feature_columns is None:
                self.feature_columns = self.predictor.feature_columns()

            for digit_name, model in models.items():
                try:
                    self.explainers[digit_name] = shap.TreeExplainer(model)
                    logger.info(f"SHAP Explainer initialized for {digit_name}")
                except Exception as exc:
                    logger.warning(f"Failed to initialize SHAP explainer for {digit_name}: {exc}")

        elif isinstance(self.predictor, EnsemblePredictor):
            # EnsemblePredictorの場合はLGBMコンポーネントを使用
            lgbm_predictor = self.predictor.lgbm
            if lgbm_predictor is not None and hasattr(lgbm_predictor, 'models'):
                models = lgbm_predictor.models()
                if self.feature_columns is None:
                    self.feature_columns = lgbm_predictor.feature_columns()

                for digit_name, model in models.items():
                    try:
                        self.explainers[digit_name] = shap.TreeExplainer(model)
                        logger.info(f"SHAP Explainer (Ensemble LGBM) initialized for {digit_name}")
                    except Exception as exc:
                        logger.warning(f"Failed to initialize SHAP explainer for {digit_name}: {exc}")

    def explain_latest(self) -> dict[str, pd.DataFrame]:
        """
        最新の予測（次回予測）に対してSHAP値を計算します。

        Returns:
            各位（n1, n2, n3）のSHAP値を含む辞書
            {
                'n1': DataFrame(columns=['feature', 'shap_value']),
                'n2': DataFrame(...),
                'n3': DataFrame(...)
            }
        """
        if not self.explainers:
            logger.warning("No SHAP explainers available.")
            return {}

        # 最新の特徴量を取得
        if isinstance(self.predictor, LGBMPredictor):
            latest_features = self.predictor.latest_features()
        elif isinstance(self.predictor, EnsemblePredictor):
            latest_features = self.predictor.lgbm.latest_features()
        else:
            logger.error("Unsupported predictor type for SHAP explanation.")
            return {}

        if latest_features is None or latest_features.empty:
            logger.warning("No features available for SHAP explanation.")
            return {}

        # 特徴量の整列
        if self.feature_columns:
            latest_features = latest_features[self.feature_columns]

        results = {}
        for digit_name, explainer in self.explainers.items():
            try:
                shap_values = explainer.shap_values(latest_features)

                # SHAP値をDataFrameに変換
                if isinstance(shap_values, list):
                    # 多クラス分類の場合は平均を取る
                    shap_values = np.mean(shap_values, axis=0)

                df_shap = pd.DataFrame({
                    'feature': self.feature_columns,
                    'shap_value': shap_values[0],  # 最新の1件
                })
                df_shap = df_shap.sort_values('shap_value', key=abs, ascending=False)
                results[digit_name] = df_shap

                logger.info(f"SHAP values computed for {digit_name}: top feature = {df_shap.iloc[0]['feature']}")

            except Exception as exc:
                logger.warning(f"SHAP computation failed for {digit_name}: {exc}")

        return results

    def explain_top_features(self, top_n: int = 5) -> pd.DataFrame:
        """
        全ての位を統合して、上位N個の特徴量を抽出します。

        Args:
            top_n: 上位何個の特徴量を抽出するか

        Returns:
            統合されたSHAP値のDataFrame
            columns: ['feature', 'n1_shap', 'n2_shap', 'n3_shap', 'total_abs_shap']
        """
        shap_results = self.explain_latest()
        if not shap_results:
            return pd.DataFrame()

        # 各位のSHAP値を統合
        merged = None
        for digit_name, df_shap in shap_results.items():
            df_temp = df_shap.rename(columns={'shap_value': f'{digit_name}_shap'})
            if merged is None:
                merged = df_temp
            else:
                merged = merged.merge(df_temp, on='feature', how='outer')

        if merged is None:
            return pd.DataFrame()

        # 欠損値を0で埋める
        merged = merged.fillna(0)

        # 絶対値の合計でソート
        merged['total_abs_shap'] = (
            merged['n1_shap'].abs() +
            merged['n2_shap'].abs() +
            merged['n3_shap'].abs()
        )
        merged = merged.sort_values('total_abs_shap', ascending=False).head(top_n)

        return merged

    def generate_explanation_text(self, top_n: int = 3) -> str:
        """
        上位N個の特徴量から自然言語の説明文を生成します。

        Args:
            top_n: 説明に含める特徴量の数

        Returns:
            説明文の文字列
        """
        top_features = self.explain_top_features(top_n=top_n)

        if top_features.empty:
            return "予測の根拠となる特徴量が取得できませんでした。"

        lines = ["### 予測の根拠（SHAP分析）"]
        lines.append("")
        lines.append("以下の特徴量が今回の予測に**最も寄与**しています：")
        lines.append("")

        for idx, row in top_features.iterrows():
            feature_name = row['feature']
            n1_val = row['n1_shap']
            n2_val = row['n2_shap']
            n3_val = row['n3_shap']
            total = row['total_abs_shap']

            # 特徴量名を人間が読みやすい形に変換（簡易版）
            readable_name = self._make_readable_feature_name(feature_name)

            # 各位への影響を記述
            impacts = []
            if abs(n1_val) > 0.01:
                direction = "プラス" if n1_val > 0 else "マイナス"
                impacts.append(f"1位目に{direction}影響")
            if abs(n2_val) > 0.01:
                direction = "プラス" if n2_val > 0 else "マイナス"
                impacts.append(f"2位目に{direction}影響")
            if abs(n3_val) > 0.01:
                direction = "プラス" if n3_val > 0 else "マイナス"
                impacts.append(f"3位目に{direction}影響")

            impact_text = "、".join(impacts) if impacts else "全体的に影響"
            lines.append(f"- **{readable_name}**: {impact_text} (寄与度: {total:.4f})")

        return "\n".join(lines)

    def _make_readable_feature_name(self, feature_name: str) -> str:
        """
        特徴量名を人間が読みやすい日本語に変換します（簡易版）。

        Args:
            feature_name: 元の特徴量名

        Returns:
            読みやすい名前
        """
        # 簡易的な変換ルール
        replacements = {
            'weekday': '曜日',
            'sum_last': '直近の合計値',
            'interval': 'インターバル',
            'hot_': '頻出数字_',
            'cold_': '低頻度数字_',
            'ema_': '指数移動平均_',
            'momentum_': 'モメンタム_',
            'gap_': 'ギャップ_',
            'pattern_': 'パターン_',
        }

        readable = feature_name
        for key, value in replacements.items():
            readable = readable.replace(key, value)

        return readable

    def save_explanation(self, output_path: Path | str) -> None:
        """
        説明結果をCSVファイルとして保存します。

        Args:
            output_path: 保存先のパス
        """
        top_features = self.explain_top_features(top_n=20)
        if not top_features.empty:
            top_features.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"SHAP explanation saved to {output_path}")


def explain_ensemble_prediction(
    ensemble_predictor: EnsemblePredictor,
    top_n: int = 5,
    save_path: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    EnsemblePredictorの予測に対してSHAP分析を実行します。

    Args:
        ensemble_predictor: EnsemblePredictor インスタンス
        top_n: 上位N個の特徴量を取得
        save_path: 結果を保存するパス（オプション）

    Returns:
        (top_features DataFrame, explanation_text)
    """
    explainer = SHAPExplainer(ensemble_predictor)
    top_features = explainer.explain_top_features(top_n=top_n)
    explanation_text = explainer.generate_explanation_text(top_n=top_n)

    if save_path:
        explainer.save_explanation(save_path)

    return top_features, explanation_text


def explain_lgbm_prediction(
    lgbm_predictor: LGBMPredictor,
    top_n: int = 5,
    save_path: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    LGBMPredictorの予測に対してSHAP分析を実行します。

    Args:
        lgbm_predictor: LGBMPredictor インスタンス
        top_n: 上位N個の特徴量を取得
        save_path: 結果を保存するパス（オプション）

    Returns:
        (top_features DataFrame, explanation_text)
    """
    explainer = SHAPExplainer(lgbm_predictor)
    top_features = explainer.explain_top_features(top_n=top_n)
    explanation_text = explainer.generate_explanation_text(top_n=top_n)

    if save_path:
        explainer.save_explanation(save_path)

    return top_features, explanation_text
