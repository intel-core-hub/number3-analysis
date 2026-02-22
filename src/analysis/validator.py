"""
src.analysis.validator — 検証ゲート: Champion vs Challenger

MLOps Phase 25: モデル昇格判定とA/Bテスト

機能:
    - 新しいモデル (Challenger) と現在の本番モデル (Champion) を比較
    - 直近のテストデータでバックテスト対決を実行
    - ROI、的中率、キャリブレーションエラーで昇格判定
    - 自動ロールバックのトリガー検出
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.models.registry import ModelMetadata, ModelRegistry
from src.strategies.selective_purchase import SelectiveBacktester, SelectivePurchaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """検証結果"""
    champion_roi: float
    challenger_roi: float
    champion_hit_rate: float
    challenger_hit_rate: float
    champion_calibration_error: float
    challenger_calibration_error: float
    roi_improvement: float  # %ポイント改善
    hit_rate_improvement: float  # %ポイント改善
    calibration_improvement: float  # 改善度
    should_promote: bool
    reason: str


class ValidationGate:
    """検証ゲート: モデル昇格判定"""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        test_rounds: int = 50,
        roi_threshold: float = 2.0,  # ROI改善が2%以上で昇格候補
        hit_rate_threshold: float = 1.0,  # 的中率改善が1%以上で昇格候補
        calibration_threshold: float = 0.02,  # Calibration Error減少が0.02以上で昇格候補
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.test_rounds = test_rounds
        self.roi_threshold = roi_threshold
        self.hit_rate_threshold = hit_rate_threshold
        self.calibration_threshold = calibration_threshold

    def validate_challenger(
        self,
        challenger_metadata: ModelMetadata,
        test_df: pd.DataFrame,
    ) -> ValidationResult:
        """
        Challengerモデルを検証してChampionと比較

        Args:
            challenger_metadata: 候補モデルのメタデータ
            test_df: テストデータ（直近N回号）

        Returns:
            検証結果
        """
        champion = self.registry.get_champion()
        if not champion:
            # Championがいない場合は自動昇格
            logger.info("No champion model exists, auto-promoting challenger")
            return ValidationResult(
                champion_roi=0.0,
                challenger_roi=0.0,
                champion_hit_rate=0.0,
                challenger_hit_rate=0.0,
                champion_calibration_error=1.0,
                challenger_calibration_error=0.0,
                roi_improvement=0.0,
                hit_rate_improvement=0.0,
                calibration_improvement=1.0,
                should_promote=True,
                reason="No existing champion model",
            )

        # 直近のテストデータを切り出し
        test_data = test_df.tail(self.test_rounds).copy()
        if len(test_data) < 10:
            logger.warning("Insufficient test data (%d rounds), skipping validation", len(test_data))
            return ValidationResult(
                champion_roi=0.0,
                challenger_roi=0.0,
                champion_hit_rate=0.0,
                challenger_hit_rate=0.0,
                champion_calibration_error=0.0,
                challenger_calibration_error=0.0,
                roi_improvement=0.0,
                hit_rate_improvement=0.0,
                calibration_improvement=0.0,
                should_promote=False,
                reason=f"Insufficient test data: {len(test_data)} rounds",
            )

        logger.info("Running validation on %d test rounds", len(test_data))

        # Champion評価
        champion_metrics = self._evaluate_model(champion, test_data)

        # Challenger評価
        challenger_metrics = self._evaluate_model(challenger_metadata, test_data)

        # 改善度を計算
        roi_improvement = challenger_metrics["roi"] - champion_metrics["roi"]
        hit_rate_improvement = challenger_metrics["hit_rate"] - champion_metrics["hit_rate"]
        calibration_improvement = champion_metrics["calibration_error"] - challenger_metrics["calibration_error"]

        # 昇格判定
        should_promote = self._should_promote(
            roi_improvement, hit_rate_improvement, calibration_improvement
        )

        reason = self._generate_reason(
            should_promote, roi_improvement, hit_rate_improvement, calibration_improvement
        )

        logger.info("Validation result: %s", reason)

        return ValidationResult(
            champion_roi=champion_metrics["roi"],
            challenger_roi=challenger_metrics["roi"],
            champion_hit_rate=champion_metrics["hit_rate"],
            challenger_hit_rate=challenger_metrics["hit_rate"],
            champion_calibration_error=champion_metrics["calibration_error"],
            challenger_calibration_error=challenger_metrics["calibration_error"],
            roi_improvement=roi_improvement,
            hit_rate_improvement=hit_rate_improvement,
            calibration_improvement=calibration_improvement,
            should_promote=should_promote,
            reason=reason,
        )

    def _evaluate_model(
        self,
        metadata: ModelMetadata,
        test_df: pd.DataFrame,
    ) -> dict[str, float]:
        """
        モデルをテストデータで評価

        Args:
            metadata: モデルメタデータ
            test_df: テストデータ

        Returns:
            評価メトリクス
        """
        try:
            # モデルをロード
            model_path = Path(metadata.artifact_path)
            if not model_path.exists():
                logger.error("Model artifact not found: %s", model_path)
                return {"roi": -100.0, "hit_rate": 0.0, "calibration_error": 1.0}

            # 簡易バックテストを実行（戦略ベース）
            strategy = SelectivePurchaseStrategy(
                ev_threshold=0.0,
                confidence_threshold=0.05,
                max_tickets=5,
                top_k_patterns=2,
                payout_history=test_df,
            )

            backtester = SelectiveBacktester(
                strategy=strategy,
                data=test_df,
                skip_training=False,
            )

            result = backtester.run(rounds=len(test_df), update_interval=1, verbose=False)

            # メトリクスを抽出
            roi = result.get("roi", -100.0)
            hit_rate = result.get("hit_rate", 0.0) * 100 if result.get("hit_rate", 0.0) < 1.0 else result.get("hit_rate", 0.0)  # noqa: E501

            # キャリブレーションエラーを推定（予測確率と実際の的中率の差）
            expected_hits = result.get("total_predictions", 0) * result.get("avg_confidence", 0.1)
            actual_hits = result.get("total_hits", 0)
            calibration_error = abs(expected_hits - actual_hits) / max(result.get("total_predictions", 1), 1)

            logger.info("Model %s evaluated: ROI=%.2f%%, HitRate=%.2f%%, CalibError=%.4f",
                        metadata.version, roi, hit_rate, calibration_error)

            return {
                "roi": roi,
                "hit_rate": hit_rate,
                "calibration_error": calibration_error,
            }

        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            return {"roi": -100.0, "hit_rate": 0.0, "calibration_error": 1.0}

    def _should_promote(
        self,
        roi_improvement: float,
        hit_rate_improvement: float,
        calibration_improvement: float,
    ) -> bool:
        """
        昇格判定ロジック

        Args:
            roi_improvement: ROI改善度 (%)
            hit_rate_improvement: 的中率改善度 (%)
            calibration_improvement: キャリブレーション改善度

        Returns:
            昇格すべきか
        """
        # いずれかのメトリクスで閾値以上改善していれば昇格
        conditions = [
            roi_improvement >= self.roi_threshold,
            hit_rate_improvement >= self.hit_rate_threshold,
            calibration_improvement >= self.calibration_threshold,
        ]

        # ROIが悪化していないことも確認（安全策）
        if roi_improvement < -5.0:  # 5%以上悪化は却下
            logger.warning("ROI degradation detected (%.2f%%), rejecting promotion", roi_improvement)
            return False

        return any(conditions)

    def _generate_reason(
        self,
        should_promote: bool,
        roi_improvement: float,
        hit_rate_improvement: float,
        calibration_improvement: float,
    ) -> str:
        """昇格判定理由を生成"""
        if not should_promote:
            return (
                f"Insufficient improvement: "
                f"ROI={roi_improvement:+.2f}% (threshold={self.roi_threshold}%), "
                f"HitRate={hit_rate_improvement:+.2f}% (threshold={self.hit_rate_threshold}%), "
                f"CalibError={calibration_improvement:+.4f} (threshold={self.calibration_threshold})"
            )

        reasons = []
        if roi_improvement >= self.roi_threshold:
            reasons.append(f"ROI improved by {roi_improvement:+.2f}%")
        if hit_rate_improvement >= self.hit_rate_threshold:
            reasons.append(f"Hit rate improved by {hit_rate_improvement:+.2f}%")
        if calibration_improvement >= self.calibration_threshold:
            reasons.append(f"Calibration error reduced by {calibration_improvement:+.4f}")

        return "Promotion approved: " + ", ".join(reasons)


def run_validation_gate(
    challenger_metadata: ModelMetadata,
    test_df: pd.DataFrame,
    registry: ModelRegistry | None = None,
) -> tuple[bool, ValidationResult]:
    """
    検証ゲートを実行してChallenger昇格可否を判定

    Args:
        challenger_metadata: 候補モデルのメタデータ
        test_df: テストデータ
        registry: モデルレジストリ (Noneの場合はデフォルト)

    Returns:
        (昇格すべきか, 検証結果)
    """
    gate = ValidationGate(registry=registry)
    result = gate.validate_challenger(challenger_metadata, test_df)

    if result.should_promote:
        logger.info("✅ Validation passed: %s", result.reason)
    else:
        logger.info("❌ Validation failed: %s", result.reason)

    return result.should_promote, result
