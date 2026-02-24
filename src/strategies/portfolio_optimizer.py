"""
src.strategies.portfolio_optimizer — ポートフォリオ最適化戦略

役割:
    - 複数の購入パターン（ストレート、ボックス、ミニ、セット）を候補とする
    - 予算制約下で期待値最大化する組み合わせを選別
    - リスク分散を考慮した配分
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.utils.config import TICKET_COST, STRAIGHT_PRIZE, BOX_PRIZE_SINGLE, BOX_PRIZE_DOUBLE, MINI_PRIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TicketCandidate:
    """購入候補チケット"""
    number: str
    bet_type: str  # 'straight', 'box', 'mini', 'set', 'hybrid'
    cost: int
    expected_value: float  # 期待値的中
    confidence: float  # 予測信頼度
    predicted_prize: float
    ev_ratio: float  # EV / 投資額
    diversification_score: float  # リスク分散スコア（0.0-1.0）


@dataclass
class PortfolioSelection:
    """ポートフォリオ選定結果"""
    tickets: List[TicketCandidate]
    total_cost: int
    total_expected_value: float
    portfolio_ev_ratio: float
    portfolio_confidence: float
    diversification_ratio: float
    reason: str


class PortfolioOptimizer:
    """ポートフォリオ最適化戦略"""

    def __init__(
        self,
        budget: int = 5000,
        min_ev_ratio: float = 1.05,
        max_tickets: int = 20,
        diversification_weight: float = 0.1,  # EV と多様性のトレードオフ
    ):
        """
        Args:
            budget: 購入予算上限
            min_ev_ratio: 最低期待値倍率
            max_tickets: 最大購入票数
            diversification_weight: 多様性の重み（0.0=EV優先, 1.0=多様性優先）
        """
        self.budget = budget
        self.min_ev_ratio = min_ev_ratio
        self.max_tickets = max_tickets
        self.diversification_weight = diversification_weight

    def generate_candidates(
        self,
        numbers: List[str],
        payouts: pd.DataFrame,  # {"number", "straight_prize", "box_prize"}
        probabilities: pd.DataFrame,  # {"number", "straight_prob", "box_prob", "mini_prob"}
    ) -> List[TicketCandidate]:
        """
        候補チケット一覧を生成

        Args:
            numbers: 検討対象の数字リスト
            payouts: 配当予測（number, straight_prize, box_prize）
            probabilities: 的中確率予測（number, straight_prob, box_prob, mini_prob）

        Returns:
            TicketCandidate リスト
        """
        candidates = []

        for number in numbers:
            num_str = str(number).zfill(3)

            # データ取得
            payout_row = payouts[payouts["number"] == num_str].iloc[0] if not payouts.empty else None
            prob_row = probabilities[probabilities["number"] == num_str].iloc[0] if not probabilities.empty else None

            if payout_row is None or prob_row is None:
                continue

            straight_prize = float(payout_row.get("straight_prize", STRAIGHT_PRIZE))
            box_prize_default = (
                BOX_PRIZE_DOUBLE if len(set(num_str)) == 2 else BOX_PRIZE_SINGLE
            )
            box_prize = float(payout_row.get("box_prize", box_prize_default))

            straight_prob = float(prob_row.get("straight_prob", 0.001))
            box_prob = float(prob_row.get("box_prob", 0.01))
            mini_prob = float(prob_row.get("mini_prob", 0.01))

            # ストレート
            # EV比 = (期待回収額) / 投資額
            # 期待利益 = 期待回収額 - 投資額 = 投資額 × (EV比 - 1)
            expected_return_straight = straight_prob * straight_prize
            ev_ratio_straight = expected_return_straight / TICKET_COST
            expected_profit_straight = expected_return_straight - TICKET_COST
            candidates.append(
                TicketCandidate(
                    number=num_str,
                    bet_type="straight",
                    cost=TICKET_COST,
                    expected_value=expected_profit_straight,
                    confidence=straight_prob,
                    predicted_prize=straight_prize,
                    ev_ratio=ev_ratio_straight,
                    diversification_score=1.0,  # Pure outcome
                )
            )

            # ボックス
            expected_return_box = box_prob * box_prize
            ev_ratio_box = expected_return_box / TICKET_COST
            expected_profit_box = expected_return_box - TICKET_COST
            candidates.append(
                TicketCandidate(
                    number=num_str,
                    bet_type="box",
                    cost=TICKET_COST,
                    expected_value=expected_profit_box,
                    confidence=box_prob,
                    predicted_prize=box_prize,
                    ev_ratio=ev_ratio_box,
                    diversification_score=0.8,
                )
            )

            # ミニ
            expected_return_mini = mini_prob * MINI_PRIZE
            ev_ratio_mini = expected_return_mini / TICKET_COST
            expected_profit_mini = expected_return_mini - TICKET_COST
            candidates.append(
                TicketCandidate(
                    number=num_str,
                    bet_type="mini",
                    cost=TICKET_COST,
                    expected_value=expected_profit_mini,
                    confidence=mini_prob,
                    predicted_prize=MINI_PRIZE,
                    ev_ratio=ev_ratio_mini,
                    diversification_score=0.5,
                )
            )

        return sorted(candidates, key=lambda x: x.ev_ratio, reverse=True)

    def select_optimal_portfolio(
        self,
        candidates: List[TicketCandidate],
    ) -> PortfolioSelection:
        """
        予算制約下で期待値最大化＆多様性バランスを取るポートフォリオを選定

        Args:
            candidates: TicketCandidate リスト

        Returns:
            PortfolioSelection
        """
        if not candidates:
            return PortfolioSelection(
                tickets=[],
                total_cost=0,
                total_expected_value=0.0,
                portfolio_ev_ratio=0.0,
                portfolio_confidence=0.0,
                diversification_ratio=0.0,
                reason="候補がありません",
            )

        # フィルタリング
        filtered = [c for c in candidates if c.ev_ratio >= self.min_ev_ratio]
        if not filtered:
            return PortfolioSelection(
                tickets=[],
                total_cost=0,
                total_expected_value=0.0,
                portfolio_ev_ratio=0.0,
                portfolio_confidence=0.0,
                diversification_ratio=0.0,
                reason=f"EV >= {self.min_ev_ratio} を満たす候補がありません",
            )

        # 貪欲法によるナップサック近似
        # ステップ1: EV比率でソート
        sorted_by_ev = sorted(filtered, key=lambda x: x.ev_ratio, reverse=True)

        selected = []
        total_cost = 0

        # ステップ2: EV優先で選択
        for cand in sorted_by_ev:
            if total_cost + cand.cost > self.budget or len(selected) >= self.max_tickets:
                break
            # 同一数字・同一bet_type の重複チェック
            duplicate = any(
                s.number == cand.number and s.bet_type == cand.bet_type for s in selected
            )
            if duplicate:
                continue
            selected.append(cand)
            total_cost += cand.cost

        if not selected:
            return PortfolioSelection(
                tickets=[],
                total_cost=0,
                total_expected_value=0.0,
                portfolio_ev_ratio=0.0,
                portfolio_confidence=0.0,
                diversification_ratio=0.0,
                reason="条件を満たす組み合わせがありません",
            )

        # メトリクス計算
        total_ev = sum(t.expected_value for t in selected)
        avg_confidence = np.mean([t.confidence for t in selected])
        portfolio_ev_ratio = total_ev / total_cost if total_cost > 0 else 0.0
        
        # 多様性スコア（異なる数字の種類とbet_typeの分散）
        unique_numbers = len(set(t.number for t in selected))
        unique_bet_types = len(set(t.bet_type for t in selected))
        diversification_ratio = (unique_numbers / len(selected)) * (unique_bet_types / 3.0)

        reason = (
            f"EV優先選定: {len(selected)}点 / "
            f"予算 {total_cost:,}円 (上限 {self.budget:,}円)"
        )

        return PortfolioSelection(
            tickets=selected,
            total_cost=total_cost,
            total_expected_value=total_ev,
            portfolio_ev_ratio=portfolio_ev_ratio,
            portfolio_confidence=avg_confidence,
            diversification_ratio=diversification_ratio,
            reason=reason,
        )

    def summary_text(self, selection: PortfolioSelection) -> str:
        """選定結果の要約"""
        if not selection.tickets:
            return f"⚠️ {selection.reason}"

        return (
            f"🎯 **ポートフォリオ選定結果**\n"
            f"- 選定票数: {len(selection.tickets)}\n"
            f"- 総コスト: {selection.total_cost:,}円\n"
            f"- 総期待値: {selection.total_expected_value:,.0f}円\n"
            f"- ポートフォリオEV比: {selection.portfolio_ev_ratio:.3f}x\n"
            f"- 平均自信度: {selection.portfolio_confidence:.2%}\n"
            f"- 多様性スコア: {selection.diversification_ratio:.2%}\n"
            f"- {selection.reason}\n"
        )
