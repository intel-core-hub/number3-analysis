"""
src.utils.paper_trader — ペーパートレーディング・ロガー

機能:
    - 予測時の購入判断を保存
    - 次回実行時に実現損益を自動照合
    - 資産推移 (Equity Curve) を生成
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

from src.utils.config import (
    BOX_PRIZE_DOUBLE,
    BOX_PRIZE_SINGLE,
    MINI_PRIZE,
    SET_BOX_PRIZE,
    SET_STRAIGHT_PRIZE,
    STRAIGHT_PRIZE,
    TICKET_COST,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PaperBet:
    bet_type: str
    number: str
    amount: int
    expected_value: float


@dataclass
class PaperTrade:
    trade_id: str
    timestamp: str
    target_round: int
    bets: list[PaperBet]
    total_cost: int
    status: str  # "open" or "settled"
    settled_round: int | None = None
    winning_number: str | None = None
    pnl: float | None = None
    bankroll_after: float | None = None


@dataclass
class PaperTraderState:
    starting_bankroll: float
    current_bankroll: float
    trades: list[PaperTrade]


class PaperTrader:
    """ペーパートレーディング管理クラス"""

    def __init__(
        self,
        results_dir: Path,
        starting_bankroll: float = 100_000.0,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trade_path = self.results_dir / "paper_trades.json"
        self.state = self._load_state(starting_bankroll)
        self.integrity_ok = self.validate_state()

    def _load_state(self, starting_bankroll: float) -> PaperTraderState:
        if not self.trade_path.exists():
            return PaperTraderState(
                starting_bankroll=starting_bankroll,
                current_bankroll=starting_bankroll,
                trades=[],
            )
        try:
            data = json.loads(self.trade_path.read_text(encoding="utf-8"))
            trades = [PaperTrade(
                trade_id=t["trade_id"],
                timestamp=t["timestamp"],
                target_round=int(t["target_round"]),
                bets=[PaperBet(**b) for b in t.get("bets", [])],
                total_cost=int(t["total_cost"]),
                status=t.get("status", "open"),
                settled_round=t.get("settled_round"),
                winning_number=t.get("winning_number"),
                pnl=t.get("pnl"),
                bankroll_after=t.get("bankroll_after"),
            ) for t in data.get("trades", [])]
            return PaperTraderState(
                starting_bankroll=float(data.get("starting_bankroll", starting_bankroll)),
                current_bankroll=float(data.get("current_bankroll", starting_bankroll)),
                trades=trades,
            )
        except Exception as exc:
            logger.warning("Failed to load paper trades: %s", exc)
            return PaperTraderState(
                starting_bankroll=starting_bankroll,
                current_bankroll=starting_bankroll,
                trades=[],
            )

    def _save_state(self) -> None:
        data = {
            "starting_bankroll": self.state.starting_bankroll,
            "current_bankroll": self.state.current_bankroll,
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "timestamp": t.timestamp,
                    "target_round": t.target_round,
                    "bets": [asdict(b) for b in t.bets],
                    "total_cost": t.total_cost,
                    "status": t.status,
                    "settled_round": t.settled_round,
                    "winning_number": t.winning_number,
                    "pnl": t.pnl,
                    "bankroll_after": t.bankroll_after,
                }
                for t in self.state.trades
            ],
        }
        data["checksum"] = self._compute_checksum(data)
        self.trade_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def record_trade(self, target_round: int, bets: list[PaperBet]) -> PaperTrade | None:
        if not bets:
            return None

        total_cost = sum(b.amount for b in bets)
        if total_cost <= 0:
            return None

        if self.state.current_bankroll < total_cost:
            logger.warning("Insufficient bankroll: %s < %s", self.state.current_bankroll, total_cost)
            return None

        trade_id = datetime.now().strftime("%Y%m%d%H%M%S")
        trade = PaperTrade(
            trade_id=trade_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            target_round=target_round,
            bets=bets,
            total_cost=total_cost,
            status="open",
        )
        self.state.current_bankroll -= total_cost
        self.state.trades.append(trade)
        self._save_state()

        logger.info("Paper trade recorded: round=%s, cost=%s", target_round, total_cost)
        return trade

    def settle_with_draws(self, draws_df: pd.DataFrame) -> int:
        if draws_df.empty or "回号" not in draws_df.columns or "当選番号" not in draws_df.columns:
            return 0

        latest_round = int(pd.to_numeric(draws_df["回号"], errors="coerce").max())
        round_map = {
            int(r["回号"]): str(r["当選番号"]).zfill(3)
            for _, r in draws_df.iterrows()
            if pd.notna(r.get("回号")) and pd.notna(r.get("当選番号"))
        }

        settled_count = 0
        for trade in self.state.trades:
            if trade.status != "open":
                continue
            if trade.target_round not in round_map:
                continue
            if trade.target_round > latest_round:
                continue

            winning_number = round_map[trade.target_round]
            payout = self._calculate_payout(trade.bets, winning_number)
            pnl = payout - trade.total_cost

            self.state.current_bankroll += payout
            trade.status = "settled"
            trade.settled_round = trade.target_round
            trade.winning_number = winning_number
            trade.pnl = pnl
            trade.bankroll_after = self.state.current_bankroll
            settled_count += 1

        if settled_count > 0:
            self._save_state()
            logger.info("Settled %d paper trades", settled_count)

        return settled_count

    def get_equity_curve(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        bankroll = self.state.starting_bankroll
        rows.append({"timestamp": None, "equity": bankroll})

        for trade in sorted(self.state.trades, key=lambda t: t.timestamp):
            bankroll -= trade.total_cost
            if trade.status == "settled" and trade.pnl is not None:
                bankroll += trade.total_cost + trade.pnl
            rows.append({"timestamp": trade.timestamp, "equity": bankroll})

        return pd.DataFrame(rows)

    def validate_state(self) -> bool:
        if not self.trade_path.exists():
            return True
        try:
            data = json.loads(self.trade_path.read_text(encoding="utf-8"))
            checksum = data.get("checksum")
            if not checksum:
                return False
            expected = self._compute_checksum({
                "starting_bankroll": data.get("starting_bankroll"),
                "current_bankroll": data.get("current_bankroll"),
                "trades": data.get("trades", []),
            })
            if checksum != expected:
                logger.warning("Paper trade checksum mismatch")
                return False
            recomputed = self.recompute_bankroll()
            if abs(recomputed - self.state.current_bankroll) > 1e-6:
                logger.warning("Paper trade bankroll mismatch: %s vs %s", recomputed, self.state.current_bankroll)
                return False
            return True
        except Exception as exc:
            logger.warning("Failed to validate paper trades: %s", exc)
            return False

    def recompute_bankroll(self) -> float:
        if not self.state.trades:
            return self.state.starting_bankroll

        rows = []
        for trade in self.state.trades:
            rows.append({
                "status": trade.status,
                "total_cost": trade.total_cost,
                "pnl": trade.pnl or 0.0,
            })
        df = pl.DataFrame(rows)

        total_cost = df.select(pl.col("total_cost").sum()).item() or 0.0
        pnl_sum = df.filter(pl.col("status") == "settled").select(pl.col("pnl").sum()).item() or 0.0
        return float(self.state.starting_bankroll - total_cost + pnl_sum)

    def get_recent_pnls(self, lookback: int = 30) -> list[float]:
        pnls = [t.pnl for t in self.state.trades if t.status == "settled" and t.pnl is not None]
        return pnls[-lookback:]

    @staticmethod
    def _compute_checksum(data: dict[str, object]) -> str:
        payload = json.dumps(data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _calculate_payout(self, bets: list[PaperBet], winning_number: str) -> float:
        payout = 0.0
        for bet in bets:
            if self._is_win(bet, winning_number):
                prize = self._get_prize(bet.bet_type, bet.number)
                payout += (bet.amount / TICKET_COST) * prize
        return payout

    @staticmethod
    def _is_win(bet: PaperBet, winning_number: str) -> bool:
        win_num = str(winning_number).zfill(3)
        bet_num = str(bet.number).zfill(3)
        if bet.bet_type in ("straight", "set_straight"):
            return win_num == bet_num
        if bet.bet_type in ("box", "set_box"):
            return sorted(win_num) == sorted(bet_num)
        if bet.bet_type == "mini":
            return win_num[-2:] == bet_num[-2:]
        return False

    @staticmethod
    def _get_prize(bet_type: str, number: str) -> int:
        if bet_type == "straight":
            return STRAIGHT_PRIZE
        if bet_type == "box":
            unique_digits = len(set(str(number).zfill(3)))
            return BOX_PRIZE_DOUBLE if unique_digits == 2 else BOX_PRIZE_SINGLE
        if bet_type == "mini":
            return MINI_PRIZE
        if bet_type == "set_straight":
            return SET_STRAIGHT_PRIZE
        if bet_type == "set_box":
            return SET_BOX_PRIZE
        return 0


def simulate_kelly_equity(
    backtest_df: pd.DataFrame,
    starting_bankroll: float = 100_000.0,
    prize_col: str = "set_prize",
    profit_col: str = "set_profit",
    window: int = 50,
    max_risk_pct: float = 0.05,
) -> pd.DataFrame:
    """
    バックテスト結果からケリー運用の仮想資産曲線を生成

    直近window回の的中率を確率として推定し、ケリー比率で賭け金を調整。
    """
    if backtest_df.empty:
        return pd.DataFrame(columns=["round", "equity"])

    bankroll = starting_bankroll
    equities = []
    hits = []

    for _, row in backtest_df.iterrows():
        prize = float(row.get(prize_col, 0))
        profit = float(row.get(profit_col, 0))
        hit = 1 if profit > 0 else 0
        hits.append(hit)

        if len(hits) <= 5:
            p = 0.1
        else:
            window_hits = hits[-window:]
            p = sum(window_hits) / len(window_hits)

        odds = prize / TICKET_COST - 1.0 if prize > 0 else 0.0
        if odds <= 0:
            bet_size = 0
        else:
            f = max(0.0, min(1.0, (p * (odds + 1.0) - 1.0) / odds))
            f = min(f, max_risk_pct)
            bet_size = int((bankroll * f) // TICKET_COST) * TICKET_COST

        payout = bet_size * (prize / TICKET_COST) if hit else 0.0
        bankroll += payout - bet_size

        equities.append({"round": int(row.get("target_round", 0)), "equity": bankroll})

    return pd.DataFrame(equities)
