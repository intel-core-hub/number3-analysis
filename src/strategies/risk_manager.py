from __future__ import annotations

from typing import Any, Optional, Tuple


class CircuitBreaker:
    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def __call__(self, df: Any) -> bool:
        return False

    def compute_max_drawdown_pct(self, equity_curve: Any) -> float:
        """Estimate maximum drawdown percentage from an equity curve.

        This is a lightweight implementation used for health checks and UI.
        """
        try:
            # equity_curve is expected to be an iterable of numbers
            values = list(equity_curve)
            if not values:
                return 0.0
            peak = values[0]
            max_dd = 0.0
            for v in values:
                if v > peak:
                    peak = v
                dd = (peak - v) / max(peak, 1e-9)
                if dd > max_dd:
                    max_dd = dd
            return float(max_dd * 100.0)
        except Exception:
            return 0.0

    def should_block(
        self,
        max_drawdown_pct: float,
        regime_confidence: Optional[float] = None,
        drift_score: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Simple blocking logic for UI; returns (triggered, reason)."""
        # If drawdown exceeds threshold, trigger
        if max_drawdown_pct >= (self.threshold * 100.0):
            return True, f"drawdown {max_drawdown_pct:.1f}% >= threshold"
        # optional additional signals
        if regime_confidence is not None and regime_confidence < 0.2:
            return True, "low regime confidence"
        if drift_score is not None and drift_score > 0.9:
            return True, "high data drift"
        return False, None
