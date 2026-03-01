from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class SelectivePurchaseStrategy:
    ev_threshold: float = 0.0
    confidence_threshold: float = 0.0
    max_tickets: int = 1
    top_k_patterns: int = 1
    payout_history: pd.DataFrame | None = None


class SelectiveBacktester:
    def __init__(
        self,
        strategy: SelectivePurchaseStrategy,
        data: pd.DataFrame | None = None,
        skip_training: bool = False,
    ) -> None:
        self.strategy = strategy
        self.data = data
        self.skip_training = skip_training

    def run(
        self, rounds: int = 0, update_interval: int = 1, verbose: bool = False
    ) -> Dict[str, Any]:
        # Minimal stub: return plausible keys used by callers
        return {
            "roi": 0.0,
            "hit_rate": 0.0,
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "total_hits": 0,
        }
