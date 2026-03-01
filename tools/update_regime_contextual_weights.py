from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.rl_agent import RLMetaAgent
from src.models.weight_evolver import DEFAULT_WEIGHTS


def _load_history_df() -> pd.DataFrame:
    for p in [
        ROOT / "numbers3_clean.csv",
        ROOT / "data" / "raw" / "numbers3_clean.csv",
    ]:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def _load_recent_pnls() -> list[float]:
    path = ROOT / "ml_backtest_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    for col in ["set_profit", "profit"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
            return [float(v) for v in vals]
    return []


def main() -> None:
    hist_df = _load_history_df()
    if hist_df.empty:
        print("No history dataframe found for regime detection")
        return

    pnls = _load_recent_pnls()
    if not pnls:
        print("No pnl series found; using neutral pseudo pnl")
        pnls = [0.0] * 20

    agent = RLMetaAgent(results_dir=ROOT / "results")
    new_weights = agent.adapt_weights_with_regime(
        history_df=hist_df,
        pnls=pnls,
        base_weights=dict(DEFAULT_WEIGHTS),
    )

    print("=== Contextual Bandit Weight Update ===")
    print(new_weights)


if __name__ == "__main__":
    main()
