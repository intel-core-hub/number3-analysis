from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.meta_learner import optimize_threshold_with_optuna


def main() -> None:
    candidates = [
        ROOT / "ml_backtest_results.csv",
        ROOT / "artifacts" / "historical_backtest_1y.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("No input csv found for threshold optimization")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"Input empty: {path}")
        return

    # Required proxy columns for optimizer
    if "miss_proba" not in df.columns:
        if "confidence" in df.columns:
            conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
            # confidence high => miss_proba low (proxy only)
            df["miss_proba"] = 1.0 - conf.clip(0.0, 1.0)
        else:
            df["miss_proba"] = 0.5

    if "profit" not in df.columns:
        if "set_profit" in df.columns:
            df["profit"] = pd.to_numeric(df["set_profit"], errors="coerce").fillna(0.0)
        else:
            df["profit"] = 0.0

    if "cost" not in df.columns:
        if "total_cost" in df.columns:
            df["cost"] = pd.to_numeric(df["total_cost"], errors="coerce").fillna(0.0)
        elif "set_profit" in df.columns:
            # fallback assumption for set bet logs
            df["cost"] = 200.0
        else:
            df["cost"] = 200.0

    result = optimize_threshold_with_optuna(
        df,
        miss_proba_col="miss_proba",
        profit_if_buy_col="profit",
        cost_if_buy_col="cost",
        threshold_min=0.50,
        threshold_max=0.95,
        opportunity_cost_weight=0.35,
        n_trials=80,
    )

    print("=== Meta Threshold Optimization (ROI Utility) ===")
    print(f"input: {path}")
    print(f"best_threshold: {result['best_threshold']:.4f}")
    print(f"best_roi: {result['best_roi']:.6f}")
    print(f"best_profit: {result['best_profit']:.2f}")
    print(f"best_cost: {result['best_cost']:.2f}")
    print(f"best_utility: {result['best_utility']:.2f}")

    out_path = ROOT / "artifacts" / "meta_threshold.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_threshold": float(result.get("best_threshold", 0.65)),
        "best_roi": float(result.get("best_roi", 0.0)),
        "best_profit": float(result.get("best_profit", 0.0)),
        "best_cost": float(result.get("best_cost", 0.0)),
        "best_utility": float(result.get("best_utility", 0.0)),
        "source_csv": str(path),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
