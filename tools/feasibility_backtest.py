import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (estimate_hit_probability, probability_edge,
                 required_hit_probability)

CSV = ROOT / "artifacts" / "historical_backtest_1y.csv"


def main() -> None:
    if not CSV.exists():
        print(f"Missing file: {CSV}")
        return

    df = pd.read_csv(CSV)
    if df.empty:
        print("No rows in historical CSV")
        return

    for col in ["ev_ratio", "confidence", "profit", "total_cost"]:
        val = df.get(col)
        if val is None:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(val, errors="coerce").fillna(0.0)

    df["estimated_hit_rate"] = df["confidence"].apply(estimate_hit_probability)
    df["required_hit_rate"] = df.apply(
        lambda r: required_hit_probability(
            r.get("ev_ratio", 0.0), r.get("estimated_hit_rate", 0.0)
        ),
        axis=1,
    )
    df["probability_edge"] = df.apply(
        lambda r: probability_edge(r["ev_ratio"], r["confidence"]), axis=1
    )

    print("=== Feasibility summary ===")
    print(f"rows: {len(df)}")
    print(f"edge>=0 rows: {(df['probability_edge'] >= 0).sum()}")
    print(f"edge<0 rows: {(df['probability_edge'] < 0).sum()}")
    print(f"edge mean: {df['probability_edge'].mean():.6f}")
    print(f"edge max: {df['probability_edge'].max():.6f}")
    print(f"edge min: {df['probability_edge'].min():.6f}")

    feasible = df[df["probability_edge"] >= 0].copy()
    if feasible.empty:
        print("No feasible rows found (estimated hit rate never exceeds break-even).")
        return

    total_profit = float(feasible["profit"].sum())
    total_cost = float(feasible["total_cost"].sum())
    roi = (total_profit / total_cost) if total_cost > 0 else 0.0
    print(f"Feasible subset rows: {len(feasible)}")
    print(f"Feasible subset ROI: {roi:.6f}")

    cols = [
        "draw_date",
        "round_idx",
        "ev_ratio",
        "confidence",
        "estimated_hit_rate",
        "required_hit_rate",
        "probability_edge",
        "profit",
    ]
    cols = [c for c in cols if c in feasible.columns]
    print(
        feasible.sort_values("probability_edge", ascending=False)[cols]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
