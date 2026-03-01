import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (apply_grade_veto, calibrate_confidence,
                 compute_composite_score, compute_grade,
                 compute_risk_adjusted_ev, estimate_hit_probability,
                 probability_edge, required_hit_probability,
                 resolve_buy_action)

DATA_PATH = ROOT / "data" / "processed" / "latest_predictions.parquet"

if not DATA_PATH.exists():
    print("No prediction file at", DATA_PATH)
    sys.exit(1)

try:
    df = pd.read_parquet(DATA_PATH)
except Exception as e:
    print("Failed to read parquet:", e)
    sys.exit(1)

# ensure numeric
for col in ["ev_ratio", "ruin_probability", "confidence"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if {"ev_ratio", "confidence", "ruin_probability"}.issubset(df.columns):
    df["confidence_calibrated"] = df["confidence"].apply(
        lambda c: calibrate_confidence(c)
    )
    df["confidence_estimated_hit_rate"] = df["confidence"].apply(
        lambda c: estimate_hit_probability(c)
    )
    df["required_hit_rate"] = df.apply(
        lambda r: required_hit_probability(
            r.get("ev_ratio", 0.0), r.get("confidence_estimated_hit_rate", 0.0)
        ),
        axis=1,
    )
    df["probability_edge"] = df.apply(
        lambda r: probability_edge(r.get("ev_ratio", 0.0), r.get("confidence", 0.0)),
        axis=1,
    )
    df["risk_adjusted_ev"] = df.apply(
        lambda r: compute_risk_adjusted_ev(
            r.get("ev_ratio", 0.0), r.get("confidence", 0.0)
        ),
        axis=1,
    )
    df["composite_score"] = df.apply(
        lambda r: compute_composite_score(
            r.get("ev_ratio", 0.0),
            r.get("confidence", 0.0),
            r.get("ruin_probability", 0.0),
            r.get("risk_adjusted_ev", 0.0),
        ),
        axis=1,
    )
    df["grade"] = df.apply(
        lambda r: apply_grade_veto(
            compute_grade(r.get("composite_score", 0.0)),
            r.get("risk_adjusted_ev", 0.0),
            r.get("ruin_probability", 0.0),
            r.get("confidence", 0.0),
            r.get("ev_ratio", 0.0),
        ),
        axis=1,
    )
    df["buy_action"] = df["grade"].map(lambda g: resolve_buy_action(str(g)))

# Select columns to display
display_cols = [
    "item_id",
    "grade",
    "buy_action",
    "composite_score",
    "predicted_number",
    "purchase_type",
    "predicted_value",
    "risk_adjusted_ev",
    "ev_ratio",
    "confidence_calibrated",
    "confidence_estimated_hit_rate",
    "required_hit_rate",
    "probability_edge",
    "ruin_probability",
    "confidence",
    "updated_at",
]
display_cols = [c for c in display_cols if c in df.columns]

print("Predictions head (first 10 rows):")
print(df[display_cols].head(10).to_string(index=False))

# show eligible by default EV threshold 1.05
eligible = df[df.get("ev_ratio", 0) >= 1.05]
print("\nEligible (EV >= 1.05) head:")
print(eligible[display_cols].head(10).to_string(index=False))
