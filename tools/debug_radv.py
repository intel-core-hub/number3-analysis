"""Debug script to check risk-adjusted EV calculation."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import calibrate_confidence, compute_risk_adjusted_ev

PRED_PATH = Path("data/processed/latest_predictions.parquet")

if not PRED_PATH.exists():
    print(f"Error: {PRED_PATH} not found.")
    sys.exit(1)

df = pd.read_parquet(PRED_PATH)
df = df[df["ev_ratio"] >= 1.05].copy()

if df.empty:
    print("No eligible rows.")
    sys.exit(0)

first = df.iloc[0]

ev = float(first.get("ev_ratio", 0.0) or 0.0)
conf_raw = float(first.get("confidence", 0.0) or 0.0)

print("=== Risk-Adjusted EV Debug ===")
print(f"item_id: {first.get('item_id', 'N/A')}")
print(f"ev_ratio (raw): {ev:.6f}")
print(f"confidence (raw): {conf_raw:.6f}")
print()

conf_score = calibrate_confidence(conf_raw)
base_edge = ev - 1.0
shrink = 0.25 + 0.75 * conf_score
radv = base_edge * shrink

print(f"base_edge = ev_ratio - 1 = {base_edge:.6f}")
print(f"confidence_score = calibrate_confidence({conf_raw:.6f}) = {conf_score:.6f}")
print(f"shrink_factor = 0.25 + 0.75 * {conf_score:.6f} = {shrink:.6f}")
print(
    f"risk_adjusted_ev = base_edge * shrink = {base_edge:.6f} * {shrink:.6f} = {radv:.6f}"
)
print()

radv_from_func = compute_risk_adjusted_ev(ev, conf_raw)
print(f"compute_risk_adjusted_ev({ev:.6f}, {conf_raw:.6f}) = {radv_from_func:.6f}")
print()

print(f"Value in dataframe: {first.get('risk_adjusted_ev', 'missing')}")
