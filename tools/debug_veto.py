"""Debug script to check veto conditions in detail."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (apply_grade_veto, compute_grade, estimate_hit_probability,
                 get_confidence_veto_threshold, probability_edge,
                 required_hit_probability)

PRED_PATH = Path("data/processed/latest_predictions.parquet")

if not PRED_PATH.exists():
    print(f"Error: {PRED_PATH} not found.")
    sys.exit(1)

df = pd.read_parquet(PRED_PATH)
df = df[df["ev_ratio"] >= 1.05].copy()

if df.empty:
    print("No eligible rows (EV >= 1.05).")
    sys.exit(0)

first_row = df.iloc[0]

print("=== Veto Debug (First Eligible Row) ===")
print(f"item_id: {first_row.get('item_id', 'N/A')}")
print(f"composite_score: {first_row.get('composite_score', 0.0):.2f}")
print(
    f"raw grade (before veto): {compute_grade(first_row.get('composite_score', 0.0))}"
)
print()
print(f"ev_ratio: {first_row.get('ev_ratio', 0.0):.6f}")
print(f"confidence (raw): {first_row.get('confidence', 0.0):.6f}")
print(f"risk_adjusted_ev: {first_row.get('risk_adjusted_ev', 0.0):.6f}")
print(f"ruin_probability: {first_row.get('ruin_probability', 0.0):.6f}")
print()

conf_raw = float(first_row.get("confidence", 0.0) or 0.0)
ev = float(first_row.get("ev_ratio", 0.0) or 0.0)
radv = float(first_row.get("risk_adjusted_ev", 0.0) or 0.0)
ruin = float(first_row.get("ruin_probability", 0.0) or 0.0)

est_hit = estimate_hit_probability(conf_raw)
req_hit = required_hit_probability(ev, est_hit)
edge = probability_edge(ev, conf_raw)
veto_thresh = get_confidence_veto_threshold()

print(f"estimated_hit_rate: {est_hit:.6f} ({est_hit*100:.2f}%)")
print(f"required_hit_rate: {req_hit:.6f} ({req_hit*100:.2f}%)")
print(f"probability_edge: {edge:.6f} ({edge*100:.2f}%)")
print(f"confidence_veto_threshold: {veto_thresh:.6f}")
print()

print("--- Veto Condition Checks ---")
print(f"radv < 0: {radv < 0} (radv={radv:.6f})")
print(f"ruin >= 0.20: {ruin >= 0.20} (ruin={ruin:.6f})")
print(
    f"est_hit < veto_thresh: {est_hit < veto_thresh} (est={est_hit:.6f}, thresh={veto_thresh:.6f})"
)
print(f"ev > 0: {ev > 0} (ev={ev:.6f})")
print(f"edge < 0 (when ev > 0): {edge < 0} (edge={edge:.6f})")
print()

grade_raw = compute_grade(first_row.get("composite_score", 0.0))
grade_final = apply_grade_veto(
    grade_raw,
    radv,
    ruin,
    conf_raw,
    ev,
)
print(f"Final grade after veto: {grade_final}")
print()

should_be_e = (
    (radv < 0) or (ruin >= 0.20) or (est_hit < veto_thresh) or (ev > 0 and edge < 0)
)
print(f"Expected veto = E: {should_be_e}")
