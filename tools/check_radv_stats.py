import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import compute_risk_adjusted_ev

p = ROOT / "artifacts" / "historical_backtest_1y.csv"
df = pd.read_csv(p)
df["ev_ratio"] = pd.to_numeric(df["ev_ratio"], errors="coerce").fillna(0.0)
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
df["risk_adjusted_ev"] = df.apply(
    lambda r: compute_risk_adjusted_ev(r["ev_ratio"], r["confidence"]), axis=1
)
print(df["risk_adjusted_ev"].describe())
print("fraction negative:", (df["risk_adjusted_ev"] < 0).mean())
