"""Show column names in latest predictions parquet."""
import sys
from pathlib import Path

import pandas as pd

PRED_PATH = Path("data/processed/latest_predictions.parquet")

if not PRED_PATH.exists():
    print(f"Error: {PRED_PATH} not found.")
    sys.exit(1)

df = pd.read_parquet(PRED_PATH)
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  - {col}")
