from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CANDIDATES = [
    Path("data/raw/latest_features.parquet"),
    Path("numbers3_clean.parquet"),
    Path("numbers3_clean.csv"),
]
OUTPUT_PATH = Path("data/processed/latest_predictions.parquet")
OUTPUT_CSV_PATH = Path("data/processed/latest_predictions.csv")
MIN_ROWS = 10


logger = logging.getLogger("batch_predict")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def load_latest_data() -> pd.DataFrame:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists() and candidate.suffix.lower() == ".parquet":
            logger.info("Loading parquet: %s", candidate)
            return pd.read_parquet(candidate)
        if candidate.exists() and candidate.suffix.lower() == ".csv":
            logger.info("Loading csv: %s", candidate)
            return pd.read_csv(candidate)
    raise FileNotFoundError("No input dataset found. Checked: " + ", ".join(str(x) for x in INPUT_CANDIDATES))


def run_inference(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    if "draw_no" in df.columns:
        item_id = df["draw_no"].astype(str)
    elif "id" in df.columns:
        item_id = df["id"].astype(str)
    else:
        item_id = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)

    base_feature = None
    for col in ("bonus", "set", "sum", "month", "day"):
        if col in df.columns:
            base_feature = pd.to_numeric(df[col], errors="coerce")
            break
    if base_feature is None:
        base_feature = pd.Series(np.arange(len(df), dtype=float), index=df.index)

    z = (base_feature - base_feature.mean()) / (base_feature.std(ddof=0) + 1e-9)
    confidence = (0.5 + 0.2 * np.tanh(z)).clip(0.05, 0.95)
    ev_ratio = (1.0 + 0.08 * np.tanh(z / 1.2)).clip(0.85, 1.25)
    ruin_probability = (0.30 - 0.22 * confidence + 0.04 * np.maximum(0, 1.05 - ev_ratio)).clip(0.01, 0.80)
    predicted_value = (10000 * ev_ratio * (0.6 + confidence)).round(0)

    result = pd.DataFrame(
        {
            "item_id": item_id,
            "predicted_value": predicted_value.astype(float),
            "ev_ratio": ev_ratio.astype(float),
            "ruin_probability": ruin_probability.astype(float),
            "confidence": confidence.astype(float),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return result.tail(max(MIN_ROWS, min(50, len(result)))).reset_index(drop=True)


def save_predictions(pred_df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(OUTPUT_PATH, index=False)
    pred_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Saved predictions: %s", OUTPUT_PATH)
    logger.info("Saved backup csv: %s", OUTPUT_CSV_PATH)


def run_batch() -> None:
    logger.info("Batch started")
    raw_df = load_latest_data()
    pred_df = run_inference(raw_df)
    if pred_df.empty:
        raise RuntimeError("Prediction result is empty")
    save_predictions(pred_df)
    logger.info("Batch finished | rows=%s", len(pred_df))


if __name__ == "__main__":
    run_batch()