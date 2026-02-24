"""
src.config - central configuration for the refactored workflow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    from src.utils.config import PREDICTION_MODELS as _LEGACY_PREDICTION_MODELS
except Exception:
    _LEGACY_PREDICTION_MODELS = {
        "hybrid": "hybrid",
    }

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "numbers3.db"
MODEL_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR = MODEL_DIR  # Phase 31: Alias for backward compatibility
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_predictor.pkl"

CSV_FALLBACK_PATH = PROJECT_ROOT / "numbers3_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

PREDICTION_MODELS: dict[str, str] = dict(_LEGACY_PREDICTION_MODELS)

RAW_COLUMNS: list[str] = [
    "回号",
    "抽せん日",
    "当選番号",
    "ストレート",
    "ボックス",
    "セット（ストレート）",
    "セット（ボックス）",
    "ミニ",
    "販売実績額",
]

COLUMN_MAP: dict[str, str] = {
    "回号": "round_no",
    "抽せん日": "draw_date",
    "当選番号": "winning_number",
    "ストレート": "straight",
    "ボックス": "box",
    "セット（ストレート）": "set_straight",
    "セット（ボックス）": "set_box",
    "ミニ": "mini",
    "販売実績額": "sales",
}

REVERSE_COLUMN_MAP: dict[str, str] = {v: k for k, v in COLUMN_MAP.items()}

DEFAULT_SCRAPE_SLEEP_SECONDS: float = 1.0
DEFAULT_SCRAPE_START_YEAR = 1994
DEFAULT_SCRAPE_START_MONTH = 10
DEFAULT_SCRAPE_START_DAY = 1
