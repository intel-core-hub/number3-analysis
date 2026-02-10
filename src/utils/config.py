"""
src.utils.config — 一元管理された定数・設定

責務:
    - CSV パス / バックアップパス
    - 賞金額・チケットコスト
    - LightGBM デフォルトハイパーパラメータ
    - スクレイピング URL テンプレート
    - 特徴量ウィンドウサイズ
"""
from __future__ import annotations

from typing import Any, Dict, List

# =====================================================================
# パス
# =====================================================================

CLEAN_CSV_PATH: str = "numbers3_clean.csv"
PERFORMANCE_CSV_PATH: str = "results/ml_performance_tracking.csv"
FEATURE_IMPORTANCE_CSV_PATH: str = "results/feature_importance.csv"
RESULTS_DIR: str = "results"

# =====================================================================
# 賞金・コスト
# =====================================================================

SET_STRAIGHT_PRIZE: int = 37_500
SET_BOX_PRIZE: int = 15_000
TICKET_COST: int = 200

# ストレート / ボックス / ミニ (参考)
STRAIGHT_PRIZE: int = 75_000
BOX_PRIZE_SINGLE: int = 12_500      # シングル (6通り)
BOX_PRIZE_DOUBLE: int = 37_500       # ダブル (3通り)
MINI_PRIZE: int = 7_500

# =====================================================================
# スクレイピング
# =====================================================================

DEFAULT_SOURCE_TEMPLATES: List[str] = [
    "https://takarakuji.rakuten.co.jp/backnumber/numbers3/{yyyymm}/",
]

SCRAPING_SLEEP_SECONDS: float = 1.0
SCRAPING_TIMEOUT: int = 10
SCRAPING_RETRIES: int = 3
SCRAPING_BACKOFF: float = 1.5

# =====================================================================
# 特徴量ウィンドウ
# =====================================================================

WINDOW_SHORT: int = 5
WINDOW_LONG: int = 10
WINDOW_MID: int = 50
WINDOW_FE_DEFAULT: int = 20
ROLLING_FREQ_WINDOW: int = 20

# =====================================================================
# LightGBM デフォルトハイパーパラメータ
# =====================================================================

DEFAULT_LGB_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
}

DEFAULT_NUM_BOOST_ROUND: int = 120
DEFAULT_VALID_SIZE: int = 180
EARLY_STOPPING_ROUNDS: int = 30

# =====================================================================
# Optuna
# =====================================================================

OPTUNA_N_TRIALS: int = 30
OPTUNA_CV_FOLDS: int = 5

# =====================================================================
# 列名
# =====================================================================

COLUMNS: List[str] = [
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
PAYOUT_COLUMNS: List[str] = COLUMNS[3:]

ALIAS_MAP: Dict[str, str] = {
    "当せん番号": "当選番号",
    "抽選日": "抽せん日",
}

# =====================================================================
# 予測モデルラベル
# =====================================================================

PREDICTION_MODELS: Dict[str, str] = {
    "ハイブリッド(遷移+合計+ハマリ)": "hybrid",
    "遷移+合計": "transition_sum",
    "ハマリ+合計": "interval_sum",
    "合計のみ": "sum_only",
    "頻度+合計": "frequency",
    "直近n回(頻度+合計)": "recent_frequency",
    "ハマリ強調(合計+ハマリ)": "interval_boost",
}
