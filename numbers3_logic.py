"""
numbers3_logic.py - backward-compatible bridge

All implementation lives under src/ submodules.
This file re-exports every public symbol for backward compatibility.
"""

# === Utils ===
from src.utils.config import (                                  # noqa: F401
    COLUMNS,
    PAYOUT_COLUMNS,
    ALIAS_MAP,
    PREDICTION_MODELS,
    SET_STRAIGHT_PRIZE,
    SET_BOX_PRIZE,
    TICKET_COST,
)

# === Data / Loader ===
from src.data.loader import (                                   # noqa: F401
    normalize_numbers3_columns,
    add_digit_columns as _add_digit_columns,
    validate_numbers3,
)

# === Data / Fetcher ===
from src.data.fetcher import (                                  # noqa: F401
    fetch_numbers3_by_month,
)

# === Data / Loader (high-level) ===
from src.data.loader import (                                   # noqa: F401
    update_numbers3_clean,
    backfill_numbers3_range,
)

# === Features / Engineer ===
from src.features.engineer import (                             # noqa: F401
    compute_common_features as _compute_common_features,
    Numbers3FeatureEngineer,
    ml_feature_columns,
    add_enhanced_ml_features,
    compute_all_ml_features,
)

# === Models / Base ===
from src.models.base import (                                   # noqa: F401
    evaluate_set_profit,
    box_key as _box_key,
    box_type as _box_type,
)

# === Models / Statistical ===
from src.models.statistical import Numbers3Predictor            # noqa: F401

# === Models / ML Wrapper ===
from src.models.ml_wrapper import (                             # noqa: F401
    Numbers3MLPredictor,
    extract_feature_importance,
)

# === Analysis / Backtester ===
from src.analysis.backtester import (                           # noqa: F401
    Numbers3Backtester,
    Numbers3MLBacktester,
    simulate_revenue,
    max_drawdown,
)

# === Analysis / Visualizer ===
from src.analysis.visualizer import (                           # noqa: F401
    configure_japanese_fonts,
    Numbers3IntervalAnalyzer,
    Numbers3TrendAnalyzer,
    Numbers3PatternAnalyzer,
    plot_feature_importance,
    plot_feature_importance_per_digit,
    plot_accuracy_trend,
    plot_cumulative_profit_plotly,
    plot_hit_miss_heatmap_plotly,
    plot_error_distribution_plotly,
    plot_weekday_accuracy,
    run_pattern_accuracy_test,
)

# pandas / numpy
import numpy as np                                              # noqa: F401
import pandas as pd                                             # noqa: F401
