"""
src.models.predictor - re-export bridge

Implementation split into:
    src.models.base        -- utility & abstract base
    src.models.statistical -- Numbers3Predictor
    src.models.ml_wrapper  -- Numbers3MLPredictor
    src.analysis.backtester -- Numbers3Backtester / Numbers3MLBacktester
"""
from __future__ import annotations

# --- base ---
from src.models.base import (        # noqa: F401
    evaluate_set_profit,
    box_key as _box_key,
    box_type as _box_type,
)

# --- config ---
from src.utils.config import (       # noqa: F401
    SET_STRAIGHT_PRIZE,
    SET_BOX_PRIZE,
    TICKET_COST,
    PREDICTION_MODELS,
)

# --- statistical ---
from src.models.statistical import Numbers3Predictor  # noqa: F401

# --- ml_wrapper ---
from src.models.ml_wrapper import (  # noqa: F401
    Numbers3MLPredictor,
    extract_feature_importance,
)

# --- backtester ---
from src.analysis.backtester import (  # noqa: F401
    Numbers3Backtester,
    Numbers3MLBacktester,
)
