from datetime import datetime, timedelta

import pandas as pd

from numbers3_logic import Numbers3MLPredictor


def _make_sample_df(count=80):
    start = datetime(2024, 1, 1)
    rows = []
    for i in range(count):
        draw_date = start + timedelta(days=i)
        num = f"{i % 10}{(i * 3) % 10}{(i * 7) % 10}"
        rows.append(
            {
                "回号": str(1000 + i),
                "抽せん日": draw_date.strftime("%Y-%m-%d"),
                "当選番号": num,
            }
        )
    return pd.DataFrame(rows)


def test_ml_predictor_train_and_predict():
    df = _make_sample_df()
    predictor = Numbers3MLPredictor(df, num_boost_round=10, random_state=0)
    predictor.train()
    result = predictor.predict_next()
    assert isinstance(result, str)
    assert len(result) == 3
    assert result.isdigit()


def test_ml_predictor_tune_hyperparams():
    df = _make_sample_df(count=120)
    predictor = Numbers3MLPredictor(df, num_boost_round=5, random_state=1)
    result = predictor.tune_hyperparams(
        param_grid={
            "learning_rate": [0.1],
            "num_leaves": [15, 31],
            "max_depth": [-1],
            "min_child_samples": [10],
        },
        valid_size=20,
    )
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "valid_logloss" in result
