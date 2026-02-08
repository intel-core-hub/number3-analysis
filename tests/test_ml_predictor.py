import os
import pytest
import pandas as pd

from numbers3_logic import Numbers3MLPredictor


def test_ml_predictor_smoke():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'numbers3_clean.csv')
    csv_path = os.path.normpath(csv_path)
    if not os.path.exists(csv_path):
        pytest.skip("numbers3_clean.csv が見つかりません。自動更新を先に実行してください。")

    df = pd.read_csv(csv_path)
    if len(df) < 60:
        pytest.skip("データ件数が少ないため ML 学習テストをスキップします。")

    ml = Numbers3MLPredictor(df, num_boost_round=20)
    ml.train()
    pred = ml.predict_next()

    assert isinstance(pred, str)
    assert len(pred) == 3
    assert pred.isdigit()
