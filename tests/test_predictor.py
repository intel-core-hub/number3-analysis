import unittest

import pandas as pd

from numbers3_logic import Numbers3Predictor


class TestPredictor(unittest.TestCase):
    def test_predict_returns_ranked_numbers(self):
        df = pd.DataFrame({
            "回号": ["0001", "0002", "0003", "0004", "0005", "0006"],
            "当選番号": ["123", "456", "789", "012", "345", "678"],
        })
        predictor = Numbers3Predictor(df)
        result = predictor.predict(top_n=5, model="sum_only", verbose=False)
        self.assertEqual(len(result), 5)
        self.assertIn("予測番号", result.columns)
        self.assertIn("総合スコア", result.columns)


if __name__ == "__main__":
    unittest.main()
