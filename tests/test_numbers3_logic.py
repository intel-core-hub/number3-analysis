import unittest

import pandas as pd

from numbers3_logic import _add_digit_columns, validate_numbers3


class TestNumbers3Logic(unittest.TestCase):
    def test_add_digit_columns(self):
        df = pd.DataFrame({"当選番号": ["007", "123", "999"]})
        result = _add_digit_columns(df)
        self.assertListEqual(result["digit_h"].tolist(), [0, 1, 9])
        self.assertListEqual(result["digit_t"].tolist(), [0, 2, 9])
        self.assertListEqual(result["digit_o"].tolist(), [7, 3, 9])

    def test_validate_numbers3_detects_invalid_number(self):
        df = pd.DataFrame({"回号": ["0001", "0002"], "当選番号": [None, "123"]})
        messages = validate_numbers3(df, strict=False, allow_missing_rounds=True)
        self.assertTrue(any("当選番号が3桁" in msg for msg in messages))

    def test_validate_numbers3_warns_missing_round(self):
        df = pd.DataFrame({"回号": ["0001", "0003"], "当選番号": ["123", "456"]})
        messages = validate_numbers3(df, strict=False, allow_missing_rounds=True)
        self.assertTrue(any("欠番" in msg for msg in messages))

    def test_validate_numbers3_detects_duplicate_round(self):
        df = pd.DataFrame({"回号": ["0001", "0001"], "当選番号": ["123", "456"]})
        messages = validate_numbers3(df, strict=False, allow_missing_rounds=True)
        self.assertTrue(any("回号の重複" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()
