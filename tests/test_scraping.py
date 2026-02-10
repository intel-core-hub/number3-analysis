import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from numbers3_logic import fetch_numbers3_by_month


class TestScraping(unittest.TestCase):
    def test_fetch_with_fallback_source(self):
        tables = [
            pd.DataFrame({
                "回号": ["0001"],
                "抽せん日": ["2020-01-01"],
                "当選番号": ["123"],
            })
        ]

        def fake_read_html(url, timeout=10, retries=3, backoff=1.5):
            if "primary" in url:
                raise ValueError("primary failed")
            return tables

        with patch("src.data.fetcher._read_html_tables", side_effect=fake_read_html):
            df, failures = fetch_numbers3_by_month(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
                source_templates=["http://primary/{yyyymm}/", "http://backup/{yyyymm}/"],
                fail_fast=False,
            )

        self.assertEqual(len(df), 1)
        self.assertEqual(failures, [])

    def test_fetch_records_failure(self):
        def fake_read_html(url, timeout=10, retries=3, backoff=1.5):
            raise ValueError("boom")

        with patch("src.data.fetcher._read_html_tables", side_effect=fake_read_html):
            df, failures = fetch_numbers3_by_month(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
                source_templates=["http://primary/{yyyymm}/"],
                fail_fast=False,
            )

        self.assertEqual(len(df), 0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["month"], "202001")


if __name__ == "__main__":
    unittest.main()
