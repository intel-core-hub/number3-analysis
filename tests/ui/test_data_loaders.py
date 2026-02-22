"""
tests.ui.test_data_loaders — データローダーのテスト

責務:
    - データベース読み込み機能のテスト
    - 最新回号情報取得のテスト
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.ui.data_loaders import (
    load_draws,
    latest_round_info,
)
from src.database import Numbers3Database


class TestLoadDraws:
    """データベース読み込み機能のテスト"""
    
    def test_load_draws_returns_dataframe(self, tmp_path):
        """データベースから抽選履歴を読み込めるか"""
        db_path = tmp_path / "test.db"
        db = Numbers3Database(str(db_path))
        
        # テストデータを挿入
        test_data = pd.DataFrame({
            "回号": [1, 2, 3],
            "当選番号": ["123", "456", "789"],
            "抽せん日": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        db.upsert_draws(test_data)
        
        # 読み込みテスト
        result = load_draws(db)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "回号" in result.columns
        assert "当選番号" in result.columns
    
    def test_load_draws_empty_database(self, tmp_path):
        """空のデータベースから読み込んだ場合"""
        db_path = tmp_path / "empty.db"
        db = Numbers3Database(str(db_path))
        
        result = load_draws(db)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestLatestRoundInfo:
    """最新回号情報取得のテスト"""
    
    def test_latest_round_info_with_valid_data(self):
        """正常なデータから最新回号と日付を取得"""
        df = pd.DataFrame({
            "回号": [1, 2, 3, 4, 5],
            "当選番号": ["123", "456", "789", "012", "345"],
            "抽せん日": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03", 
                "2024-01-04", "2024-01-05"
            ]),
        })
        
        latest_round, latest_date = latest_round_info(df)
        
        assert latest_round == 5
        assert latest_date == pd.Timestamp("2024-01-05")
    
    def test_latest_round_info_without_date_column(self):
        """抽せん日カラムがない場合"""
        df = pd.DataFrame({
            "回号": [1, 2, 3],
            "当選番号": ["123", "456", "789"],
        })
        
        latest_round, latest_date = latest_round_info(df)
        
        assert latest_round == 3
        assert latest_date is None
    
    def test_latest_round_info_empty_dataframe(self):
        """空のDataFrameの場合"""
        df = pd.DataFrame()
        
        latest_round, latest_date = latest_round_info(df)
        
        assert latest_round is None
        assert latest_date is None
    
    def test_latest_round_info_invalid_round_numbers(self):
        """無効な回号データの場合"""
        df = pd.DataFrame({
            "回号": ["invalid", None, "abc"],
            "当選番号": ["123", "456", "789"],
        })
        
        latest_round, latest_date = latest_round_info(df)
        
        assert latest_round is None
        assert latest_date is None
    
    def test_latest_round_info_mixed_valid_invalid(self):
        """有効・無効な回号が混在する場合"""
        df = pd.DataFrame({
            "回号": [1, "invalid", 3, None, 5],
            "当選番号": ["123", "456", "789", "012", "345"],
            "抽せん日": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03", 
                "2024-01-04", "2024-01-05"
            ]),
        })
        
        latest_round, latest_date = latest_round_info(df)
        
        # 有効な回号の最大値を返す
        assert latest_round == 5
        assert isinstance(latest_date, pd.Timestamp)
