"""
tests.ui.test_analysis_helpers — 分析ヘルパー関数のテスト

責務:
    - ストレステスト実行のテスト
    - レジーム検出のテスト
    - エクイティ指標計算のテスト
"""
import pytest
import pandas as pd
import numpy as np

from src.ui.analysis_helpers import (
    compute_equity_metrics,
)


class TestComputeEquityMetrics:
    """エクイティ指標計算のテスト"""
    
    def test_compute_with_positive_profits(self):
        """プラスの利益データで計算"""
        df = pd.DataFrame({
            "profit": [100, 200, 150, 300, 250],
        })
        
        result = compute_equity_metrics(df)
        
        assert result is not None
        assert "total_profit" in result
        assert "max_drawdown" in result
        assert "sharpe_ratio" in result
        assert result["total_profit"] == 1000
        assert result["max_drawdown"] <= 0  # ドローダウンは負の値
    
    def test_compute_with_negative_profits(self):
        """マイナスの利益データで計算"""
        df = pd.DataFrame({
            "profit": [-100, -200, -150, -300, -250],
        })
        
        result = compute_equity_metrics(df)
        
        assert result is not None
        assert result["total_profit"] == -1000
        assert result["max_drawdown"] < 0
    
    def test_compute_with_mixed_profits(self):
        """プラスとマイナスが混在"""
        df = pd.DataFrame({
            "profit": [100, -50, 200, -100, 150],
        })
        
        result = compute_equity_metrics(df)
        
        assert result is not None
        assert result["total_profit"] == 300
    
    def test_compute_empty_dataframe(self):
        """空のDataFrameの場合"""
        df = pd.DataFrame()
        
        result = compute_equity_metrics(df)
        
        # 空の場合はデフォルト値を返す
        assert result is not None
        assert result["total_profit"] == 0
        assert result["max_drawdown"] == 0
    
    def test_compute_missing_profit_column(self):
        """profitカラムが存在しない場合"""
        df = pd.DataFrame({
            "other_column": [1, 2, 3],
        })
        
        result = compute_equity_metrics(df)
        
        # エラーハンドリング: デフォルト値を返す
        assert result is not None
        assert result["total_profit"] == 0
    
    def test_compute_single_trade(self):
        """単一の取引データの場合"""
        df = pd.DataFrame({
            "profit": [500],
        })
        
        result = compute_equity_metrics(df)
        
        assert result is not None
        assert result["total_profit"] == 500
        # ドローダウンは1件だけなので0
        assert result["max_drawdown"] == 0
    
    def test_compute_with_zero_profits(self):
        """利益がすべて0の場合"""
        df = pd.DataFrame({
            "profit": [0, 0, 0, 0],
        })
        
        result = compute_equity_metrics(df)
        
        assert result is not None
        assert result["total_profit"] == 0
        assert result["max_drawdown"] == 0
        # シャープレシオは計算不可（標準偏差が0）
        assert np.isnan(result["sharpe_ratio"]) or result["sharpe_ratio"] == 0
    
    def test_compute_sharpe_ratio_calculation(self):
        """シャープレシオが正しく計算されるか"""
        # 平均リターン=100, 標準偏差=50
        df = pd.DataFrame({
            "profit": [50, 100, 150, 100, 100],
        })
        
        result = compute_equity_metrics(df)
        
        assert result is not None
        assert "sharpe_ratio" in result
        # シャープレシオ = 平均 / 標準偏差 (リスクフリーレート無視)
        # 平均=100, std≈35.36, sharpe≈2.83
        assert result["sharpe_ratio"] > 0
    
    def test_compute_max_drawdown_calculation(self):
        """最大ドローダウンが正しく計算されるか"""
        df = pd.DataFrame({
            "profit": [100, 200, -300, 150, -100],  # 累積: 100, 300, 0, 150, 50
        })
        
        result = compute_equity_metrics(df)
        
        assert result is not None
        # 300 → 0 の下落が最大ドローダウン(-300)
        assert result["max_drawdown"] <= -200
    
    def test_compute_with_nan_values(self):
        """NaN値を含むデータの場合"""
        df = pd.DataFrame({
            "profit": [100, np.nan, 200, 150],
        })
        
        result = compute_equity_metrics(df)
        
        # NaNを除外して計算
        assert result is not None
        assert result["total_profit"] == 450  # 100+200+150
    
    def test_compute_return_structure(self):
        """返り値の構造が正しいか"""
        df = pd.DataFrame({
            "profit": [100, 200, 150],
        })
        
        result = compute_equity_metrics(df)
        
        # 必須キーの存在確認
        assert "total_profit" in result
        assert "max_drawdown" in result
        assert "sharpe_ratio" in result
        assert "win_rate" in result
        assert "profit_factor" in result
        
        # 型の確認
        assert isinstance(result["total_profit"], (int, float))
        assert isinstance(result["max_drawdown"], (int, float))
        assert isinstance(result["sharpe_ratio"], (int, float)) or np.isnan(result["sharpe_ratio"])
        assert isinstance(result["win_rate"], (int, float))
        assert isinstance(result["profit_factor"], (int, float)) or np.isnan(result["profit_factor"])
