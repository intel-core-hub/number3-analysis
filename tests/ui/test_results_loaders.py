"""
tests.ui.test_results_loaders — 結果ファイルローダーのテスト

責務:
    - バックテスト結果読み込み機能のテスト
    - EV閾値推奨機能のテスト
    - 特徴量重要度読み込みのテスト
    - 最適化結果読み込みのテスト
"""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.ui.results_loaders import (
    load_latest_backtest_results,
    recommend_ev_threshold,
    load_feature_importance,
    load_optimization_results,
    load_latest_selective_summary,
)


class TestLoadLatestBacktestResults:
    """バックテスト結果読み込みのテスト"""
    
    def test_load_existing_results(self, tmp_path):
        """既存の結果ファイルを読み込めるか"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # テスト用CSVを作成
        test_df = pd.DataFrame({
            "回号": [1, 2, 3],
            "actual": [123, 456, 789],
            "predicted": [100, 450, 800],
            "profit": [100, -50, 200],
        })
        csv_path = results_dir / "backtest_results_20240101.csv"
        test_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        # 読み込みテスト
        result = load_latest_backtest_results(str(results_dir))
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "profit" in result.columns
    
    def test_load_nonexistent_directory(self):
        """存在しないディレクトリの場合"""
        result = load_latest_backtest_results("/nonexistent/path/results")
        
        assert result is None
    
    def test_load_empty_directory(self, tmp_path):
        """空のディレクトリの場合"""
        results_dir = tmp_path / "empty_results"
        results_dir.mkdir()
        
        result = load_latest_backtest_results(str(results_dir))
        
        assert result is None
    
    def test_load_latest_among_multiple_files(self, tmp_path):
        """複数ファイルがある場合、最新を読み込む"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # 古いファイル
        old_df = pd.DataFrame({"data": [1, 2]})
        (results_dir / "backtest_results_20230101.csv").write_text(
            old_df.to_csv(index=False), encoding="utf-8-sig"
        )
        
        # 新しいファイル
        new_df = pd.DataFrame({"data": [3, 4, 5]})
        (results_dir / "backtest_results_20240101.csv").write_text(
            new_df.to_csv(index=False), encoding="utf-8-sig"
        )
        
        result = load_latest_backtest_results(str(results_dir))
        
        assert len(result) == 3


class TestRecommendEvThreshold:
    """EV閾値推奨機能のテスト"""
    
    def test_recommend_with_valid_data(self):
        """正常なデータでEV閾値を推奨"""
        df = pd.DataFrame({
            "expected_value": [0.5, 1.0, 1.5, 2.0, 2.5],
            "hit": [0, 0, 1, 1, 1],
            "profit": [-200, -100, 500, 800, 1000],
        })
        
        threshold = recommend_ev_threshold(df)
        
        assert isinstance(threshold, float)
        assert threshold >= 0
    
    def test_recommend_missing_columns(self):
        """必要なカラムがない場合"""
        df = pd.DataFrame({
            "other_column": [1, 2, 3],
        })
        
        threshold = recommend_ev_threshold(df)
        
        # デフォルト値を返す
        assert threshold == 1.05
    
    def test_recommend_empty_dataframe(self):
        """空のDataFrameの場合"""
        df = pd.DataFrame()
        
        threshold = recommend_ev_threshold(df)
        
        assert threshold == 1.05
    
    def test_recommend_all_losing_bets(self):
        """すべて負けている場合"""
        df = pd.DataFrame({
            "expected_value": [0.5, 1.0, 1.5],
            "hit": [0, 0, 0],
            "profit": [-200, -100, -300],
        })
        
        threshold = recommend_ev_threshold(df)
        
        # 高めの閾値を返すべき
        assert threshold >= 1.05


class TestLoadFeatureImportance:
    """特徴量重要度読み込みのテスト"""
    
    def test_load_existing_file(self, tmp_path):
        """既存のファイルを読み込めるか"""
        file_path = tmp_path / "feature_importance.csv"
        test_df = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "importance": [0.5, 0.3, 0.2],
        })
        test_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        
        result = load_feature_importance(str(file_path))
        
        assert result is not None
        assert len(result) == 3
        assert "feature" in result.columns
    
    def test_load_nonexistent_file(self):
        """存在しないファイルの場合"""
        result = load_feature_importance("/nonexistent/file.csv")
        
        assert result is None


class TestLoadOptimizationResults:
    """最適化結果読み込みのテスト"""
    
    def test_load_valid_json(self, tmp_path):
        """正常なJSONファイルを読み込めるか"""
        file_path = tmp_path / "optimization_results.json"
        test_data = {
            "best_params": {"alpha": 0.5, "beta": 0.3},
            "best_score": 0.85,
            "iterations": 100,
        }
        file_path.write_text(json.dumps(test_data), encoding="utf-8")
        
        result = load_optimization_results(str(file_path))
        
        assert result is not None
        assert result["best_score"] == 0.85
        assert "best_params" in result
    
    def test_load_invalid_json(self, tmp_path):
        """無効なJSONファイルの場合"""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not a json", encoding="utf-8")
        
        result = load_optimization_results(str(file_path))
        
        assert result is None
    
    def test_load_nonexistent_file(self):
        """存在しないファイルの場合"""
        result = load_optimization_results("/nonexistent/optimization.json")
        
        assert result is None


class TestLoadLatestSelectiveSummary:
    """最新selective_summary読み込みのテスト"""
    
    def test_load_existing_summary(self, tmp_path):
        """既存のsummaryファイルを読み込めるか"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        test_df = pd.DataFrame({
            "model": ["model1", "model2"],
            "accuracy": [0.85, 0.90],
            "profit": [1000, 1500],
        })
        csv_path = results_dir / "selective_summary_20240101.csv"
        test_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        result = load_latest_selective_summary(str(results_dir))
        
        assert result is not None
        assert len(result) == 2
        assert "accuracy" in result.columns
    
    def test_load_from_nonexistent_directory(self):
        """存在しないディレクトリの場合"""
        result = load_latest_selective_summary("/nonexistent/directory")
        
        assert result is None
    
    def test_load_latest_among_multiple_summaries(self, tmp_path):
        """複数のsummaryファイルから最新を選択"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # 古いファイル
        old_df = pd.DataFrame({"data": [1]})
        (results_dir / "selective_summary_20230101.csv").write_text(
            old_df.to_csv(index=False), encoding="utf-8-sig"
        )
        
        # 新しいファイル
        new_df = pd.DataFrame({"data": [2, 3]})
        (results_dir / "selective_summary_20240101.csv").write_text(
            new_df.to_csv(index=False), encoding="utf-8-sig"
        )
        
        result = load_latest_selective_summary(str(results_dir))
        
        assert len(result) == 2
