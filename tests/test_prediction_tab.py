"""prediction_tab のテスト"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.ui.components.prediction_tab import (
    render_prediction_tab,
    load_optimization_results,
    recommend_ev_threshold,
)
from src.ui.results_loaders import load_latest_backtest_results

class TestPredictionTabHelpers:
    """Helper 関数のテスト"""
    
    def test_load_optimization_results_valid(self, tmp_path):
        """有効なJSONファイルをロード"""
        json_file = tmp_path / "strategy_optimization.json"
        json_file.write_text('{"aggressive": {"roi": 0.15}, "conservative": {"roi": 0.05}}')
        
        result = load_optimization_results(str(json_file))
        assert result is not None
        assert "aggressive" in result
        assert result["aggressive"]["roi"] == 0.15
    
    def test_load_optimization_results_invalid(self, tmp_path):
        """無効なJSONファイル"""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not json")
        
        result = load_optimization_results(str(json_file))
        assert result is None
    
    def test_load_optimization_results_missing(self, tmp_path):
        """ファイルが存在しない場合"""
        result = load_optimization_results(str(tmp_path / "nonexistent.json"))
        assert result is None
    
    def test_load_latest_backtest_results_valid(self, tmp_path):
        """有効なバックテスト結果（glob検索）"""
        csv_file = tmp_path / "backtest_results_20250219.csv"
        df = pd.DataFrame({
            "expected_value": [10.0, 20.0, 15.0],
            "roi": [0.05, 0.10, 0.08],
        })
        df.to_csv(csv_file, index=False)
        
        result = load_latest_backtest_results(str(tmp_path))
        assert result is not None
        assert len(result) == 3
        assert "expected_value" in result.columns
    
    def test_load_latest_backtest_results_missing(self, tmp_path):
        """ファイルが存在しない場合"""
        result = load_latest_backtest_results(str(tmp_path))
        assert result is None
    
    def test_recommend_ev_threshold_valid(self):
        """有効なデータで推奨EV閾値を計算"""
        df = pd.DataFrame({
            "expected_value": [5.0, 10.0, 15.0, 20.0],
        })
        threshold = recommend_ev_threshold(df)
        assert abs(threshold - 16.25) < 0.01  # 四分位数75%
    
    def test_recommend_ev_threshold_empty(self):
        """空のDataFrame"""
        df = pd.DataFrame({"expected_value": []})
        threshold = recommend_ev_threshold(df)
        assert threshold == 1.0
    
    def test_recommend_ev_threshold_missing_columns(self):
        """カラムが存在しない場合"""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        threshold = recommend_ev_threshold(df)
        assert threshold == 1.0


class TestRenderPredictionTab:
    """render_prediction_tab 関数のテスト"""
    
    @patch('src.ui.components.prediction_tab.st')
    def test_render_with_empty_dataframe(self, mock_st):
        """空のDataFrameでレンダリング"""
        df = pd.DataFrame()
        
        # mock_st.columns をセットアップ
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        # ボタンをクリック状態に（Empty チェック前提）
        mock_st.button.return_value = True
        mock_st.checkbox.return_value = False
        
        render_prediction_tab(df)
        
        # エラーメッセージが表示されることを確認
        mock_st.error.assert_called_with("データが空です")
    
    @patch('src.ui.components.prediction_tab.st')
    def test_render_with_valid_dataframe(self, mock_st):
        """有効なDataFrameでレンダリング"""
        df = pd.DataFrame({
            "result": [100, 200, 300],
            "round": [1, 2, 3],
        })
        
        # mock_st.columns を正しくセットアップ
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        # mock_st.button, st.checkbox などの戻り値を設定
        mock_st.button.return_value = False
        mock_st.checkbox.return_value = False
        
        # ヘッダーが呼び出されることを確認
        render_prediction_tab(df)
        mock_st.header.assert_called()
    
    @patch('src.ui.components.prediction_tab.st')
    def test_render_shows_parameters(self, mock_st):
        """パラメータスライダーが表示される"""
        df = pd.DataFrame({"result": [100, 200]})
        
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.checkbox.return_value = False
        
        render_prediction_tab(df)
        
        # slider が最低1回は呼び出されることを確認（最大購入点数など）
        assert mock_st.slider.call_count >= 3
